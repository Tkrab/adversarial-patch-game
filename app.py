"""
Flask patch placement + ResNet34 classification (slim demo).
Drag a pretrained adversarial patch onto a base image and classify.
"""
from __future__ import annotations

import json
import os
import random
import urllib.request
import uuid
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from flask import Flask, jsonify, render_template, request, session
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths & bootstrap downloads
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
BASE_IMAGE_DIR = STATIC_DIR / "image"
PATCH_PREVIEW_DIR = STATIC_DIR / "Patches"
RESULTS_DIR = STATIC_DIR / "results"
DATASET_PATH = BASE_DIR / "data"
CHECKPOINT_PATH = BASE_DIR / "saved_models" / "tutorial10"

# Prefer existing assets from the reference project when present.
REF_ROOT = Path(r"D:\project\Patch-Game-Codebase")
REF_DATA = REF_ROOT / "data"
REF_CKPT = REF_ROOT / "saved_models" / "tutorial10"

DOWNLOAD_BASE = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial10/"
PRETRAINED_FILES = [
    (DATASET_PATH, "TinyImageNet.zip"),
    (CHECKPOINT_PATH, "patches.zip"),
]

CLASS_NAMES = ["toaster", "goldfish", "school bus", "lipstick", "pineapple"]
PATCH_SIZES = [32, 48, 64]
IMAGE_SIZE = 224
OFFERED_PATCH_COUNT = 6
MAX_SUBMISSIONS = 3
TARGET_SCORE = 95.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])
TENSOR_MEANS = torch.FloatTensor(NORM_MEAN)[:, None, None]
TENSOR_STD = torch.FloatTensor(NORM_STD)[:, None, None]


def ensure_dirs() -> None:
    for d in (
        STATIC_DIR,
        BASE_IMAGE_DIR,
        PATCH_PREVIEW_DIR,
        RESULTS_DIR,
        DATASET_PATH,
        CHECKPOINT_PATH,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _copy_tree_if_needed(src: Path, dst: Path) -> None:
    """Reuse reference assets by copying missing files (shallow for zips/json)."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            continue
        if item.is_file():
            target.write_bytes(item.read_bytes())
        elif item.is_dir():
            # For TinyImageNet image folders, symlink/junction would be ideal;
            # fall back to letting download populate if images are missing.
            pass


def ensure_assets() -> None:
    """Reuse reference project assets when possible, otherwise download."""
    ensure_dirs()
    _copy_tree_if_needed(REF_DATA, DATASET_PATH)
    _copy_tree_if_needed(REF_CKPT, CHECKPOINT_PATH)

    # Copy TinyImageNet label file if we only have that.
    ref_labels = REF_DATA / "TinyImageNet" / "label_list.json"
    local_imagenet = DATASET_PATH / "TinyImageNet"
    local_imagenet.mkdir(parents=True, exist_ok=True)
    if ref_labels.is_file() and not (local_imagenet / "label_list.json").is_file():
        (local_imagenet / "label_list.json").write_bytes(ref_labels.read_bytes())

    for dir_path, file_name in PRETRAINED_FILES:
        file_path = dir_path / file_name
        if not file_path.is_file():
            url = DOWNLOAD_BASE + file_name
            print(f"Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, str(file_path))
            except HTTPError as exc:
                raise RuntimeError(
                    f"Failed to download {url}. "
                    f"Place the file manually at {file_path}.\n{exc}"
                ) from exc

        # Extract zip when contents are missing.
        if file_name == "TinyImageNet.zip":
            marker = dir_path / "TinyImageNet" / "label_list.json"
            # Also need actual images; ImageFolder needs class subdirs.
            has_images = any(
                p.is_file()
                for p in (dir_path / "TinyImageNet").rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPEG", ".JPG"}
            )
            if not has_images:
                print(f"Extracting {file_path} ...")
                with zipfile.ZipFile(file_path, "r") as zf:
                    zf.extractall(dir_path)
            elif not marker.is_file():
                print(f"Extracting {file_path} (labels) ...")
                with zipfile.ZipFile(file_path, "r") as zf:
                    zf.extractall(dir_path)

        if file_name == "patches.zip":
            expected = dir_path / "toaster_32_patch.pt"
            if not expected.is_file():
                # May already be extracted under a subfolder.
                found = list(dir_path.rglob("toaster_32_patch.pt"))
                if not found:
                    print(f"Extracting {file_path} ...")
                    with zipfile.ZipFile(file_path, "r") as zf:
                        zf.extractall(dir_path)


# ---------------------------------------------------------------------------
# Model & data
# ---------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ensure_assets()
os.environ["TORCH_HOME"] = str(CHECKPOINT_PATH)

pretrained_model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
pretrained_model = pretrained_model.to(device)
pretrained_model.eval()
for p in pretrained_model.parameters():
    p.requires_grad = False

plain_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
)

imagenet_path = DATASET_PATH / "TinyImageNet"
assert imagenet_path.is_dir(), f"TinyImageNet not found at {imagenet_path}"

with open(imagenet_path / "label_list.json", "r", encoding="utf-8") as f:
    label_names = json.load(f)


# ---------------------------------------------------------------------------
# Base image helpers (static/image)
# ---------------------------------------------------------------------------
def list_base_image_files() -> list[Path]:
    """Return image files under static/image."""
    BASE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p
        for p in BASE_IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No images found in {BASE_IMAGE_DIR}. "
            f"Add .jpg/.png files to static/image/."
        )
    return files


def load_base_image_tensor(image_path: Path) -> torch.Tensor:
    """Load an image, resize to 224x224, return normalized CHW tensor."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = transforms.functional.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        return plain_transforms(img)


def predict_image_label(img_tensor: torch.Tensor) -> int:
    with torch.no_grad():
        logits = pretrained_model(img_tensor.unsqueeze(0).to(device))
        return int(logits.argmax(dim=-1).item())


def choose_random_base_image(
    exclude_relpaths: set[str] | None = None,
    last_relpath: str | None = None,
) -> tuple[torch.Tensor, int, str]:
    """
    Pick a random image from static/image, infer its class, write static/base.png.
    Excludes paths in exclude_relpaths. If no candidates remain, avoids last_relpath
    when possible so the same image is not picked twice in a row.
    Returns (tensor, label_index, relpath under static/).
    """
    files = list_base_image_files()

    def to_relpath(path: Path) -> str:
        return path.relative_to(STATIC_DIR).as_posix()

    exclude = set(exclude_relpaths or ())
    pool = [p for p in files if to_relpath(p) not in exclude]

    if not pool and last_relpath and len(files) > 1:
        pool = [p for p in files if to_relpath(p) != last_relpath]

    if not pool:
        pool = files

    image_path = random.choice(pool)
    tensor = load_base_image_tensor(image_path)
    label = predict_image_label(tensor)
    save_base_image(tensor, STATIC_DIR / "base.png")
    relpath = to_relpath(image_path)
    print(f"Base image: {relpath}, class: {label_names[label]}")
    return tensor, label, relpath


def set_session_base_image() -> tuple[int, str]:
    """Pick a random base image (not used before in this session) and store in session."""
    used = set(session.get("used_base_images", []))
    current = session.get("base_image_relpath")
    exclude = used | ({current} if current else set())

    _, label, relpath = choose_random_base_image(
        exclude_relpaths=exclude,
        last_relpath=current,
    )

    used.add(relpath)
    session["used_base_images"] = list(used)
    session["base_image_relpath"] = relpath
    session["selected_label"] = label
    version = uuid.uuid4().hex[:8]
    session["base_version"] = version
    return label, version


def load_session_base_image() -> tuple[torch.Tensor, int]:
    """Reload the current session base image tensor and label."""
    relpath = session.get("base_image_relpath")
    label = session.get("selected_label")
    if not relpath or label is None:
        raise RuntimeError("No base image in session. Visit /play first.")
    image_path = STATIC_DIR / relpath
    if not image_path.is_file():
        raise FileNotFoundError(f"Session base image not found: {image_path}")
    return load_base_image_tensor(image_path), int(label)


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------
def patch_forward(patch: torch.Tensor) -> torch.Tensor:
    """Map patch parameters from (-inf, inf) into ImageNet-normalized pixel space."""
    return (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)


def place_patch_with_position(
    img: torch.Tensor, patch: torch.Tensor, position: tuple[int, int]
) -> torch.Tensor:
    """Overlay patch onto a single CHW image at (y, x)."""
    img = img.clone()
    y_offset, x_offset = position
    max_y = img.shape[1] - patch.shape[1]
    max_x = img.shape[2] - patch.shape[2]
    y_offset = max(0, min(y_offset, max_y))
    x_offset = max(0, min(x_offset, max_x))
    img[:, y_offset : y_offset + patch.shape[1], x_offset : x_offset + patch.shape[2]] = (
        patch_forward(patch)
    )
    return img


def find_patch_file(class_name: str, patch_size: int) -> Path:
    candidates = [
        f"{class_name}_{patch_size}_patch.pt",
        f"{class_name.replace(' ', '_')}_{patch_size}_patch.pt",
    ]
    for candidate in candidates:
        direct = CHECKPOINT_PATH / candidate
        if direct.is_file():
            return direct
    for path in CHECKPOINT_PATH.rglob("*_patch.pt"):
        if path.name in candidates:
            return path
    raise FileNotFoundError(
        f"Could not find patch for class={class_name!r}, size={patch_size} under {CHECKPOINT_PATH}"
    )


def normalise_loaded_patch(patch, file_name: str) -> torch.Tensor:
    if isinstance(patch, dict):
        for key in ("patch", "tensor", "state_dict"):
            if key in patch:
                patch = patch[key]
                break
        else:
            raise ValueError(f"Unknown patch dict in {file_name}: {list(patch.keys())}")
    if not isinstance(patch, torch.Tensor):
        raise TypeError(f"Patch file did not contain a Tensor: {file_name}")
    if patch.dim() == 4 and patch.shape[0] == 1:
        patch = patch.squeeze(0)
    if patch.dim() != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(patch.shape)} from {file_name}")
    return patch.detach().cpu().float()


def get_patches(class_names, patch_sizes):
    result = {}
    for name in class_names:
        result[name] = {}
        for patch_size in patch_sizes:
            file_path = find_patch_file(name, patch_size)
            print(f"Loading patch: {file_path}")
            raw = torch.load(file_path, map_location="cpu", weights_only=False)
            patch = normalise_loaded_patch(raw, str(file_path))
            result[name][patch_size] = {"patch": patch}
    return result


def tensor_to_display_np(img: torch.Tensor) -> np.ndarray:
    if img.dim() == 4:
        img = img.squeeze(0)
    arr = img.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * NORM_STD[None, None]) + NORM_MEAN[None, None]
    return np.clip(arr, 0.0, 1.0)


def save_base_image(img_tensor: torch.Tensor, filename: Path) -> None:
    arr = tensor_to_display_np(img_tensor)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr)
    ax.axis("off")
    fig.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)


def save_prediction_figure(
    img: torch.Tensor, label: int, pred: torch.Tensor, filename: Path, k: int = 5
) -> None:
    arr = tensor_to_display_np(img)
    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(k, dim=-1)
    topk_vals = topk_vals.cpu().numpy().squeeze(0)
    topk_idx = topk_idx.cpu().numpy().squeeze(0)
    if isinstance(label, torch.Tensor):
        label = int(label.item())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1]})
    ax[0].imshow(arr)
    ax[0].set_title(label_names[label] if 0 <= label < len(label_names) else str(label))
    ax[0].axis("off")

    colors = ["C2" if int(topk_idx[i]) == label else "C0" for i in range(k)]
    ax[1].barh(np.arange(k), topk_vals * 100.0, align="center", color=colors)
    ax[1].set_yticks(np.arange(k))
    ax[1].set_yticklabels([label_names[int(c)] for c in topk_idx])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Confidence (%)")
    ax[1].set_title("Top-5 Predictions")
    fig.savefig(filename, bbox_inches="tight", dpi=120)
    plt.close(fig)


def export_patch_previews(patch_dict) -> None:
    """Export RGB preview PNGs from .pt patches for the drag UI."""
    PATCH_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for cname, sizes in patch_dict.items():
        safe = cname.replace(" ", "_")
        for psize, entry in sizes.items():
            patch = entry["patch"]
            rgb = (torch.tanh(patch) + 1) / 2
            rgb = rgb.cpu().permute(1, 2, 0).numpy()
            rgb = np.clip(rgb, 0.0, 1.0)
            out = PATCH_PREVIEW_DIR / f"{safe}_{psize}.png"
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(rgb)
            ax.axis("off")
            fig.savefig(out, bbox_inches="tight", pad_inches=0, dpi=80)
            plt.close(fig)


def get_topk_with_target(model, img: torch.Tensor, target_class: str, k: int = 5):
    batch = img.unsqueeze(0) if img.dim() == 3 else img
    with torch.no_grad():
        logits = model(batch.to(device))
        probs = torch.softmax(logits, dim=-1)
    topk_vals, topk_idxs = torch.topk(probs, k, dim=-1)
    topk_idxs = topk_idxs.cpu().numpy()[0]
    topk_vals = topk_vals.cpu().numpy()[0]
    top5 = [
        {"label": label_names[int(idx)], "confidence": float(val)}
        for idx, val in zip(topk_idxs, topk_vals)
    ]
    target_confidence = 0.0
    for item in top5:
        if item["label"] == target_class:
            target_confidence = item["confidence"]
            break
    # Also check full distribution if not in top-5
    if target_confidence == 0.0 and target_class in label_names:
        tidx = label_names.index(target_class)
        target_confidence = float(probs[0, tidx].cpu().item())
    return top5, target_confidence, probs


def size_bonus(patch_size: int) -> float:
    """Extra points by patch size: 32→+20, 48→+10, 64→+0."""
    return {32: 20.0, 48: 10.0, 64: 0.0}.get(int(patch_size), 0.0)


def update_score(
    patch_confidence: float,
    base_confidence: float,
    patch_size: int,
) -> tuple[float, bool, float]:
    """
    Success if patch-class confidence > original-class confidence.
    score = 0.3 * patch_confidence(%) + (size bonus only on success).
    Returns (round_score, success, applied_bonus).
    """
    confidence_pct = float(patch_confidence) * 100.0
    success = float(patch_confidence) > float(base_confidence)
    bonus = size_bonus(patch_size) if success else 0.0
    round_score = 0.3 * confidence_pct + bonus
    return float(round_score), success, float(bonus)


# ---------------------------------------------------------------------------
# Load patches
# ---------------------------------------------------------------------------
patch_dict = get_patches(CLASS_NAMES, PATCH_SIZES)
export_patch_previews(patch_dict)
# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "patch-classify-dev-secret")


def all_patch_options():
    return [
        {"id": f"{name}|{size}", "class_name": name, "patch_size": size}
        for name in CLASS_NAMES
        for size in PATCH_SIZES
    ]


def patch_preview_url(class_name: str, patch_size: int) -> str:
    safe = class_name.replace(" ", "_")
    return f"static/Patches/{safe}_{patch_size}.png"


def _make_offered_item(class_name: str, patch_size: int) -> dict:
    return {
        "id": f"{class_name}|{patch_size}",
        "class_name": class_name,
        "patch_size": int(patch_size),
        "preview": patch_preview_url(class_name, patch_size),
    }


def deal_random_patches(count: int = OFFERED_PATCH_COUNT):
    """
    Sample `count` unique patches such that every class and every size appears
    at least once. Requires count >= max(#classes, #sizes).
    """
    n_classes = len(CLASS_NAMES)
    n_sizes = len(PATCH_SIZES)
    min_needed = max(n_classes, n_sizes)
    if count < min_needed:
        raise ValueError(
            f"Need at least {min_needed} patches to cover all classes and sizes, got {count}"
        )

    used: set[tuple[str, int]] = set()
    selected: list[dict] = []

    # 1) One patch per class; cycle a shuffled size list so all sizes appear.
    classes = CLASS_NAMES[:]
    random.shuffle(classes)
    size_cycle = PATCH_SIZES[:]
    random.shuffle(size_cycle)
    for i, cname in enumerate(classes):
        size = size_cycle[i % n_sizes]
        used.add((cname, size))
        selected.append(_make_offered_item(cname, size))

    # 2) Cover any size still missing.
    covered_sizes = {size for _, size in used}
    for size in PATCH_SIZES:
        if size in covered_sizes:
            continue
        candidates = [c for c in CLASS_NAMES if (c, size) not in used]
        if not candidates:
            continue
        cname = random.choice(candidates)
        used.add((cname, size))
        selected.append(_make_offered_item(cname, size))

    # 3) Fill remaining slots with unused combinations.
    pool = [
        (c, s)
        for c in CLASS_NAMES
        for s in PATCH_SIZES
        if (c, s) not in used
    ]
    random.shuffle(pool)
    while len(selected) < count and pool:
        cname, size = pool.pop()
        used.add((cname, size))
        selected.append(_make_offered_item(cname, size))

    # Safety: if over-filled while covering sizes, keep a valid subset of size `count`.
    if len(selected) > count:
        selected = _trim_offered_keeping_coverage(selected, count)

    random.shuffle(selected)

    offered_classes = {item["class_name"] for item in selected}
    offered_sizes = {int(item["patch_size"]) for item in selected}
    assert offered_classes == set(CLASS_NAMES), offered_classes
    assert offered_sizes == set(PATCH_SIZES), offered_sizes
    return selected


def _trim_offered_keeping_coverage(selected: list[dict], count: int) -> list[dict]:
    """Reduce to `count` items while keeping all classes and sizes if possible."""
    selected = selected[:]
    random.shuffle(selected)
    keep: list[dict] = []
    used: set[tuple[str, int]] = set()

    # Prefer items that introduce new classes/sizes.
    def coverage_gain(item):
        c, s = item["class_name"], int(item["patch_size"])
        gain = 0
        if c not in {x["class_name"] for x in keep}:
            gain += 2
        if s not in {int(x["patch_size"]) for x in keep}:
            gain += 2
        return gain

    remaining = selected[:]
    while remaining and len(keep) < count:
        remaining.sort(key=coverage_gain, reverse=True)
        item = remaining.pop(0)
        key = (item["class_name"], int(item["patch_size"]))
        if key in used:
            continue
        used.add(key)
        keep.append(item)
    return keep


@app.route("/")
def intro():
    return render_template(
        "intro.html",
        max_submissions=MAX_SUBMISSIONS,
        target_score=TARGET_SCORE,
    )


@app.route("/play")
def play():
    offered = deal_random_patches()
    session["used_base_images"] = []
    label, base_version = set_session_base_image()
    session["offered_patches"] = offered
    session["submissions_used"] = 0
    session["total_score"] = 0.0
    return render_template(
        "index.html",
        base_label=label_names[label],
        base_version=base_version,
        offered_patches=offered,
        max_submissions=MAX_SUBMISSIONS,
        submissions_used=0,
        total_score=0.0,
        target_score=TARGET_SCORE,
    )


@app.route("/classify", methods=["POST"])
def classify():
    submissions_used = int(session.get("submissions_used", 0))
    total_score = float(session.get("total_score", 0.0))
    if submissions_used >= MAX_SUBMISSIONS:
        return jsonify(
            {
                "error": f"Submission limit reached ({MAX_SUBMISSIONS}).",
                "submissions_used": submissions_used,
                "submissions_left": 0,
                "max_submissions": MAX_SUBMISSIONS,
                "total_score": total_score,
            }
        ), 429

    payload = request.get_json(silent=True) or {}
    class_name = payload.get("class_name")
    try:
        patch_size = int(payload.get("patch_size"))
        patch_x = int(payload.get("patch_x", 0))
        patch_y = int(payload.get("patch_y", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid patch_size / patch_x / patch_y"}), 400

    offered = session.get("offered_patches") or []
    allowed = {
        (item["class_name"], int(item["patch_size"])) for item in offered
    }
    if (class_name, patch_size) not in allowed:
        return jsonify({"error": "Selected patch is not in your offered set."}), 400

    if class_name not in patch_dict or patch_size not in patch_dict[class_name]:
        return jsonify({"error": f"Unknown patch: {class_name} size {patch_size}"}), 400

    try:
        selected_img, selected_label = load_session_base_image()
    except (RuntimeError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400

    patch = patch_dict[class_name][patch_size]["patch"]
    patched = place_patch_with_position(selected_img, patch, (patch_y, patch_x))

    top5, target_confidence, probs = get_topk_with_target(
        pretrained_model, patched, class_name
    )
    base_confidence = float(probs[0, selected_label].cpu().item())
    round_score, attack_success, applied_bonus = update_score(
        target_confidence, base_confidence, patch_size
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rel_path = f"static/results/{uuid.uuid4().hex[:8]}.png"
    abs_path = BASE_DIR / rel_path
    batch = patched.unsqueeze(0) if patched.dim() == 3 else patched
    with torch.no_grad():
        logits = pretrained_model(batch.to(device))
    save_prediction_figure(patched, selected_label, logits, abs_path)

    submissions_used += 1
    total_score += round_score
    session["submissions_used"] = submissions_used
    session["total_score"] = total_score
    submissions_left = max(0, MAX_SUBMISSIONS - submissions_used)

    finished = submissions_left == 0
    passed = bool(finished and total_score > TARGET_SCORE)
    result_message_key = "passed" if passed else "failed" if finished else None

    # After each submission: new patches + new random base image for the next round.
    new_offered = deal_random_patches()
    session["offered_patches"] = new_offered
    next_label, next_base_version = set_session_base_image()
    next_base_label = label_names[next_label]

    return jsonify(
        {
            "patched_image": rel_path,
            "top5": top5,
            "target_class": class_name,
            "target_confidence": target_confidence,
            "base_label": label_names[selected_label],
            "base_confidence": base_confidence,
            "attack_success": attack_success,
            "position": [patch_y, patch_x],
            "submissions_used": submissions_used,
            "submissions_left": submissions_left,
            "max_submissions": MAX_SUBMISSIONS,
            "round_score": round_score,
            "size_bonus": applied_bonus,
            "total_score": total_score,
            "target_score": TARGET_SCORE,
            "finished": finished,
            "passed": passed,
            "result_message_key": result_message_key,
            "offered_patches": new_offered,
            "next_base_label": next_base_label,
            "base_version": next_base_version,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
