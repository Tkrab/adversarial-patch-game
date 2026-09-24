(() => {
  const REAL_SIZE = 224;
  const config = window.APP_CONFIG || {
    maxSubmissions: 3,
    submissionsUsed: 0,
    totalScore: 0,
    targetScore: 95,
    baseLabel: "",
  };

  const patchPreview = document.getElementById("patchPreview");
  const baseImage = document.getElementById("baseImage");
  const baseLabelEl = document.getElementById("baseLabel");
  const headerSubtitle = document.getElementById("headerSubtitle");
  const baseContainer = document.getElementById("baseImageContainer");
  const patchOptionsRoot = document.getElementById("patchOptions");
  const classInput = document.getElementById("className");
  const sizeInput = document.getElementById("patchSize");
  const patchXInput = document.getElementById("patchX");
  const patchYInput = document.getElementById("patchY");
  const classifyBtn = document.getElementById("classifyBtn");
  const resetBtn = document.getElementById("resetBtn");
  const statusEl = document.getElementById("status");
  const resultImage = document.getElementById("resultImage");
  const resultPlaceholder = document.getElementById("resultPlaceholder");
  const top5List = document.getElementById("top5List");
  const remainingCount = document.getElementById("remainingCount");
  const roundScoreEl = document.getElementById("roundScore");
  const totalScoreEl = document.getElementById("totalScore");
  const finalResultBox = document.getElementById("finalResultBox");
  const finalResultEl = document.getElementById("finalResult");
  const replayBtn = document.getElementById("replayBtn");

  let placed = false;
  let submissionsUsed = Number(config.submissionsUsed) || 0;
  let totalScore = Number(config.totalScore) || 0;
  let currentBaseLabel = config.baseLabel || "";
  const maxSubmissions = Number(config.maxSubmissions) || 3;
  const targetScore = Number(config.targetScore) || 95;

  let paletteDrag = null;
  let repositionDrag = null;

  const ghost = document.createElement("img");
  ghost.id = "dragGhost";
  ghost.alt = "";
  ghost.draggable = false;
  document.body.appendChild(ghost);

  function t(key, vars) {
    return window.I18N ? I18N.t(key, vars) : key;
  }

  function updateHeaderSubtitle() {
    if (!headerSubtitle) return;
    headerSubtitle.innerHTML =
      t("subtitle", {
        left: remaining(),
        max: maxSubmissions,
        target: targetScore,
      }) + `<strong id="baseLabel">${currentBaseLabel}</strong>`;
  }

  function updateBaseImage(baseVersion, baseLabel) {
    if (baseVersion && baseImage) {
      baseImage.src = `/static/base.png?v=${baseVersion}`;
    }
    if (baseLabel) {
      currentBaseLabel = baseLabel;
      updateHeaderSubtitle();
    }
  }

  function patchOptionButtons() {
    return patchOptionsRoot.querySelectorAll(".patch-option");
  }

  function remaining() {
    return Math.max(0, maxSubmissions - submissionsUsed);
  }

  function updateScoreUI(roundScore, total) {
    if (typeof roundScore === "number") {
      roundScoreEl.textContent = roundScore.toFixed(2);
    }
    totalScoreEl.textContent = Number(total).toFixed(2);
  }

  function showFinalResult(passed, messageKey, total) {
    if (!finalResultEl || !finalResultBox) return;
    finalResultBox.hidden = false;
    const message = messageKey ? t(messageKey) : t(passed ? "passed" : "failed");
    finalResultEl.textContent = message;
    finalResultEl.classList.toggle("passed", !!passed);
    finalResultEl.classList.toggle("failed", !passed);
    setStatus(t("statusFinal", { message, total: Number(total).toFixed(2), target: targetScore }));
  }

  function updateAttemptsUI() {
    remainingCount.textContent = String(remaining());
    updateHeaderSubtitle();
    const exhausted = remaining() <= 0;
    classifyBtn.disabled = exhausted || !placed;
    patchOptionButtons().forEach((btn) => {
      btn.disabled = exhausted;
    });
  }

  function scaleFactor() {
    return baseContainer.getBoundingClientRect().width / REAL_SIZE;
  }

  function displaySizeFor(patchSize) {
    return Math.max(8, Math.round(Number(patchSize) * scaleFactor()));
  }

  function setStatus(text) {
    statusEl.textContent = text || "";
  }

  function highlightOption(className, patchSize) {
    patchOptionButtons().forEach((el) => {
      const match =
        el.dataset.className === className &&
        String(el.dataset.patchSize) === String(patchSize);
      el.classList.toggle("selected", match);
      el.setAttribute("aria-selected", match ? "true" : "false");
    });
  }

  function bindPatchOption(btn) {
    btn.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || btn.disabled) return;
      e.preventDefault();
      startPaletteDrag(btn, e.clientX, e.clientY);
    });
  }

  function renderOfferedPatches(offered) {
    if (!Array.isArray(offered)) return;
    patchOptionsRoot.innerHTML = "";
    offered.forEach((patch) => {
      const preview = String(patch.preview || "").startsWith("/")
        ? patch.preview
        : `/${patch.preview}`;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "patch-option";
      btn.setAttribute("role", "option");
      btn.dataset.className = patch.class_name;
      btn.dataset.patchSize = String(patch.patch_size);
      btn.dataset.preview = preview;
      btn.setAttribute("aria-selected", "false");
      btn.innerHTML =
        `<img src="${preview}" alt="${patch.class_name} ${patch.patch_size}" />` +
        `<span>${patch.class_name} · ${patch.patch_size}</span>`;
      bindPatchOption(btn);
      patchOptionsRoot.appendChild(btn);
    });
    updateAttemptsUI();
  }

  function showPlacedPreview(left, top) {
    const size = displaySizeFor(sizeInput.value);
    patchPreview.src = classInput.value
      ? `/static/Patches/${String(classInput.value).replace(/ /g, "_")}_${sizeInput.value}.png`
      : patchPreview.src;
    patchPreview.style.width = `${size}px`;
    patchPreview.style.height = `${size}px`;
    patchPreview.style.left = `${left}px`;
    patchPreview.style.top = `${top}px`;
    patchPreview.hidden = false;
    patchPreview.style.display = "block";
    clampPatchInBounds();
    syncHiddenCoords();
  }

  function hidePlacedPreview() {
    patchPreview.hidden = true;
    patchPreview.style.display = "none";
    patchPreview.removeAttribute("src");
  }

  function clampPatchInBounds() {
    const rect = baseContainer.getBoundingClientRect();
    let x = parseInt(patchPreview.style.left, 10) || 0;
    let y = parseInt(patchPreview.style.top, 10) || 0;
    const w = patchPreview.offsetWidth || displaySizeFor(sizeInput.value);
    const h = patchPreview.offsetHeight || w;
    x = Math.max(0, Math.min(x, rect.width - w));
    y = Math.max(0, Math.min(y, rect.height - h));
    patchPreview.style.left = `${x}px`;
    patchPreview.style.top = `${y}px`;
  }

  function syncHiddenCoords() {
    if (!placed) return;
    const rect = baseContainer.getBoundingClientRect();
    const displayedX = parseInt(patchPreview.style.left, 10) || 0;
    const displayedY = parseInt(patchPreview.style.top, 10) || 0;
    patchXInput.value = String(Math.round((displayedX / rect.width) * REAL_SIZE));
    patchYInput.value = String(Math.round((displayedY / rect.height) * REAL_SIZE));
  }

  function placePatchFromPalette(className, patchSize, preview, clientX, clientY) {
    const rect = baseContainer.getBoundingClientRect();
    const size = displaySizeFor(patchSize);
    let left = clientX - rect.left - size / 2;
    let top = clientY - rect.top - size / 2;
    left = Math.max(0, Math.min(left, rect.width - size));
    top = Math.max(0, Math.min(top, rect.height - size));

    classInput.value = className;
    sizeInput.value = String(patchSize);
    patchPreview.src = preview;
    placed = true;
    resetBtn.disabled = false;
    classifyBtn.disabled = remaining() <= 0;

    highlightOption(className, patchSize);
    showPlacedPreview(left, top);
    setStatus(t("statusPlaced", { name: className, size: patchSize }));
    updateAttemptsUI();
  }

  function clearPlacement(statusText) {
    placed = false;
    classInput.value = "";
    sizeInput.value = "";
    patchXInput.value = "0";
    patchYInput.value = "0";
    hidePlacedPreview();
    highlightOption(null, null);
    resetBtn.disabled = true;
    classifyBtn.disabled = true;
    setStatus(statusText || t("statusDrag"));
    updateAttemptsUI();
  }

  function pointInContainer(clientX, clientY) {
    const rect = baseContainer.getBoundingClientRect();
    return (
      clientX >= rect.left &&
      clientX <= rect.right &&
      clientY >= rect.top &&
      clientY <= rect.bottom
    );
  }

  function showGhost(preview, size, clientX, clientY) {
    ghost.src = preview;
    ghost.style.width = `${size}px`;
    ghost.style.height = `${size}px`;
    ghost.style.left = `${clientX - size / 2}px`;
    ghost.style.top = `${clientY - size / 2}px`;
    ghost.classList.add("visible");
  }

  function moveGhost(clientX, clientY) {
    if (!paletteDrag) return;
    const size = paletteDrag.displaySize;
    ghost.style.left = `${clientX - size / 2}px`;
    ghost.style.top = `${clientY - size / 2}px`;
    baseContainer.classList.toggle("drop-target", pointInContainer(clientX, clientY));
  }

  function hideGhost() {
    ghost.classList.remove("visible");
    baseContainer.classList.remove("drop-target");
  }

  function startPaletteDrag(btn, clientX, clientY) {
    if (remaining() <= 0) return;
    const patchSize = Number(btn.dataset.patchSize);
    paletteDrag = {
      className: btn.dataset.className,
      patchSize,
      preview: btn.dataset.preview,
      displaySize: displaySizeFor(patchSize),
      sourceBtn: btn,
    };
    btn.classList.add("dragging");
    showGhost(paletteDrag.preview, paletteDrag.displaySize, clientX, clientY);
    setStatus(t("statusDrop"));
  }

  function endPaletteDrag(clientX, clientY) {
    if (!paletteDrag) return;
    const { className, patchSize, preview, sourceBtn } = paletteDrag;
    sourceBtn.classList.remove("dragging");
    hideGhost();
    const dropOk = pointInContainer(clientX, clientY);
    paletteDrag = null;
    if (dropOk) {
      placePatchFromPalette(className, patchSize, preview, clientX, clientY);
    } else if (!placed) {
      setStatus(t("statusDrag"));
    }
  }

  function renderTop5(top5) {
    top5List.innerHTML = "";
    (top5 || []).forEach((item, i) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = `${i + 1}. ${item.label}`;
      const conf = document.createElement("span");
      conf.className = "conf";
      conf.textContent = `${(item.confidence * 100).toFixed(2)}%`;
      li.appendChild(name);
      li.appendChild(conf);
      top5List.appendChild(li);
    });
  }

  async function runClassify() {
    if (!placed) {
      setStatus(t("statusNeedPatch"));
      return;
    }
    if (remaining() <= 0) {
      updateAttemptsUI();
      return;
    }

    syncHiddenCoords();
    classifyBtn.disabled = true;
    setStatus(t("statusInference"));

    try {
      const res = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          class_name: classInput.value,
          patch_size: Number(sizeInput.value),
          patch_x: Number(patchXInput.value),
          patch_y: Number(patchYInput.value),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      if (typeof data.submissions_used === "number") {
        submissionsUsed = data.submissions_used;
      } else {
        submissionsUsed += 1;
      }

      if (typeof data.total_score === "number") {
        totalScore = data.total_score;
      }
      const roundScore = typeof data.round_score === "number" ? data.round_score : 0;
      const left =
        typeof data.submissions_left === "number" ? data.submissions_left : remaining();

      resultPlaceholder.hidden = true;
      resultImage.hidden = false;
      resultImage.src = `/${data.patched_image}?t=${Date.now()}`;
      renderTop5(data.top5);
      updateScoreUI(roundScore, totalScore);

      if (Array.isArray(data.offered_patches)) {
        renderOfferedPatches(data.offered_patches);
      }
      if (data.base_version) {
        updateBaseImage(data.base_version, data.next_base_label);
      }

      if (data.finished) {
        clearPlacement();
        showFinalResult(!!data.passed, data.result_message_key, totalScore);
      } else {
        clearPlacement(t("statusNextRound", { left }));
        const outcome = data.attack_success ? t("success") : t("failure");
        const bonusNote = data.attack_success
          ? t("bonusApplied", { n: Number(data.size_bonus || 0).toFixed(0) })
          : t("noBonus");
        setStatus(
          t("statusRoundDone", {
            outcome,
            bonus: bonusNote,
            round: roundScore.toFixed(2),
            total: totalScore.toFixed(2),
            left,
          })
        );
      }
      updateAttemptsUI();
    } catch (err) {
      setStatus(t("httpFailed", { msg: err.message }));
      updateAttemptsUI();
    }
  }

  patchOptionButtons().forEach(bindPatchOption);

  patchPreview.addEventListener("mousedown", (e) => {
    if (!placed || remaining() <= 0 || paletteDrag) return;
    repositionDrag = { offsetX: e.offsetX, offsetY: e.offsetY };
    patchPreview.style.cursor = "grabbing";
    e.preventDefault();
    e.stopPropagation();
  });

  document.addEventListener("mousemove", (e) => {
    if (paletteDrag) {
      moveGhost(e.clientX, e.clientY);
      return;
    }
    if (!repositionDrag) return;
    const rect = baseContainer.getBoundingClientRect();
    let x = e.clientX - rect.left - repositionDrag.offsetX;
    let y = e.clientY - rect.top - repositionDrag.offsetY;
    const w = patchPreview.offsetWidth;
    const h = patchPreview.offsetHeight;
    x = Math.max(0, Math.min(x, rect.width - w));
    y = Math.max(0, Math.min(y, rect.height - h));
    patchPreview.style.left = `${x}px`;
    patchPreview.style.top = `${y}px`;
  });

  document.addEventListener("mouseup", (e) => {
    if (paletteDrag) {
      endPaletteDrag(e.clientX, e.clientY);
      return;
    }
    if (repositionDrag) {
      repositionDrag = null;
      patchPreview.style.cursor = "grab";
      syncHiddenCoords();
    }
  });

  classifyBtn.addEventListener("click", runClassify);
  resetBtn.addEventListener("click", () => clearPlacement());
  if (replayBtn) {
    replayBtn.addEventListener("click", () => {
      window.location.href = "/play";
    });
  }
  window.addEventListener("resize", () => {
    if (!placed) return;
    const left = parseInt(patchPreview.style.left, 10) || 0;
    const top = parseInt(patchPreview.style.top, 10) || 0;
    showPlacedPreview(left, top);
  });

  document.addEventListener("languagechange", () => {
    if (window.I18N) I18N.apply();
    updateHeaderSubtitle();
    if (!placed && !finalResultBox.hidden) {
      /* keep final result text */
    } else if (!placed) {
      setStatus(t("statusDrag"));
    }
  });

  updateHeaderSubtitle();
  clearPlacement();
})();
