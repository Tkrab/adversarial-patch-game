(() => {
  const STORAGE_KEY = "patch-game-lang";

  const MESSAGES = {
    en: {
      pageTitle: "Patch Drag Classification",
      rulesTitle: "Patch Game Rules",
      subtitle:
        "Drag a patch onto the image and classify. Submissions left: {left} / {max}. Target score: {target}. Base image class: ",
      imagePatch: "Image + Patch",
      offeredPatches: "Offered patches (drag onto the image)",
      classify: "Classify",
      clearPatch: "Clear patch",
      submissionsLeft: "Submissions left: ",
      round: "Round: ",
      total: "Total: ",
      target: "Target: ",
      statusDrag: "Drag a patch from below onto the image.",
      statusDrop: "Drop the patch onto the image.",
      statusPlaced: 'Placed "{name}" ({size}). Adjust position if needed, then classify.',
      statusInference: "Running inference…",
      statusNeedPatch: "Please drag a patch onto the image first.",
      statusNextRound: "New image and patches ready. Submissions left: {left}.",
      statusRoundDone:
        "{outcome} ({bonus}). Round {round}, total {total}. Next round ready. Submissions left: {left}.",
      statusFinal: "{message} Final score: {total} / {target}.",
      success: "Success",
      failure: "Failure",
      bonusApplied: "bonus +{n}",
      noBonus: "no bonus",
      resultTitle: "Classification Result",
      resultPlaceholder: "Click Classify to see the result image and Top-5",
      passed: "Congratulations, you cleared the level!",
      failed: "Sorry, you did not clear the level.",
      playAgain: "Play again",
      backToRules: "Back to rules",
      introLead:
        "Place adversarial patches on an image and try to fool the ResNet34 classifier. Reach the target score within limited submissions to clear the level.",
      howToPlay: "How to play",
      rule1:
        "You are offered 6 random patches. Every deal includes all patch classes and all patch sizes (32 / 48 / 64).",
      rule2: "Drag one patch from the bottom onto the image, adjust its position, then click Classify.",
      rule3: "You have {max} submissions in total.",
      rule4: "After each classification, a new random base image and a new set of 6 patches are dealt.",
      rule5: "If your total score is greater than {target} after all submissions, you win.",
      successVsFailure: "Success vs failure",
      successVsFailureDesc:
        "After classification, compare the patch-class confidence with the original-class confidence on the patched image:",
      successRule: "Success: patch-class confidence > original-class confidence",
      failureRule: "Failure: otherwise",
      scoring: "Scoring",
      scoringFormula: "Each classification scores:",
      formula: "score = 0.3 × patch-class confidence(%) + size bonus (success only)",
      bonusOnlySuccess: "Size bonus is added only on success.",
      bonus32: "Patch size 32: +20 bonus",
      bonus48: "Patch size 48: +10 bonus",
      bonus64: "Patch size 64: +0 bonus",
      bonusFail: "On failure, no size bonus is added.",
      totalSum: "Your total is the sum of all submission scores. Target to clear the level: {target} points.",
      startGame: "Start Game",
      httpFailed: "Request failed: {msg}",
    },
    zh: {
      pageTitle: "补丁拖放分类",
      rulesTitle: "游戏规则",
      subtitle:
        "将补丁拖到图片上并分类。剩余次数：{left} / {max}。目标分数：{target}。原图类别：",
      imagePatch: "原图 + 补丁",
      offeredPatches: "可选补丁（拖到图片上）",
      classify: "分类",
      clearPatch: "清除补丁",
      submissionsLeft: "剩余次数：",
      round: "本轮：",
      total: "总分：",
      target: "目标：",
      statusDrag: "请从下方拖一个补丁到图片上。",
      statusDrop: "将补丁放到图片上。",
      statusPlaced: "已放置「{name}」（{size}），可调整位置后点击分类。",
      statusInference: "推理中…",
      statusNeedPatch: "请先将补丁拖到图片上。",
      statusNextRound: "已更换新底图与补丁。剩余次数：{left}。",
      statusRoundDone:
        "{outcome}（{bonus}）。本轮 {round}，总分 {total}。下一轮已就绪，剩余次数：{left}。",
      statusFinal: "{message} 最终得分：{total} / {target}。",
      success: "成功",
      failure: "失败",
      bonusApplied: "加分 +{n}",
      noBonus: "无加分",
      resultTitle: "分类结果",
      resultPlaceholder: "点击「分类」查看结果图与 Top-5",
      passed: "恭喜你过关！",
      failed: "很抱歉，你没能过关。",
      playAgain: "再玩一局",
      backToRules: "返回规则",
      introLead:
        "将对抗补丁贴到图片上，尝试欺骗 ResNet34 分类器。在有限次数内达到目标分数即可过关。",
      howToPlay: "玩法说明",
      rule1: "每次提供 6 个随机补丁，且包含全部类别与全部尺寸（32 / 48 / 64）。",
      rule2: "从下方拖一个补丁到图片上，调整位置后点击「分类」。",
      rule3: "总共 {max} 次提交机会。",
      rule4: "每次分类后，会随机更换一张新底图，并重新发放 6 个补丁。",
      rule5: "全部提交结束后，若总分大于 {target} 则过关。",
      successVsFailure: "成功与失败",
      successVsFailureDesc: "分类后，比较贴补丁图像上「补丁类别置信度」与「原图类别置信度」：",
      successRule: "成功：补丁类别置信度 > 原图类别置信度",
      failureRule: "失败：否则",
      scoring: "计分方式",
      scoringFormula: "每次分类得分：",
      formula: "得分 = 0.3 × 补丁类别置信度(%) + 尺寸加分（仅成功时）",
      bonusOnlySuccess: "仅成功时才有尺寸加分。",
      bonus32: "尺寸 32：+20",
      bonus48: "尺寸 48：+10",
      bonus64: "尺寸 64：+0",
      bonusFail: "失败时不加尺寸分。",
      totalSum: "总分为各次得分之和。过关目标：{target} 分。",
      startGame: "开始游戏",
      httpFailed: "请求失败：{msg}",
    },
  };

  function getLang() {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved === "zh" ? "zh" : "en";
  }

  function format(template, vars = {}) {
    return String(template).replace(/\{(\w+)\}/g, (_, key) =>
      vars[key] !== undefined ? String(vars[key]) : `{${key}}`
    );
  }

  function t(key, vars) {
    const lang = getLang();
    const text = MESSAGES[lang][key] ?? MESSAGES.en[key] ?? key;
    return vars ? format(text, vars) : text;
  }

  function apply(root = document) {
    root.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = t(el.dataset.i18n);
    });
    root.querySelectorAll("[data-i18n-html]").forEach((el) => {
      el.innerHTML = t(el.dataset.i18nHtml);
    });
    root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    document.documentElement.lang = getLang() === "zh" ? "zh-CN" : "en";
    document.querySelectorAll(".lang-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.lang === getLang());
    });
  }

  function setLang(lang) {
    localStorage.setItem(STORAGE_KEY, lang === "zh" ? "zh" : "en");
    apply();
    document.dispatchEvent(new CustomEvent("languagechange"));
  }

  function initSwitcher() {
    document.querySelectorAll(".lang-btn").forEach((btn) => {
      btn.addEventListener("click", () => setLang(btn.dataset.lang));
    });
  }

  window.I18N = { t, apply, setLang, getLang, format, initSwitcher };
  document.addEventListener("DOMContentLoaded", () => {
    initSwitcher();
    apply();
  });
})();
