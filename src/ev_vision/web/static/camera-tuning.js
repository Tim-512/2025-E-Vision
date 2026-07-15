"use strict";

const $ = (id) => document.getElementById(id);
const parameterKeys = ["exposure_us", "gain_db", "acquisition_fps", "auto_exposure", "auto_gain", "auto_white_balance"];
const state = { applied: null, defaults: null, bounds: null, paused: false, previewUrl: "", initialized: false };

function setMessage(message, kind = "info") {
  const node = $("status-message");
  node.textContent = message;
  node.className = `status-message ${kind}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  let body = null;
  try { body = await response.json(); } catch (_) { body = null; }
  if (!response.ok) {
    const detail = body && body.detail;
    const message = typeof detail === "string" ? detail : detail && detail.message ? detail.message : `请求失败 (${response.status})`;
    throw new Error(message);
  }
  return body;
}

function reportError(context, error) {
  console.error(context, error);
  setMessage(`${context}：${error instanceof Error ? error.message : String(error)}`, "error");
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalized(parameters) {
  if (!parameters) return null;
  return {
    exposure_us: finiteNumber(parameters.exposure_us),
    gain_db: finiteNumber(parameters.gain_db),
    acquisition_fps: finiteNumber(parameters.acquisition_fps),
    auto_exposure: Boolean(parameters.auto_exposure),
    auto_gain: Boolean(parameters.auto_gain),
    auto_white_balance: Boolean(parameters.auto_white_balance),
  };
}

function readDraft() {
  return normalized({
    exposure_us: $("exposure-us").value,
    gain_db: $("gain-db").value,
    acquisition_fps: $("acquisition-fps").value,
    auto_exposure: $("auto-exposure").checked,
    auto_gain: $("auto-gain").checked,
    auto_white_balance: $("auto-white-balance").checked,
  });
}

function setDraft(parameters) {
  const value = normalized(parameters);
  if (!value) return;
  $("exposure-us").value = value.exposure_us;
  $("gain-db").value = value.gain_db;
  $("acquisition-fps").value = value.acquisition_fps;
  $("auto-exposure").checked = value.auto_exposure;
  $("auto-gain").checked = value.auto_gain;
  $("auto-white-balance").checked = value.auto_white_balance;
  updateManualControls();
  updateDraftState();
}

function sameParameters(a, b) {
  const left = normalized(a);
  const right = normalized(b);
  return Boolean(left && right && parameterKeys.every((key) => left[key] === right[key]));
}

function updateAppliedLabels() {
  if (!state.applied) return;
  $("applied-exposure").textContent = `当前 ${state.applied.exposure_us} μs`;
  $("applied-gain").textContent = `当前 ${state.applied.gain_db} dB`;
  $("applied-fps").textContent = `当前 ${state.applied.acquisition_fps} FPS`;
}

function updateDraftState() {
  if (!state.applied) return;
  const changed = !sameParameters(readDraft(), state.applied);
  $("draft-state").textContent = changed ? "有未应用修改" : "已同步";
  $("draft-state").classList.toggle("changed", changed);
  $("apply-parameters").disabled = !changed;
}

function updateManualControls() {
  $("exposure-us").disabled = $("auto-exposure").checked;
  $("gain-db").disabled = $("auto-gain").checked;
}

function queryFlag(id) { return $(id).checked ? "true" : "false"; }
function buildPreviewUrl() {
  const query = new URLSearchParams({
    overlay: queryFlag("overlay-enabled"),
    detection: queryFlag("detection-enabled"),
    show_board_outline: queryFlag("show-board-outline"),
    show_corners: queryFlag("show-corners"),
    show_center: queryFlag("show-center"),
    show_crosshair: queryFlag("show-crosshair"),
    show_detection_text: queryFlag("show-detection-text"),
    show_center_roi: queryFlag("show-center-roi"),
    max_width: "960",
    nonce: String(Date.now()),
  });
  return `/api/preview.mjpg?${query}`;
}

function refreshPreview() {
  state.previewUrl = buildPreviewUrl();
  if (!state.paused) $("preview").src = state.previewUrl;
}

function pausePreview() {
  const image = $("preview");
  const frozen = $("frozen-preview");
  if (!state.paused) {
    if (image.naturalWidth) {
      frozen.width = image.naturalWidth;
      frozen.height = image.naturalHeight;
      frozen.getContext("2d").drawImage(image, 0, 0);
    }
    frozen.hidden = false;
    image.hidden = true;
    image.removeAttribute("src");
    state.paused = true;
    $("pause-preview").textContent = "恢复画面";
    setMessage("仅浏览器画面已暂停；相机采集仍在继续。", "warning");
  } else {
    frozen.hidden = true;
    image.hidden = false;
    state.paused = false;
    $("pause-preview").textContent = "暂停画面";
    refreshPreview();
    setMessage("实时画面已恢复。", "success");
  }
}

function setInputBounds(bounds) {
  if (!bounds) return;
  const mappings = [
    ["exposure-us", "exposure_min_us", "exposure_max_us"],
    ["gain-db", "gain_min_db", "gain_max_db"],
    ["acquisition-fps", "acquisition_fps_min", "acquisition_fps_max"],
  ];
  mappings.forEach(([id, minKey, maxKey]) => { $(id).min = bounds[minKey]; $(id).max = bounds[maxKey]; });
}

function formatRate(value) { return Number.isFinite(Number(value)) ? Number(value).toFixed(1) : "--"; }
function updateStatus(payload) {
  const runtime = payload.runtime || {};
  const rawState = runtime.state || "Starting";
  const stateName = String(rawState).toLowerCase();
  $("camera-state").textContent = rawState;
  $("camera-identity").textContent = [payload.camera?.model, payload.camera?.serial].filter(Boolean).join(" · ") || "相机信息不可用";
  $("connection-dot").className = `state-dot state-${stateName}`;
  $("acquisition-rate").textContent = formatRate(runtime.acquisition_fps);
  $("preview-rate").textContent = formatRate(runtime.preview_fps);
  $("detection-rate").textContent = formatRate(runtime.detection_fps);
  $("diagnostics-rate").textContent = formatRate(runtime.diagnostics_fps);
  $("frame-age").textContent = formatRate(runtime.frame_age_ms);
  $("fault-counters").textContent = `${runtime.timeout_count ?? 0} / ${runtime.sequence_gap_count ?? 0}`;
  const fixed = payload.fixed_format || {};
  $("fixed-format").textContent = `${fixed.width ?? 1280} × ${fixed.height ?? 1024} · ${fixed.pixel_format ?? "BayerRG8"} · Buffer ${fixed.buffer_size ?? 2}`;
  if (payload.applied) {
    const remote = normalized(payload.applied);
    const draftWasApplied = sameParameters(readDraft(), state.applied);
    state.applied = remote;
    if (!state.initialized || draftWasApplied) setDraft(remote);
    updateAppliedLabels();
    updateDraftState();
  }
  if (runtime.last_error) setMessage(`运行异常：${runtime.last_error}`, "error");
}

function drawHistogram(id, series) {
  const canvas = $(id);
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#061017";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#1b3442";
  ctx.lineWidth = 1;
  for (let line = 1; line < 4; line += 1) {
    const y = Math.round((height * line) / 4) + .5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  const max = Math.max(1, ...series.flatMap((item) => item.values || []));
  series.forEach(({ values, color }) => {
    if (!Array.isArray(values) || values.length === 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = (index / Math.max(1, values.length - 1)) * width;
      const y = height - (finiteNumber(value) / max) * (height - 5);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function drawRoi(diagnostics) {
  const canvas = $("center-roi");
  const ctx = canvas.getContext("2d");
  const image = state.paused ? $("frozen-preview") : $("preview");
  const roi = diagnostics?.roi_px;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#020608";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!Array.isArray(roi) || roi.length !== 4 || !image.width) {
    ctx.fillStyle = "#78909c";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("等待可用的中心区域", canvas.width / 2, canvas.height / 2);
    return;
  }
  try {
    const [x, y, w, h] = roi.map(Number);
    ctx.drawImage(image, x, y, w, h, 0, 0, canvas.width, canvas.height);
    $("roi-description").textContent = `${w} × ${h} px · (${x}, ${y})`;
  } catch (error) {
    $("roi-description").textContent = "中心区域暂不可绘制";
  }
}

function updateDiagnostics(payload) {
  const data = payload.diagnostics;
  const detection = payload.detection || {};
  if (!data) return;
  $("diagnostic-sequence").textContent = `序列 ${data.source_sequence}`;
  $("dark-percent").textContent = `${finiteNumber(data.dark_percent).toFixed(2)}%`;
  $("bright-percent").textContent = `${finiteNumber(data.bright_percent).toFixed(2)}%`;
  $("focus-score").textContent = finiteNumber(data.focus_score).toFixed(1);
  $("detection-state").textContent = !detection.enabled ? "已关闭" : detection.error ? "异常" : detection.detected ? "已检测" : "未检测";
  drawHistogram("gray-histogram", [{ values: data.gray_histogram, color: "#dce8ec" }]);
  drawHistogram("rgb-histogram", [
    { values: data.red_histogram, color: "#ff6b6b" },
    { values: data.green_histogram, color: "#51d88a" },
    { values: data.blue_histogram, color: "#55a8ff" },
  ]);
  drawRoi(data);
}

async function poll() {
  try {
    const [status, diagnostics] = await Promise.all([api("/api/status"), api("/api/diagnostics")]);
    updateStatus(status);
    updateDiagnostics(diagnostics);
    state.initialized = true;
  } catch (error) { reportError("状态刷新失败", error); }
}

async function loadParameters() {
  const payload = await api("/api/parameters");
  state.applied = normalized(payload.applied);
  state.defaults = normalized(payload.project_defaults);
  state.bounds = payload.bounds;
  setInputBounds(state.bounds);
  setDraft(state.applied);
  updateAppliedLabels();
}

async function applyParameters() {
  try {
    const payload = await api("/api/parameters", { method: "PUT", body: JSON.stringify(readDraft()) });
    state.applied = normalized(payload.applied);
    setDraft(state.applied);
    updateAppliedLabels();
    setMessage("参数已由相机确认并生效。", "success");
  } catch (error) { reportError("应用参数失败", error); }
}

async function capture() {
  const button = $("capture-button");
  button.disabled = true;
  try {
    const payload = await api("/api/captures", { method: "POST", body: JSON.stringify({
      overlay: $("overlay-enabled").checked,
      show_board_outline: $("show-board-outline").checked,
      show_corners: $("show-corners").checked,
      show_center: $("show-center").checked,
      show_crosshair: $("show-crosshair").checked,
      show_detection_text: $("show-detection-text").checked,
      show_center_roi: $("show-center-roi").checked,
    }) });
    $("capture-result").textContent = `已保存：${payload.capture}`;
    setMessage("完整快照保存成功。", "success");
  } catch (error) { reportError("保存快照失败", error); }
  finally { button.disabled = false; }
}

async function refreshProfiles(selected = "") {
  const payload = await api("/api/profiles");
  const select = $("profile-select");
  select.replaceChildren(new Option("请选择", ""));
  (payload.profiles || []).forEach((item) => {
    const name = typeof item === "string" ? item : item.name;
    const label = typeof item === "string" ? item : item.display_name || item.name;
    select.add(new Option(label, name));
  });
  if (selected) select.value = selected;
}

async function loadProfile() {
  const name = $("profile-select").value;
  if (!name) return setMessage("请先选择配置档。", "warning");
  try {
    const payload = await api(`/api/profiles/${encodeURIComponent(name)}`);
    setDraft(payload.draft);
    $("profile-name").value = payload.name;
    $("profile-display-name").value = payload.display_name || "";
    setMessage("配置档仅载入草稿，尚未应用到相机。", "warning");
  } catch (error) { reportError("载入配置档失败", error); }
}

async function saveProfile() {
  const name = $("profile-name").value.trim();
  if (!name) return setMessage("请输入配置档名称。", "warning");
  try {
    await api(`/api/profiles/${encodeURIComponent(name)}`, { method: "PUT", body: JSON.stringify({ display_name: $("profile-display-name").value.trim() || null, parameters: readDraft() }) });
    await refreshProfiles(name);
    setMessage("当前草稿已保存为配置档；相机参数未因此改变。", "success");
  } catch (error) { reportError("保存配置档失败", error); }
}

async function deleteProfile() {
  const name = $("profile-select").value;
  if (!name) return setMessage("请先选择配置档。", "warning");
  if (!window.confirm(`确认删除配置档“${name}”？`)) return;
  try {
    await api(`/api/profiles/${encodeURIComponent(name)}`, { method: "DELETE" });
    await refreshProfiles();
    setMessage("配置档已删除。", "success");
  } catch (error) { reportError("删除配置档失败", error); }
}

function bindEvents() {
  ["exposure-us", "gain-db", "acquisition-fps", "auto-exposure", "auto-gain", "auto-white-balance"].forEach((id) => {
    $(id).addEventListener("input", () => { updateManualControls(); updateDraftState(); });
  });
  ["overlay-enabled", "show-board-outline", "show-corners", "show-center", "show-crosshair", "show-detection-text", "show-center-roi", "detection-enabled"].forEach((id) => $(id).addEventListener("change", refreshPreview));
  $("pause-preview").addEventListener("click", pausePreview);
  $("apply-parameters").addEventListener("click", applyParameters);
  $("revert-draft").addEventListener("click", () => { setDraft(state.applied); setMessage("草稿已恢复为当前生效值。", "success"); });
  $("restore-defaults").addEventListener("click", () => { setDraft(state.defaults); setMessage("项目默认值已载入草稿，尚未应用。", "warning"); });
  $("capture-button").addEventListener("click", capture);
  $("load-profile").addEventListener("click", loadProfile);
  $("save-profile").addEventListener("click", saveProfile);
  $("delete-profile").addEventListener("click", deleteProfile);
  $("preview").addEventListener("load", () => { $("preview-placeholder").hidden = true; });
  $("preview").addEventListener("error", () => { if (!state.paused) $("preview-placeholder").hidden = false; });
}

async function initialize() {
  bindEvents();
  try {
    await loadParameters();
    await refreshProfiles();
    refreshPreview();
    await poll();
    state.initialized = true;
    setMessage("页面已连接，可开始调参。", "success");
  } catch (error) { reportError("页面初始化失败", error); }
  window.setInterval(poll, 500);
}

document.addEventListener("DOMContentLoaded", initialize);
