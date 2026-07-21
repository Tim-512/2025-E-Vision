"use strict";

const $ = (id) => document.getElementById(id);
const parameterKeys = ["exposure_us", "gain_db", "acquisition_fps", "auto_exposure", "auto_gain", "auto_white_balance"];
const state = { applied: null, defaults: null, bounds: null, paused: false, previewUrl: "", initialized: false };
let detectionConfig = null;
let latestDetectionSequence = null;

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
    const payload = body;
    const detail = payload && payload.detail;
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
  $("detection-processing-rate").textContent = formatRate(runtime.detection_fps);
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

function updateBackendVisibility(config) {
  const classical = config.backend === "classical";
  const classicalControls = $("classical-controls");
  const hybridControls = $("hybrid-controls");
  const modelReloadButton = $("reload-detection-model");
  classicalControls.hidden = !classical;
  hybridControls.hidden = classical;
  modelReloadButton.hidden = classical;
  $("hybrid-model-status").hidden = classical;
  $("classical-debug-grid").hidden = !classical;
  $("detection-backend-badge").textContent = classical ? "CLASSICAL" : "HYBRID";
}

function renderDetectionConfig(config) {
  detectionConfig = config;
  updateBackendVisibility(config);
  $("detection-confidence-threshold").value = config.model.confidence_threshold;
  $("detection-max-candidates").value = config.model.max_candidates;
  $("detection-canny-low").value = config.roi_geometry.canny_low;
  $("detection-canny-high").value = config.roi_geometry.canny_high;
  $("detection-min-edge-support").value = config.roi_geometry.min_edge_support;
  $("detection-min-geometry-score").value = config.roi_geometry.min_geometry_score;
  $("detection-model-weight").value = config.candidate_scoring.model_weight;
  $("detection-geometry-weight").value = config.candidate_scoring.geometry_weight;
  $("detection-structure-weight").value = config.candidate_scoring.structure_weight;
  $("detection-temporal-weight").value = config.candidate_scoring.temporal_weight;
  $("detection-ambiguity-margin").value = config.candidate_scoring.ambiguity_margin;
  $("clahe-clip-limit").value = config.normalization.clahe_clip_limit;
  $("min-white-occupancy").value = config.white_board.min_white_occupancy;
  $("ring-ratio-tolerance").value = config.rings.ratio_tolerance;
  $("min-arc-coverage").value = config.rings.min_arc_coverage;
  $("classical-tracking-threshold").value = config.classical_scoring.tracking_threshold;
  $("classical-acquisition-threshold").value = config.classical_scoring.acquisition_threshold;
  $("ring-only-mode").checked = config.ring_first.ring_only;
  $("ring-immediate-strong").checked = config.ring_first.immediate_strong_acquisition;
  $("ring-medium-confirm-frames").value = config.ring_first.medium_confirm_frames;
  $("ring-medium-common-center").value = config.ring_first.medium_common_center_score;
  $("ring-medium-ratio-score").value = config.ring_first.medium_ratio_score;
  $("ring-medium-coverage-score").value = config.ring_first.medium_coverage_score;
  $("ring-roi-min-half-extent").value = config.ring_first.roi_min_half_extent_px;
  $("ring-roi-outer-extent-scale").value = config.ring_first.roi_outer_extent_per_scale;
  $("ring-roi-prediction-padding").value = config.ring_first.roi_prediction_padding_px;
  $("ring-roi-safety-factor").value = config.ring_first.roi_safety_factor;
  $("ring-roi-miss-expand").value = config.ring_first.roi_miss_expand_px;
  $("ring-roi-full-frame-misses").value = config.ring_first.roi_full_frame_after_misses;
  $("detection-confirm-frames").value = config.tracking.confirm_frames;
  $("detection-predict-frames").value = config.tracking.predict_frames;
  $("predict-max-frames").value = config.tracking.predict_max_frames;
  $("predict-max-ms").value = config.tracking.predict_max_ms;
  $("detection-lost-frames").value = config.tracking.lost_frames;
  $("detection-max-center-jump-px").value = config.tracking.max_center_jump_px;
  $("detection-max-result-age-ms").value = config.tracking.max_result_age_ms;
}

function readDetectionConfigForm() {
  const classical = detectionConfig.backend === "classical";
  const value = {
    backend: detectionConfig.backend,
    tracking: {
      confirm_frames: Number($("detection-confirm-frames").value),
      predict_frames: Number($("detection-predict-frames").value),
      predict_max_frames: Number($("predict-max-frames").value),
      predict_max_ms: Number($("predict-max-ms").value),
      lost_frames: Number($("detection-lost-frames").value),
      max_center_jump_px: Number($("detection-max-center-jump-px").value),
      max_result_age_ms: Number($("detection-max-result-age-ms").value),
    },
  };
  if (classical) {
    value.normalization = {clahe_clip_limit: Number($("clahe-clip-limit").value)};
    value.white_board = {min_white_occupancy: Number($("min-white-occupancy").value)};
    value.rings = {
      ratio_tolerance: Number($("ring-ratio-tolerance").value),
      min_arc_coverage: Number($("min-arc-coverage").value),
    };
    value.classical_scoring = {
      tracking_threshold: Number($("classical-tracking-threshold").value),
      acquisition_threshold: Number($("classical-acquisition-threshold").value),
    };
    value.ring_first = {
      ring_only: $("ring-only-mode").checked,
      immediate_strong_acquisition: $("ring-immediate-strong").checked,
      medium_confirm_frames: Number($("ring-medium-confirm-frames").value),
      medium_common_center_score: Number($("ring-medium-common-center").value),
      medium_ratio_score: Number($("ring-medium-ratio-score").value),
      medium_coverage_score: Number($("ring-medium-coverage-score").value),
      roi_min_half_extent_px: Number($("ring-roi-min-half-extent").value),
      roi_outer_extent_per_scale: Number($("ring-roi-outer-extent-scale").value),
      roi_prediction_padding_px: Number($("ring-roi-prediction-padding").value),
      roi_safety_factor: Number($("ring-roi-safety-factor").value),
      roi_miss_expand_px: Number($("ring-roi-miss-expand").value),
      roi_full_frame_after_misses: Number($("ring-roi-full-frame-misses").value),
    };
  } else {
    value.model = {
      confidence_threshold: Number($("detection-confidence-threshold").value),
      max_candidates: Number($("detection-max-candidates").value),
    };
    value.roi_geometry = {
      ...detectionConfig.roi_geometry,
      canny_low: Number($("detection-canny-low").value),
      canny_high: Number($("detection-canny-high").value),
      min_edge_support: Number($("detection-min-edge-support").value),
      min_geometry_score: Number($("detection-min-geometry-score").value),
    };
    value.candidate_scoring = {
      model_weight: Number($("detection-model-weight").value),
      geometry_weight: Number($("detection-geometry-weight").value),
      structure_weight: Number($("detection-structure-weight").value),
      temporal_weight: Number($("detection-temporal-weight").value),
      ambiguity_margin: Number($("detection-ambiguity-margin").value),
    };
  }
  return value;
}

function renderDetectionStatus(status) {
  latestDetectionSequence = status.source_sequence;
  $("detection-model-state").textContent = status.model_state || "--";
  $("detection-model-backend").textContent = status.model_backend || "--";
  $("detection-model-path").textContent = status.model_path || "--";
  $("detection-tracking-state").textContent = status.tracking_state || "--";
  $("detection-failure-reason").textContent = status.failure_reason || "NONE";
  $("observation-source").textContent = status.observation_source || "NONE";
  $("detection-source-age").textContent = `Source age ${status.source_age_us == null ? "--" : (finiteNumber(status.source_age_us) / 1000).toFixed(1)} ms`;
  $("detection-target-valid").textContent = status.target_valid ? "目标有效" : "目标无效";
  $("detection-confirmation-count").textContent = status.confirmation_count ?? 0;
  $("detection-miss-count").textContent = status.miss_count ?? 0;
  $("predicted-frames").textContent = status.predicted_frames ?? 0;
  $("detection-result-age").textContent = `Result age ${status.result_age_ms == null ? "--" : finiteNumber(status.result_age_ms).toFixed(1)} ms`;
  $("detection-edge-flags").textContent = `Near edge ${Boolean(status.near_image_edge)} / partially outside ${Boolean(status.partially_outside)}`;
  $("detection-confidence").textContent = finiteNumber(status.confidence).toFixed(3);
  const velocity = Array.isArray(status.velocity_px_s) ? status.velocity_px_s.map((item) => finiteNumber(item).toFixed(1)).join(", ") : "--";
  $("detection-motion").textContent = `?? ${velocity} px/s / ?? ${status.scale_px_per_mm == null ? "--" : finiteNumber(status.scale_px_per_mm).toFixed(3)} px/mm`;
  $("detection-rejection-reasons").textContent = (status.rejection_reasons || []).join("; ") || "NONE";
  $("detection-candidate-count").textContent = status.candidate_count ?? 0;
  $("detection-latency").textContent = `?? ${finiteNumber(status.inference_ms).toFixed(1)} / ?? ${finiteNumber(status.geometry_ms).toFixed(1)} / ?? ${finiteNumber(status.total_ms).toFixed(1)} ms`;
  $("detection-score-components").textContent = `M ${finiteNumber(status.model_confidence).toFixed(3)} ? G ${finiteNumber(status.geometry_score).toFixed(3)} ? E ${finiteNumber(status.edge_support_score).toFixed(3)} ? S ${finiteNumber(status.structure_score).toFixed(3)} ? T ${finiteNumber(status.temporal_score).toFixed(3)} ? C ${finiteNumber(status.combined_score).toFixed(3)}`;
  const list = $("detection-candidate-list"); list.replaceChildren();
  (status.candidates || []).forEach((candidate, index) => {
    const item = document.createElement("li");
    if (!candidate.accepted) item.className = "rejected";
    item.textContent = `#${index + 1} C ${finiteNumber(candidate.combined_score).toFixed(3)} ${candidate.failure_reason || "ACCEPTED"}`;
    list.append(item);
  });
  if (!list.children.length) list.append(document.createElement("li"));
}

async function refreshDetectionStatus() {
  const status = await api("/api/detection/status");
  if (!status) return;
  renderDetectionStatus(status);
  if (detectionConfig && detectionConfig.backend === "classical") refreshClassicalDebug(status.source_sequence);
}

async function applyDetectionConfig() {
  const button = $("apply-detection-config"); button.disabled = true;
  try {
    const applied = await api("/api/detection/config", {method: "PUT", body: JSON.stringify(readDetectionConfigForm())});
    renderDetectionConfig(applied);
    setMessage("Detection settings applied without restarting camera acquisition.", "success");
  } catch (error) { reportError("Reload model failed", error); }
  finally { button.disabled = false; }
}

async function reloadDetectionModel() {
  const button = $("reload-detection-model"); button.disabled = true;
  try { renderDetectionStatus(await api("/api/detection/model/reload", {method:"POST", body:"{}"})); setMessage("Reload model failed", "success"); }
  catch (error) { reportError("Reload model failed", error); }
  finally { button.disabled = false; }
}

function setDebugImage(name, sourceSequence) {
  const query = sourceSequence == null
    ? ""
    : "?sequence=" + encodeURIComponent(sourceSequence);
  const selected = $("detection-debug-image");
  selected.src = "/api/detection/debug/" + encodeURIComponent(name) + query + (query ? "&" : "?") + "t=" + Date.now();
}

function refreshClassicalDebug(sourceSequence) {
  if (sourceSequence == null) return;
  const names = ["normalized-gray", "white-mask", "edge-mask", "ring-arcs", "candidate-scores"];
  names.forEach((name) => {
    const image = $("debug-" + name);
    const query = "?sequence=" + encodeURIComponent(sourceSequence);
    image.src = "/api/detection/debug/" + encodeURIComponent(name) + query + "&t=" + Date.now();
  });
}

async function refreshDetectionDebug() {
  const view = $("detection-debug-view").value;
  if (!view || latestDetectionSequence == null) { $("detection-debug-image").hidden = true; return; }
  setDebugImage(view, latestDetectionSequence);
  $("detection-debug-image").hidden = false;
}

async function poll() {
  try {
    const [status, diagnostics] = await Promise.all([api("/api/status"), api("/api/diagnostics"), refreshDetectionStatus()]);
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
  $("apply-detection-config").addEventListener("click", applyDetectionConfig);
  $("restore-detection-defaults").addEventListener("click", () => renderDetectionConfig(detectionConfig));
  $("reload-detection-model").addEventListener("click", reloadDetectionModel);
  $("refresh-detection-debug").addEventListener("click", refreshDetectionDebug);
  $("detection-debug-view").addEventListener("change", refreshDetectionDebug);
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
    renderDetectionConfig(await api("/api/detection/config"));
    await refreshProfiles();
    refreshPreview();
    await poll();
    state.initialized = true;
    setMessage("页面已连接，可开始调参。", "success");
  } catch (error) { reportError("页面初始化失败", error); }
  window.setInterval(poll, 500);
}

document.addEventListener("DOMContentLoaded", initialize);
