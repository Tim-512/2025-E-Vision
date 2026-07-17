# Classical White-Board Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-neural, grayscale-only detector and tracker that acquires the complete white A4 target, follows its concentric-ring center through partial visibility, predicts for at most 3 frames/150 ms, exposes live tuning/debug views, and emits the approved laser-free protocol V2 payloads.

**Architecture:** Keep the existing latest-frame MVS acquisition, tuning service, web application, and single `BoardTracker` state machine. Add focused classical-vision modules for normalization, white-board proposals, concentric-ring geometry, scoring, partial fusion, and debug rendering; expose full and partial observations through a shared result contract. Preserve the old hybrid detector as an optional compatibility backend, make `classical` the default, and version serial payloads explicitly so V1 bytes are never decoded as V2.

**Tech Stack:** Python 3.10, NumPy, OpenCV, dataclasses/enums, FastAPI, vanilla HTML/CSS/JavaScript, PyYAML, pytest, CRC-16/CCITT-FALSE, Hikrobot MVS SDK on Jetson.

---

## Delivery boundary and execution rules

- Work only in the existing worktree `D:\projects\2026电赛练习项目\2025电赛E题\.worktrees\classical-vision` on branch `feature/classical-vision`.
- Use TDD for every behavior: add a focused failing test, run it, implement the minimum production code, rerun the focused test, then run neighboring regressions.
- Do not add HSV red thresholds, red-channel scores, `red_*` configuration fields, or any condition requiring the rings to appear red.
- Do not create a second tracking state machine. Full, partial, single-arc, and predicted observations all pass through `BoardTracker`.
- Detector configuration updates call `DetectorPort.apply_config()` only. They must not close/reopen the Hikrobot camera or replace its active session.
- The 405 nm laser is hardware-always-on when powered. Protocol V2 contains no laser mode, laser permission, or laser confidence field.
- Keep hybrid/YOLO modules importable for compatibility, but the default Jetson dashboard path uses the classical detector without loading Ultralytics.

## File responsibility map

### New runtime files

- `src/ev_vision/detection/contracts.py` — source enums and classical candidate/result contracts.
- `src/ev_vision/detection/image_normalization.py` — grayscale conversion, CLAHE, illumination normalization, masks and edges.
- `src/ev_vision/detection/white_board.py` — complete A4-like white-paper proposal generation and measurements.
- `src/ev_vision/detection/ring_geometry.py` — grayscale arc/ellipse fitting and 1:2:3:4:5 ratio checks.
- `src/ev_vision/detection/candidate_scoring.py` — acquisition/tracking score composition and ambiguity rejection.
- `src/ev_vision/detection/partial_board.py` — predicted ROI, partial white-region matching, arc/history fusion.
- `src/ev_vision/detection/debug_rendering.py` — named debug products and overlay rendering.
- `src/ev_vision/detection/classical_board.py` — `DetectorPort` orchestration.
- `tools/evaluate_classical_detector.py` — deterministic image/capture replay.

### Modified runtime files

- `src/ev_vision/config.py`, `config/default.yaml` — classical sections and validation.
- `src/ev_vision/detection/failures.py` — shared rejection reasons.
- `tests/fixtures/synthetic_board.py` — five-ring target, perspective, clipping, shadows, blur and distractors.
- `src/ev_vision/tracking/board_tracker.py`, `src/ev_vision/tracking/predictor.py` — source-aware gates and bounded prediction.
- `src/ev_vision/tuning/models.py`, `src/ev_vision/tuning/service.py` — generic snapshots and metrics.
- `src/ev_vision/web/camera_tuning_server.py`, `src/ev_vision/web/camera_tuning_app.py` — backend factory and API.
- `src/ev_vision/web/static/camera-tuning.html`, `.css`, `.js` — classical controls and debug views.
- `src/ev_vision/tuning/storage.py` — expanded capture archive.
- `src/ev_vision/protocol.py` — explicit V2 control/feedback payloads.
- `src/ev_vision/vision_result.py`, `src/ev_vision/control.py`, `src/ev_vision/models.py` — source-aware safe control mapping.
- `README.md`, `docs/jetson-classical-vision-acceptance.md` — deployment and acceptance.

---

### Task 0: Record the clean baseline

**Files:**
- Inspect: `docs/superpowers/specs/2026-07-17-classical-white-board-tracking-design.md`
- Inspect: `src/ev_vision/web/camera_tuning_server.py`
- Inspect: `src/ev_vision/tuning/service.py`
- Inspect: `src/ev_vision/tracking/board_tracker.py`
- Inspect: `src/ev_vision/protocol.py`

- [ ] **Step 1: Confirm branch and worktree state**

```bash
git branch --show-current
git status --short --branch
git log -3 --oneline
```

Expected: branch is `feature/classical-vision`; design commit `5f1a771` is present; only known inaccessible cache warnings may appear.

- [ ] **Step 2: Run the focused baseline**

```bash
python -m pytest tests/test_config.py tests/detection tests/tracking/test_board_tracker.py tests/tuning/test_service.py tests/web/test_detection_api.py tests/test_protocol.py tests/test_vision_result.py -q -p no:cacheprovider
```

Expected: existing tests pass. Record the count; this is a software baseline, not Jetson hardware acceptance.

- [ ] **Step 3: Confirm no baseline artifacts**

Run: `git status --short`

Expected: no tracked file changed. Do not commit in this task.

---

### Task 1: Add validated classical configuration

**Files:**
- Modify: `src/ev_vision/config.py`
- Modify: `config/default.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_default_detection_backend_is_classical():
    config = load_config("config/default.yaml")
    assert config.detection.backend == "classical"
    assert config.detection.normalization.clahe_clip_limit == 2.0
    assert config.detection.white_board.expected_aspect_ratio == pytest.approx(210 / 297)
    assert config.detection.rings.expected_radius_ratios == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert config.detection.tracking.predict_max_frames == 3
    assert config.detection.tracking.predict_max_ms == 150.0
    assert not any("red" in f.name.lower() for f in dataclasses.fields(config.detection.rings))


def test_rejects_invalid_prediction_limit(tmp_path):
    path = write_config(tmp_path, {"detection": {"tracking": {"predict_max_frames": 0}}})
    with pytest.raises(ConfigError, match="predict_max_frames"):
        load_config(path)


def test_rejects_non_monotonic_ring_ratios(tmp_path):
    path = write_config(tmp_path, {"detection": {"rings": {"expected_radius_ratios": [1, 2, 2, 4, 5]}}})
    with pytest.raises(ConfigError, match="expected_radius_ratios"):
        load_config(path)
```

- [ ] **Step 2: Verify intended failure**

Run: `python -m pytest tests/test_config.py -q -p no:cacheprovider`

Expected: FAIL because classical sections do not exist and the default remains `hybrid`.

- [ ] **Step 3: Add dataclasses and validation**

```python
@dataclass(frozen=True)
class ImageNormalizationConfig:
    gaussian_kernel: int = 3
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    illumination_kernel: int = 81
    white_percentile: float = 72.0
    white_local_offset: float = 10.0
    saturation_threshold: int = 250
    canny_low: int = 40
    canny_high: int = 120


@dataclass(frozen=True)
class WhiteBoardConfig:
    expected_aspect_ratio: float = 210.0 / 297.0
    aspect_ratio_tolerance: float = 0.24
    min_area_fraction: float = 0.015
    max_area_fraction: float = 0.92
    min_white_occupancy: float = 0.58
    max_texture_std: float = 58.0
    min_convexity: float = 0.90
    min_side_px: float = 45.0
    border_band_fraction: float = 0.045


@dataclass(frozen=True)
class RingGeometryConfig:
    expected_radius_ratios: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    ratio_tolerance: float = 0.18
    center_tolerance_fraction: float = 0.08
    min_arc_coverage: float = 0.18
    min_multiple_arcs: int = 2
    max_single_arc_frames: int = 2
    saturation_mask_radius_px: int = 12


@dataclass(frozen=True)
class ClassicalScoringConfig:
    white_weight: float = 0.24
    geometry_weight: float = 0.22
    ring_weight: float = 0.30
    border_weight: float = 0.08
    temporal_weight: float = 0.16
    acquisition_threshold: float = 0.66
    tracking_threshold: float = 0.52
    ambiguity_margin: float = 0.08
    max_texture_penalty: float = 0.20


@dataclass(frozen=True)
class BoardTrackingConfig:
    confirm_frames: int = 3
    predict_max_frames: int = 3
    predict_max_ms: float = 150.0
    lost_frames: int = 4
    max_single_arc_frames: int = 2
    max_center_jump_px: float = 160.0
    max_scale_jump_fraction: float = 0.30
    max_velocity_px_s: float = 5000.0
    max_acceleration_px_s2: float = 30000.0
    max_result_age_ms: float = 100.0
```

Add `normalization`, `white_board`, `rings`, and `classical_scoring` to `DetectionConfig` and `_DETECTION_SECTIONS`; set backend default to `classical`. Validate odd kernels, byte ranges, strictly increasing positive ratios, score weights summing to `1.0 ± 1e-6`, positive prediction limits, and `max_single_arc_frames <= predict_max_frames`. Add the exact defaults above to YAML.

- [ ] **Step 4: Run configuration tests**

Run: `python -m pytest tests/test_config.py -q -p no:cacheprovider`

Expected: PASS, including legacy YAML with absent new sections.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/config.py config/default.yaml tests/test_config.py
git commit -m "feat: add classical detector configuration"
```

---

### Task 2: Define shared classical contracts

**Files:**
- Create: `src/ev_vision/detection/contracts.py`
- Modify: `src/ev_vision/detection/failures.py`
- Create: `tests/detection/test_classical_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_source_priority():
    order = [
        ObservationSource.FULL_BOARD,
        ObservationSource.CONCENTRIC_ARCS,
        ObservationSource.FUSED_PARTIAL,
        ObservationSource.SINGLE_ARC,
        ObservationSource.WHITE_REGION,
        ObservationSource.PREDICTED,
    ]
    assert [item.priority for item in order] == sorted(
        [item.priority for item in order], reverse=True
    )


def test_partial_result_needs_no_four_corners():
    result = ClassicalBoardResult(
        timestamp_ns=1_000_000, source_sequence=7, detected=True,
        target_valid=True, tracking_state="TRACKING",
        observation_source=ObservationSource.CONCENTRIC_ARCS,
        confidence=0.81, center_px=(320.0, 240.0), corners_px=(),
    )
    assert result.homography_valid is False


def test_contract_has_no_red_or_laser_fields():
    names = {f.name for f in dataclasses.fields(ClassicalBoardResult)}
    assert not any("red" in name or "laser" in name for name in names)
```

- [ ] **Step 2: Verify collection failure**

Run: `python -m pytest tests/detection/test_classical_contracts.py -q -p no:cacheprovider`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement contracts**

```python
class ObservationSource(str, Enum):
    NONE = "NONE"
    FULL_BOARD = "FULL_BOARD"
    CONCENTRIC_ARCS = "CONCENTRIC_ARCS"
    SINGLE_ARC = "SINGLE_ARC"
    WHITE_REGION = "WHITE_REGION"
    FUSED_PARTIAL = "FUSED_PARTIAL"
    PREDICTED = "PREDICTED"

    @property
    def priority(self) -> int:
        return {self.NONE: 0, self.PREDICTED: 1, self.WHITE_REGION: 2,
                self.SINGLE_ARC: 3, self.FUSED_PARTIAL: 4,
                self.CONCENTRIC_ARCS: 5, self.FULL_BOARD: 6}[self]


@dataclass(frozen=True)
class ClassicalCandidateEvaluation:
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float] | None
    white_score: float
    geometry_score: float
    ring_score: float
    border_score: float
    temporal_score: float
    texture_penalty: float
    jump_penalty: float
    combined_score: float
    accepted: bool
    failure_reason: DetectionFailure | None = None
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassicalBoardResult:
    timestamp_ns: int
    source_sequence: int
    detected: bool
    target_valid: bool
    tracking_state: str
    observation_source: ObservationSource
    confidence: float
    center_px: tuple[float, float] | None
    corners_px: tuple[tuple[float, float], ...] = ()
    scale_px_per_mm: float | None = None
    velocity_px_s: tuple[float, float] | None = None
    predicted_frames: int = 0
    source_age_us: int = 0
    homography_valid: bool = False
    target_x_mm: float | None = None
    target_y_mm: float | None = None
    near_image_edge: bool = False
    partially_outside: bool = False
    failure_reason: DetectionFailure | None = None
    candidates: tuple[ClassicalCandidateEvaluation, ...] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    debug_images: Mapping[str, np.ndarray] = field(default_factory=dict, compare=False)
```

Extend `DetectionFailure` with `NO_WHITE_CANDIDATE`, `WHITE_OCCUPANCY_LOW`, `A4_GEOMETRY_INVALID`, `RING_RATIO_INVALID`, `RING_CENTER_INCONSISTENT`, `PARTIAL_HISTORY_REQUIRED`, `SINGLE_ARC_LIMIT`, `EXCESSIVE_SCALE_JUMP`, `EXCESSIVE_VELOCITY`, and `PREDICTION_EXPIRED`. Preserve hybrid types.

- [ ] **Step 4: Run focused and compatibility tests**

Run: `python -m pytest tests/detection/test_classical_contracts.py tests/detection/test_hybrid_board.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/detection/contracts.py src/ev_vision/detection/failures.py tests/detection/test_classical_contracts.py
git commit -m "feat: define classical detection contracts"
```

---

### Task 3: Extend the synthetic target fixture

**Files:**
- Modify: `tests/fixtures/synthetic_board.py`
- Create: `tests/fixtures/test_synthetic_board.py`

- [ ] **Step 1: Write failing fixture tests**

```python
def test_ring_target_contains_five_grayscale_rings():
    target = render_ring_target(image_size=(720, 960), ring_gray=92)
    assert target.center_px == pytest.approx((480.0, 360.0), abs=1.0)
    assert np.asarray(target.radii_px) / target.radii_px[0] == pytest.approx(
        [1, 2, 3, 4, 5], rel=0.03
    )


def test_ring_target_supports_partial_visibility_and_shadow():
    target = render_ring_target(
        image_size=(480, 640), board_center=(-20.0, 240.0),
        perspective=0.10, shadow_strength=0.45, blur_sigma=1.2,
    )
    assert target.image.shape == (480, 640, 3)
    assert target.partially_outside is True
    assert target.center_px[0] < 0.0
```

- [ ] **Step 2: Verify missing API**

Run: `python -m pytest tests/fixtures/test_synthetic_board.py -q -p no:cacheprovider`

Expected: FAIL because `render_ring_target` does not exist.

- [ ] **Step 3: Implement deterministic rendering**

Add `SyntheticRingTarget(image, center_px, corners_px, radii_px, partially_outside)`. Render canonical 210×297 paper, black tape, and:

```python
for radius_mm in (20.0, 40.0, 60.0, 80.0, 100.0):
    radius_px = int(round(radius_mm * pixels_per_mm))
    cv2.circle(canonical, canonical_center, radius_px,
               (ring_gray, ring_gray, ring_gray), ring_thickness_px, cv2.LINE_AA)
```

Apply perspective, illumination gradient, polygon shadow, blur and distractors. Seed noise with `np.random.default_rng(seed)`.

- [ ] **Step 4: Run fixture and legacy geometry tests**

Run: `python -m pytest tests/fixtures/test_synthetic_board.py tests/detection/test_board_geometry.py tests/detection/test_roi_board_geometry.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/synthetic_board.py tests/fixtures/test_synthetic_board.py
git commit -m "test: add grayscale concentric target fixtures"
```

---

### Task 4: Implement shared grayscale normalization

**Files:**
- Create: `src/ev_vision/detection/image_normalization.py`
- Create: `tests/detection/test_image_normalization.py`

- [ ] **Step 1: Write failing tests**

```python
def test_normalization_recovers_white_paper_under_shadow():
    target = render_ring_target(shadow_strength=0.55, ring_gray=105)
    result = normalize_frame(target.image, ImageNormalizationConfig())
    mask = np.zeros(result.white_mask.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(target.corners_px, np.int32), 255)
    occupancy = np.count_nonzero(result.white_mask & mask) / np.count_nonzero(mask)
    assert occupancy > 0.58


def test_saturated_spot_is_removed_from_ring_edges():
    image = render_ring_target().image.copy()
    cv2.circle(image, (480, 360), 10, (255, 255, 255), -1)
    result = normalize_frame(image, ImageNormalizationConfig(), mask_radius_px=12)
    assert np.count_nonzero(result.saturated_mask[348:373, 468:493]) > 0
    assert np.count_nonzero(result.ring_edge_mask[348:373, 468:493]) == 0
```

- [ ] **Step 2: Verify missing module**

Run: `python -m pytest tests/detection/test_image_normalization.py -q -p no:cacheprovider`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement pipeline**

```python
@dataclass(frozen=True)
class NormalizedFrame:
    gray: np.ndarray
    denoised_gray: np.ndarray
    normalized_gray: np.ndarray
    white_mask: np.ndarray
    gradient: np.ndarray
    edge_mask: np.ndarray
    saturated_mask: np.ndarray
    ring_edge_mask: np.ndarray


def normalize_frame(image, config, *, mask_radius_px=12):
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("image must be a non-empty gray or BGR array")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    denoised = cv2.GaussianBlur(gray, (config.gaussian_kernel,) * 2, 0)
    local = cv2.createCLAHE(config.clahe_clip_limit, (config.clahe_grid_size,) * 2).apply(denoised)
    background = cv2.GaussianBlur(local, (config.illumination_kernel,) * 2, 0)
    normalized = cv2.addWeighted(local, 1.0, background, -1.0, 128.0)
    threshold = max(0.0, np.percentile(normalized, config.white_percentile) - config.white_local_offset)
    white = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)[1]
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, 3)
    gradient = cv2.magnitude(gx, gy)
    edges = cv2.Canny(normalized, config.canny_low, config.canny_high)
    saturated = cv2.threshold(gray, config.saturation_threshold, 255, cv2.THRESH_BINARY)[1]
    saturated = cv2.dilate(saturated, disk_kernel(mask_radius_px))
    ring_edges = cv2.bitwise_and(edges, cv2.bitwise_not(saturated))
    return NormalizedFrame(gray, denoised, normalized, white, gradient, edges, saturated, ring_edges)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/detection/test_image_normalization.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/detection/image_normalization.py tests/detection/test_image_normalization.py
git commit -m "feat: add grayscale target normalization"
```

---

### Task 5: Detect complete white A4 candidates

**Files:**
- Create: `src/ev_vision/detection/white_board.py`
- Create: `tests/detection/test_white_board.py`

- [ ] **Step 1: Write failing complete-board tests**

```python
def candidates_for(image):
    frame = normalize_frame(image, ImageNormalizationConfig())
    return find_white_board_candidates(frame, WhiteBoardConfig())


def test_complete_a4_board_is_a_candidate():
    target = render_ring_target(perspective=0.12, shadow_strength=0.35)
    best = max(candidates_for(target.image), key=lambda item: item.geometry_score)
    assert best.white_occupancy >= 0.58
    assert best.aspect_ratio_error <= 0.24
    assert best.center_px == pytest.approx(target.center_px, abs=10.0)


def test_plain_white_wall_is_not_a_candidate():
    assert candidates_for(np.full((720, 960, 3), 220, np.uint8)) == ()


def test_small_white_distractor_is_rejected():
    image = np.full((720, 960, 3), 35, np.uint8)
    cv2.rectangle(image, (20, 20), (55, 70), (230, 230, 230), -1)
    assert candidates_for(image) == ()
```

- [ ] **Step 2: Verify missing module**

Run: `python -m pytest tests/detection/test_white_board.py -q -p no:cacheprovider`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement extraction and measurements**

```python
@dataclass(frozen=True)
class WhiteBoardCandidate:
    contour: np.ndarray = field(compare=False)
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float]
    bbox_xyxy: tuple[int, int, int, int]
    area_fraction: float
    white_occupancy: float
    convexity: float
    aspect_ratio_error: float
    texture_std: float
    border_support: float
    geometry_score: float


def find_white_board_candidates(normalized, config, *, roi_xyxy=None):
    contours, _ = cv2.findContours(
        normalized.white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    measured = [
        _measure_candidate(contour, normalized, config, roi_xyxy)
        for contour in contours
    ]
    accepted = [candidate for candidate in measured if candidate is not None]
    return tuple(sorted(accepted, key=lambda item: item.geometry_score, reverse=True))
```

Use hull-area convexity, `approxPolyDP` with `minAreaRect` fallback, ordered corners, and a 420×594 rectified plane. Measure A4 aspect error, white occupancy after closing tape/ring gaps, texture standard deviation after 15×15 blur, and an inside/outside edge band for optional dark-border support. Border support is never a hard rejection.

- [ ] **Step 4: Run white-board and normalization tests**

Run: `python -m pytest tests/detection/test_white_board.py tests/detection/test_image_normalization.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/detection/white_board.py tests/detection/test_white_board.py
git commit -m "feat: detect complete white A4 candidates"
```

---

### Task 6: Fit grayscale concentric ring geometry

**Files:**
- Create: `src/ev_vision/detection/ring_geometry.py`
- Create: `tests/detection/test_ring_geometry.py`

- [ ] **Step 1: Write failing ring tests**

```python
def test_five_gray_rings_pass_common_center_and_ratio_checks():
    target = render_ring_target(ring_gray=118, perspective=0.08)
    edges = normalize_frame(target.image, ImageNormalizationConfig()).ring_edge_mask
    result = detect_concentric_arcs(edges, RingGeometryConfig())
    assert result.valid is True
    assert result.center_px == pytest.approx(target.center_px, abs=8.0)
    assert len(result.arcs) >= 3
    assert result.ratio_score > 0.75
    assert result.common_center_score > 0.75


def test_non_red_gray_intensities_are_equivalent():
    dark = render_ring_target(ring_gray=70)
    light = render_ring_target(ring_gray=145)
    a = detect_concentric_arcs(normalize_frame(dark.image, ImageNormalizationConfig()).ring_edge_mask, RingGeometryConfig())
    b = detect_concentric_arcs(normalize_frame(light.image, ImageNormalizationConfig()).ring_edge_mask, RingGeometryConfig())
    assert a.valid and b.valid
    assert a.center_px == pytest.approx(b.center_px, abs=3.0)


def test_wrong_radius_ratios_are_rejected():
    image = np.zeros((720, 960), np.uint8)
    for radius in (35, 69, 111, 143, 209):
        cv2.circle(image, (480, 360), radius, 255, 2)
    result = detect_concentric_arcs(image, RingGeometryConfig(ratio_tolerance=0.10))
    assert result.valid is False
    assert result.ratio_score < 0.7


def test_partial_multiple_arcs_recover_center_near_edge():
    target = render_ring_target(board_center=(5.0, 360.0))
    edges = normalize_frame(target.image, ImageNormalizationConfig()).ring_edge_mask
    result = detect_concentric_arcs(edges, RingGeometryConfig(min_arc_coverage=0.12))
    assert result.visible_arc_count >= 2
    assert result.center_px == pytest.approx(target.center_px, abs=15.0)
```

- [ ] **Step 2: Verify missing module**

Run: `python -m pytest tests/detection/test_ring_geometry.py -q -p no:cacheprovider`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement robust arc grouping**

```python
@dataclass(frozen=True)
class ArcFit:
    center_px: tuple[float, float]
    axes_px: tuple[float, float]
    angle_deg: float
    equivalent_radius_px: float
    coverage: float
    residual_px: float


@dataclass(frozen=True)
class RingGeometryResult:
    valid: bool
    center_px: tuple[float, float] | None
    arcs: tuple[ArcFit, ...]
    visible_arc_count: int
    common_center_score: float
    ratio_score: float
    coverage_score: float
    scale_px_per_mm: float | None
    failure_reason: DetectionFailure | None


def detect_concentric_arcs(edge_mask, config, *, expected_center_px=None,
                            expected_scale_px_per_mm=None, roi_xyxy=None):
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    fits = tuple(_fit_arc(c) for c in contours if len(c) >= 12)
    fits = tuple(f for f in fits if f is not None and f.coverage >= config.min_arc_coverage)
    return _select_concentric_group(
        fits, config, expected_center_px, expected_scale_px_per_mm
    )
```

Fit ellipses with `cv2.fitEllipse`, compute occupied angular-bin coverage, reject extreme eccentricity/residual, cluster centers using a fraction of median radius, and match observed radii to subsets of `(20, 40, 60, 80, 100)` mm with one least-squares scale. Single arcs may be returned as evidence, but `valid` requires `min_multiple_arcs`.

- [ ] **Step 4: Run ring tests**

Run: `python -m pytest tests/detection/test_ring_geometry.py -q -p no:cacheprovider`

Expected: PASS on gray, clipped and rejection fixtures.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/detection/ring_geometry.py tests/detection/test_ring_geometry.py
git commit -m "feat: fit grayscale concentric target rings"
```

---

### Task 7: Score candidates and reject ambiguous false locks

**Files:**
- Create: `src/ev_vision/detection/candidate_scoring.py`
- Create: `tests/detection/test_candidate_scoring.py`

- [ ] **Step 1: Write failing scoring tests**

```python
def evidence(**changes):
    values = dict(
        white_score=0.90, geometry_score=0.85, ring_score=0.90,
        border_score=0.40, temporal_score=0.0,
        texture_penalty=0.02, jump_penalty=0.0,
    )
    values.update(changes)
    return CandidateEvidence(**values)


def test_search_accepts_without_history_but_requires_ring_identity():
    assert rank_candidates([evidence()], ClassicalScoringConfig(), tracking=False).best
    rejected = rank_candidates(
        [evidence(ring_score=0.0)], ClassicalScoringConfig(), tracking=False
    )
    assert rejected.best is None


def test_tracking_rejects_large_jump():
    ranked = rank_candidates(
        [evidence(jump_penalty=0.8)], ClassicalScoringConfig(), tracking=True
    )
    assert ranked.best is None
    assert ranked.evaluations[0].failure_reason == DetectionFailure.EXCESSIVE_POSITION_JUMP


def test_near_equal_candidates_are_ambiguous():
    ranked = rank_candidates(
        [evidence(ring_score=0.90), evidence(ring_score=0.88)],
        ClassicalScoringConfig(ambiguity_margin=0.08), tracking=False,
    )
    assert ranked.best is None
    assert ranked.failure_reason == DetectionFailure.AMBIGUOUS_CANDIDATES
```

- [ ] **Step 2: Verify missing module**

Run: `python -m pytest tests/detection/test_candidate_scoring.py -q -p no:cacheprovider`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement scoring**

```python
@dataclass(frozen=True)
class CandidateEvidence:
    white_score: float
    geometry_score: float
    ring_score: float
    border_score: float
    temporal_score: float
    texture_penalty: float
    jump_penalty: float


def combined_score(item, config):
    positive = (
        config.white_weight * item.white_score
        + config.geometry_weight * item.geometry_score
        + config.ring_weight * item.ring_score
        + config.border_weight * item.border_score
        + config.temporal_weight * item.temporal_score
    )
    return max(0.0, min(1.0, positive - item.texture_penalty - item.jump_penalty))
```

`rank_candidates` sorts descending, applies acquisition/tracking thresholds, requires ring identity during acquisition, applies hard jump/scale gates before weighted scoring, and returns `AMBIGUOUS_CANDIDATES` when accepted top-two scores differ by less than the margin. Populate explicit rejection reasons.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/detection/test_candidate_scoring.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/detection/candidate_scoring.py tests/detection/test_candidate_scoring.py
git commit -m "feat: score classical target candidates"
```

---

### Task 8: Fuse partial white regions, arcs and history

**Files:**
- Create: `src/ev_vision/detection/partial_board.py`
- Create: `tests/detection/test_partial_board.py`

- [ ] **Step 1: Write failing fusion tests**

```python
def history():
    return BoardHistory(
        center_px=(320.0, 240.0), velocity_px_s=(100.0, 0.0),
        scale_px_per_mm=2.0,
        corners_px=((110.0, -57.0), (530.0, -57.0),
                    (530.0, 537.0), (110.0, 537.0)),
        timestamp_ns=1_000_000_000,
    )


def test_multiple_arcs_and_white_region_are_fused():
    result = fuse_partial_observation(
        history=history(), timestamp_ns=1_020_000_000,
        image_size=(640, 480), arc_center_px=(323.0, 241.0),
        arc_count=3, arc_confidence=0.84,
        white_center_px=(328.0, 239.0), white_confidence=0.70,
    )
    assert result.source == ObservationSource.FUSED_PARTIAL
    assert result.center_px == pytest.approx((324.9, 240.6), abs=2.0)
    assert result.partially_outside is True


def test_partial_requires_confirmed_history():
    result = fuse_partial_observation(
        history=None, timestamp_ns=1, image_size=(640, 480),
        arc_center_px=(320.0, 240.0), arc_count=3, arc_confidence=0.9,
        white_center_px=None, white_confidence=0.0,
    )
    assert result.valid is False
    assert result.failure_reason == DetectionFailure.PARTIAL_HISTORY_REQUIRED


def test_single_arc_is_strictly_gated():
    result = fuse_partial_observation(
        history=history(), timestamp_ns=1_020_000_000,
        image_size=(640, 480), arc_center_px=(520.0, 420.0),
        arc_count=1, arc_confidence=0.95, observed_scale_px_per_mm=4.0,
        white_center_px=None, white_confidence=0.0,
    )
    assert result.valid is False
```

- [ ] **Step 2: Verify missing module**

Run: `python -m pytest tests/detection/test_partial_board.py -q -p no:cacheprovider`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement history, ROI and fusion**

Expose `BoardHistory`, `PartialObservation`, `predicted_roi()` and `fuse_partial_observation()`. Predict by constant velocity, expand the historical board box by 20%, clip to frame, set `near_image_edge` within 8% of either dimension, and set `partially_outside` if transformed historical corners cross a boundary.

```python
def weighted_center(items):
    total = sum(weight for _, weight in items)
    return (
        sum(point[0] * weight for point, weight in items) / total,
        sum(point[1] * weight for point, weight in items) / total,
    )

sources = [(predicted_center, 0.35)]
if arc_center_px is not None:
    sources.append((arc_center_px, 0.55 * arc_confidence))
if white_center_px is not None:
    sources.append((white_center_px, 0.25 * white_confidence))
```

Choose `CONCENTRIC_ARCS` for multi-arc-only evidence, `FUSED_PARTIAL` for combined evidence, `SINGLE_ARC` only for history-consistent one-arc evidence, and `WHITE_REGION` only for strong local white agreement. Never establish a target without history.

- [ ] **Step 4: Run partial tests**

Run: `python -m pytest tests/detection/test_partial_board.py -q -p no:cacheprovider`

Expected: PASS, including parametrized clipping from all four sides.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/detection/partial_board.py tests/detection/test_partial_board.py
git commit -m "feat: fuse partial target observations"
```

---

### Task 9: Extend the existing tracker for bounded prediction

**Files:**
- Modify: `src/ev_vision/tracking/board_tracker.py`
- Modify: `src/ev_vision/tracking/predictor.py`
- Modify: `tests/tracking/test_board_tracker.py`

- [ ] **Step 1: Add failing state-machine tests**

```python
def test_partial_is_valid_only_after_full_confirmation():
    tracker = BoardTracker(BoardTrackingConfig(confirm_frames=2))
    assert tracker.update(full(1, 1_000_000_000)).state == TrackingState.CONFIRMING
    assert tracker.update(full(2, 1_010_000_000)).state == TrackingState.TRACKING
    result = tracker.update(partial(3, 1_020_000_000, ObservationSource.CONCENTRIC_ARCS))
    assert result.state == TrackingState.TRACKING
    assert result.target_valid is True


def test_prediction_expires_on_frame_limit():
    tracker = confirmed_tracker(BoardTrackingConfig(predict_max_frames=3, predict_max_ms=150.0))
    for sequence in (3, 4, 5):
        result = tracker.update(miss(sequence, 1_000_000_000 + sequence * 10_000_000))
        assert result.state == TrackingState.PREDICTING
        assert result.target_valid is True
    assert tracker.update(miss(6, 1_060_000_000)).state == TrackingState.LOST


def test_prediction_expires_on_time_limit_first():
    tracker = confirmed_tracker(BoardTrackingConfig(predict_max_frames=10, predict_max_ms=150.0))
    result = tracker.update(miss(3, 1_200_000_001))
    assert result.state == TrackingState.LOST
    assert result.target_valid is False


def test_lost_cannot_reacquire_from_partial_arcs():
    tracker = lost_tracker()
    assert tracker.update(partial(20, 2_000_000_000, ObservationSource.CONCENTRIC_ARCS)).state == TrackingState.LOST
    assert tracker.update(full(21, 2_010_000_000)).state == TrackingState.CONFIRMING


def test_single_arc_limit_and_scale_jump_are_rejected():
    tracker = confirmed_tracker(BoardTrackingConfig(max_single_arc_frames=2))
    assert tracker.update(partial(3, 1_030_000_000, ObservationSource.SINGLE_ARC)).target_valid
    assert tracker.update(partial(4, 1_040_000_000, ObservationSource.SINGLE_ARC)).target_valid
    assert tracker.update(partial(5, 1_050_000_000, ObservationSource.SINGLE_ARC)).target_valid is False
    assert tracker.update(full(6, 1_060_000_000, scale=3.0)).failure_reason == DetectionFailure.EXCESSIVE_SCALE_JUMP
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/tracking/test_board_tracker.py -q -p no:cacheprovider`

Expected: FAIL because observations have no source/scale/confidence and prediction has no time limit.

- [ ] **Step 3: Extend the same state machine**

```python
@dataclass(frozen=True)
class TrackObservation:
    timestamp_ns: int
    source_sequence: int
    detected: bool
    source: ObservationSource = ObservationSource.NONE
    center_px: tuple[float, float] | None = None
    corners_px: tuple[tuple[float, float], ...] = ()
    scale_px_per_mm: float | None = None
    confidence: float = 0.0
    failure_reason: DetectionFailure | None = None


@dataclass(frozen=True)
class TrackedBoardResult:
    state: TrackingState
    target_valid: bool
    observation: TrackObservation | None
    observation_source: ObservationSource
    predicted_center_px: tuple[float, float] | None
    confirmation_count: int
    miss_count: int
    predicted_frames: int
    source_age_us: int
    velocity_px_s: tuple[float, float] | None
    scale_px_per_mm: float | None
    failure_reason: DetectionFailure | None
```

Rules: `SEARCHING`/`LOST` accept only `FULL_BOARD` with four finite corners; confirmed tracking accepts gated full/partial sources; full board refresh resets prediction/single-arc counters; prediction remains valid only while both frame and time limits pass; expiry clears history and enters `LOST`; reject stale/non-finite/jump/scale/velocity/acceleration observations before remembering them.

Update `MotionPredictor`:

```python
def predict(self, target_ns: int, *, max_horizon_ns: int):
    if self._latest is None or target_ns - self._latest[1] > max_horizon_ns:
        return None
    return self._constant_velocity_prediction(target_ns)
```

- [ ] **Step 4: Run tracker and hybrid regressions**

Run: `python -m pytest tests/tracking/test_board_tracker.py tests/detection/test_hybrid_board.py tests/test_vision_result.py -q -p no:cacheprovider`

Expected: PASS; adapt hybrid observations to `FULL_BOARD` without changing hybrid behavior.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/tracking/board_tracker.py src/ev_vision/tracking/predictor.py tests/tracking/test_board_tracker.py
git commit -m "feat: track partial targets with bounded prediction"
```

---

### Task 10: Orchestrate the complete classical detector

**Files:**
- Create: `src/ev_vision/detection/debug_rendering.py`
- Create: `src/ev_vision/detection/classical_board.py`
- Modify: `src/ev_vision/detection/board_solution.py`
- Create: `tests/detection/test_classical_board.py`

- [ ] **Step 1: Write failing end-to-end tests**

```python
def test_complete_board_confirms_and_uses_a4_center():
    target = render_ring_target(perspective=0.10, shadow_strength=0.30)
    subject = ClassicalBoardDetector(DetectionConfig())
    subject.detect(target.image, captured_ns=1_000_000_000, source_sequence=1)
    subject.detect(target.image, captured_ns=1_010_000_000, source_sequence=2)
    result = subject.detect(
        target.image, captured_ns=1_020_000_000,
        source_sequence=3, include_debug=True,
    )
    assert result.target_valid is True
    assert result.observation_source == ObservationSource.FULL_BOARD
    assert result.center_px == pytest.approx(target.center_px, abs=10.0)
    assert result.homography_valid is True
    assert set(result.debug_images) >= {
        "normalized-gray", "white-mask", "edge-mask",
        "ring-arcs", "candidate-scores",
    }


def test_confirmed_detector_tracks_clipped_target():
    subject = confirmed_detector()
    target = render_ring_target(board_center=(10.0, 360.0))
    result = subject.detect(target.image, captured_ns=1_040_000_000, source_sequence=4)
    assert result.target_valid is True
    assert result.observation_source in {
        ObservationSource.CONCENTRIC_ARCS,
        ObservationSource.FUSED_PARTIAL,
    }
    assert result.partially_outside is True


def test_lost_detector_rejects_partial_only_reacquisition():
    subject = expired_detector()
    target = render_ring_target(board_center=(-30.0, 360.0))
    result = subject.detect(target.image, captured_ns=2_000_000_000, source_sequence=20)
    assert result.target_valid is False
    assert result.tracking_state == "LOST"


def test_config_apply_and_reload_need_no_model():
    subject = ClassicalBoardDetector(DetectionConfig())
    subject.apply_config(dataclasses.replace(
        DetectionConfig(), rings=RingGeometryConfig(ratio_tolerance=0.20)
    ))
    subject.reload_model()
    assert subject.model_state == "READY"
    assert subject.model_backend == "classical"
```

- [ ] **Step 2: Verify missing orchestrator**

Run: `python -m pytest tests/detection/test_classical_board.py -q -p no:cacheprovider`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement full-search and tracking-ROI paths**

```python
class ClassicalBoardDetector:
    model_state = "READY"
    model_backend = "classical"
    model_path = None

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._tracker = BoardTracker(config.tracking)
        self._history: BoardHistory | None = None

    def detect(self, image, *, captured_ns, source_sequence,
               include_debug=False, update_tracker=True):
        normalized = normalize_frame(
            image, self._config.normalization,
            mask_radius_px=self._config.rings.saturation_mask_radius_px,
        )
        acquisition = self._tracker.latest.state in {
            TrackingState.SEARCHING, TrackingState.CONFIRMING, TrackingState.LOST
        }
        observation, candidates = (
            self._acquire_full_board(normalized, image.shape)
            if acquisition else
            self._track_in_predicted_roi(normalized, image.shape, captured_ns)
        )
        tracked = (
            self._tracker.update(observation, now_ns=captured_ns)
            if update_tracker else
            self._tracker.preview(observation, now_ns=captured_ns)
        )
        return self._build_result(tracked, candidates, normalized, include_debug)

    def reset(self):
        self._tracker.reset()
        self._history = None

    def apply_config(self, config):
        self._config = config
        self._tracker.apply_config(config.tracking)

    def reload_model(self):
        return None
```

Full acquisition: find white candidates, rectify each, evaluate ring geometry in the rectified plane, score, choose an unambiguous candidate, solve homography, and derive center from canonical A4 `(105, 148.5)` mm. Tracking: search predicted ROI for a full refresh first; otherwise run ring detection and local white-region fusion; if no real observation survives, feed a miss to the same tracker.

`debug_rendering.py` exports:

```python
DEBUG_IMAGE_NAMES = (
    "normalized-gray", "white-mask", "edge-mask",
    "ring-arcs", "candidate-scores",
)
```

Candidate debug prints total score and every rejection reason. Ring debug draws fitted arcs, common center, predicted center, and ROI; display colors do not affect acceptance.

- [ ] **Step 4: Run detector and geometry tests**

Run: `python -m pytest tests/detection/test_classical_board.py tests/detection/test_white_board.py tests/detection/test_ring_geometry.py tests/detection/test_board_solution.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/detection/classical_board.py src/ev_vision/detection/debug_rendering.py src/ev_vision/detection/board_solution.py tests/detection/test_classical_board.py
git commit -m "feat: orchestrate classical target detection"
```

---

### Task 11: Select the classical detector in server and service

**Files:**
- Modify: `src/ev_vision/web/camera_tuning_server.py`
- Modify: `src/ev_vision/tuning/service.py`
- Modify: `src/ev_vision/tuning/models.py`
- Modify: `tests/web/test_camera_tuning_tool.py`
- Modify: `tests/tuning/test_service.py`

- [ ] **Step 1: Write failing factory and camera-continuity tests**

```python
def test_build_detector_uses_classical_without_model_factory():
    detector = build_detector(
        DetectionConfig(backend="classical"), board=BoardConfig(),
        backend_factory=lambda *_: pytest.fail("YOLO factory called"),
    )
    assert isinstance(detector, ClassicalBoardDetector)


def test_detector_config_update_does_not_replace_camera(running_service):
    original_camera = running_service.active_camera
    original_count = running_service.runtime_snapshot().frame_count
    changed = dataclasses.replace(
        running_service.detection_config(),
        rings=RingGeometryConfig(ratio_tolerance=0.20),
    )
    running_service.apply_detection_config(changed)
    assert running_service.active_camera is original_camera
    wait_until(lambda: running_service.runtime_snapshot().frame_count > original_count)
    assert running_service.runtime_snapshot().state == "Connected"


def test_classical_reload_is_clean_no_op(running_service):
    running_service.reload_detection_model()
    status = running_service.latest_detection()
    assert status.model_state == "READY"
    assert status.model_backend == "classical"
```

- [ ] **Step 2: Verify failures**

Run: `python -m pytest tests/web/test_camera_tuning_tool.py tests/tuning/test_service.py -q -p no:cacheprovider`

Expected: FAIL because construction and service mapping are hybrid-specific.

- [ ] **Step 3: Implement generic integration**

```python
def build_detector(config, *, board, backend_factory=None):
    if config.backend == "classical":
        return ClassicalBoardDetector(config)
    if config.backend == "hybrid":
        return _build_hybrid_detector(
            config, board=board, backend_factory=backend_factory
        )
    raise ValueError(f"unsupported detection backend: {config.backend}")
```

Change `DetectorPort.detect()` to return `HybridBoardResult | ClassicalBoardResult`. Replace `_detect_hybrid` with `_detect_board` and type-dispatch snapshot conversion. Extend `DetectionSnapshot` with:

```python
observation_source: str = "NONE"
confidence: float = 0.0
scale_px_per_mm: float | None = None
velocity_px_s: tuple[float, float] | None = None
predicted_frames: int = 0
source_age_us: int = 0
near_image_edge: bool = False
partially_outside: bool = False
rejection_reasons: tuple[str, ...] = ()
```

Keep old model fields with defaults. `apply_detection_config()` mutates only detector/config and invalidates derived detection/debug snapshots; it must not call camera close/open. Update startup text to state that the laser is hardware-always-on when powered, rather than claiming software can keep it off.

- [ ] **Step 4: Run service and camera regressions**

Run: `python -m pytest tests/tuning/test_service.py tests/web/test_camera_tuning_tool.py tests/camera/test_latest_frame.py tests/camera/test_hikrobot_adapter.py -q -p no:cacheprovider`

Expected: PASS, including camera close-timeout tests.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/web/camera_tuning_server.py src/ev_vision/tuning/service.py src/ev_vision/tuning/models.py tests/web/test_camera_tuning_tool.py tests/tuning/test_service.py
git commit -m "feat: integrate classical detector service"
```

---

### Task 12: Expose classical configuration, status and debug API

**Files:**
- Modify: `src/ev_vision/web/camera_tuning_app.py`
- Modify: `tests/web/test_detection_api.py`
- Modify: `tests/web/test_camera_tuning_api.py`

- [ ] **Step 1: Write failing API tests**

```python
def test_config_exposes_classical_sections(client):
    payload = client.get("/api/detection/config").json()
    assert payload["backend"] == "classical"
    assert payload["normalization"]["clahe_clip_limit"] == 2.0
    assert payload["rings"]["expected_radius_ratios"] == [1, 2, 3, 4, 5]
    assert "red_threshold" not in json.dumps(payload).lower()


def test_put_config_does_not_restart_camera(client, service):
    generation = service.camera_generation
    response = client.put("/api/detection/config", json={
        "backend": "classical",
        "normalization": {"clahe_clip_limit": 2.4},
        "white_board": {"min_white_occupancy": 0.55},
        "rings": {"ratio_tolerance": 0.20},
        "classical_scoring": {"tracking_threshold": 0.50},
        "tracking": {"predict_max_frames": 3, "predict_max_ms": 150.0},
    })
    assert response.status_code == 200
    assert service.camera_generation == generation


def test_status_reports_source_and_prediction(client):
    payload = client.get("/api/detection/status").json()
    assert set(payload) >= {
        "tracking_state", "observation_source", "confidence",
        "source_age_us", "predicted_frames",
    }


@pytest.mark.parametrize("name", [
    "normalized-gray", "white-mask", "edge-mask",
    "ring-arcs", "candidate-scores",
])
def test_classical_debug_route(client, seeded_debug, name):
    response = client.get(f"/api/detection/debug/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
```

- [ ] **Step 2: Verify API failures**

Run: `python -m pytest tests/web/test_detection_api.py tests/web/test_camera_tuning_api.py -q -p no:cacheprovider`

Expected: FAIL because request models and debug names expose only hybrid settings.

- [ ] **Step 3: Add exact request/response models**

Add Pydantic request classes mirroring classical dataclasses. Preserve unspecified fields by replacing submitted sections against current config, then run the same validation used by YAML. Use:

```python
DEBUG_IMAGE_NAMES = frozenset({
    "model-candidates", "geometry-accepted", "geometry-rejected",
    "normalized-gray", "white-mask", "edge-mask",
    "ring-arcs", "candidate-scores",
})
```

Return classical status fields. Keep reload routes for compatibility, but for classical return `{"status":"ready","backend":"classical","reloaded":false}`.

- [ ] **Step 4: Run all web tests**

Run: `python -m pytest tests/web -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/web/camera_tuning_app.py tests/web/test_detection_api.py tests/web/test_camera_tuning_api.py
git commit -m "feat: expose classical detector API"
```

---

### Task 13: Add classical controls and debug views to dashboard

**Files:**
- Modify: `src/ev_vision/web/static/camera-tuning.html`
- Modify: `src/ev_vision/web/static/camera-tuning.css`
- Modify: `src/ev_vision/web/static/camera-tuning.js`
- Modify: `tests/web/test_camera_tuning_tool.py`

- [ ] **Step 1: Write failing static UI assertions**

```python
def test_dashboard_contains_classical_controls_and_status():
    joined = (
        packaged_static("camera-tuning.html")
        + packaged_static("camera-tuning.js")
    ).lower()
    for token in (
        "clahe-clip-limit", "min-white-occupancy",
        "ring-ratio-tolerance", "min-arc-coverage",
        "predict-max-frames", "predict-max-ms",
        "observation-source", "predicted-frames",
        "normalized-gray", "white-mask", "edge-mask",
        "ring-arcs", "candidate-scores",
    ):
        assert token in joined
    assert "red-threshold" not in joined
    assert "hsv" not in joined


def test_classical_hides_model_reload():
    script = packaged_static("camera-tuning.js")
    assert 'backend === "classical"' in script
    assert 'modelReloadButton.hidden = classical' in script
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/web/test_camera_tuning_tool.py -q -p no:cacheprovider`

Expected: FAIL because classical UI is absent.

- [ ] **Step 3: Implement grouped controls and live status**

Add groups for light normalization, white board, concentric rings, and tracking/prediction. Show state, source, confidence, result/source age, velocity, scale, prediction count, edge/partial flags, and rejection reasons. Debug image requests include current sequence:

```javascript
function setDebugImage(name, sourceSequence) {
  const query = sourceSequence == null
    ? ""
    : "?sequence=" + encodeURIComponent(sourceSequence);
  elements.debugImage.src =
    "/api/detection/debug/" + encodeURIComponent(name) + query;
}

function updateBackendVisibility(config) {
  const classical = config.backend === "classical";
  elements.classicalControls.hidden = !classical;
  elements.hybridControls.hidden = classical;
  elements.modelReloadButton.hidden = classical;
}
```

Applying detector settings calls only `PUT /api/detection/config`; it must not call camera parameter or reload routes.

- [ ] **Step 4: Run UI/package tests**

Run: `python -m pytest tests/web/test_camera_tuning_tool.py tests/test_tuning_packaging.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/web/static/camera-tuning.html src/ev_vision/web/static/camera-tuning.css src/ev_vision/web/static/camera-tuning.js tests/web/test_camera_tuning_tool.py
git commit -m "feat: add classical tuning dashboard"
```

---

### Task 14: Save expanded classical debug captures safely

**Files:**
- Modify: `src/ev_vision/tuning/storage.py`
- Modify: `src/ev_vision/web/camera_tuning_app.py`
- Modify: `tests/tuning/test_storage.py`
- Modify: `tests/web/test_camera_tuning_api.py`

- [ ] **Step 1: Write failing capture tests**

```python
def test_classical_capture_saves_debug_products(storage, classical_snapshot):
    capture = storage.save_capture(
        classical_snapshot, overlay_image=classical_snapshot.frame.image
    )
    expected = {
        "original.png", "overlay.png", "normalized-gray.png",
        "white-mask.png", "edge-mask.png", "ring-arcs.png",
        "candidate-scores.png", "metadata.yaml",
    }
    assert expected <= {path.name for path in capture.iterdir()}
    metadata = yaml.safe_load((capture / "metadata.yaml").read_text("utf-8"))
    assert metadata["detection"]["observation_source"] == "FULL_BOARD"


def test_capture_api_lists_classical_files(client):
    response = client.post("/api/captures", json={"enabled": True})
    assert response.status_code == 200
    assert "ring-arcs.png" in response.json()["files"]
```

- [ ] **Step 2: Verify missing files**

Run: `python -m pytest tests/tuning/test_storage.py tests/web/test_camera_tuning_api.py -q -p no:cacheprovider`

Expected: FAIL because whitelists contain only hybrid names.

- [ ] **Step 3: Extend whitelist, atomic write and cleanup**

```python
_CAPTURE_DEBUG_FILES = {
    "model-candidates": "model-candidates.png",
    "geometry-accepted": "geometry-accepted.png",
    "geometry-rejected": "geometry-rejected.png",
    "normalized-gray": "normalized-gray.png",
    "white-mask": "white-mask.png",
    "edge-mask": "edge-mask.png",
    "ring-arcs": "ring-arcs.png",
    "candidate-scores": "candidate-scores.png",
}
```

Use existing temporary-file/atomic-replace flow. Add every `.tmp.png` and final name to safe cleanup, and final names to API response whitelist. Save only debug images matching the captured frame sequence.

- [ ] **Step 4: Run storage/API regressions**

Run: `python -m pytest tests/tuning/test_storage.py tests/web/test_camera_tuning_api.py -q -p no:cacheprovider`

Expected: PASS, including path/symlink safety tests.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/tuning/storage.py src/ev_vision/web/camera_tuning_app.py tests/tuning/test_storage.py tests/web/test_camera_tuning_api.py
git commit -m "feat: archive classical debug captures"
```

---

### Task 15: Introduce protocol V2 without breaking explicit V1 compatibility

**Files:**
- Modify: `src/ev_vision/protocol.py`
- Modify: `tests/test_protocol.py`

- [ ] **Step 1: Write failing V2 layout, round-trip and safety tests**

Append tests that pin the byte layout, enum values, all flag bits, V1 compatibility, and the zero-rate invariant:

```python
from ev_vision.protocol import (
    PROTOCOL_VERSION, PROTOCOL_VERSION_V1, PROTOCOL_VERSION_V2,
    GimbalFeedbackPayloadV2, ObservationSourceCode, TrackingStateCode,
    VisionControlFlagsV2, VisionControlPayloadV2,
)


def test_protocol_v2_versions_and_payload_sizes() -> None:
    assert (PROTOCOL_VERSION_V1, PROTOCOL_VERSION_V2, PROTOCOL_VERSION) == (1, 2, 2)
    control = VisionControlPayloadV2(
        operating_mode=OperatingMode.TRACK, target_valid=True,
        tracking_state=TrackingStateCode.TRACKING,
        observation_source=ObservationSourceCode.FULL_BOARD,
        flags=VisionControlFlagsV2.CAMERA_HEALTHY,
        confidence_permille=900, yaw_rate_cdeg_s=100,
        pitch_rate_cdeg_s=-100, error_yaw_mdeg=250,
        error_pitch_mdeg=-125, target_x_px=640, target_y_px=512,
        source_age_us=25_000, source_frame_sequence=17,
    )
    assert len(control.pack()) == 28
    assert len(GimbalFeedbackPayloadV2().pack()) == 24


def test_protocol_v2_code_values_are_frozen() -> None:
    assert [int(value) for value in TrackingStateCode] == [0, 1, 2, 3, 4, 5]
    assert [int(value) for value in ObservationSourceCode] == [0, 1, 2, 3, 4, 5, 6]
    assert int(VisionControlFlagsV2.CAMERA_HEALTHY) == 1 << 0
    assert int(VisionControlFlagsV2.RESERVED) == 1 << 14
    assert int(VisionControlFlagsV2.EMERGENCY_STOP) == 1 << 15


def test_vision_control_v2_round_trip_has_no_laser_fields() -> None:
    payload = VisionControlPayloadV2(
        operating_mode=OperatingMode.TRACK, target_valid=True,
        tracking_state=TrackingStateCode.TRACKING,
        observation_source=ObservationSourceCode.CONCENTRIC_ARCS,
        flags=(VisionControlFlagsV2.CAMERA_HEALTHY |
               VisionControlFlagsV2.MULTIPLE_ARCS_VALID),
        confidence_permille=812, yaw_rate_cdeg_s=-123,
        pitch_rate_cdeg_s=456, error_yaw_mdeg=-50,
        error_pitch_mdeg=75, target_x_px=701, target_y_px=481,
        source_age_us=2_500, source_frame_sequence=0x10203040,
    )
    assert VisionControlPayloadV2.unpack(payload.pack()) == payload
    assert not any("laser" in field.name for field in dataclasses.fields(payload))


def test_invalid_v2_payload_always_serializes_zero_rates() -> None:
    payload = VisionControlPayloadV2(
        operating_mode=OperatingMode.SEARCH, target_valid=False,
        tracking_state=TrackingStateCode.LOST,
        observation_source=ObservationSourceCode.NONE,
        flags=VisionControlFlagsV2.CAMERA_HEALTHY,
        confidence_permille=0, yaw_rate_cdeg_s=900,
        pitch_rate_cdeg_s=-800, error_yaw_mdeg=0,
        error_pitch_mdeg=0, target_x_px=0, target_y_px=0,
        source_age_us=200_000, source_frame_sequence=9,
    )
    decoded = VisionControlPayloadV2.unpack(payload.pack())
    assert decoded.target_valid is False
    assert decoded.yaw_rate_cdeg_s == decoded.pitch_rate_cdeg_s == 0


def test_gimbal_feedback_v2_round_trip() -> None:
    payload = GimbalFeedbackPayloadV2(
        state=2, fault_flags=3, ack_sequence=513,
        yaw_angle_mdeg=-12_345, pitch_angle_mdeg=67_890,
        yaw_rate_cdeg_s=-210, pitch_rate_cdeg_s=345,
        control_latency_us=2_500, reserved=0,
        controller_time_us=0x10203040,
    )
    assert GimbalFeedbackPayloadV2.unpack(payload.pack()) == payload


def test_v1_payload_stays_explicitly_available() -> None:
    payload = VisionControlPayload(
        mode=OperatingMode.TRACK, target_valid=True, laser_mode=LaserMode.ON,
        flags=ControlFlags.CAMERA_HEALTHY, yaw_rate_cdeg_s=1,
        pitch_rate_cdeg_s=2, error_yaw_mdeg=3, error_pitch_mdeg=4,
        board_confidence_permille=5, laser_confidence_permille=6,
        source_age_us=7,
    )
    frame = encode_frame(MessageType.VISION_CONTROL, 8, payload.pack(),
                         version=PROTOCOL_VERSION_V1)
    assert decode_frame(frame).version == PROTOCOL_VERSION_V1
    assert VisionControlPayload.unpack(decode_frame(frame).payload) == payload
```

Add `import dataclasses` to the test module.

- [ ] **Step 2: Run protocol tests to verify failure**

Run: `python -m pytest tests/test_protocol.py -q -p no:cacheprovider`

Expected: FAIL because the V2 constants, enums and payload classes do not exist.

- [ ] **Step 3: Implement the fixed V2 wire contract**

Keep `VisionControlPayload` and `_CONTROL_STRUCT` unchanged as V1 compatibility. Add these exact declarations:

```python
PROTOCOL_VERSION_V1 = 1
PROTOCOL_VERSION_V2 = 2
PROTOCOL_VERSION = PROTOCOL_VERSION_V2

_CONTROL_V2_STRUCT = struct.Struct("<BBBBHHhhhhhhII")  # 28 bytes
_GIMBAL_V2_STRUCT = struct.Struct("<BBHiihhHHI")       # 24 bytes


class TrackingStateCode(IntEnum):
    SEARCHING = 0
    CONFIRMING = 1
    TRACKING = 2
    PREDICTING = 3
    LOST = 4
    FAULT = 5


class ObservationSourceCode(IntEnum):
    NONE = 0
    FULL_BOARD = 1
    CONCENTRIC_ARCS = 2
    SINGLE_ARC = 3
    WHITE_REGION = 4
    FUSED_PARTIAL = 5
    PREDICTED = 6


class VisionControlFlagsV2(IntFlag):
    CAMERA_HEALTHY = 1 << 0
    FULL_BOARD_VISIBLE = 1 << 1
    HOMOGRAPHY_VALID = 1 << 2
    MULTIPLE_ARCS_VALID = 1 << 3
    SINGLE_ARC_VALID = 1 << 4
    WHITE_REGION_VALID = 1 << 5
    USING_PREDICTION = 1 << 6
    TARGET_NEAR_IMAGE_EDGE = 1 << 7
    TARGET_PARTIALLY_OUTSIDE = 1 << 8
    OBSERVATION_STALE = 1 << 9
    CENTER_JUMP_REJECTED = 1 << 10
    SCALE_JUMP_REJECTED = 1 << 11
    FEEDBACK_STALE = 1 << 12
    GIMBAL_FAULT_RECEIVED = 1 << 13
    RESERVED = 1 << 14
    EMERGENCY_STOP = 1 << 15
```

Import `IntFlag`. Implement frozen dataclasses in exact struct order. The control payload must apply the zero-rate invariant in `pack()`:

```python
@dataclass(frozen=True)
class VisionControlPayloadV2:
    operating_mode: OperatingMode
    target_valid: bool
    tracking_state: TrackingStateCode
    observation_source: ObservationSourceCode
    flags: VisionControlFlagsV2
    confidence_permille: int
    yaw_rate_cdeg_s: int
    pitch_rate_cdeg_s: int
    error_yaw_mdeg: int
    error_pitch_mdeg: int
    target_x_px: int
    target_y_px: int
    source_age_us: int
    source_frame_sequence: int

    def pack(self) -> bytes:
        yaw_rate = self.yaw_rate_cdeg_s if self.target_valid else 0
        pitch_rate = self.pitch_rate_cdeg_s if self.target_valid else 0
        return _CONTROL_V2_STRUCT.pack(
            int(self.operating_mode), int(self.target_valid),
            int(self.tracking_state), int(self.observation_source),
            int(self.flags), self.confidence_permille,
            yaw_rate, pitch_rate, self.error_yaw_mdeg,
            self.error_pitch_mdeg, self.target_x_px, self.target_y_px,
            self.source_age_us, self.source_frame_sequence,
        )
```

Its strict `unpack()` checks `_CONTROL_V2_STRUCT.size` and reconstructs enum/flag values. Implement `GimbalFeedbackPayloadV2` with zero defaults, fields `state`, `fault_flags`, `ack_sequence`, `yaw_angle_mdeg`, `pitch_angle_mdeg`, `yaw_rate_cdeg_s`, `pitch_rate_cdeg_s`, `control_latency_us`, `reserved`, `controller_time_us`, and strict 24-byte `pack()`/`unpack()` using `_GIMBAL_V2_STRUCT`.

- [ ] **Step 4: Run protocol regressions**

Run: `python -m pytest tests/test_protocol.py -q -p no:cacheprovider`

Expected: PASS; V1 and V2 payload tests both remain green.

- [ ] **Step 5: Commit**

```bash
git add src/ev_vision/protocol.py tests/test_protocol.py
git commit -m "feat: add laser-free protocol v2"
```

---

### Task 16: Map detector results and servo commands into protocol V2

**Files:**
- Modify: `src/ev_vision/vision_result.py`
- Modify: `src/ev_vision/control.py`
- Modify: `src/ev_vision/models.py`
- Modify: `tests/test_vision_result.py`
- Modify: `tests/test_control.py`
- Create: `tests/test_protocol_mapping.py`

- [ ] **Step 1: Write failing generic result-mapping tests**

Add a classical-result factory to `tests/test_vision_result.py`, then pin full, partial, prediction and invalid-state semantics:

```python
from ev_vision.detection.contracts import ObservationSource


def classical_result(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "timestamp_ns": 50_000_000, "source_sequence": 17,
        "target_valid": True, "tracking_state": "TRACKING",
        "observation_source": ObservationSource.FULL_BOARD,
        "confidence": 0.92, "center_px": (700.0, 480.0),
        "corners_px": CORNERS, "homography_valid": True,
        "target_x_mm": 0.0, "target_y_mm": 0.0,
        "predicted_frames": 0, "near_image_edge": False,
        "partially_outside": False, "failure_reason": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_generic_full_board_requires_four_corners_and_homography() -> None:
    valid = VisionTargetResult.from_detection(
        classical_result(), image_size=(1280, 1024), now_ns=90_000_000
    )
    assert valid.target_valid is True
    assert valid.observation_source == "FULL_BOARD"
    for overrides in ({"corners_px": CORNERS[:3]}, {"homography_valid": False}):
        invalid = VisionTargetResult.from_detection(
            classical_result(**overrides),
            image_size=(1280, 1024), now_ns=90_000_000,
        )
        assert invalid.target_valid is False


def test_confirmed_partial_source_needs_center_but_not_corners() -> None:
    result = VisionTargetResult.from_detection(
        classical_result(
            observation_source=ObservationSource.CONCENTRIC_ARCS,
            corners_px=(), homography_valid=False,
            target_x_mm=None, target_y_mm=None,
        ), image_size=(1280, 1024), now_ns=90_000_000,
    )
    assert result.target_valid is True
    assert result.corners == ()


def test_prediction_validity_uses_both_three_frames_and_150_ms() -> None:
    base = dict(
        tracking_state="PREDICTING",
        observation_source=ObservationSource.PREDICTED,
        corners_px=(), homography_valid=False,
        target_x_mm=None, target_y_mm=None,
    )
    assert VisionTargetResult.from_detection(
        classical_result(**base, predicted_frames=3),
        image_size=(1280, 1024), now_ns=150_000_000,
        predict_max_frames=3, predict_max_ms=150.0,
    ).target_valid
    assert not VisionTargetResult.from_detection(
        classical_result(**base, predicted_frames=4),
        image_size=(1280, 1024), now_ns=150_000_000,
        predict_max_frames=3, predict_max_ms=150.0,
    ).target_valid
    assert not VisionTargetResult.from_detection(
        classical_result(**base, timestamp_ns=0, predicted_frames=1),
        image_size=(1280, 1024), now_ns=150_000_001,
        predict_max_frames=3, predict_max_ms=150.0,
    ).target_valid


@pytest.mark.parametrize("state", ["SEARCHING", "CONFIRMING", "LOST", "FAULT"])
def test_noncontrolling_states_are_invalid(state: str) -> None:
    result = VisionTargetResult.from_detection(
        classical_result(tracking_state=state),
        image_size=(1280, 1024), now_ns=60_000_000,
    )
    assert result.target_valid is False
```

Retain every existing `from_hybrid()` test; it becomes a compatibility wrapper over `from_detection()`.

- [ ] **Step 2: Write failing control-to-payload tests**

Create `tests/test_protocol_mapping.py` with an explicit `mapped_result()` factory and these cases:

```python
from ev_vision.control import make_vision_control_v2
from ev_vision.models import LaserMode, OperatingMode, RateCommand
from ev_vision.protocol import (
    ObservationSourceCode, TrackingStateCode, VisionControlFlagsV2,
)


def test_full_board_command_maps_units_and_flags() -> None:
    result = mapped_result(
        target_valid=True, tracking_state="TRACKING",
        observation_source="FULL_BOARD", confidence=0.812,
        center_x_px=700.4, center_y_px=480.6,
        homography_valid=True, frame_age_ms=25.0,
    )
    command = RateCommand(
        yaw_rate_deg_s=-1.23, pitch_rate_deg_s=4.56,
        target_valid=True, source_age_us=25_000,
    )
    payload = make_vision_control_v2(
        result, command, operating_mode=OperatingMode.TRACK,
        angular_error_deg=(-0.050, 0.075), camera_healthy=True,
    )
    assert payload.tracking_state == TrackingStateCode.TRACKING
    assert payload.observation_source == ObservationSourceCode.FULL_BOARD
    assert payload.confidence_permille == 812
    assert payload.yaw_rate_cdeg_s == -123
    assert payload.pitch_rate_cdeg_s == 456
    assert payload.error_yaw_mdeg == -50
    assert payload.error_pitch_mdeg == 75
    assert payload.target_x_px == 700
    assert payload.target_y_px == 481
    assert payload.flags & VisionControlFlagsV2.CAMERA_HEALTHY
    assert payload.flags & VisionControlFlagsV2.FULL_BOARD_VISIBLE
    assert payload.flags & VisionControlFlagsV2.HOMOGRAPHY_VALID


def test_invalid_or_stale_result_forces_zero_rates() -> None:
    result = mapped_result(
        target_valid=False, tracking_state="LOST",
        observation_source="NONE", frame_age_ms=200.0,
    )
    command = RateCommand(
        yaw_rate_deg_s=9.0, pitch_rate_deg_s=-8.0,
        target_valid=True, source_age_us=200_000,
    )
    payload = make_vision_control_v2(
        result, command, operating_mode=OperatingMode.SEARCH,
        angular_error_deg=(1.0, -1.0), camera_healthy=True,
    )
    assert payload.target_valid is False
    assert payload.yaw_rate_cdeg_s == payload.pitch_rate_cdeg_s == 0
    assert payload.flags & VisionControlFlagsV2.OBSERVATION_STALE


def test_v2_mapping_ignores_legacy_laser_mode() -> None:
    result = mapped_result(target_valid=True)
    command = RateCommand(
        target_valid=True, laser_mode=LaserMode.ON,
        yaw_rate_deg_s=1.0, pitch_rate_deg_s=2.0,
    )
    payload = make_vision_control_v2(
        result, command, operating_mode=OperatingMode.TRACK,
        angular_error_deg=(0.1, 0.2), camera_healthy=True,
    )
    assert not hasattr(payload, "laser_mode")
    assert not hasattr(payload, "laser_confidence")


@pytest.mark.parametrize(
    ("state", "source"),
    [("SEARCHING", "NONE"), ("CONFIRMING", "FULL_BOARD"),
     ("LOST", "NONE"), ("FAULT", "NONE")],
)
def test_invalid_states_never_emit_motion(state: str, source: str) -> None:
    payload = make_vision_control_v2(
        mapped_result(target_valid=False, tracking_state=state,
                      observation_source=source),
        RateCommand(yaw_rate_deg_s=20.0, pitch_rate_deg_s=20.0,
                    target_valid=True),
        operating_mode=OperatingMode.TRACK,
        angular_error_deg=(2.0, 2.0), camera_healthy=True,
    )
    assert payload.target_valid is False
    assert payload.yaw_rate_cdeg_s == payload.pitch_rate_cdeg_s == 0
```

Also add a focused `tests/test_control.py` assertion that a stale `VisualServo.update()` returns zero rates. Legacy `RateCommand.laser_mode` may remain for V1 callers but cannot influence V2.

- [ ] **Step 3: Run result/control mapping tests to verify failure**

Run: `python -m pytest tests/test_vision_result.py tests/test_control.py tests/test_protocol_mapping.py -q -p no:cacheprovider`

Expected: FAIL because `from_detection()` and `make_vision_control_v2()` do not exist and the neutral result lacks classical metadata.

- [ ] **Step 4: Implement generic validity and V2 mapping**

Extend `VisionTargetResult` after `frame_age_ms` with defaults so existing direct construction stays compatible:

```python
observation_source: str = "NONE"
homography_valid: bool = False
predicted_frames: int = 0
near_image_edge: bool = False
partially_outside: bool = False
failure_reason: str | None = None
laser_permission: bool = False  # deprecated V1-only compatibility; never serialized in V2
```

Implement:

```python
def _enum_name(value: object, default: str) -> str:
    raw = getattr(value, "value", value)
    return default if raw is None else str(raw).upper()


@classmethod
def from_detection(
    cls, detection: Any, *, image_size: tuple[int, int], now_ns: int,
    max_result_age_ms: float = BoardTrackingConfig().max_result_age_ms,
    predict_max_frames: int = 3, predict_max_ms: float = 150.0,
) -> "VisionTargetResult":
    timestamp_ns = int(detection.timestamp_ns)
    frame_age_ms = max(0.0, (int(now_ns) - timestamp_ns) / 1_000_000.0)
    state = _enum_name(getattr(detection, "tracking_state", None), "SEARCHING")
    source = _enum_name(getattr(detection, "observation_source", None), "FULL_BOARD")

    center = getattr(detection, "center_px", None)
    if center is None:
        center_x_px = center_y_px = offset_x_px = offset_y_px = None
    else:
        center_x_px, center_y_px = float(center[0]), float(center[1])
        width, height = image_size
        offset_x_px = center_x_px - width / 2.0
        offset_y_px = center_y_px - height / 2.0

    raw_corners = getattr(detection, "corners_px", None)
    corners = tuple((float(x), float(y)) for x, y in raw_corners) if raw_corners else ()
    homography_valid = bool(getattr(detection, "homography_valid", False))
    target_x_mm = getattr(detection, "target_x_mm", None)
    target_y_mm = getattr(detection, "target_y_mm", None)
    predicted_frames = int(getattr(detection, "predicted_frames", 0))
    failure = getattr(detection, "failure_reason", None)
    failure_value = _enum_name(failure, "") if failure is not None else None

    fresh = frame_age_ms <= float(max_result_age_ms)
    full_valid = (
        source == "FULL_BOARD" and state == "TRACKING" and center is not None
        and len(corners) == 4 and homography_valid
        and target_x_mm is not None and target_y_mm is not None
    )
    partial_valid = (
        source in {"CONCENTRIC_ARCS", "SINGLE_ARC", "WHITE_REGION", "FUSED_PARTIAL"}
        and state == "TRACKING" and center is not None
    )
    predicted_valid = (
        source == "PREDICTED" and state == "PREDICTING" and center is not None
        and predicted_frames <= predict_max_frames and frame_age_ms <= predict_max_ms
    )
    target_valid = bool(
        getattr(detection, "target_valid", False)
        and fresh and failure_value not in _INVALID_FAILURES
        and (full_valid or partial_valid or predicted_valid)
    )

    confidence = getattr(detection, "confidence", None)
    if confidence is None:
        confidence = getattr(detection, "combined_score", 0.0)
    return cls(
        timestamp_ms=timestamp_ns // 1_000_000,
        frame_sequence=int(detection.source_sequence),
        target_valid=target_valid, tracking_state=state,
        confidence=float(confidence), center_x_px=center_x_px,
        center_y_px=center_y_px, offset_x_px=offset_x_px,
        offset_y_px=offset_y_px,
        target_x_mm=float(target_x_mm) if target_x_mm is not None else None,
        target_y_mm=float(target_y_mm) if target_y_mm is not None else None,
        corners=corners, frame_age_ms=frame_age_ms,
        observation_source=source, homography_valid=homography_valid,
        predicted_frames=predicted_frames,
        near_image_edge=bool(getattr(detection, "near_image_edge", False)),
        partially_outside=bool(getattr(detection, "partially_outside", False)),
        failure_reason=failure_value, laser_permission=False,
    )
```

Normalize enum/string state and source values. Compute age from original `timestamp_ns`; prediction never refreshes it. Apply these exact rules:

```python
fresh = frame_age_ms <= max_result_age_ms
full_valid = (
    source == "FULL_BOARD" and state == "TRACKING" and center is not None
    and len(corners) == 4 and homography_valid
    and target_x_mm is not None and target_y_mm is not None
)
partial_valid = (
    source in {"CONCENTRIC_ARCS", "SINGLE_ARC", "WHITE_REGION", "FUSED_PARTIAL"}
    and state == "TRACKING" and center is not None
)
predicted_valid = (
    source == "PREDICTED" and state == "PREDICTING" and center is not None
    and predicted_frames <= predict_max_frames and frame_age_ms <= predict_max_ms
)
target_valid = bool(
    getattr(detection, "target_valid", False)
    and fresh and failure_value not in _INVALID_FAILURES
    and (full_valid or partial_valid or predicted_valid)
)
```

`from_hybrid()` calls `from_detection()` after treating a missing observation source as `FULL_BOARD`; existing hybrid behavior remains unchanged.

In `control.py`, add integer clamp/round helpers and:

```python
def _clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def make_vision_control_v2(
    result: VisionTargetResult, command: RateCommand, *,
    operating_mode: OperatingMode,
    angular_error_deg: tuple[float, float], camera_healthy: bool,
    feedback_stale: bool = False, gimbal_fault_received: bool = False,
    emergency_stop: bool = False,
) -> VisionControlPayloadV2:
    state = TrackingStateCode[result.tracking_state]
    source = ObservationSourceCode[result.observation_source]
    flags = VisionControlFlagsV2(0)
    if camera_healthy:
        flags |= VisionControlFlagsV2.CAMERA_HEALTHY
    source_flag = {
        "FULL_BOARD": VisionControlFlagsV2.FULL_BOARD_VISIBLE,
        "CONCENTRIC_ARCS": VisionControlFlagsV2.MULTIPLE_ARCS_VALID,
        "SINGLE_ARC": VisionControlFlagsV2.SINGLE_ARC_VALID,
        "WHITE_REGION": VisionControlFlagsV2.WHITE_REGION_VALID,
        "FUSED_PARTIAL": (
            VisionControlFlagsV2.MULTIPLE_ARCS_VALID
            | VisionControlFlagsV2.WHITE_REGION_VALID
        ),
        "PREDICTED": VisionControlFlagsV2.USING_PREDICTION,
    }.get(result.observation_source, VisionControlFlagsV2(0))
    flags |= source_flag
    if result.homography_valid:
        flags |= VisionControlFlagsV2.HOMOGRAPHY_VALID
    if result.near_image_edge:
        flags |= VisionControlFlagsV2.TARGET_NEAR_IMAGE_EDGE
    if result.partially_outside:
        flags |= VisionControlFlagsV2.TARGET_PARTIALLY_OUTSIDE
    if result.frame_age_ms > BoardTrackingConfig().max_result_age_ms:
        flags |= VisionControlFlagsV2.OBSERVATION_STALE
    if result.failure_reason == "EXCESSIVE_POSITION_JUMP":
        flags |= VisionControlFlagsV2.CENTER_JUMP_REJECTED
    if result.failure_reason == "EXCESSIVE_SCALE_JUMP":
        flags |= VisionControlFlagsV2.SCALE_JUMP_REJECTED
    if feedback_stale:
        flags |= VisionControlFlagsV2.FEEDBACK_STALE
    if gimbal_fault_received:
        flags |= VisionControlFlagsV2.GIMBAL_FAULT_RECEIVED
    if emergency_stop:
        flags |= VisionControlFlagsV2.EMERGENCY_STOP

    target_valid = bool(
        result.target_valid and command.target_valid and camera_healthy
        and not feedback_stale and not gimbal_fault_received
        and not emergency_stop
    )
    yaw_rate = _clamp_int(command.yaw_rate_deg_s * 100.0, -32768, 32767)
    pitch_rate = _clamp_int(command.pitch_rate_deg_s * 100.0, -32768, 32767)
    return VisionControlPayloadV2(
        operating_mode=operating_mode, target_valid=target_valid,
        tracking_state=state, observation_source=source, flags=flags,
        confidence_permille=_clamp_int(result.confidence * 1000.0, 0, 1000),
        yaw_rate_cdeg_s=yaw_rate if target_valid else 0,
        pitch_rate_cdeg_s=pitch_rate if target_valid else 0,
        error_yaw_mdeg=_clamp_int(angular_error_deg[0] * 1000.0, -32768, 32767),
        error_pitch_mdeg=_clamp_int(angular_error_deg[1] * 1000.0, -32768, 32767),
        target_x_px=_clamp_int(result.center_x_px or 0.0, -32768, 32767),
        target_y_px=_clamp_int(result.center_y_px or 0.0, -32768, 32767),
        source_age_us=_clamp_int(result.frame_age_ms * 1000.0, 0, 0xFFFFFFFF),
        source_frame_sequence=_clamp_int(result.frame_sequence, 0, 0xFFFFFFFF),
    )
```

Map state/source by name. Build flags from camera health, source type, homography, prediction, image-edge status, staleness, `EXCESSIVE_POSITION_JUMP`, `EXCESSIVE_SCALE_JUMP`, feedback status, gimbal fault and emergency stop. Final validity is:

```python
target_valid = bool(
    result.target_valid and command.target_valid and camera_healthy
    and not feedback_stale and not gimbal_fault_received
    and not emergency_stop
)
```

Use hundredths of degree per second for rates, thousandths of degree for errors, rounded pixels for center, clamped signed 16-bit fields, confidence `0..1000`, and uint32 age/frame sequence. Set both rates to zero whenever invalid. Do not read `command.laser_mode` and do not add V2 laser fields.

- [ ] **Step 5: Run result, control and protocol regressions**

Run: `python -m pytest tests/test_vision_result.py tests/test_control.py tests/test_protocol_mapping.py tests/test_protocol.py -q -p no:cacheprovider`

Expected: PASS; invalid, stale, lost and fault results all serialize zero motion.

- [ ] **Step 6: Commit**

```bash
git add src/ev_vision/vision_result.py src/ev_vision/control.py src/ev_vision/models.py tests/test_vision_result.py tests/test_control.py tests/test_protocol_mapping.py
git commit -m "feat: map tracked targets to protocol v2"
```

---

### Task 17: Add a hardware-independent classical replay evaluator

**Files:**
- Create: `tools/evaluate_classical_detector.py`
- Create: `tests/tools/test_evaluate_classical_detector.py`
- Create: `docs/classical-detector-evaluation.md`
- Optional create, only if repository size and image ownership are acceptable: `tests/fixtures/real_classical_target.jpg`

- [ ] **Step 1: Write failing iterator, metrics and dependency-boundary tests**

```python
import ast
import json
from pathlib import Path

import cv2
import numpy as np

from tools import evaluate_classical_detector as subject


def write_frames(root: Path, count: int = 4) -> None:
    root.mkdir()
    for index in range(count):
        image = np.full((120, 160, 3), 20 + index, dtype=np.uint8)
        assert cv2.imwrite(str(root / f"{index:04d}.png"), image)


def test_iter_replay_frames_orders_image_directory(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 3)
    frames = list(subject.iter_replay_frames(replay, fps=20.0, max_frames=None))
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert [frame.captured_ns for frame in frames] == [0, 50_000_000, 100_000_000]


def test_evaluate_reports_measured_sources_latency_and_fps(tmp_path, monkeypatch) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 4)
    monkeypatch.setattr(subject, "ClassicalBoardDetector", FakeDetector)
    report = subject.evaluate(
        input_path=replay, output_path=tmp_path / "report.json",
        config_path=Path("config/default.yaml"), fps=20.0,
        save_debug=None, max_frames=None,
    )
    assert report["frames"] == 4
    assert report["source_counts"] == {
        "CONCENTRIC_ARCS": 1, "FULL_BOARD": 1,
        "NONE": 1, "PREDICTED": 1,
    }
    assert report["valid_frames"] == 3
    assert report["lost_frames"] == 1
    assert report["mean_processing_ms"] >= 0.0
    assert report["p95_processing_ms"] >= 0.0
    assert report["measured_detection_fps"] > 0.0
    assert json.loads((tmp_path / "report.json").read_text("utf-8")) == report


def test_debug_export_uses_sequence_named_subdirectories(tmp_path, monkeypatch) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 1)
    monkeypatch.setattr(subject, "ClassicalBoardDetector", FakeDebugDetector)
    subject.evaluate(
        input_path=replay, output_path=tmp_path / "report.json",
        config_path=Path("config/default.yaml"), fps=20.0,
        save_debug=tmp_path / "debug", max_frames=1,
    )
    assert (tmp_path / "debug" / "000000" / "overlay.png").is_file()
    for name in (
        "normalized-gray", "white-mask", "edge-mask",
        "ring-arcs", "candidate-scores",
    ):
        assert (tmp_path / "debug" / "000000" / f"{name}.png").is_file()


def test_evaluator_does_not_import_mvs_or_ultralytics() -> None:
    tree = ast.parse(Path(subject.__file__).read_text("utf-8"))
    imports = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("MvCamera" in name or "ultralytics" in name for name in imports)
```

`FakeDetector` deterministically returns `FULL_BOARD`, `CONCENTRIC_ARCS`, `PREDICTED`, then `LOST/NONE`. `FakeDebugDetector` returns the five named classical debug images. Automated tests generate images in `tmp_path`; they never reference an absolute Downloads path.

- [ ] **Step 2: Verify evaluator tests fail**

Run: `python -m pytest tests/tools/test_evaluate_classical_detector.py -q -p no:cacheprovider`

Expected: FAIL with `ImportError` because the evaluator does not exist.

- [ ] **Step 3: Implement replay loading, evaluation and CLI**

The module imports only standard library, `cv2`, `numpy`, `load_config`, `ClassicalBoardDetector`, `Frame`, and classical debug rendering. Define:

```python
@dataclass(frozen=True)
class ReplayFrame:
    sequence: int
    captured_ns: int
    image: np.ndarray
    source_name: str


def _iter_image_paths(root: Path) -> Iterator[Path]:
    extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    yield from sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination)


def iter_replay_frames(
    input_path: Path, *, fps: float, max_frames: int | None
) -> Iterator[ReplayFrame]:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    image_extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if input_path.is_dir() or input_path.suffix.lower() in image_extensions:
        if fps <= 0:
            raise ValueError("fps must be positive for image input")
        paths = [input_path] if input_path.is_file() else list(_iter_image_paths(input_path))
        for sequence, path in enumerate(paths[:max_frames]):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            yield ReplayFrame(
                sequence, round(sequence * 1_000_000_000 / fps), image, path.name
            )
        return
    if input_path.name.endswith((".tar", ".tar.gz", ".tgz")):
        if fps <= 0:
            raise ValueError("fps must be positive for archive input")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _safe_extract(input_path, root)
            yield from iter_replay_frames(root, fps=fps, max_frames=max_frames)
        return

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"unsupported or unreadable input: {input_path}")
    source_fps = fps if fps > 0 else float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        raise ValueError("video FPS is unavailable; pass --fps")
    sequence = 0
    try:
        while max_frames is None or sequence < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            yield ReplayFrame(
                sequence, round(sequence * 1_000_000_000 / source_fps),
                image, f"frame-{sequence:06d}",
            )
            sequence += 1
    finally:
        capture.release()


def save_frame_debug(root: Path, frame: ReplayFrame, result: object) -> None:
    destination = root / f"{frame.sequence:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    debug_images = dict(getattr(result, "debug_images", {}))
    products = {"overlay": debug_images.get("overlay", frame.image)}
    for name in DEBUG_IMAGE_NAMES:
        if name not in debug_images:
            raise ValueError(
                f"frame {frame.sequence} is missing debug image {name}"
            )
        products[name] = debug_images[name]
    for name, image in products.items():
        if not cv2.imwrite(str(destination / f"{name}.png"), image):
            raise OSError(f"cannot write debug image {name} for frame {frame.sequence}")


def evaluate(
    *, input_path: Path, output_path: Path, config_path: Path,
    fps: float, save_debug: Path | None, max_frames: int | None,
) -> dict[str, object]:
    config = load_config(config_path)
    detector = ClassicalBoardDetector(config.detection)
    durations_ms: list[float] = []
    source_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    valid_count = full_count = partial_count = predicted_count = lost_count = 0
    prediction_streak = max_prediction_streak = frame_count = 0

    for frame in iter_replay_frames(input_path, fps=fps, max_frames=max_frames):
        started = time.perf_counter_ns()
        result = detector.detect(
            frame.image, captured_ns=frame.captured_ns,
            source_sequence=frame.sequence, include_debug=save_debug is not None,
        )
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        source = getattr(result.observation_source, "value", result.observation_source)
        state = getattr(result.tracking_state, "value", result.tracking_state)
        source_counts[str(source)] += 1
        state_counts[str(state)] += 1
        frame_count += 1
        valid_count += int(result.target_valid)
        full_count += int(source == "FULL_BOARD")
        partial_count += int(source in {
            "CONCENTRIC_ARCS", "SINGLE_ARC", "WHITE_REGION", "FUSED_PARTIAL"
        })
        predicted_count += int(source == "PREDICTED")
        lost_count += int(state == "LOST")
        prediction_streak = prediction_streak + 1 if source == "PREDICTED" else 0
        max_prediction_streak = max(max_prediction_streak, prediction_streak)
        if save_debug is not None:
            save_frame_debug(save_debug, frame, result)

    mean_ms = float(np.mean(durations_ms)) if durations_ms else 0.0
    p95_ms = float(np.percentile(durations_ms, 95)) if durations_ms else 0.0
    report = {
        "input": str(input_path), "config": str(config_path),
        "frames": frame_count, "valid_frames": valid_count,
        "full_board_frames": full_count, "partial_frames": partial_count,
        "predicted_frames": predicted_count, "lost_frames": lost_count,
        "source_counts": dict(sorted(source_counts.items())),
        "state_counts": dict(sorted(state_counts.items())),
        "max_prediction_streak": max_prediction_streak,
        "mean_processing_ms": mean_ms, "p95_processing_ms": p95_ms,
        "measured_detection_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
```

Accepted inputs are a single image, a lexically ordered image directory, an OpenCV-readable video, or a `.tar`/`.tar.gz` archive containing images. Archive extraction rejects absolute member paths and every member containing `..`, and uses `TemporaryDirectory` only. Images get timestamps `round(sequence * 1e9 / fps)`. Video uses requested positive `--fps`; otherwise it uses positive container FPS.

Measure only `detector.detect(...)` with `time.perf_counter_ns()`. Collect:

```python
{
    "input": str(input_path), "config": str(config_path),
    "frames": frame_count, "valid_frames": valid_count,
    "full_board_frames": full_count, "partial_frames": partial_count,
    "predicted_frames": predicted_count, "lost_frames": lost_count,
    "source_counts": dict(sorted(source_counts.items())),
    "state_counts": dict(sorted(state_counts.items())),
    "max_prediction_streak": max_prediction_streak,
    "mean_processing_ms": mean_ms, "p95_processing_ms": p95_ms,
    "measured_detection_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
}
```

When `save_debug` is set, detect with `include_debug=True`, save `overlay.png` and the five classical debug products under `<save-debug>/<sequence:06d>/`, and never reuse an image from another source sequence.

Build the exact CLI:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the classical white-board detector without camera hardware"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--save-debug", type=Path)
    parser.add_argument("--max-frames", type=int)
    return parser
```

Validate `fps > 0` for image/archive input and `max_frames > 0` when supplied. Write stable UTF-8 JSON using `indent=2`, `sort_keys=True`, print the same report, and return `0`.

- [ ] **Step 4: Document repeatable replay commands**

Create `docs/classical-detector-evaluation.md` with repository-relative examples:

```bash
python tools/evaluate_classical_detector.py \
  --input artifacts/classical-replay \
  --output artifacts/classical-metrics.json \
  --config config/default.yaml \
  --fps 20 \
  --save-debug artifacts/classical-debug

python tools/evaluate_classical_detector.py \
  --input latest-camera-capture.tar.gz \
  --output artifacts/latest-camera-capture-metrics.json \
  --config config/default.yaml \
  --fps 20 \
  --max-frames 300
```

Explain every metric, state that `measured_detection_fps` depends on machine and input, and require recording the actual Jetson result rather than copying the design target. State that the tool opens neither Hikrobot/MVS nor Ultralytics.

- [ ] **Step 5: Run evaluator and adjacent detector tests**

Run: `python -m pytest tests/tools/test_evaluate_classical_detector.py tests/detection/test_classical_board.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/evaluate_classical_detector.py tests/tools/test_evaluate_classical_detector.py docs/classical-detector-evaluation.md
git commit -m "feat: evaluate classical detector replays"
```

If a deliberately approved real fixture is added, include only `tests/fixtures/real_classical_target.jpg`; never commit the user's whole capture archive.

---

### Task 18: Document Jetson launch, safety and one-pass final acceptance

**Files:**
- Modify: `README.md`
- Create: `docs/jetson-classical-vision-acceptance.md`
- Create: `tests/test_classical_documentation.py`

- [ ] **Step 1: Write failing documentation safety tests**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text("utf-8")
ACCEPTANCE_PATH = ROOT / "docs" / "jetson-classical-vision-acceptance.md"
ACCEPTANCE = ACCEPTANCE_PATH.read_text("utf-8") if ACCEPTANCE_PATH.exists() else ""


def test_readme_does_not_claim_jetson_controls_laser() -> None:
    assert "由 Jetson 控制开关" not in README
    assert "激光" in README
    assert "上电" in README


def test_acceptance_contains_exact_environment_and_launch_commands() -> None:
    for command in (
        "cd ~/2025-E-Vision/2025-E-Vision",
        "conda activate 2025-e-vision",
        'export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"',
        'export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
        "export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "python tools/camera_tuning_server.py --config config/default.yaml --host 0.0.0.0 --port 8000",
    ):
        assert command in ACCEPTANCE


def test_acceptance_pins_tracking_and_safety_checks() -> None:
    for text in (
        "3 frames", "150 ms", "LOST", "FUSED_PARTIAL",
        "FULL_BOARD", "five classical debug images",
        "measured_detection_fps", "software cannot make it safe",
    ):
        assert text in ACCEPTANCE
```

- [ ] **Step 2: Run documentation tests to verify failure**

Run: `python -m pytest tests/test_classical_documentation.py -q -p no:cacheprovider`

Expected: FAIL because the acceptance document is absent and README still describes Jetson-controlled laser switching.

- [ ] **Step 3: Correct README and write the Jetson acceptance runbook**

Update every relevant README safety statement: the 405 nm laser is physically always on whenever its power rail is energized; Jetson has no laser switch authority; V2 has no laser-control field; software cannot make an energized laser safe. Remove instructions that depend on `LaserMode.OFF` or Jetson GPIO for safety. Label any remaining V1 laser material as legacy and unused by the competition V2 path.

Create `docs/jetson-classical-vision-acceptance.md` with these ordered sections.

1. **Close camera owners and enter the project**

```bash
pkill -f MvViewer || true
cd ~/2025-E-Vision/2025-E-Vision
conda activate 2025-e-vision
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

2. **Compile and run the full suite**

```bash
PYTHONPYCACHEPREFIX="$(mktemp -d)" \
python -m compileall -q src tests tools

TEST_TMP="$(mktemp -d)"
python -m pytest \
  -q \
  -p no:cacheprovider \
  --basetemp "$TEST_TMP"
```

3. **Start the existing single tuning service**

```bash
python tools/camera_tuning_server.py --config config/default.yaml --host 0.0.0.0 --port 8000
```

Confirm `/api/status` reports `Connected`, `detection.backend=classical`, no camera error, an increasing frame count, and finite acquisition/detection FPS. Detector configuration changes use only `PUT /api/detection/config` and must not restart or close MVS acquisition.

4. **Full-board acquisition matrix**

Record at least three consecutive confirmations at image center, top, bottom, left and right. For every position record source, state, target validity, confidence, source age and center error. Control entry requires `FULL_BOARD`; black tape is supporting evidence only.

5. **Partial tracking matrix**

After full confirmation, move every board edge partly outside the frame and cover parts of the white sheet. Verify accepted evidence follows `CONCENTRIC_ARCS > FUSED_PARTIAL > SINGLE_ARC > WHITE_REGION > PREDICTED`; color never determines acceptance. Include center localization from concentric arcs around the target center.

6. **Bounded prediction and loss**

Remove all usable evidence and verify prediction stops at the first reached limit: `3 frames` or `150 ms`. The next result is `LOST`, `target_valid=false`, `yaw_rate=0`, `pitch_rate=0`. Partial arcs or white regions while LOST cannot reacquire. Restore a complete board and require multi-frame `FULL_BOARD` confirmation.

7. **Negative scenes**

Present a white wall, white book, monitor, blank A4, black background rectangle, one circle, non-concentric circles, wrong radius ratios, a saturated laser spot and a circle outside historical ROI. Record rejection reason and verify no sustained valid lock.

8. **Snapshot evidence**

Create captures for full tracking, partial tracking and a rejected negative. Each contains the five classical debug images `normalized-gray.png`, `white-mask.png`, `edge-mask.png`, `ring-arcs.png`, `candidate-scores.png`, plus original, overlay and metadata from the same source sequence.

9. **Measure, do not invent, performance**

```bash
python tools/evaluate_classical_detector.py \
  --input artifacts/classical-replay \
  --output artifacts/jetson-classical-metrics.json \
  --config config/default.yaml \
  --fps 20 \
  --save-debug artifacts/jetson-classical-debug
```

Copy actual `measured_detection_fps`, mean/p95 processing time, valid ratio, source/state counts and observation age into the acceptance table. Compare 10, 15, 20, 30 and 50 ms exposure. Do not claim the 15 FPS design target unless measured on Jetson.

10. **Laser hardware warning**

Include this exact English sentence: `The laser is physically always on whenever powered; software cannot make it safe.` Also provide a Chinese warning that power isolation, eyewear, beam stops and personnel control are mandatory because Jetson cannot turn the laser off.

- [ ] **Step 4: Run documentation tests and full clean verification**

```bash
python -m pytest tests/test_classical_documentation.py -q -p no:cacheprovider

PYTHONPYCACHEPREFIX="$(mktemp -d)" \
python -m compileall -q src tests tools

TEST_TMP="$(mktemp -d)"
python -m pytest \
  -q \
  -p no:cacheprovider \
  --basetemp "$TEST_TMP"
```

Expected: documentation test PASS, compileall exits `0`, and the complete suite passes without collection errors.

- [ ] **Step 5: Perform one implementation self-review and correct findings**

Compare implementation against the approved design once. Check: no red-color requirement; no V2 laser field; LOST cannot reacquire from partial evidence; prediction uses frame and time limits; detector configuration cannot restart acquisition; protocol payloads are exactly 28/24 bytes; debug products share one source sequence; README no longer promises software laser control. Fix findings and rerun Step 4 once. Do not perform a duplicate second review.

- [ ] **Step 6: Commit final documentation and verification guards**

```bash
git add README.md docs/jetson-classical-vision-acceptance.md tests/test_classical_documentation.py
git commit -m "docs: add jetson classical vision acceptance"
```

---

## Final acceptance criteria

- [ ] `config/default.yaml` selects `detection.backend: classical`; the competition path does not require Ultralytics.
- [ ] Full acquisition uses a complete white A4 board and multi-frame confirmation; black tape is supporting evidence only.
- [ ] Ring/arc checks use grayscale geometry, a common center and `1:2:3:4:5` radius relation; no red-color threshold is required.
- [ ] After full acquisition, evidence priority is `FULL_BOARD > CONCENTRIC_ARCS > FUSED_PARTIAL > SINGLE_ARC > WHITE_REGION > PREDICTED`.
- [ ] `LOST` accepts no partial-only reacquisition; a complete white board must pass multi-frame confirmation and reset drift.
- [ ] Prediction is valid only while both `predicted_frames <= 3` and original source age `<= 150 ms`; expiry produces `LOST` and zero rates.
- [ ] Camera disconnect, detector fault, stale observation, stale feedback, gimbal fault and emergency stop invalidate control and serialize zero rates.
- [ ] Protocol V2 uses exact 28-byte control and 24-byte feedback payloads, has no laser fields, and retains explicit V1 compatibility.
- [ ] The dashboard displays state, source, validity, confidence, age, center, velocity, prediction count, rejection reasons and five debug views.
- [ ] Detector parameter updates apply atomically without closing the Hikrobot camera or restarting MVS acquisition.
- [ ] Captures save original, overlay, metadata and five debug images from one source sequence with path/symlink protections intact.
- [ ] The replay evaluator accepts relative images/directories/videos/archives, imports neither MVS nor Ultralytics, and reports measured latency/FPS.
- [ ] Synthetic tests cover perspective, uneven light, shadow, blur, noise, over/underexposure, four-edge clipping, random ring color, partial arcs, prediction, loss and full reacquisition.
- [ ] Negative tests reject white walls/books/monitors, blank A4, black rectangles, incompatible circles, saturated spots and out-of-history distractors.
- [ ] Clean `compileall` and pytest commands pass on Windows development and Jetson Python 3.10.
- [ ] Jetson evidence records full/partial/lost behavior, debug captures, observation age and measured FPS rather than invented performance.
- [ ] Safety documentation states that the laser is hardware always-on whenever powered, Jetson cannot switch it, V2 cannot control it, and software cannot make it safe.
