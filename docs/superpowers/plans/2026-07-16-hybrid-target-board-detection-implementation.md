# Hybrid Target-Board Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a competition-oriented hybrid target-board detector that uses a one-class YOLO Nano model for clutter-resistant localization, ROI geometry for accurate corners, temporal tracking for fail-safe validity, and the existing Jetson tuning dashboard for live diagnosis.

**Architecture:** The camera service remains latest-frame-only and independent from browser clients. A portable YOLO adapter returns up to three board boxes; each box is expanded and refined by ROI-only geometry, scored, disambiguated, and passed through an explicit temporal state machine before any result can become valid. Model failures, stale frames, jumps, ambiguity, and classical-only fallback always publish an invalid target and never authorize the 405 nm laser.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PyYAML, Ultralytics YOLO, ONNX/TensorRT artifacts, FastAPI, vanilla HTML/CSS/JavaScript, pytest, Hikrobot MVS on Jetson Orin NX Super.

---

## Delivery boundary and execution order

The critical path is:

1. dataset preparation and validation;
2. detection configuration;
3. multi-candidate model adapter;
4. ROI geometry and real-capture regression;
5. hybrid scoring and temporal tracker;
6. tuning-service/API/dashboard integration;
7. training, export, backend fallback, and Jetson acceptance.

The transport packet for the gimbal and actual laser control are outside this implementation. This plan only defines a semantic vision result and safety rules so the communication task can use stable fields the next day. The 405 nm laser must remain physically disconnected or OFF throughout all work in this plan.

## File responsibility map

### New runtime files

- `src/ev_vision/detection/roi_board_geometry.py`: expanded-ROI extraction, edge maps, quadrilateral gates, scores, corner ordering, and optional in-memory debug products.
- `src/ev_vision/detection/failures.py`: stable detection failure enum shared by detector, tracker, service, and API without circular imports.
- `src/ev_vision/detection/hybrid_board.py`: model candidate orchestration, geometry calls, score combination, ambiguity handling, timing, backend state, and atomic model ownership.
- `src/ev_vision/tracking/board_tracker.py`: tracker-owned input/result DTOs plus `SEARCHING`, `CONFIRMING`, `TRACKING`, `PREDICTING`, and `LOST` transitions; freshness and jump gates.
- `src/ev_vision/vision_result.py`: transport-neutral semantic result passed to the future gimbal integration.

### Modified runtime files

- `src/ev_vision/config.py`: validated nested detection configuration.
- `src/ev_vision/detection/yolo_board.py`: multi-candidate output and lazy Ultralytics artifact backend.
- `src/ev_vision/tuning/models.py`: richer immutable detection/debug snapshots.
- `src/ev_vision/tuning/service.py`: hybrid detector result publication, latest-only debug data, tracker reset, and live detection configuration.
- `src/ev_vision/tuning/storage.py`: optional hybrid debug PNGs and metadata in manual captures.
- `src/ev_vision/web/camera_tuning_app.py`: detection configuration/status/debug/reload API and richer overlays.
- `src/ev_vision/web/camera_tuning_server.py`: engine-to-ONNX fallback wiring and safe diagnostic-classical fallback.
- `src/ev_vision/web/static/camera-tuning.html`: detection controls and status panels.
- `src/ev_vision/web/static/camera-tuning.css`: compact status/debug layout.
- `src/ev_vision/web/static/camera-tuning.js`: detection API polling, config apply, model reload, and debug image selection.

### Dataset, model, tool, and documentation files

- `datasets/target_board/dataset.yaml`: YOLO dataset definition.
- `datasets/target_board/README.md`: collection, scene split, labeling, and directory contract.
- `datasets/target_board/split-manifest.csv`: source image to scene/split mapping; header committed, rows created by the preparation tool.
- `models/README.md`: artifact names, source model, export, checksum, and Jetson engine rules.
- `tools/__init__.py`: make tool modules importable consistently in tests and direct Python execution.
- `tools/prepare_target_dataset.py`: ingest dashboard capture directories and negatives into deterministic staging names.
- `tools/validate_target_dataset.py`: validate images, YOLO labels, split leakage, class IDs, and box bounds.
- `tools/train_target_detector.py`: Windows training entry point with lazy Ultralytics import.
- `tools/evaluate_target_detector.py`: holdout metrics and threshold-oriented report.
- `tools/export_target_detector.py`: export `.pt` to `.onnx`; TensorRT export is run on Jetson.
- `tools/inspect_target_predictions.py`: render predictions for unseen images without changing labels.
- `docs/gimbal-vision-interface.md`: exact gimbal input/output semantics and laser-invalid conditions.
- `docs/hybrid-detector-acceptance.md`: Windows, offline capture, and Jetson acceptance commands.

### New and modified tests

- `tests/test_config.py`
- `tests/detection/test_yolo_adapter.py`
- `tests/detection/test_roi_board_geometry.py`
- `tests/detection/test_real_capture_roi.py`
- `tests/detection/test_hybrid_board.py`
- `tests/tracking/test_board_tracker.py`
- `tests/tuning/test_service.py`
- `tests/tuning/test_storage.py`
- `tests/web/test_detection_api.py`
- `tests/web/test_camera_tuning_tool.py`
- `tests/tools/test_target_dataset.py`
- `tests/tools/test_target_training_tools.py`
- `tests/test_vision_result.py`
- `tests/fixtures/real_board_roi.jpg`

---

### Task 0: Record the clean baseline

**Files:**
- Read: `pyproject.toml`
- Read: `tests/`
- No production file changes

- [ ] **Step 1: Confirm the worktree and branch**

Run on Windows PowerShell:

```powershell
$git = 'D:\Coding\anaconda\pkgs\git-2.51.0-haa95532_2\Library\bin\git.exe'
& $git status --short --branch
& $git log -3 --oneline
```

Expected: branch `feature/classical-vision`; no uncommitted source edits; design commit `f488a76` is present.

- [ ] **Step 2: Run the current focused detection tests**

Run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -q -p no:cacheprovider tests/detection
```

Expected: all existing detection tests pass. If the active Windows interpreter lacks project dependencies, activate the project Conda environment first and rerun the same command.

- [ ] **Step 3: Run the complete baseline suite**

Run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$testTmp = Join-Path $env:TEMP ("ev-vision-baseline-" + [guid]::NewGuid())
python -m compileall -q src tests tools
python -m pytest -q -p no:cacheprovider --basetemp $testTmp
```

Expected: the existing suite passes. Record the pass count in the execution notes; do not claim Jetson hardware verification from this Windows run.

- [ ] **Step 4: Preserve the baseline without a commit**

Run:

```powershell
& $git status --short
```

Expected: no files changed by the baseline commands.

---

### Task 1: Create the target-board dataset contract and validator

**Files:**
- Create: `datasets/target_board/dataset.yaml`
- Create: `datasets/target_board/README.md`
- Create: `datasets/target_board/split-manifest.csv`
- Create: `tools/__init__.py`
- Create: `tools/prepare_target_dataset.py`
- Create: `tools/validate_target_dataset.py`
- Create: `tests/tools/test_target_dataset.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing tests for capture ingestion and YOLO validation**

Create `tests/tools/test_target_dataset.py` with focused cases equivalent to:

```python
from pathlib import Path

from tools.prepare_target_dataset import prepare_dataset
from tools.validate_target_dataset import validate_dataset


def test_prepare_uses_only_original_png_and_deterministic_names(tmp_path: Path) -> None:
    capture = tmp_path / "captures" / "20260716T100000Z"
    capture.mkdir(parents=True)
    (capture / "original.png").write_bytes(b"image")
    (capture / "overlay.png").write_bytes(b"must-not-copy")

    manifest = prepare_dataset(
        sources=[capture.parent],
        output=tmp_path / "staging",
        scene="desk-left",
        split="train",
        difficulty="clear",
        copy_file=lambda source, target: target.write_bytes(source.read_bytes()),
    )

    assert [item.destination.name for item in manifest] == [
        "desk-left-000001.png"
    ]
    assert not (tmp_path / "staging" / "overlay.png").exists()


def test_validator_rejects_out_of_bounds_and_scene_leakage(tmp_path: Path) -> None:
    dataset = build_minimal_dataset(tmp_path)
    write_label(dataset / "labels/train/a.txt", "0 1.10 0.50 0.20 0.20\n")
    append_manifest(
        dataset,
        "a.png,scene-1,train,clear\na2.png,scene-1,val,clear\n",
    )

    report = validate_dataset(dataset)

    assert "a.txt: normalized coordinates must be within [0, 1]" in report.errors
    assert "scene-1 appears in multiple splits: train, val" in report.errors
```

The test helper creates valid small PNGs with OpenCV, one label per positive, empty label files for negatives, and a manifest with columns `image,scene,split,difficulty`. Allowed difficulty values are `clear`, `difficult`, and `negative`; positive images use `clear` or `difficult`, while empty-label images use `negative`.

- [ ] **Step 2: Run the tests and verify the missing modules fail**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tools/test_target_dataset.py
```

Expected: collection fails because `tools.prepare_target_dataset` and `tools.validate_target_dataset` do not exist.

- [ ] **Step 3: Implement deterministic preparation and strict validation**

Use these public contracts:

```python
@dataclass(frozen=True)
class PreparedImage:
    source: Path
    destination: Path
    scene: str
    split: str
    difficulty: str


def prepare_dataset(
    *,
    sources: Sequence[Path],
    output: Path,
    scene: str,
    split: Literal["train", "val", "test"],
    difficulty: Literal["clear", "difficult", "negative"],
    copy_file: Callable[[Path, Path], None] = shutil.copy2,
) -> tuple[PreparedImage, ...]:
    originals = sorted(
        path for root in sources for path in root.rglob("original.png")
    )
    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, source in enumerate(originals, start=1):
        target = output / f"{scene}-{index:06d}.png"
        copy_file(source, target)
        prepared.append(PreparedImage(source, target, scene, split, difficulty))
    return tuple(prepared)
```

The validator must return a concrete report:

```python
@dataclass(frozen=True)
class DatasetReport:
    images_by_split: Mapping[str, int]
    positive_by_split: Mapping[str, int]
    negative_by_split: Mapping[str, int]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    clear_positive_count: int
    difficult_positive_count: int
    negative_count: int

    @property
    def valid(self) -> bool:
        return not self.errors
```

Validate all of the following:

- every image is readable;
- each image has exactly one corresponding label file;
- label lines have five fields;
- class ID is exactly `0`;
- center/width/height values are finite and normalized;
- width and height are greater than zero;
- manifest columns are exactly `image,scene,split,difficulty`, difficulty is one of `clear|difficult|negative`, and difficulty agrees with whether the label is empty;
- a scene appears in only one split;
- duplicate image hashes are reported across splits;
- counts for clear positives, difficult positives, and negatives are printed by the CLI;
- nonzero exit code is returned when errors exist.

Create `datasets/target_board/dataset.yaml`:

```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: target_board
```

Create `datasets/target_board/split-manifest.csv` with only:

```csv
image,scene,split,difficulty
```

Update `.gitignore` so image and label contents stay local while the contract files remain versioned:

```gitignore
# Local target-board dataset contents
datasets/target_board/images/
datasets/target_board/labels/
datasets/target_board/staging/
datasets/target_board/*.cache
```

Document 150 normal positives, 60–80 difficult positives, 30–50 negatives, complete-board tight boxes, `clear|difficult|negative` manifest labels, and scene-based 70/20/10 splitting in `datasets/target_board/README.md`. Create an empty `tools/__init__.py` so `from tools...` imports behave consistently across pytest and direct execution.

- [ ] **Step 4: Run the dataset tool tests and CLI smoke checks**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tools/test_target_dataset.py
python tools/prepare_target_dataset.py --help
python tools/validate_target_dataset.py --help
```

Expected: tests pass and both tools print usage with exit code 0.

- [ ] **Step 5: Commit the dataset contract**

```powershell
& $git add .gitignore datasets/target_board tools/__init__.py tools/prepare_target_dataset.py tools/validate_target_dataset.py tests/tools/test_target_dataset.py
& $git commit -m "feat: add target board dataset workflow"
```

---

### Task 2: Add validated hybrid-detection configuration

**Files:**
- Modify: `src/ev_vision/config.py`
- Modify: `config/default.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Append tests that assert defaults, normalization, and rejection:

```python
def test_detection_defaults_and_weight_normalization(tmp_path: Path) -> None:
    path = write_config(tmp_path, base_config_without_detection())
    config = load_config(path)

    assert config.detection.model.max_candidates == 3
    assert config.detection.roi_geometry.padding_fraction == pytest.approx(0.08)
    assert sum(config.detection.candidate_scoring.weights) == pytest.approx(1.0)
    assert config.detection.tracking.confirm_frames == 3


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"model": {"confidence_threshold": 1.2}}, "confidence_threshold"),
        ({"model": {"max_candidates": 0}}, "max_candidates"),
        ({"roi_geometry": {"canny_low": 200, "canny_high": 100}}, "canny_low"),
        ({"candidate_scoring": {"model_weight": -1.0}}, "weight"),
        ({"tracking": {"max_result_age_ms": 0}}, "max_result_age_ms"),
    ],
)
def test_invalid_detection_configuration_is_rejected(
    tmp_path: Path, patch: dict[str, object], message: str
) -> None:
    path = write_detection_config(tmp_path, patch)
    with pytest.raises(ConfigError, match=message):
        load_config(path)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_config.py -k detection
```

Expected: failure because `AppConfig` has no `detection` field.

- [ ] **Step 3: Implement nested immutable configuration**

Add these dataclasses with the shown defaults:

```python
@dataclass(frozen=True)
class ModelDetectionConfig:
    path: str = "models/target-board.engine"
    fallback_path: str = "models/target-board.onnx"
    input_width: int = 640
    input_height: int = 640
    confidence_threshold: float = 0.45
    max_candidates: int = 3
    device: int = 0


@dataclass(frozen=True)
class RoiGeometryConfig:
    padding_fraction: float = 0.08
    canny_low: int = 60
    canny_high: int = 180
    min_edge_support: float = 0.45
    min_geometry_score: float = 0.55
    expected_aspect_ratio: float = 0.707
    aspect_ratio_tolerance: float = 0.35
    minimum_side_px: float = 40.0
    minimum_area_fraction: float = 0.25
    maximum_area_fraction: float = 1.15


@dataclass(frozen=True)
class CandidateScoringConfig:
    model_weight: float = 0.45
    geometry_weight: float = 0.30
    structure_weight: float = 0.15
    temporal_weight: float = 0.10
    ambiguity_margin: float = 0.08

    @property
    def weights(self) -> tuple[float, float, float, float]:
        return (
            self.model_weight,
            self.geometry_weight,
            self.structure_weight,
            self.temporal_weight,
        )


@dataclass(frozen=True)
class BoardTrackingConfig:
    confirm_frames: int = 3
    predict_frames: int = 2
    lost_frames: int = 3
    max_center_jump_px: float = 160.0
    max_result_age_ms: float = 100.0


@dataclass(frozen=True)
class DetectionConfig:
    backend: str = "hybrid"
    model: ModelDetectionConfig = field(default_factory=ModelDetectionConfig)
    roi_geometry: RoiGeometryConfig = field(default_factory=RoiGeometryConfig)
    candidate_scoring: CandidateScoringConfig = field(default_factory=CandidateScoringConfig)
    tracking: BoardTrackingConfig = field(default_factory=BoardTrackingConfig)
```

Extend `_build` with a dedicated `_build_detection()` that rejects unknown nested keys and normalizes the four positive weights by dividing each by their finite positive sum. Validate confidence and score values in `[0, 1]`, `canny_low < canny_high`, positive dimensions/counts/ages, non-negative ambiguity margin, and `backend in {"hybrid", "classical"}`. Add the exact design values to `config/default.yaml`.

- [ ] **Step 4: Run config tests and full config regression**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_config.py
```

Expected: all configuration tests pass, including loading older YAML files with no `detection` section.

- [ ] **Step 5: Commit detection configuration**

```powershell
& $git add src/ev_vision/config.py config/default.yaml tests/test_config.py
& $git commit -m "feat: add hybrid detection configuration"
```

---

### Task 3: Extend YOLO to multiple candidates and portable artifacts

**Files:**
- Modify: `src/ev_vision/detection/yolo_board.py`
- Modify: `tests/detection/test_yolo_adapter.py`

- [ ] **Step 1: Write failing multi-candidate and backend tests**

Add tests equivalent to:

```python
def test_detect_candidates_filters_sorts_and_limits() -> None:
    image = np.zeros((480, 640, 3), np.uint8)
    backend = FakeInference([
        RawDetection((10, 10, 110, 110), 0.70, 0),
        RawDetection((20, 20, 120, 120), 0.95, 1),
        RawDetection((30, 30, 130, 130), 0.90, 0),
        RawDetection((40, 40, 40, 140), 0.99, 0),
        RawDetection((50, 50, 150, 150), float("nan"), 0),
        RawDetection((60, 60, 160, 160), 0.80, 0),
    ])

    results = YoloBoardDetector(
        backend, confidence_threshold=0.5, max_candidates=2
    ).detect_candidates(image)

    assert [item.confidence for item in results] == [0.90, 0.80]
    assert YoloBoardDetector(backend, max_candidates=2).detect(image) == results[0]


def test_ultralytics_backend_accepts_onnx_without_importing_at_module_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = fake_ultralytics_module(expected_path="board.onnx")
    monkeypatch.setitem(sys.modules, "ultralytics", module)

    backend = UltralyticsBackend("board.onnx", device=0)

    assert backend.artifact_kind == "onnx"
    assert list(backend.infer(np.zeros((1, 3, 640, 640), np.float32)))
```

- [ ] **Step 2: Run and verify the new API is absent**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/detection/test_yolo_adapter.py
```

Expected: failure for missing `detect_candidates`, `max_candidates`, or `UltralyticsBackend`.

- [ ] **Step 3: Implement the compatible multi-candidate adapter**

Use this public shape:

```python
class YoloBoardDetector:
    def __init__(
        self,
        backend: InferencePort,
        *,
        confidence_threshold: float = 0.5,
        board_class_id: int = 0,
        max_candidates: int = 3,
    ) -> None:
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.board_class_id = board_class_id
        self.max_candidates = max_candidates

    def detect_candidates(self, image: np.ndarray) -> tuple[BoardSearchResult, ...]:
        prepared, transform = letterbox(image, self.backend.input_size)
        valid = []
        for detection in self.backend.infer(self._tensor(prepared)):
            if detection.class_id != self.board_class_id:
                continue
            if not math.isfinite(detection.confidence):
                continue
            if detection.confidence < self.confidence_threshold:
                continue
            box = transform.to_original_box(np.asarray(detection.xyxy, dtype=np.float64))
            if not np.isfinite(box).all():
                continue
            x0, y0, x1, y1 = map(float, box)
            if x1 <= x0 or y1 <= y0:
                continue
            valid.append(BoardSearchResult((x0, y0, x1, y1), detection.confidence))
        valid.sort(key=lambda item: item.confidence, reverse=True)
        return tuple(valid[: self.max_candidates])

    def detect(self, image: np.ndarray) -> BoardSearchResult | None:
        candidates = self.detect_candidates(image)
        return candidates[0] if candidates else None
```

Rename `TensorRTBackend` to `UltralyticsBackend`, retain `TensorRTBackend = UltralyticsBackend` as a compatibility alias, expose `artifact_kind` from suffix, and keep the Ultralytics import inside `__init__`. The backend must accept `.pt`, `.onnx`, and `.engine`; unsupported suffixes raise `ValueError` before model construction.

- [ ] **Step 4: Run YOLO adapter tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/detection/test_yolo_adapter.py
```

Expected: all adapter tests pass, including the original single-result compatibility tests.

- [ ] **Step 5: Commit the adapter**

```powershell
& $git add src/ev_vision/detection/yolo_board.py tests/detection/test_yolo_adapter.py
& $git commit -m "feat: return multiple board model candidates"
```

---

### Task 4: Implement ROI-only board geometry primitives

**Files:**
- Create: `src/ev_vision/detection/roi_board_geometry.py`
- Create: `tests/detection/test_roi_board_geometry.py`
- Modify: `tests/fixtures/synthetic_board.py`

- [ ] **Step 1: Write failing primitive and acceptance tests**

Create tests for expansion, ordering, restoration, and rejection:

```python
def test_expand_roi_clips_to_image_and_maps_points_back() -> None:
    roi = expand_roi((5.0, 10.0, 95.0, 90.0), (100, 80), padding_fraction=0.20)
    assert roi.xyxy_px == (0, 0, 100, 80)
    assert roi.to_source(((1.0, 2.0), (50.0, 40.0))) == (
        (1.0, 2.0),
        (50.0, 40.0),
    )


def test_order_corners_is_tl_tr_br_bl() -> None:
    points = np.asarray([[80, 70], [20, 10], [15, 75], [85, 15]], np.float32)
    assert order_corners(points) == pytest.approx(
        ((20, 10), (85, 15), (80, 70), (15, 75))
    )


def test_refine_accepts_synthetic_board_inside_cluttered_roi() -> None:
    image, expected = synthetic_cluttered_board()
    result = RoiBoardGeometry().refine(image, model_box=expected.model_box)

    assert result.accepted is True
    assert result.failure_reason is None
    assert result.geometry_score >= 0.55
    assert result.edge_support_score >= 0.45
    assert np.allclose(result.corners_px, expected.corners, atol=8.0)


@pytest.mark.parametrize(
    ("fixture_name", "reason"),
    [
        ("non_convex", GeometryFailure.NO_VALID_QUADRILATERAL),
        ("truncated", GeometryFailure.TRUNCATED_QUADRILATERAL),
        ("undersized", GeometryFailure.UNDERSIZED_QUADRILATERAL),
        ("weak_edges", GeometryFailure.LOW_EDGE_SUPPORT),
    ],
)
def test_refine_rejects_hard_geometry_failures(fixture_name: str, reason: GeometryFailure) -> None:
    image, model_box = geometry_failure_fixture(fixture_name)
    result = RoiBoardGeometry().refine(image, model_box=model_box)
    assert result.accepted is False
    assert result.failure_reason is reason
```

- [ ] **Step 2: Run and verify the geometry module is missing**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/detection/test_roi_board_geometry.py
```

Expected: collection fails because `ev_vision.detection.roi_board_geometry` does not exist.

- [ ] **Step 3: Add stable geometry types and ROI mapping**

Implement these public types first:

```python
class GeometryFailure(str, Enum):
    INVALID_ROI = "INVALID_ROI"
    NO_VALID_QUADRILATERAL = "NO_VALID_QUADRILATERAL"
    TRUNCATED_QUADRILATERAL = "TRUNCATED_QUADRILATERAL"
    UNDERSIZED_QUADRILATERAL = "UNDERSIZED_QUADRILATERAL"
    INVALID_ASPECT_RATIO = "INVALID_ASPECT_RATIO"
    LOW_EDGE_SUPPORT = "LOW_EDGE_SUPPORT"
    LOW_INTERNAL_STRUCTURE = "LOW_INTERNAL_STRUCTURE"
    AMBIGUOUS_GEOMETRY = "AMBIGUOUS_GEOMETRY"
    CORNER_ORDER_FAILED = "CORNER_ORDER_FAILED"


@dataclass(frozen=True)
class RoiWindow:
    xyxy_px: tuple[int, int, int, int]

    def to_source(
        self, points: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, float], ...]:
        x0, y0, _, _ = self.xyxy_px
        return tuple((float(x) + x0, float(y) + y0) for x, y in points)


@dataclass(frozen=True)
class GeometryDebugImages:
    roi_bgr: np.ndarray
    edges: np.ndarray
    candidates_bgr: np.ndarray


@dataclass(frozen=True)
class GeometryResult:
    accepted: bool
    corners_px: tuple[tuple[float, float], ...] | None
    center_px: tuple[float, float] | None
    geometry_score: float
    edge_support_score: float
    structure_score: float
    roi_xyxy_px: tuple[int, int, int, int]
    failure_reason: GeometryFailure | None
    debug: GeometryDebugImages | None = None
```

`expand_roi()` must interpret `image_size` as `(width, height)`, expand by `padding_fraction` on both dimensions, floor the start, ceil the end, and clip to half-open image bounds. `order_corners()` must reject duplicates, non-finite points, and zero-area orderings.

- [ ] **Step 4: Implement the minimal contour refinement pipeline**

`RoiBoardGeometry.refine(image, model_box, *, include_debug=False)` must:

1. extract only the expanded ROI;
2. convert to gray and apply `cv2.GaussianBlur(gray, (5, 5), 0)`;
3. build two edge maps using `(canny_low, canny_high)` and a second pair scaled by `0.75`, then OR them;
4. close small gaps with a 3×3 morphology kernel;
5. call `cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)`;
6. approximate contours at epsilon fractions `0.015`, `0.025`, and `0.04`;
7. accept only four-point convex candidates;
8. hard-gate minimum side, ROI-boundary truncation, area/model-box agreement, aspect tolerance, edge support, and minimum geometry score;
9. calculate internal-frame structure from dark pixels sampled in an inward border band versus the center region;
10. score each survivor and require a unique winner;
11. optionally run `cv2.cornerSubPix` within a 5×5 window;
12. order and restore source coordinates.

Use normalized scores clamped to `[0, 1]`. For the first release, geometry score is the mean of area agreement, opposite-side consistency, perspective-tolerant aspect score, and minimum-side score. Candidate uniqueness requires a score gap of at least `0.05`. Do not add the deferred line-intersection fallback in this task.

- [ ] **Step 5: Run focused and existing geometry tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/detection/test_roi_board_geometry.py tests/detection/test_board_geometry.py
```

Expected: both the new ROI geometry tests and old diagnostic classical tests pass.

- [ ] **Step 6: Commit ROI geometry**

```powershell
& $git add src/ev_vision/detection/roi_board_geometry.py tests/detection/test_roi_board_geometry.py tests/fixtures/synthetic_board.py
& $git commit -m "feat: refine board corners inside model roi"
```

---

### Task 5: Add the supplied real-capture ROI regression

**Files:**
- Create: `tests/fixtures/real_board_roi.jpg`
- Create: `tests/detection/test_real_capture_roi.py`
- Read: `../../artifacts/analysis/latest-camera-capture/original.png` (source retained in the parent workspace and Git-ignored)

- [ ] **Step 1: Create a compact documented fixture from the supplied capture**

Use the analyzed source corners, reordered to project convention `TL, TR, BR, BL`:

```text
TL=(384,343), TR=(938,315), BR=(962,702), BL=(412,728)
```

Run this one-time command from the worktree root; it writes only a compressed fixture, not the 2 MB source capture:

```powershell
python -c "import cv2; from pathlib import Path; source=Path('../../artifacts/analysis/latest-camera-capture/original.png'); image=cv2.imread(str(source)); assert image is not None, source; crop=image[235:805,285:1045]; target=Path('tests/fixtures/real_board_roi.jpg'); target.parent.mkdir(parents=True,exist_ok=True); assert cv2.imwrite(str(target),crop,[cv2.IMWRITE_JPEG_QUALITY,88]); print(target,crop.shape)"
```

Expected: `tests/fixtures/real_board_roi.jpg` is created at 760×570 and is substantially smaller than the original capture.

- [ ] **Step 2: Write the failing real-capture test**

Create:

```python
def test_real_capture_geometry_recovers_board_when_given_correct_model_roi() -> None:
    image = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
    # Full-image coordinates minus crop origin (285, 235), with loose model padding.
    expected = np.asarray(((99, 108), (653, 80), (677, 467), (127, 493)), np.float32)
    model_box = (75.0, 60.0, 700.0, 515.0)

    result = RoiBoardGeometry().refine(image, model_box=model_box, include_debug=True)

    assert result.accepted is True
    assert result.failure_reason is None
    assert np.allclose(result.corners_px, expected, atol=24.0)
    assert result.debug is not None
```

- [ ] **Step 3: Run the regression and tune only evidence-backed ROI defaults**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/detection/test_real_capture_roi.py -vv
```

Expected initially: the test may fail on a concrete geometry gate. Adjust only ROI morphology, contour epsilon, dark-frame sampling bands, or score thresholds that also preserve Task 4 rejection tests. Do not weaken convexity, truncation, freshness, or corner-order safety gates.

- [ ] **Step 4: Run all geometry regression tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/detection/test_roi_board_geometry.py tests/detection/test_real_capture_roi.py tests/detection/test_board_geometry.py
```

Expected: all pass; the old full-frame detector is not required to pass on the real fixture.

- [ ] **Step 5: Commit the regression fixture**

```powershell
& $git add tests/fixtures/real_board_roi.jpg tests/detection/test_real_capture_roi.py src/ev_vision/detection/roi_board_geometry.py
& $git commit -m "test: cover real target board roi geometry"
```

---

### Task 6: Orchestrate model candidates, geometry, scoring, and ambiguity

**Files:**
- Create: `src/ev_vision/detection/failures.py`
- Create: `src/ev_vision/detection/hybrid_board.py`
- Create: `tests/detection/test_hybrid_board.py`

- [ ] **Step 1: Write failing hybrid-result tests using fake ports**

Cover no candidate, rejected geometry, ranking, ambiguity, and model errors:

```python
def test_unique_candidate_combines_scores_and_returns_corners() -> None:
    model = FakeModel([candidate(0.80), candidate(0.60, x=400)])
    geometry = FakeGeometry({0: accepted_geometry(0.90, 0.80, 0.70), 1: rejected_geometry()})
    detector = HybridBoardDetector(model=model, geometry=geometry)

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=7)

    assert result.detected is True
    assert result.candidate_count == 2
    assert result.combined_score == pytest.approx(
        0.45 * 0.80 + 0.30 * 0.90 + 0.15 * 0.70
    )
    assert result.failure_reason is None


def test_close_candidate_scores_are_ambiguous() -> None:
    detector = detector_with_combined_scores(0.81, 0.75, ambiguity_margin=0.08)
    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=8)
    assert result.detected is False
    assert result.failure_reason is DetectionFailure.AMBIGUOUS_CANDIDATES


def test_model_exception_returns_safe_invalid_result() -> None:
    detector = HybridBoardDetector(model=RaisingModel(), geometry=FakeGeometry({}))
    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=9)
    assert result.detected is False
    assert result.failure_reason is DetectionFailure.MODEL_ERROR
    assert result.target_valid is False
```

- [ ] **Step 2: Run and verify the hybrid module is missing**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/detection/test_hybrid_board.py
```

Expected: collection fails because `ev_vision.detection.hybrid_board` does not exist.

- [ ] **Step 3: Implement stable failure/result types**

Put the failure enum in `src/ev_vision/detection/failures.py` so the tracker and hybrid detector do not import each other:

```python
class DetectionFailure(str, Enum):
    NO_MODEL_CANDIDATE = "NO_MODEL_CANDIDATE"
    LOW_MODEL_CONFIDENCE = "LOW_MODEL_CONFIDENCE"
    NO_VALID_QUADRILATERAL = "NO_VALID_QUADRILATERAL"
    INVALID_ASPECT_RATIO = "INVALID_ASPECT_RATIO"
    LOW_EDGE_SUPPORT = "LOW_EDGE_SUPPORT"
    LOW_INTERNAL_STRUCTURE = "LOW_INTERNAL_STRUCTURE"
    AMBIGUOUS_CANDIDATES = "AMBIGUOUS_CANDIDATES"
    EXCESSIVE_POSITION_JUMP = "EXCESSIVE_POSITION_JUMP"
    STALE_FRAME = "STALE_FRAME"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_ERROR = "MODEL_ERROR"


@dataclass(frozen=True)
class CandidateEvaluation:
    model: BoardSearchResult
    geometry: GeometryResult
    temporal_score: float
    combined_score: float


@dataclass(frozen=True)
class HybridBoardResult:
    timestamp_ns: int
    source_sequence: int
    detected: bool
    target_valid: bool
    tracking_state: str
    model_state: str
    model_backend: str
    model_path: str | None
    model_confidence: float
    geometry_score: float
    edge_support_score: float
    structure_score: float
    temporal_score: float
    combined_score: float
    candidate_count: int
    corners_px: tuple[tuple[float, float], ...] | None
    center_px: tuple[float, float] | None
    failure_reason: DetectionFailure | None
    inference_ms: float
    geometry_ms: float
    total_ms: float
    candidates: tuple[CandidateEvaluation, ...] = ()
    debug_images: Mapping[str, np.ndarray] = field(default_factory=dict, compare=False)
```

At this stage `target_valid` is always false and `tracking_state` is `SEARCHING`; Task 7 owns validity.

- [ ] **Step 4: Implement deterministic candidate evaluation**

`HybridBoardDetector.detect(image, *, captured_ns, source_sequence, include_debug=False, update_tracker=True)` must:

- time model and geometry separately with an injectable `clock_ns`;
- process only the model adapter's already limited candidates;
- map geometry failures to the stable detection enum;
- combine configured model, geometry, structure, and provided temporal scores;
- sort by combined score;
- reject a best/second score gap below `ambiguity_margin`;
- preserve candidate evaluations for the overlay;
- catch model exceptions as `MODEL_ERROR` without throwing into camera acquisition;
- return `MODEL_UNAVAILABLE` when no backend is installed;
- create debug images only when `include_debug=True`;
- skip tracker mutation when `update_tracker=False`, while still computing model, geometry, scores, corners, and debug products for the requested frame.

Do not use a model box alone as a detected board. `detected=True` requires accepted ROI geometry and a unique winner.

- [ ] **Step 5: Run hybrid and adapter tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/detection/test_hybrid_board.py tests/detection/test_yolo_adapter.py tests/detection/test_roi_board_geometry.py
```

Expected: all pass.

- [ ] **Step 6: Commit hybrid scoring**

```powershell
& $git add src/ev_vision/detection/failures.py src/ev_vision/detection/hybrid_board.py tests/detection/test_hybrid_board.py
& $git commit -m "feat: combine model and geometry board candidates"
```

---

### Task 7: Add temporal confirmation, prediction, loss, jump, and freshness gates

**Files:**
- Create: `src/ev_vision/tracking/board_tracker.py`
- Create: `tests/tracking/test_board_tracker.py`
- Modify: `src/ev_vision/detection/hybrid_board.py`
- Modify: `tests/detection/test_hybrid_board.py`

- [ ] **Step 1: Write failing state-transition tests**

Create a table-driven tracker test plus explicit safety cases:

```python
def test_three_hits_track_two_predictions_then_loss() -> None:
    tracker = BoardTracker(config())

    states = [
        tracker.update(hit(1, 0)).state,
        tracker.update(hit(2, 20_000_000)).state,
        tracker.update(hit(3, 40_000_000)).state,
        tracker.update(miss(4, 60_000_000)).state,
        tracker.update(miss(5, 80_000_000)).state,
        tracker.update(miss(6, 100_000_000)).state,
        tracker.update(miss(7, 120_000_000)).state,
    ]

    assert states == [
        TrackingState.CONFIRMING,
        TrackingState.CONFIRMING,
        TrackingState.TRACKING,
        TrackingState.PREDICTING,
        TrackingState.PREDICTING,
        TrackingState.LOST,
        TrackingState.LOST,
    ]
    assert tracker.latest.target_valid is False


def test_prediction_never_grants_target_valid() -> None:
    tracker = tracking_tracker()
    predicted = tracker.update(miss(4, 60_000_000))
    assert predicted.state is TrackingState.PREDICTING
    assert predicted.target_valid is False
    assert predicted.predicted_center_px is not None


def test_jump_and_stale_hit_force_reconfirmation() -> None:
    tracker = tracking_tracker(center=(100.0, 100.0))
    jumped = tracker.update(hit(4, 60_000_000, center=(400.0, 100.0)))
    assert jumped.failure_reason is DetectionFailure.EXCESSIVE_POSITION_JUMP
    assert jumped.target_valid is False

    stale = tracker.update(hit(5, 0), now_ns=200_000_000)
    assert stale.failure_reason is DetectionFailure.STALE_FRAME
    assert stale.target_valid is False
```

- [ ] **Step 2: Run and verify the tracker module is missing**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tracking/test_board_tracker.py
```

Expected: collection fails because `ev_vision.tracking.board_tracker` does not exist.

- [ ] **Step 3: Implement the state machine and tracked result**

Use:

```python
class TrackingState(str, Enum):
    SEARCHING = "SEARCHING"
    CONFIRMING = "CONFIRMING"
    TRACKING = "TRACKING"
    PREDICTING = "PREDICTING"
    LOST = "LOST"


@dataclass(frozen=True)
class TrackObservation:
    timestamp_ns: int
    source_sequence: int
    detected: bool
    center_px: tuple[float, float] | None
    corners_px: tuple[tuple[float, float], ...] | None
    failure_reason: DetectionFailure | None


@dataclass(frozen=True)
class TrackedBoardResult:
    state: TrackingState
    target_valid: bool
    observation: TrackObservation | None
    predicted_center_px: tuple[float, float] | None
    confirmation_count: int
    miss_count: int
    failure_reason: DetectionFailure | None


class BoardTracker:
    def update(
        self,
        result: TrackObservation,
        *,
        now_ns: int | None = None,
    ) -> TrackedBoardResult:
        """Advance one observation while enforcing age, jump, and miss gates."""
        raise NotImplementedError

    def reset(self) -> None:
        """Return to SEARCHING and clear accepted history."""
        raise NotImplementedError

    def temporal_score(self, center_px: tuple[float, float]) -> float:
        """Return [0, 1] consistency with the latest accepted motion."""
        raise NotImplementedError
```

`board_tracker.py` imports `DetectionFailure` only from `ev_vision.detection.failures`; it must not import `HybridBoardResult`. `HybridBoardDetector` converts its selected result into `TrackObservation`, preventing a detector/tracker circular import. Prediction is constant-velocity center extrapolation from the two most recent accepted centers. Predicted corners may be translated for diagnostics, but `target_valid` must be false in `PREDICTING`. Only a fresh, non-jumping, geometry-accepted result in `TRACKING` is valid. A jump clears confirmation history and returns `CONFIRMING` only after a subsequent accepted hit. `LOST` transitions to `CONFIRMING` on the next hit.

- [ ] **Step 4: Feed tracker temporal consistency into hybrid ranking**

Inject `BoardTracker` into `HybridBoardDetector`. Before final ranking, call `tracker.temporal_score(candidate.geometry.center_px)` for each accepted geometry. Convert the unique selection (or miss) to `TrackObservation`, call `tracker.update(track_observation)` only when `update_tracker=True`, then return a copy of `HybridBoardResult` with tracker state, validity, and tracker failure reason. If no unique selected candidate exists, still advance a miss when `update_tracker=True`. When `update_tracker=False`, return the current tracker state with `target_valid=False` and do not change confirmation, miss, velocity, or accepted-history state.

- [ ] **Step 5: Run tracker and hybrid tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tracking/test_board_tracker.py tests/detection/test_hybrid_board.py
```

Expected: all state, freshness, jump, scoring, and ambiguity tests pass.

- [ ] **Step 6: Commit temporal tracking**

```powershell
& $git add src/ev_vision/tracking/board_tracker.py tests/tracking/test_board_tracker.py src/ev_vision/detection/hybrid_board.py tests/detection/test_hybrid_board.py
& $git commit -m "feat: validate board detections over time"
```

---

### Task 8: Solve the board homography and target-plane coordinates

**Files:**
- Create: `src/ev_vision/detection/board_solution.py`
- Create: `tests/detection/test_board_solution.py`
- Modify: `src/ev_vision/detection/hybrid_board.py`
- Modify: `src/ev_vision/geometry.py`
- Modify: `tests/test_geometry.py`
- Modify: `tests/detection/test_hybrid_board.py`

- [ ] **Step 1: Write failing homography and physical-coordinate tests**

Create tests that reuse the existing `TargetGeometry` instead of adding another homography solver:

```python
def test_board_solution_maps_ordered_corners_to_physical_board_mm() -> None:
    corners = ((100.0, 80.0), (520.0, 100.0), (500.0, 694.0), (120.0, 674.0))
    solution = solve_board_plane(
        corners,
        board=BoardConfig(width_cm=21.0, height_cm=29.7, rectified_px_per_cm=40.0),
    )

    assert solution.homography_valid is True
    assert solution.center_px == pytest.approx((310.0, 387.0), abs=2.0)
    assert solution.image_to_target_mm(corners[0]) == pytest.approx((-105.0, 148.5))
    assert solution.image_to_target_mm(corners[2]) == pytest.approx((105.0, -148.5))
    assert solution.image_to_target_mm(solution.center_px) == pytest.approx((0.0, 0.0), abs=1e-5)


def test_degenerate_corners_return_invalid_solution_without_coordinates() -> None:
    solution = solve_board_plane(
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)),
        board=BoardConfig(),
    )
    assert solution.homography_valid is False
    assert solution.target_center_mm is None
```

- [ ] **Step 2: Run and verify the board-solution module is missing**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/detection/test_board_solution.py
```

Expected: collection fails because `ev_vision.detection.board_solution` does not exist.

- [ ] **Step 3: Implement the solution as a thin validated wrapper**

Use:

```python
@dataclass(frozen=True)
class BoardPlaneSolution:
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float] | None
    homography_valid: bool
    target_center_mm: tuple[float, float] | None
    _geometry: TargetGeometry | None = field(default=None, repr=False, compare=False)

    def image_to_target_mm(self, point_px: tuple[float, float]) -> tuple[float, float]:
        if self._geometry is None:
            raise HomographyError("board homography is invalid")
        x_cm, y_cm = self._geometry.image_to_target_cm(point_px)
        return x_cm * 10.0, y_cm * 10.0


def solve_board_plane(
    corners_px: Sequence[Sequence[float]],
    *,
    board: BoardConfig,
) -> BoardPlaneSolution:
    corners = tuple((float(x), float(y)) for x, y in corners_px)
    try:
        geometry = TargetGeometry.from_image_corners(
            corners,
            px_per_cm=board.rectified_px_per_cm,
            width_cm=board.width_cm,
            height_cm=board.height_cm,
        )
        center = geometry.target_cm_to_image((0.0, 0.0))
        return BoardPlaneSolution(corners, center, True, (0.0, 0.0), geometry)
    except (HomographyError, ValueError, TypeError):
        return BoardPlaneSolution(corners, None, False, None, None)
```

Extend `TargetGeometry.from_image_corners()` with this backward-compatible signature and preserve the selected dimensions in the returned object:

```python
@classmethod
def from_image_corners(
    cls,
    corners: Iterable[tuple[float, float]],
    px_per_cm: float = 40.0,
    *,
    width_cm: float = 21.0,
    height_cm: float = 29.7,
) -> "TargetGeometry":
    base = cls(px_per_cm=px_per_cm, width_cm=width_cm, height_cm=height_cm)
    # Keep the existing validation and homography solve, using base.target_corners_cm.
    return cls(
        px_per_cm=px_per_cm,
        width_cm=width_cm,
        height_cm=height_cm,
        _target_to_image=target_to_image,
        _image_to_target=image_to_target,
    )
```

The implementation above must use the configured dimensions through this public call:

```python
geometry = TargetGeometry.from_image_corners(
    corners,
    px_per_cm=board.rectified_px_per_cm,
    width_cm=board.width_cm,
    height_cm=board.height_cm,
)
```

Update `src/ev_vision/geometry.py` and `tests/test_geometry.py` in this same step to preserve defaults and verify non-A4 dimensions. This is the only approved change to the shared geometry module.

- [ ] **Step 4: Attach homography fields to the hybrid result**

Add to `HybridBoardResult`:

```python
homography_valid: bool = False
target_x_mm: float | None = None
target_y_mm: float | None = None
```

After the tracker selects a fresh refined quadrilateral, call `solve_board_plane()`. A failed homography forces `target_valid=False` even if the tracker is in `TRACKING`; it does not throw. For this board-center detection release, `target_x_mm` and `target_y_mm` intentionally equal `(0.0, 0.0)` whenever the homography is valid: they describe the selected aim point in the board coordinate frame, whose origin is the physical board center. Pixel offsets remain the fields used to steer the gimbal toward that center. Keep `image_to_target_mm()` available for the later laser-spot/circle task.

- [ ] **Step 5: Run homography, hybrid, and shared geometry tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_geometry.py tests/detection/test_board_solution.py tests/detection/test_hybrid_board.py
```

Expected: all pass; ordered corners map to `±105 mm` and `±148.5 mm`, and degenerate homography never leaves a valid target.

- [ ] **Step 6: Commit the board-plane solution**

```powershell
& $git add src/ev_vision/geometry.py src/ev_vision/detection/board_solution.py src/ev_vision/detection/hybrid_board.py tests/test_geometry.py tests/detection/test_board_solution.py tests/detection/test_hybrid_board.py
& $git commit -m "feat: solve board homography and target coordinates"
```

---
### Task 9: Define the transport-neutral vision result for the gimbal team

**Files:**
- Create: `src/ev_vision/vision_result.py`
- Create: `tests/test_vision_result.py`
- Create: `docs/gimbal-vision-interface.md`

- [ ] **Step 1: Write failing semantic-result tests**

Create:

```python
def test_tracking_result_maps_to_valid_gimbal_semantics() -> None:
    result = VisionTargetResult.from_hybrid(
        tracked_hybrid_result(center=(700.0, 480.0), tracking_state="TRACKING"),
        image_size=(1280, 1024),
        now_ns=90_000_000,
    )

    assert result.target_valid is True
    assert result.offset_x_px == pytest.approx(60.0)
    assert result.offset_y_px == pytest.approx(-32.0)
    assert result.target_x_mm == pytest.approx(0.0)
    assert result.target_y_mm == pytest.approx(0.0)


def test_prediction_stale_or_nontracking_result_is_invalid() -> None:
    for state in ("SEARCHING", "CONFIRMING", "PREDICTING", "LOST"):
        result = VisionTargetResult.from_hybrid(
            hybrid_result(tracking_state=state, target_valid=True),
            image_size=(1280, 1024),
            now_ns=200_000_000,
        )
        assert result.target_valid is False
        assert result.laser_permission is False
```

- [ ] **Step 2: Run and verify the semantic module is missing**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_vision_result.py
```

Expected: collection fails because `ev_vision.vision_result` does not exist.

- [ ] **Step 3: Implement immutable semantic fields without changing the wire protocol**

Create:

```python
@dataclass(frozen=True)
class VisionTargetResult:
    timestamp_ms: int
    frame_sequence: int
    target_valid: bool
    tracking_state: str
    confidence: float
    center_x_px: float | None
    center_y_px: float | None
    offset_x_px: float | None
    offset_y_px: float | None
    target_x_mm: float | None
    target_y_mm: float | None
    corners: tuple[tuple[float, float], ...]
    frame_age_ms: float
    laser_permission: bool = False
```

`from_hybrid()` must force `target_valid=False` unless state is exactly `TRACKING`, the hybrid result is valid, corners/center exist, and age is within the configured maximum. `laser_permission` remains false in this project stage even for a valid target; the future system-level safety controller owns laser permission.

Do not modify `src/ev_vision/protocol.py` or replace the existing rate-command packet in this task.

- [ ] **Step 4: Document both sides of the future link**

`docs/gimbal-vision-interface.md` must state that the gimbal sends:

```text
timestamp_ms
gimbal_ready
yaw_angle_deg
pitch_angle_deg
yaw_rate_deg_s
pitch_rate_deg_s
motion_state
fault_flags
```

and vision provides the exact `VisionTargetResult` fields above. State explicitly:

- `PREDICTING`, stale, ambiguous, model-error, camera-error, and excessive-jump results are invalid;
- missing/late gimbal feedback prevents laser permission but does not stop detection;
- the concrete UART/CAN framing, endianness, CRC, and rates are decided with the gimbal team in the next communication task;
- the 405 nm laser is physically disconnected or OFF during detector work.

- [ ] **Step 5: Run semantic tests and commit**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_vision_result.py
```

Expected: all pass.

```powershell
& $git add src/ev_vision/vision_result.py tests/test_vision_result.py docs/gimbal-vision-interface.md
& $git commit -m "feat: define gimbal vision result semantics"
```

---

### Task 10: Integrate rich hybrid results into the latest-only tuning service

**Files:**
- Modify: `src/ev_vision/tuning/models.py`
- Modify: `src/ev_vision/tuning/service.py`
- Modify: `tests/tuning/test_service.py`

- [ ] **Step 1: Write failing service tests for rich results and fail-safe behavior**

Add tests equivalent to:

```python
def test_service_publishes_hybrid_snapshot_without_blocking_acquisition() -> None:
    detector = SequencedHybridDetector([tracking_result(sequence=9)])
    service = make_service(detector=detector, detection_fps=1000)
    service.start()

    wait_until(lambda: service.latest_detection().source_sequence == 9)
    snapshot = service.latest_detection()

    assert snapshot.target_valid is True
    assert snapshot.tracking_state == "TRACKING"
    assert snapshot.model_backend == "onnx"
    assert snapshot.combined_score == pytest.approx(0.83)
    assert service.runtime_snapshot().state == "Streaming"


def test_detector_exception_invalidates_result_and_camera_keeps_streaming() -> None:
    service = make_service(detector=RaisingHybridDetector())
    service.start()

    wait_until(lambda: service.latest_detection().failure_reason == "MODEL_ERROR")

    assert service.latest_detection().target_valid is False
    assert service.runtime_snapshot().frame_count > 0


def test_disabling_detection_and_camera_restart_reset_tracker() -> None:
    detector = ResettableHybridDetector()
    service = make_service(detector=detector)
    service.start()
    service.set_detection_enabled(False)
    service.apply_parameters(changed_parameters())

    assert detector.reset_calls >= 2


def test_detection_config_update_does_not_reopen_camera() -> None:
    camera = FakeCamera()
    detector = ConfigurableHybridDetector()
    service = make_service(camera=camera, detector=detector)
    service.start()

    service.apply_detection_config(updated_detection_config())

    assert camera.close_calls == 0
    assert detector.config == updated_detection_config()
```

- [ ] **Step 2: Run focused service tests and verify shape mismatch**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tuning/test_service.py -k "hybrid or detector_exception or detection_config or reset_tracker"
```

Expected: failures because `DetectionSnapshot` and `DetectorPort` still use `BoardObservation | None`.

- [ ] **Step 3: Extend immutable tuning snapshots compatibly**

Replace the small detection snapshot with defaults that preserve old callers:

```python
@dataclass(frozen=True)
class DetectionSnapshot:
    enabled: bool
    detected: bool
    source_sequence: int | None = None
    observation: BoardObservation | None = None
    result_age_ms: float | None = None
    error: str | None = None
    target_valid: bool = False
    tracking_state: str = "SEARCHING"
    model_state: str = "UNAVAILABLE"
    model_backend: str = "none"
    model_path: str | None = None
    model_confidence: float = 0.0
    geometry_score: float = 0.0
    edge_support_score: float = 0.0
    structure_score: float = 0.0
    combined_score: float = 0.0
    candidate_count: int = 0
    confirmation_count: int = 0
    miss_count: int = 0
    failure_reason: str | None = None
    inference_ms: float = 0.0
    geometry_ms: float = 0.0
    total_ms: float = 0.0
```

Add immutable candidate and geometry fields needed by the ordinary overlay without rerunning inference:

```python
@dataclass(frozen=True)
class DetectionCandidateSnapshot:
    xyxy_px: tuple[float, float, float, float]
    accepted: bool
    model_confidence: float
    geometry_score: float
    edge_support_score: float
    structure_score: float
    temporal_score: float
    combined_score: float
    failure_reason: str | None = None
```

Append these defaulted fields to `DetectionSnapshot` after the timing fields:

```python
    temporal_score: float = 0.0
    homography_valid: bool = False
    target_x_mm: float | None = None
    target_y_mm: float | None = None
    corners_px: tuple[tuple[float, float], ...] = ()
    center_px: tuple[float, float] | None = None
    candidates: tuple[DetectionCandidateSnapshot, ...] = ()
```

Add a non-serialized latest-only container:

```python
@dataclass(frozen=True)
class DetectionDebugSnapshot:
    source_sequence: int
    images: Mapping[str, np.ndarray]
```

Extend `CaptureSnapshot` with `detection_debug: DetectionDebugSnapshot | None = None` at the end so current constructors continue to work.

- [ ] **Step 4: Adapt the detector port and loop**

Change the service port to:

```python
class DetectorPort(Protocol):
    def detect(
        self,
        image: np.ndarray,
        *,
        captured_ns: int,
        source_sequence: int,
        include_debug: bool = False,
        update_tracker: bool = True,
    ) -> HybridBoardResult:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def apply_config(self, config: DetectionConfig) -> None:
        raise NotImplementedError

    def reload_model(self) -> None:
        raise NotImplementedError
```

Add the exact copy/error helpers before the public methods:

```python
class StaleDetectionFrameError(RuntimeError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"requested detection sequence {expected}, latest is {actual}")
        self.expected = expected
        self.actual = actual


def copy_frame(frame: Frame | None) -> Frame | None:
    if frame is None:
        return None
    return Frame(frame.sequence, frame.captured_ns, frame.image.copy())


def copy_debug_images(images: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return {name: image.copy() for name, image in images.items()}
```

Add public methods:

```python
def detection_config(self) -> DetectionConfig:
    return self._detection_config


def apply_detection_config(self, config: DetectionConfig) -> DetectionConfig:
    self._detector.apply_config(config)
    self._detection_config = config
    return config


def detection_frame_for_latest(
    self,
    *,
    expected_sequence: int | None = None,
) -> tuple[Frame, DetectionSnapshot] | None:
    with self._lock:
        frame = copy_frame(self._latest_detection_frame)
        snapshot = self._latest_detection
    if frame is None or snapshot.source_sequence != frame.sequence:
        return None
    if expected_sequence is not None and frame.sequence != expected_sequence:
        raise StaleDetectionFrameError(expected_sequence, frame.sequence)
    return frame, snapshot


def detection_debug_for_latest(
    self,
    *,
    expected_sequence: int | None = None,
) -> DetectionDebugSnapshot | None:
    """Inspect the retained detection input frame without advancing the tracker."""
    pair = self.detection_frame_for_latest(expected_sequence=expected_sequence)
    if pair is None:
        return None
    frame, _snapshot = pair
    with self._lock:
        cached = self._latest_detection_debug
    if cached is not None and cached.source_sequence == frame.sequence:
        return cached
    result = self._detector.detect(
        frame.image,
        captured_ns=frame.captured_ns,
        source_sequence=frame.sequence,
        include_debug=True,
        update_tracker=False,
    )
    snapshot = DetectionDebugSnapshot(frame.sequence, copy_debug_images(result.debug_images))
    with self._lock:
        if self._latest_detection_frame is not None and self._latest_detection_frame.sequence == frame.sequence:
            self._latest_detection_debug = snapshot
    return snapshot


def reload_detection_model(self) -> None:
    self._detector.reload_model()
```

The detection loop must still copy only the latest frame and skip obsolete results. When publishing a result, retain exactly one copied input frame as `_latest_detection_frame`, map each `CandidateEvaluation` to `DetectionCandidateSnapshot`, and map the winning corners/center/homography fields to `DetectionSnapshot`. Do not generate or encode debug images in the normal detection loop. `detection_debug_for_latest()` is the only live-debug inference path; it reuses the retained detection input frame, uses `update_tracker=False`, and caches only the most recently requested debug result. Map `HybridBoardResult` to both `BoardObservation` compatibility data and the richer fields. A detector exception publishes `MODEL_ERROR` and invalidates the result; it must not exit acquisition or diagnostics. Reset detector/tracker when detection is disabled, a camera session is replaced, the camera disconnects, or service shutdown begins. Repeated model errors may set `model_state="ERROR"`, but the camera runtime state remains independent.

- [ ] **Step 5: Run all tuning-service tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tuning/test_service.py
```

Expected: all old camera apply/close-latency tests and new hybrid tests pass.

- [ ] **Step 6: Commit service integration**

```powershell
& $git add src/ev_vision/tuning/models.py src/ev_vision/tuning/service.py tests/tuning/test_service.py
& $git commit -m "feat: publish hybrid detector state in tuning service"
```

---

### Task 11: Add detection configuration, status, debug, and reload APIs

**Files:**
- Modify: `src/ev_vision/web/camera_tuning_app.py`
- Create: `tests/web/test_detection_api.py`

- [ ] **Step 1: Write failing API contract tests**

Create tests using the existing `TestClient` pattern:

```python
def test_detection_status_exposes_scores_tracker_and_model(client: TestClient) -> None:
    response = client.get("/api/detection/status")
    assert response.status_code == 200
    assert response.json() == {
        "enabled": True,
        "detected": True,
        "source_sequence": 42,
        "target_valid": True,
        "tracking_state": "TRACKING",
        "model_state": "READY",
        "model_backend": "onnx",
        "model_path": "models/target-board.onnx",
        "candidate_count": 2,
        "model_confidence": 0.91,
        "geometry_score": 0.88,
        "edge_support_score": 0.81,
        "structure_score": 0.72,
        "temporal_score": 0.75,
        "combined_score": 0.86,
        "confirmation_count": 3,
        "miss_count": 0,
        "failure_reason": None,
        "inference_ms": 18.0,
        "geometry_ms": 4.0,
        "total_ms": 22.0,
        "result_age_ms": 12.0,
        "homography_valid": True,
        "target_x_mm": 0.0,
        "target_y_mm": 0.0,
        "corners_px": [[100.0, 80.0], [520.0, 100.0], [500.0, 694.0], [120.0, 674.0]],
        "center_px": [310.0, 387.0],
        "candidates": fake_candidate_payloads(),
    }


def test_detection_config_update_is_separate_from_camera_parameters(client: TestClient) -> None:
    payload = client.get("/api/detection/config").json()
    payload["model"]["confidence_threshold"] = 0.50
    response = client.put("/api/detection/config", json=payload)
    assert response.status_code == 200
    assert response.json()["model"]["confidence_threshold"] == 0.50
    assert service.apply_calls == []


def test_invalid_detection_config_returns_422(client: TestClient) -> None:
    payload = valid_detection_payload()
    payload["roi_geometry"]["canny_low"] = 200
    payload["roi_geometry"]["canny_high"] = 100
    response = client.put("/api/detection/config", json=payload)
    assert response.status_code == 422
    assert "canny_low" in response.json()["detail"]


def test_debug_image_rejects_unknown_name_and_stale_sequence(client: TestClient) -> None:
    assert client.get("/api/detection/debug/not-a-view").status_code == 404
    response = client.get("/api/detection/debug/roi-edges", params={"sequence": 1})
    assert response.status_code == 409


def test_debug_request_does_not_advance_tracker(client: TestClient) -> None:
    before = client.get("/api/detection/status").json()
    response = client.get(
        "/api/detection/debug/roi-geometry",
        params={"sequence": before["source_sequence"]},
    )
    after = client.get("/api/detection/status").json()
    assert response.status_code == 200
    assert after["confirmation_count"] == before["confirmation_count"]
    assert after["miss_count"] == before["miss_count"]


def test_model_reload_failure_is_safe_conflict(client: TestClient) -> None:
    service.reload_error = RuntimeError("bad engine")
    response = client.post("/api/detection/model/reload")
    assert response.status_code == 409
    assert response.json()["detail"] == "model reload failed: bad engine"
```

- [ ] **Step 2: Run and verify endpoints return 404**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/web/test_detection_api.py
```

Expected: API route tests fail with 404.

- [ ] **Step 3: Add typed request models and explicit routes**

Add Pydantic request models mirroring the four detection config dataclasses, rejecting non-finite values. Add:

```text
GET  /api/detection/config
PUT  /api/detection/config
GET  /api/detection/status
GET  /api/detection/debug/{image_name}?sequence=N
POST /api/detection/model/reload
```

The only allowed debug names are:

```python
DEBUG_IMAGE_NAMES = {
    "model-candidates",
    "roi",
    "roi-edges",
    "roi-geometry",
    "final-overlay",
}
```

For the four detector views, call `service.detection_debug_for_latest(expected_sequence=sequence)`; for `final-overlay`, call `service.detection_frame_for_latest(expected_sequence=sequence)` and render the returned frame with its matching `DetectionSnapshot`. This ensures the status sequence and debug pixels refer to the same detector input even while acquisition continues. Encode only the selected image to JPEG in memory and use `Cache-Control: no-store`. Return 404 for an unknown/unavailable view, 409 for a requested sequence that is no longer retained, 422 for validation, and 409 for model reload failure. Never expose arbitrary filesystem paths.

- [ ] **Step 4: Extend the existing status response without breaking clients**

Keep `/api/status` and its current keys. Add the rich detection payload under its existing `detection` key. `/api/detection/status` returns that same detection dictionary, avoiding two schemas.

- [ ] **Step 5: Run API and existing web regressions**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/web/test_detection_api.py tests/web/test_camera_tuning_api.py
```

Expected: all pass.

- [ ] **Step 6: Commit the API**

```powershell
& $git add src/ev_vision/web/camera_tuning_app.py tests/web/test_detection_api.py tests/web/test_camera_tuning_api.py
& $git commit -m "feat: expose hybrid detection dashboard api"
```

---

### Task 12: Render candidate decisions and tracking state on the dashboard

**Files:**
- Modify: `src/ev_vision/web/camera_tuning_app.py`
- Modify: `src/ev_vision/web/static/camera-tuning.html`
- Modify: `src/ev_vision/web/static/camera-tuning.css`
- Modify: `src/ev_vision/web/static/camera-tuning.js`
- Modify: `tests/web/test_camera_tuning_tool.py`

- [ ] **Step 1: Write failing packaging/UI marker tests**

Add assertions for the exact controls and labels:

```python
def test_dashboard_contains_hybrid_detection_controls() -> None:
    html = static_text("camera-tuning.html")
    script = static_text("camera-tuning.js")

    assert 'id="detection-model-state"' in html
    assert 'id="detection-tracking-state"' in html
    assert 'id="detection-confidence-threshold"' in html
    assert 'id="detection-ambiguity-margin"' in html
    assert 'id="detection-debug-view"' in html
    assert 'id="reload-detection-model"' in html
    assert "/api/detection/status" in script
    assert "/api/detection/config" in script
    assert "/api/detection/model/reload" in script
```

Add an overlay test that inspects pixels or a fake OpenCV call recorder to prove:

- model candidate box uses yellow;
- geometry-rejected candidate box uses red;
- validated quadrilateral uses green;
- text includes tracking state, score, and failure code.

- [ ] **Step 2: Run and verify the UI markers are missing**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/web/test_camera_tuning_tool.py -k "hybrid or overlay"
```

Expected: assertions fail because the hybrid controls are absent.

- [ ] **Step 3: Extend the no-build dashboard HTML/CSS**

Add a compact “靶面检测” section showing:

- model state/backend/path;
- inference FPS from runtime and inference/geometry/total latency;
- candidate count and four score components;
- tracker state, confirmation count, miss count, target validity, age, and failure code;
- editable confidence, max-candidate, Canny, edge/geometry threshold, score-weight, ambiguity, and tracker fields;
- buttons `应用检测参数`, `恢复项目默认值`, and `重新加载模型`;
- debug image selector with the five allowed names and an explicit `刷新调试图` button.

Keep camera parameter controls separate so applying detection configuration cannot trigger camera reopen.

- [ ] **Step 4: Implement polling and apply behavior in vanilla JavaScript**

Use one detection-status poll at the existing status cadence. Keep the debug image stopped by default; request one selected debug view only when the user changes the selector or clicks an explicit `刷新调试图` button. Do not attach a changing debug URL to every status poll:

```javascript
async function refreshDetectionStatus() {
  const status = await requestJson('/api/detection/status');
  latestDetectionSequence = status.source_sequence;
  renderDetectionStatus(status);
}

async function refreshDetectionDebug() {
  if (!selectedDebugView || latestDetectionSequence == null) return;
  debugImage.src = `/api/detection/debug/${selectedDebugView}`
    + `?sequence=${latestDetectionSequence}&t=${Date.now()}`;
}

async function applyDetectionConfig() {
  const payload = readDetectionConfigForm();
  const applied = await requestJson('/api/detection/config', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  renderDetectionConfig(applied);
}
```

Disable apply/reload buttons while requests are pending. Show the server `detail` text on 409/422 rather than the old generic “请求失败”. Do not continuously request every debug image.

- [ ] **Step 5: Extend overlay rendering**

In `render_overlay()` draw all candidate decisions from `DetectionSnapshot.candidates`, plus `corners_px` and `center_px`: yellow model boxes, red rejected boxes, green winning corners, center, image-center offset line, and text. Color is supplemental; text must identify `TRACKING`, `PREDICTING`, invalid state, combined score, and failure reason.

- [ ] **Step 6: Run static and API regressions**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/web/test_camera_tuning_tool.py tests/web/test_detection_api.py tests/web/test_camera_tuning_api.py
```

Expected: all pass.

- [ ] **Step 7: Commit dashboard changes**

```powershell
& $git add src/ev_vision/web/camera_tuning_app.py src/ev_vision/web/static/camera-tuning.html src/ev_vision/web/static/camera-tuning.css src/ev_vision/web/static/camera-tuning.js tests/web/test_camera_tuning_tool.py
& $git commit -m "feat: add hybrid detector dashboard controls"
```

---

### Task 13: Save optional hybrid debug products in manual capture bundles

**Files:**
- Modify: `src/ev_vision/tuning/models.py`
- Modify: `src/ev_vision/tuning/storage.py`
- Modify: `src/ev_vision/tuning/service.py`
- Modify: `src/ev_vision/web/camera_tuning_app.py`
- Modify: `tests/tuning/test_storage.py`
- Modify: `tests/tuning/test_service.py`
- Modify: `tests/web/test_camera_tuning_api.py`

- [ ] **Step 1: Write failing storage and capture-response tests**

Add:

```python
def test_capture_saves_available_hybrid_debug_images(tmp_path: Path) -> None:
    snapshot = capture_snapshot(
        detection_debug=DetectionDebugSnapshot(
            source_sequence=42,
            images={
                "model-candidates": image(10),
                "roi": image(20),
                "roi-edges": gray_image(30),
                "roi-geometry": image(40),
            },
        )
    )

    capture = storage(tmp_path).save_capture(snapshot, overlay_image=image(50))

    assert sorted(path.name for path in capture.glob("*.png")) == [
        "model-candidates.png",
        "original.png",
        "overlay.png",
        "roi-edges.png",
        "roi-geometry.png",
        "roi.png",
    ]


def test_capture_with_no_hybrid_debug_preserves_original_contract(tmp_path: Path) -> None:
    capture = storage(tmp_path).save_capture(capture_snapshot(), overlay_image=image(50))
    assert sorted(path.name for path in capture.iterdir()) == [
        "metadata.yaml", "original.png", "overlay.png"
    ]
```

Add an API assertion that `POST /api/captures` returns the actual files written, not a fixed list.

- [ ] **Step 2: Run and verify only the three old files are saved**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tuning/test_storage.py tests/web/test_camera_tuning_api.py -k capture
```

Expected: the hybrid-debug capture test fails.

- [ ] **Step 3: Save only available latest-frame debug products atomically**

When `capture_snapshot()` is called, pass its copied frame to the same private inspection helper used by `detection_debug_for_latest()`, with `include_debug=True` and `update_tracker=False`. Attach `DetectionDebugSnapshot` only if its `source_sequence` equals the captured frame sequence. `TuningStorage.save_capture()` writes each allowed image through the existing temporary-file plus atomic rename path. If any write fails, remove the entire newly created capture directory as before.

Map names safely:

```python
_CAPTURE_DEBUG_FILES = {
    "model-candidates": "model-candidates.png",
    "roi": "roi.png",
    "roi-edges": "roi-edges.png",
    "roi-geometry": "roi-geometry.png",
}
```

Include detection backend, scores, tracking state, failure reason, timings, and debug source sequence in `metadata.yaml`. No continuous writes are allowed; files are created only by `POST /api/captures`.

- [ ] **Step 4: Return the real capture file list**

After storage completes, enumerate only expected regular files inside the new capture directory and return their names sorted. Preserve the safe relative capture identifier and do not expose the host path.

- [ ] **Step 5: Run storage, service, and API tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tuning/test_storage.py tests/tuning/test_service.py tests/web/test_camera_tuning_api.py
```

Expected: all pass, including atomic cleanup and path-safety tests.

- [ ] **Step 6: Commit capture expansion**

```powershell
& $git add src/ev_vision/tuning/models.py src/ev_vision/tuning/storage.py src/ev_vision/tuning/service.py src/ev_vision/web/camera_tuning_app.py tests/tuning/test_storage.py tests/tuning/test_service.py tests/web/test_camera_tuning_api.py
& $git commit -m "feat: include hybrid debug views in captures"
```

---

### Task 14: Add training, evaluation, export, and prediction-inspection tools

**Files:**
- Create: `tools/train_target_detector.py`
- Create: `tools/evaluate_target_detector.py`
- Create: `tools/export_target_detector.py`
- Create: `tools/inspect_target_predictions.py`
- Create: `tests/tools/test_target_training_tools.py`
- Create: `models/README.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing tests around lazy imports and argument forwarding**

Create tests with a fake `ultralytics.YOLO`:

```python
def test_train_forwards_reproducible_nano_settings(monkeypatch, tmp_path: Path) -> None:
    fake = install_fake_ultralytics(monkeypatch)
    exit_code = train_main([
        "--data", str(tmp_path / "dataset.yaml"),
        "--model", "yolo11n.pt",
        "--epochs", "80",
        "--imgsz", "640",
        "--device", "0",
        "--project", str(tmp_path / "runs"),
        "--name", "target-board-v1",
    ])

    assert exit_code == 0
    assert fake.train_calls == [{
        "data": str(tmp_path / "dataset.yaml"),
        "epochs": 80,
        "imgsz": 640,
        "device": "0",
        "project": str(tmp_path / "runs"),
        "name": "target-board-v1",
        "single_cls": True,
        "seed": 20250716,
        "deterministic": True,
        "patience": 20,
        "batch": -1,
        "hsv_h": 0.015,
        "hsv_s": 0.50,
        "hsv_v": 0.40,
        "degrees": 7.0,
        "translate": 0.08,
        "scale": 0.25,
        "shear": 0.0,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 0.50,
        "mosaic": 0.20,
        "close_mosaic": 10,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
    }]


def test_missing_ultralytics_prints_install_command(monkeypatch, capsys) -> None:
    block_import(monkeypatch, "ultralytics")
    assert export_main(["--model", "best.pt", "--format", "onnx"]) == 2
    assert "pip install -e .[training]" in capsys.readouterr().err
```

Also test evaluation writes a JSON report and inspection uses `save=False` plus an explicit output directory.

- [ ] **Step 2: Run and verify tool modules are missing**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tools/test_target_training_tools.py
```

Expected: collection fails for missing tool modules.

- [ ] **Step 3: Add an optional training dependency and lazy tool entry points**

Modify `pyproject.toml`:

```toml
training = ["ultralytics>=8.3"]
```

Do not add Ultralytics to mandatory runtime dependencies. Every tool imports `YOLO` inside `main()` through a helper:

```python
def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required; install with: pip install -e .[training]"
        ) from exc
    return YOLO
```

`train_target_detector.py` defaults:

```text
model=yolo11n.pt
epochs=80
imgsz=640
device=0
single_cls=True
seed=20250716
deterministic=True
patience=20
batch=-1
hsv_h=0.015
hsv_s=0.50
hsv_v=0.40
degrees=7.0
translate=0.08
scale=0.25
shear=0.0
perspective=0.0005
flipud=0.0
fliplr=0.50
mosaic=0.20
close_mosaic=10
mixup=0.0
cutmix=0.0
copy_paste=0.0
```

Use `--model` to allow replacing the Nano checkpoint if the installed Ultralytics release uses another supported Nano name; the committed default remains explicit and tested. These fixed augmentations provide moderate color/brightness, rotation, translation, scale, perspective, horizontal flip, and limited mosaic. Vertical flip, mixup, cutmix, copy-paste, severe crop, and unrealistic deformation are disabled. The one-day first release does not synthesize blur or sensor noise automatically; collect natural defocus, motion blur, and noisy exposures as difficult training scenes instead, so validation/test images remain untouched and no derivative can leak across scene splits.

- [ ] **Step 4: Implement evaluation, export, and inspection contracts**

- evaluation calls `model.val(data=..., split="test", imgsz=640, device=...)`, then runs image-level prediction over the test manifest and writes JSON containing precision, recall, mAP50, mAP50-95, speed, model/data paths, threshold arguments, `clear_recall`, `difficult_recall`, `negative_false_positive_images`, and `max_negative_high_confidence_streak`;
- ONNX export calls `model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=12)`;
- TensorRT export is allowed only when `--format engine` is explicitly run on Jetson and uses `half=True` after a successful FP32/ONNX validation;
- `clear_recall` and `difficult_recall` count a positive image as found when at least one class-0 box meets the configured confidence and IoU threshold; negative streaks are measured in sorted order within each negative scene;
- inspection predicts an image directory, writes annotated images to the selected output, and never edits labels; its review CSV contains `image`, `difficulty`, predicted confidence, matched IoU, `full_board_contained` (`yes|no`), and notes for the manual full-board-box check.

- [ ] **Step 5: Document artifacts and exact commands**

In `models/README.md`, define:

```text
target-board.pt      selected best Windows checkpoint
target-board.onnx    portable deployment fallback
target-board.engine  Jetson-local TensorRT build; never copied from Windows
```

Record model source, dataset manifest hash, training command, metrics JSON, export command, SHA-256, JetPack/TensorRT version, and calibration/test date for each selected release.

- [ ] **Step 6: Run tool tests and help smoke tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/tools/test_target_training_tools.py
python tools/train_target_detector.py --help
python tools/evaluate_target_detector.py --help
python tools/export_target_detector.py --help
python tools/inspect_target_predictions.py --help
```

Expected: tests pass and help commands do not require Ultralytics to be installed.

- [ ] **Step 7: Commit training tools**

```powershell
& $git add pyproject.toml tools/train_target_detector.py tools/evaluate_target_detector.py tools/export_target_detector.py tools/inspect_target_predictions.py tests/tools/test_target_training_tools.py models/README.md
& $git commit -m "feat: add target board model training tools"
```

---

### Task 15: Load engine, fall back to ONNX, and preserve diagnostic preview

**Files:**
- Modify: `src/ev_vision/web/camera_tuning_server.py`
- Modify: `src/ev_vision/detection/hybrid_board.py`
- Modify: `tests/web/test_camera_tuning_tool.py`
- Modify: `tests/detection/test_hybrid_board.py`

- [ ] **Step 1: Write failing backend-selection tests**

Add tests:

```python
def test_server_prefers_engine_then_falls_back_to_onnx(monkeypatch, tmp_path: Path) -> None:
    engine = tmp_path / "target-board.engine"
    onnx = tmp_path / "target-board.onnx"
    engine.write_bytes(b"engine")
    onnx.write_bytes(b"onnx")
    loader = FakeBackendLoader(fail_paths={engine})

    detector = build_detector(detection_config(engine, onnx), model_loader=loader)

    assert loader.paths == [engine, onnx]
    assert detector.model_backend == "onnx"


def test_missing_models_builds_invalid_diagnostic_detector_without_blocking_app(tmp_path: Path) -> None:
    app = build_application(
        args_with_missing_models(tmp_path),
        native_api_factory=lambda: FakeNativeApi(),
        backend_factory=FakeBackendFactory(),
    )
    status = app.state.service.latest_detection()
    assert status.model_state == "UNAVAILABLE"
    assert status.target_valid is False
```

Test reload success atomically swaps the backend and reload failure keeps acquisition alive while invalidating target results.

- [ ] **Step 2: Run and verify the server still wires classical geometry directly**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/web/test_camera_tuning_tool.py tests/detection/test_hybrid_board.py -k "engine or onnx or missing_models or reload"
```

Expected: failures because `build_application()` always constructs `BoardGeometryDetector()`.

- [ ] **Step 3: Implement an injectable backend builder**

Add the explicit selection type and builder:

```python
@dataclass(frozen=True)
class DetectionBackendSelection:
    backend: InferencePort | None
    path: Path | None
    artifact_kind: str
    errors: tuple[str, ...]


def build_detection_backend(
    config: DetectionConfig,
    *,
    backend_factory: Callable[..., InferencePort] = UltralyticsBackend,
) -> DetectionBackendSelection:
    errors = []
    for path in (config.model.path, config.model.fallback_path):
        if not Path(path).is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            backend = backend_factory(
                path,
                input_size=(config.model.input_width, config.model.input_height),
                device=config.model.device,
            )
            return DetectionBackendSelection(backend, Path(path), backend.artifact_kind, ())
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return DetectionBackendSelection(None, None, "classical-diagnostic", tuple(errors))
```

Change the server entry point to an injectable signature:

```python
def build_application(
    args: argparse.Namespace,
    *,
    native_api_factory: Callable[[], MvsApi] = create_native_api,
    backend_factory: Callable[..., InferencePort] = UltralyticsBackend,
):
    native_api = native_api_factory()
    detector = build_detector(
        config.detection,
        model_loader=lambda detection_config: build_detection_backend(
            detection_config,
            backend_factory=backend_factory,
        ),
    )
    # Keep the existing camera, service, storage, and FastAPI construction below.
```

`build_detector()` constructs `YoloBoardDetector`, `RoiBoardGeometry`, `BoardTracker`, and `HybridBoardDetector` from the loaded config. If no model loads, retain `BoardGeometryDetector` only as a dashboard diagnostic source wrapped by a safe adapter that always returns `MODEL_UNAVAILABLE`, `model_backend="classical-diagnostic"`, and `target_valid=False`.

- [ ] **Step 4: Make model reload atomic and safe**

Give `HybridBoardDetector` a `model_loader: Callable[[DetectionConfig], DetectionBackendSelection]` callback and make it the sole owner of the active backend. `HybridBoardDetector.reload_model()` loads and smoke-tests the new backend before acquiring its own short swap lock, then swaps model/backend metadata and resets the tracker atomically. The tuning service only calls `detector.reload_model()`; the server does not mutate a live detector after construction. On failure, retain camera acquisition and preview, set model state/error, clear old valid target data, and return a reload error to the API. Do not continue using stale valid coordinates from the previous backend.

- [ ] **Step 5: Run server, API, and detector tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/web/test_camera_tuning_tool.py tests/web/test_detection_api.py tests/detection/test_hybrid_board.py
```

Expected: all pass.

- [ ] **Step 6: Commit backend wiring**

```powershell
& $git add src/ev_vision/web/camera_tuning_server.py src/ev_vision/detection/hybrid_board.py tests/web/test_camera_tuning_tool.py tests/detection/test_hybrid_board.py
& $git commit -m "feat: load portable board detector backends"
```

---

### Task 16: Run the first Windows training iteration and freeze evaluation evidence

**Files:**
- Local only: `datasets/target_board/images/`
- Local only: `datasets/target_board/labels/`
- Local only: `runs/target-board/`
- Local/generated: `models/target-board.pt`
- Local/generated: `models/target-board.onnx`
- Create or update: `models/README.md`

- [ ] **Step 1: Install training dependencies in the Windows project environment**

Run in the project Conda environment:

```powershell
python -m pip install -e ".[dev,vision,training]"
python -c "import cv2, torch, ultralytics; print('OpenCV', cv2.__version__); print('CUDA', torch.cuda.is_available()); print('Ultralytics', ultralytics.__version__)"
```

Expected: imports succeed and CUDA reports `True` on the RTX 4070 Laptop GPU. If CUDA is false, install the appropriate PyTorch CUDA wheel before training; do not silently train the one-day model on CPU.

- [ ] **Step 2: Prepare captures by scene and label only original images**

For each capture session, run:

```powershell
python tools/prepare_target_dataset.py --source "D:\path\to\capture-root" --output datasets/target_board/staging --scene desk-left --split train --difficulty clear
```

Then label `target_board` class `0` with tight complete-board boxes. Move/export by the scene-assigned split into `images/{train,val,test}` and `labels/{train,val,test}` and fill `split-manifest.csv`. Do not label `overlay.png`, masks, or debug images.

Expected final collection: approximately 240–280 total images, including 30–50 hard negatives and no scene crossing splits.

- [ ] **Step 3: Validate before training**

Run:

```powershell
python tools/validate_target_dataset.py --dataset datasets/target_board
```

Expected: exit code 0; no invalid labels, missing pairs, duplicate cross-split images, or scene leakage. Counts are printed per split and for clear positives, difficult positives, and negatives.

- [ ] **Step 4: Train iteration 1**

Run:

```powershell
python tools/train_target_detector.py --data datasets/target_board/dataset.yaml --model yolo11n.pt --epochs 80 --imgsz 640 --device 0 --project runs/target-board --name iteration-1
```

Expected: a best checkpoint is created under `runs/target-board/iteration-1/weights/best.pt`. Preserve console output and results CSV.

- [ ] **Step 5: Evaluate only on the unseen test split and inspect failures**

Run:

```powershell
python tools/evaluate_target_detector.py --model runs/target-board/iteration-1/weights/best.pt --data datasets/target_board/dataset.yaml --split test --device 0 --output runs/target-board/iteration-1/test-metrics.json
python tools/inspect_target_predictions.py --model runs/target-board/iteration-1/weights/best.pt --source datasets/target_board/images/test --output runs/target-board/iteration-1/test-predictions --device 0
```

Expected: metrics JSON and annotated predictions exist. The JSON explicitly reports clear/difficult image recall and the maximum high-confidence false-positive streak on negative scenes. Complete the review CSV for full-board containment and manually list false positives, missed difficult positives, and rectangular distractors.

- [ ] **Step 6: Collect hard failures and train iteration 2**

Add 30–60 images focused on observed misses and false positives, keep new scenes isolated to a single split, validate again, then run:

```powershell
python tools/validate_target_dataset.py --dataset datasets/target_board
python tools/train_target_detector.py --data datasets/target_board/dataset.yaml --model runs/target-board/iteration-1/weights/best.pt --epochs 50 --imgsz 640 --device 0 --project runs/target-board --name iteration-2
python tools/evaluate_target_detector.py --model runs/target-board/iteration-2/weights/best.pt --data datasets/target_board/dataset.yaml --split test --device 0 --output runs/target-board/iteration-2/test-metrics.json
```

Acceptance target: `clear_recall >= 0.95`, `difficult_recall >= 0.85`, every accepted model ROI contains the full board sufficiently for geometric refinement, and `max_negative_high_confidence_streak == 0` at the selected operating threshold. The hybrid regression must also reject model boxes whose ROI geometry cannot verify a board. Iteration 2 must not regress clear-board recall and should reduce observed distractor false positives. If metrics conflict with visual safety evidence, select the checkpoint with fewer sustained false positives and record the reason.

- [ ] **Step 7: Export and record the selected portable artifacts**

Run:

```powershell
Copy-Item runs/target-board/iteration-2/weights/best.pt models/target-board.pt
python tools/export_target_detector.py --model models/target-board.pt --format onnx --imgsz 640
Get-FileHash models/target-board.pt -Algorithm SHA256
Get-FileHash models/target-board.onnx -Algorithm SHA256
```

The exporter must produce `models/target-board.onnx` beside the selected checkpoint. Record both hashes, selected iteration, metrics path, and commands in `models/README.md`.

- [ ] **Step 8: Commit only policy-approved small artifacts**

Check sizes:

```powershell
Get-Item models/target-board.pt, models/target-board.onnx | Select-Object Name,Length
```

Repository default is to commit `models/README.md` only and distribute `.pt`/`.onnx` as release artifacts because `.onnx` is Git-ignored. If the team explicitly chooses to version the small `.pt`, update `.gitignore` narrowly and use Git LFS; never commit `.engine`.

```powershell
& $git add models/README.md
& $git commit -m "docs: record selected target board model"
```

---

### Task 17: Write the Jetson deployment and acceptance runbook

**Files:**
- Create: `docs/hybrid-detector-acceptance.md`
- Modify: `README.md`

- [ ] **Step 1: Write a runbook-content test before documentation**

Add to `tests/web/test_camera_tuning_tool.py`:

```python
def test_hybrid_acceptance_runbook_contains_required_safety_and_commands() -> None:
    text = Path("docs/hybrid-detector-acceptance.md").read_text(encoding="utf-8")
    for marker in (
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "models/target-board.onnx",
        "models/target-board.engine",
        "ev-camera-tuning",
        "TRACKING",
        "15-20 Hz",
        "20 minutes",
        "405 nm",
        "physically disconnected",
    ):
        assert marker in text
```

- [ ] **Step 2: Run and verify the runbook is absent**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/web/test_camera_tuning_tool.py -k acceptance_runbook
```

Expected: failure because the document does not exist.

- [ ] **Step 3: Document exact Jetson installation and verification**

Include these commands, correcting only environment-specific MVS paths if the installed report proves different:

```bash
cd ~/2025-E-Vision/2025-E-Vision
conda activate 2025-e-vision
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pip install -e ".[dev,vision,tuning,hardware,training]"
python -m compileall -q src tests tools
TEST_TMP="$(mktemp -d)"
python -m pytest -q -p no:cacheprovider --basetemp "$TEST_TMP"
```

Copy both `models/target-board.onnx` and `models/target-board.pt` to Jetson. Validate ONNX fallback loading first, then build the TensorRT engine locally from the selected `.pt`:

```bash
python tools/export_target_detector.py \
  --model models/target-board.pt \
  --format engine \
  --imgsz 640 \
  --device 0
```

Record the exact JetPack, CUDA, TensorRT, Ultralytics, and model hashes. Never use a Windows-generated engine. If engine export fails, keep the tested ONNX backend as the explicit safe fallback instead of using an unverified engine.

- [ ] **Step 4: Document exact dashboard launch and safety check**

Use:

```bash
python -m ev_vision.web.camera_tuning_server \
  --config config/default.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --output artifacts/camera-tuning \
  --preview-fps 20 \
  --detection-fps 20
```

Before launch, require:

```text
405 nm laser physically disconnected or OFF
camera visible in lsusb and MVS closed
no old uvicorn/ev-camera-tuning process on port 8000
model hashes match the recorded release
```

- [ ] **Step 5: Document the measurable acceptance sequence**

The runbook must require and record:

1. engine selected, or explicit ONNX fallback selected;
2. normal Hikrobot preview and camera parameter control;
3. yellow box on the true board;
4. green quadrilateral and center after geometry acceptance;
5. three accepted frames enter `TRACKING`;
6. removing the board reaches `LOST` and clears `target_valid`;
7. rectangular distractors do not sustain `TRACKING`;
8. `PREDICTING` remains invalid;
9. effective result rate approximately 15–20 Hz using an exposure short enough for expected motion;
10. 20-minute run with no camera loss, worker exit, or continuously growing resident memory;
11. model removal/reload failure preserves preview but publishes invalid target;
12. all capture evidence and config/model hashes are archived.

Provide copyable status checks:

```bash
curl -sS http://127.0.0.1:8000/api/detection/status | python -m json.tool
curl -sS http://127.0.0.1:8000/api/detection/config | python -m json.tool
ps -o pid,rss,etime,cmd -C python
```

- [ ] **Step 6: Link the runbook from README and run docs test**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/web/test_camera_tuning_tool.py -k acceptance_runbook
```

Expected: pass.

- [ ] **Step 7: Commit the runbook**

```powershell
& $git add docs/hybrid-detector-acceptance.md README.md tests/web/test_camera_tuning_tool.py
& $git commit -m "docs: add hybrid detector Jetson acceptance"
```

---

### Task 18: Final regression, safety audit, and branch delivery

**Files:**
- Verify: all changed source, tests, tools, config, and docs
- No new feature scope

- [ ] **Step 1: Compile all Python sources in a disposable cache**

Windows PowerShell:

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP ("ev-vision-pycache-" + [guid]::NewGuid())
python -m compileall -q src tests tools
```

Expected: exit code 0 and no syntax errors.

- [ ] **Step 2: Run focused hybrid suites**

Run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -q -p no:cacheprovider tests/detection tests/tracking tests/tools tests/test_vision_result.py tests/tuning/test_service.py tests/tuning/test_storage.py tests/web/test_detection_api.py tests/web/test_camera_tuning_tool.py
```

Expected: all pass.

- [ ] **Step 3: Run the complete project suite with a disposable base temp**

Run:

```powershell
$testTmp = Join-Path $env:TEMP ("ev-vision-final-" + [guid]::NewGuid())
python -m pytest -q -p no:cacheprovider --basetemp $testTmp
```

Expected: all tests pass. Save the exact count in the final delivery note.

- [ ] **Step 4: Run configuration and dataset smoke checks**

Run:

```powershell
python -m ev_vision.cli validate-config --config config/default.yaml
python tools/validate_target_dataset.py --dataset datasets/target_board
```

Expected: config validation succeeds. Dataset validation succeeds when local labeled data are present; if the repository contains only the empty contract, the validator must report “dataset contains no images” clearly and return nonzero without affecting code-test success.

- [ ] **Step 5: Audit fail-safe invariants in code and tests**

Search:

```powershell
Select-String -Path src\ev_vision\**\*.py,tests\**\*.py -Pattern 'target_valid|laser_permission|PREDICTING|MODEL_UNAVAILABLE|STALE_FRAME'
```

Confirm with concrete tests that:

- only `TRACKING` can make `target_valid` true;
- prediction never authorizes laser;
- model/classical-only failure is invalid;
- stale and excessive-jump results are invalid;
- camera and model reload failures clear old valid results;
- the semantic DTO keeps `laser_permission=False` in this delivery.

- [ ] **Step 6: Check formatting, placeholders, generated files, and diff**

Run:

```powershell
& $git diff --check
& $git status --short
$markers = @('TO' + 'DO', 'TB' + 'D', 'implement ' + 'later', 'fill in ' + 'details')
Get-ChildItem src,tests,tools,docs -Recurse -File | Select-String -Pattern $markers
```

Expected: `git diff --check` produces no output; no cache, dataset image, run, engine, or temporary artifact is staged; placeholder scan returns no unresolved implementation marker.

- [ ] **Step 7: Create the final regression commit only if verification changed tracked files**

If documentation or test fixtures were adjusted during final verification:

```powershell
& $git add README.md docs src tests tools config pyproject.toml .gitignore datasets models
& $git commit -m "test: verify hybrid target board pipeline"
```

If the worktree is already clean, do not create an empty commit.

- [ ] **Step 8: Inspect the delivery history and push when authorized**

Run:

```powershell
& $git status --short --branch
& $git log --oneline --decorate -20
```

Expected: clean `feature/classical-vision`, ahead of `origin/feature/classical-vision` by the completed commits. After user authorization or the already established repository workflow:

```powershell
& $git push origin feature/classical-vision
```

On Jetson, update only after the push succeeds:

```bash
cd ~/2025-E-Vision/2025-E-Vision
git switch feature/classical-vision
git pull --ff-only origin feature/classical-vision
```

Do not claim hardware acceptance until the Task 17 Jetson checklist is executed on the actual camera and board.

---

## One-day operator schedule after implementation

Use this schedule only after Tasks 1–15 are implemented and their tests pass; Tasks 16–17 are the operator execution described below:

| Time | Operator action | Produced evidence |
|---|---|---|
| 0–1.5 h | collect about 250 varied positives/negatives on Jetson dashboard | capture directories and scene list |
| 1.5–3 h | prepare, label, scene-split, and validate on Windows | valid dataset report and manifest |
| 3–4 h | train iteration 1 and inspect unseen test predictions | checkpoint, metrics JSON, failure list |
| 4–5 h | collect/label hard negatives and misses | expanded validated dataset |
| 5–6 h | train iteration 2, select, and export PT/ONNX | hashes and selected model record |
| 6–8 h | copy ONNX/PT, build Jetson engine, start dashboard | backend status and live overlay |
| 8–10 h | tune thresholds and run distractor/loss/motion checks | saved config and capture bundles |
| remaining | run 20-minute stability test and archive evidence | acceptance record and fallback artifacts |

Training may overlap with organizing additional captures. Do not overlap model work with live laser testing; the laser remains physically disconnected or OFF for the entire schedule.

