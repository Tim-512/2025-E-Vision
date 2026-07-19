# Gimbal USB Calibration and Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a low-memory Jetson workflow that captures and solves an 8×5-inner-corner chessboard calibration, converts the detected target center to optical-axis yaw/pitch error, and safely sends it to the gimbal board over the independent A5 5A USB CDC-ACM protocol at 50 Hz.

**Architecture:** Keep the existing classical detector, tracker, local preview, and legacy AA55 protocol unchanged. Add a camera-only calibration-capture package and an isolated `ev_vision.gimbal_usb` package containing strict configuration, A5 5A encoding, single-point angle conversion, safety gating, reconnecting serial transport, a latest-only 50 Hz worker, and a CLI with optional OpenCV preview. Missing calibration, stale/invalid detection, excessive prediction, serial failure, and shutdown all fail closed to an all-zero `tracking=0` frame.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PyYAML, pyserial, pytest, Hikrobot MVS SDK through the existing `ev_vision_mvs_adapter`.

---

## Fixed decisions used by every task

- Work only on `feature/classical-white-board-tracking`; do not merge to `main`.
- Camera: MV-CA013-21UC serial `00G02809155`, 1280×1024 BayerRG8, buffer 2.
- Runtime: exposure `15000 us`, gain `14 dB`, acquisition `50 FPS`, detection `50 FPS`, optional display `45 FPS`.
- Chessboard: 6×9 physical squares means OpenCV `(8, 5)` inner corners; square side `22.0 mm`.
- Calibration: exact 1280×1024, at least 10 usable poses, recommended 15+, RMS ≤ `0.5 px`.
- Port: `/dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00`, 115200 8N1, no flow control.
- Output: 50 Hz, 26-byte A5 5A frame, CRC16/MODBUS.
- `distance_m=0.0`, `fire=0`, `target_id=0`, and `reserved=0` are encoder constants.
- Pure `PREDICTED` control is allowed for frames 1–3 only, age ≤60 ms, per-axis step ≤1.5°.
- A recovered real observation is immediately eligible; do not add reacquisition jump checks or two-frame reconfirmation.
- Do not add YOLO, red-color logic, full-frame undistortion, or web services.
- The 405 nm laser is hardware-always-on; tests require physical disconnection or a reliable cover.
- Leave unrelated `.tmp/` data untouched and unstaged.

## Task 1: Report usable chessboard poses accurately

**Files:**
- Modify: `src/ev_vision/calibration.py:106-136`
- Modify: `tools/calibrate_camera.py:1-34`
- Modify: `tests/test_calibration.py`
- Create: `tests/tools/test_calibrate_camera.py`
- Modify: `config/camera_calibration.example.yaml`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing solve-report tests**

Add to `tests/test_calibration.py`:

```python
from ev_vision.calibration import ChessboardCalibrationResult, solve_chessboard_with_report


def test_solve_report_counts_only_complete_corner_sets(monkeypatch) -> None:
    images = [np.zeros((1024, 1280, 3), np.uint8) for _ in range(12)]
    calls = iter([(i < 10, np.zeros((40, 1, 2), np.float32)) for i in range(12)])
    monkeypatch.setattr("ev_vision.calibration.cv2.findChessboardCorners", lambda *a, **k: next(calls))
    monkeypatch.setattr("ev_vision.calibration.cv2.cornerSubPix", lambda gray, corners, *a: corners)
    monkeypatch.setattr(
        "ev_vision.calibration.cv2.calibrateCamera",
        lambda *a, **k: (0.31, calibration().camera_matrix, calibration().distortion, [], []),
    )

    report = solve_chessboard_with_report(images, pattern_size=(8, 5), square_size_mm=22.0)

    assert isinstance(report, ChessboardCalibrationResult)
    assert (report.input_images, report.usable_poses, report.rejected_images) == (12, 10, 2)
    assert report.calibration.rms_px == pytest.approx(0.31)


def test_solve_rejects_mixed_image_sizes() -> None:
    images = [np.zeros((1024, 1280, 3), np.uint8) for _ in range(9)]
    images.append(np.zeros((720, 1280, 3), np.uint8))
    with pytest.raises(CalibrationError, match="same size"):
        solve_chessboard_with_report(images, pattern_size=(8, 5), square_size_mm=22.0)
```

Create `tests/tools/test_calibrate_camera.py` using the repository's existing file-based tool import pattern:

```python
def test_cli_prints_loaded_and_usable_counts(tmp_path, monkeypatch, capsys) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(12):
        (image_dir / f"capture-{index:03d}.png").write_bytes(b"image")
    monkeypatch.setattr(module.cv2, "imread", lambda *a: np.zeros((1024, 1280, 3), np.uint8))
    monkeypatch.setattr(
        module,
        "solve_chessboard_with_report",
        lambda *a, **k: ChessboardCalibrationResult(calibration(), 12, 10, 2),
    )
    monkeypatch.setattr(module, "save_calibration", lambda path, value: Path(path).write_text("saved"))

    assert module.main([
        str(image_dir), "--output", str(tmp_path / "camera.yaml"),
        "--columns", "8", "--rows", "5", "--square-mm", "22", "--max-rms", "0.5",
    ]) == 0
    assert "images=12 usable_poses=10 rejected=2 rms_px=0.3100" in capsys.readouterr().out
```

- [ ] **Step 2: Run and confirm the new API is missing**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/test_calibration.py tests/tools/test_calibrate_camera.py
```

Expected: FAIL because `ChessboardCalibrationResult`, `solve_chessboard_with_report()`, and `main(argv)` do not exist.

- [ ] **Step 3: Implement the report without breaking `solve_chessboard()`**

Add this exact public shape to `src/ev_vision/calibration.py`:

```python
@dataclass(frozen=True)
class ChessboardCalibrationResult:
    calibration: Calibration
    input_images: int
    usable_poses: int
    rejected_images: int


def solve_chessboard_with_report(
    images: list[np.ndarray],
    *,
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> ChessboardCalibrationResult:
    if not images:
        raise CalibrationError("no calibration images")
    sizes = []
    for image in images:
        if image.ndim not in (2, 3):
            raise CalibrationError("calibration images must be gray or BGR arrays")
        sizes.append((int(image.shape[1]), int(image.shape[0])))
    if len(set(sizes)) != 1:
        raise CalibrationError("all calibration images must have the same size")
    image_size = sizes[0]
    object_template = Calibration.chessboard_object_points(pattern_size, square_size_mm)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found or corners is None:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
    usable = len(image_points)
    if usable < 10:
        raise CalibrationError(f"need at least 10 usable chessboard poses, got {usable}")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    value = Calibration(image_size, matrix, distortion, float(rms))
    return ChessboardCalibrationResult(value, len(images), usable, len(images) - usable)
```

Keep `solve_chessboard()` as a compatibility wrapper returning `.calibration`. Change `tools/calibrate_camera.py` to `main(argv: Sequence[str] | None = None)`, call the report function, validate RMS, save, and print:

```python
print(
    f"images={report.input_images} usable_poses={report.usable_poses} "
    f"rejected={report.rejected_images} rms_px={report.calibration.rms_px:.4f} "
    f"output={args.output}"
)
```

- [ ] **Step 4: Correct the example and ignore the real calibration**

Replace the example with:

```yaml
schema_version: 1
image_size: [1280, 1024]
camera_matrix:
  - [900.0, 0.0, 640.0]
  - [0.0, 900.0, 512.0]
  - [0.0, 0.0, 1.0]
distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
rms_px: 0.5
```

Append `config/camera_calibration.yaml` to `.gitignore`. The example is schema-only; the Jetson-generated file is lens/camera-instance-specific.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/test_calibration.py tests/tools/test_calibrate_camera.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add .gitignore config/camera_calibration.example.yaml src/ev_vision/calibration.py tools/calibrate_camera.py tests/test_calibration.py tests/tools/test_calibrate_camera.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: improve camera calibration reporting'
```

Expected: focused tests pass and one commit is created.

## Task 2: Analyze and save calibration-capture frames

**Files:**
- Create: `src/ev_vision/calibration_capture/__init__.py`
- Create: `src/ev_vision/calibration_capture/local.py`
- Create: `tests/calibration_capture/__init__.py`
- Create: `tests/calibration_capture/test_local.py`

- [ ] **Step 1: Write failing analysis and storage tests**

Create tests for this public contract:

```python
from ev_vision.calibration_capture.local import (
    analyze_chessboard_frame,
    poses_are_similar,
    save_original_frame,
    remove_last_saved,
)


def complete_corners() -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(280, 1000, 8), np.linspace(260, 760, 5))
    return np.stack((xs, ys), axis=-1).reshape(-1, 1, 2).astype(np.float32)


def test_complete_sharp_centered_board_is_saveable(monkeypatch) -> None:
    monkeypatch.setattr(module.cv2, "findChessboardCorners", lambda *a, **k: (True, complete_corners()))
    monkeypatch.setattr(module.cv2, "cornerSubPix", lambda gray, corners, *a: corners)
    monkeypatch.setattr(module.cv2, "Laplacian", lambda *a, **k: np.array([0.0, 100.0]))
    result = analyze_chessboard_frame(
        np.zeros((1024, 1280, 3), np.uint8), pattern_size=(8, 5)
    )
    assert result.found and result.edge_margin_ok and result.save_allowed
    assert result.coverage_fraction > 0.02 and result.pose_signature is not None


def test_pose_similarity_uses_center_area_and_rotation() -> None:
    assert poses_are_similar((.50, .50, .20, 5.0), (.54, .53, .22, 10.0))
    assert not poses_are_similar((.50, .50, .20, 5.0), (.80, .50, .20, 5.0))


def test_saved_image_keeps_original_size_and_remove_is_session_scoped(tmp_path) -> None:
    path = save_original_frame(
        tmp_path, np.zeros((1024, 1280, 3), np.uint8), index=1
    )
    assert cv2.imread(str(path)).shape[:2] == (1024, 1280)
    assert remove_last_saved([path]) == path
    assert not path.exists()
```

Also test missing corners, focus below 40, coverage below 2%, and a corner within 3% of an image edge all produce `save_allowed=False` with a concrete `reason`.

- [ ] **Step 2: Run and confirm import failure**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/calibration_capture/test_local.py
```

Expected: FAIL because the package is absent.

- [ ] **Step 3: Implement deterministic analysis and overlay helpers**

Define in `local.py`:

```python
@dataclass(frozen=True)
class ChessboardFrameAnalysis:
    found: bool
    corners: np.ndarray | None
    focus_score: float
    coverage_fraction: float
    edge_margin_ok: bool
    pose_signature: tuple[float, float, float, float] | None
    save_allowed: bool
    reason: str


def analyze_chessboard_frame(
    image: np.ndarray,
    *,
    pattern_size: tuple[int, int],
    min_focus_score: float = 40.0,
    min_coverage_fraction: float = 0.02,
    edge_margin_fraction: float = 0.03,
) -> ChessboardFrameAnalysis:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image must be grayscale or BGR")
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if not found or corners is None or len(corners) != pattern_size[0] * pattern_size[1]:
        return ChessboardFrameAnalysis(False, None, focus, 0.0, False, None, False, "corners_not_found")
    refined = cv2.cornerSubPix(
        gray,
        np.asarray(corners, np.float32),
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    ).reshape(-1, 2)
    x_min, y_min = refined.min(axis=0)
    x_max, y_max = refined.max(axis=0)
    height, width = gray.shape
    coverage = float(max(0.0, x_max - x_min) * max(0.0, y_max - y_min) / (width * height))
    margin_x = width * edge_margin_fraction
    margin_y = height * edge_margin_fraction
    edge_ok = bool(x_min >= margin_x and x_max < width - margin_x and y_min >= margin_y and y_max < height - margin_y)
    top_left = refined[0]
    top_right = refined[pattern_size[0] - 1]
    angle = math.degrees(math.atan2(float(top_right[1] - top_left[1]), float(top_right[0] - top_left[0])))
    signature = (
        float((x_min + x_max) / (2.0 * width)),
        float((y_min + y_max) / (2.0 * height)),
        coverage,
        angle,
    )
    reason = "ready"
    if focus < min_focus_score:
        reason = "focus_too_low"
    elif coverage < min_coverage_fraction:
        reason = "coverage_too_small"
    elif not edge_ok:
        reason = "corners_too_close_to_edge"
    return ChessboardFrameAnalysis(
        True,
        refined,
        focus,
        coverage,
        edge_ok,
        signature,
        reason == "ready",
        reason,
    )


def poses_are_similar(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    *,
    center_tolerance: float = 0.08,
    area_ratio_tolerance: float = 0.25,
    angle_tolerance_deg: float = 12.0,
) -> bool:
    first_area = max(first[2], 1e-9)
    area_ratio_delta = abs(second[2] / first_area - 1.0)
    angle_delta = abs((second[3] - first[3] + 180.0) % 360.0 - 180.0)
    return (
        math.hypot(second[0] - first[0], second[1] - first[1]) <= center_tolerance
        and area_ratio_delta <= area_ratio_tolerance
        and angle_delta <= angle_tolerance_deg
    )


def calibration_image_path(output: Path, index: int) -> Path:
    return output / f"calibration-{index:03d}.png"


def save_original_frame(output: Path, image: np.ndarray, *, index: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = calibration_image_path(output, index)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write calibration image: {path}")
    return path


def remove_last_saved(session_paths: list[Path]) -> Path | None:
    if not session_paths:
        return None
    path = session_paths.pop()
    path.unlink(missing_ok=False)
    return path


def render_capture_overlay(
    image: np.ndarray,
    analysis: ChessboardFrameAnalysis,
    saved_count: int,
    duplicate_warning: bool,
) -> np.ndarray:
    rendered = image.copy()
    if analysis.corners is not None:
        cv2.drawChessboardCorners(rendered, (8, 5), analysis.corners.reshape(-1, 1, 2), analysis.found)
    color = (0, 220, 0) if analysis.save_allowed else (0, 0, 255)
    lines = [
        f"saved={saved_count} focus={analysis.focus_score:.1f} coverage={analysis.coverage_fraction:.3f}",
        f"status={analysis.reason}",
        "SPACE save | R remove last session image | Q/ESC quit",
    ]
    if duplicate_warning:
        lines.append("warning: pose is similar to an image already saved")
    for row, line in enumerate(lines, start=1):
        cv2.putText(rendered, line, (16, 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return rendered
```

Use refined complete corners. Focus is `variance(Laplacian(gray, CV_64F))`; coverage is corner bounding-box area/full image area; pose signature is normalized box center, coverage, and top-row angle. Similar pose is a warning only. `save_original_frame()` creates the output directory, writes the untouched original array, checks `cv2.imwrite()` success, and returns the path. R may remove only a path held in the current process list.

- [ ] **Step 4: Run and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/calibration_capture/test_local.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add src/ev_vision/calibration_capture tests/calibration_capture
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: add chessboard capture analysis'
```

Expected: all new tests pass.

## Task 3: Add the Jetson-local calibration capture command

**Files:**
- Create: `src/ev_vision/calibration_capture/cli.py`
- Create: `tools/camera_calibration_capture.py`
- Create: `tests/calibration_capture/test_cli.py`
- Modify: `pyproject.toml:19-23`

- [ ] **Step 1: Write failing parser and lifecycle tests**

```python
def test_parser_uses_approved_board_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config == Path("config/jetson-local.yaml")
    assert args.serial == "00G02809155"
    assert (args.columns, args.rows, args.square_mm) == (8, 5, 22.0)
    assert args.output == Path("artifacts/calibration/images")


def test_capture_session_closes_camera_and_window() -> None:
    camera = FakeCamera([frame(1)])
    cv = FakeCv(keys=[ord("q")])
    assert run_capture_session(
        camera,
        output=Path("images"),
        pattern_size=(8, 5),
        square_size_mm=22.0,
        max_width=960,
        cv=cv,
    ) == 0
    assert camera.open_calls == 1 and camera.close_calls == 1
    assert cv.destroyed == ["EV Vision - camera calibration capture"]


def test_linux_cli_rejects_missing_display(monkeypatch, capsys) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert main([]) == 2
    assert "DISPLAY is not set" in capsys.readouterr().err
```

Also assert `pyproject.toml` contains:

```toml
ev-camera-calibration-capture = "ev_vision.calibration_capture.cli:main"
```

- [ ] **Step 2: Run and confirm the CLI is absent**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/calibration_capture/test_cli.py
```

Expected: FAIL on missing CLI symbols and script entry.

- [ ] **Step 3: Implement the camera-only capture lifecycle**

Expose:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ev-camera-calibration-capture")
    parser.add_argument("--config", type=Path, default=Path("config/jetson-local.yaml"))
    parser.add_argument("--serial", default="00G02809155")
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--square-mm", type=float, default=22.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration/images"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--timeout-ms", type=int, default=100)
    return parser


def build_camera(config_path: Path, serial: str) -> HikrobotCamera:
    config = load_config(config_path)
    require_fixed_format(config.camera)
    return HikrobotCamera(create_native_api(), config.camera, serial_number=serial)


def run_capture_session(
    camera: HikrobotCamera,
    *,
    output: Path,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    max_width: int,
    read_timeout_ms: int = 100,
    cv: Any = cv2,
) -> int:
    del square_size_mm
    saved_paths: list[Path] = []
    signatures: list[tuple[float, float, float, float]] = []
    window = "EV Vision - camera calibration capture"
    camera.open()
    try:
        while True:
            frame = camera.read(timeout_ms=read_timeout_ms)
            analysis = analyze_chessboard_frame(frame.image, pattern_size=pattern_size)
            duplicate = bool(
                analysis.pose_signature is not None
                and any(poses_are_similar(analysis.pose_signature, previous) for previous in signatures)
            )
            rendered = render_capture_overlay(frame.image, analysis, len(saved_paths), duplicate)
            scale = min(1.0, max_width / rendered.shape[1])
            if scale < 1.0:
                rendered = cv.resize(rendered, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            cv.imshow(window, rendered)
            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                return 0
            if key == ord(" ") and analysis.save_allowed:
                path = save_original_frame(output, frame.image, index=len(saved_paths) + 1)
                saved_paths.append(path)
                signatures.append(analysis.pose_signature)
            elif key in (ord("r"), ord("R")) and saved_paths:
                remove_last_saved(saved_paths)
                signatures.pop()
    finally:
        camera.close()
        cv.destroyWindow(window)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("calibration capture failed: DISPLAY is not set", file=sys.stderr)
        return 2
    if args.columns <= 1 or args.rows <= 1 or args.square_mm <= 0 or args.width <= 0 or args.timeout_ms <= 0:
        print("calibration capture failed: board and runtime values must be positive", file=sys.stderr)
        return 2
    try:
        camera = build_camera(args.config, args.serial)
        config = camera.config
        print(f"camera exposure_us={config.exposure_us} gain_db={config.gain_db} acquisition_fps={config.acquisition_fps}")
        print("Capture 20-25 images and keep at least 15 usable center/corner/near/far/rotated/tilted poses.")
        print("SAFETY: physically disconnect or reliably cover the hardware-always-on 405 nm laser.")
        return run_capture_session(
            camera,
            output=args.output,
            pattern_size=(args.columns, args.rows),
            square_size_mm=args.square_mm,
            max_width=args.width,
            read_timeout_ms=args.timeout_ms,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"calibration capture failed: {exc}", file=sys.stderr)
        return 3
```

Behavior: require `DISPLAY` on Linux; open once and close in `finally`; analyze original frames; resize only rendered preview; Space saves only `save_allowed` frames; R removes only the session's last image; Q/Esc/window close exits; similar pose produces a warning but remains manually saveable. Startup prints the loaded exposure/gain/FPS, recommends 20–25 captures with at least 15 usable across center/corners/near/far/rotation/tilt, and warns that the laser must be physically disconnected or covered.

Create `tools/camera_calibration_capture.py`:

```python
from ev_vision.calibration_capture.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/calibration_capture
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add pyproject.toml src/ev_vision/calibration_capture tools/camera_calibration_capture.py tests/calibration_capture
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: add Jetson calibration capture command'
```

Expected: all capture tests pass.

## Task 4: Add strict gimbal configuration and Jetson profiles

**Files:**
- Create: `src/ev_vision/gimbal_usb/__init__.py`
- Create: `src/ev_vision/gimbal_usb/config.py`
- Create: `tests/gimbal_usb/__init__.py`
- Create: `tests/gimbal_usb/test_config.py`
- Create: `config/gimbal_usb.yaml`
- Create: `config/jetson-local.yaml`

- [ ] **Step 1: Write failing configuration tests**

```python
from ev_vision.gimbal_usb.config import GimbalUsbConfig, GimbalUsbConfigError, load_gimbal_usb_config


def test_loads_approved_gimbal_values(tmp_path) -> None:
    path = tmp_path / "gimbal.yaml"
    path.write_text("""
port: /dev/serial/by-id/test
baudrate: 115200
output_hz: 50
reconnect_interval_s: 1
calibration_path: config/camera_calibration.yaml
max_calibration_rms_px: 0.5
max_result_age_ms: 60
predicted_control_max_frames: 3
predicted_control_max_age_ms: 60
predicted_max_angle_step_deg: 1.5
yaw_sign: 1
pitch_sign: 1
""", encoding="utf-8")
    value = load_gimbal_usb_config(path)
    assert value.port == "/dev/serial/by-id/test"
    assert value.output_hz == 50.0 and value.predicted_control_max_frames == 3


@pytest.mark.parametrize("field,value", [
    ("yaw_sign", 0), ("pitch_sign", 2), ("output_hz", 0),
    ("max_result_age_ms", float("nan")),
])
def test_rejects_unsafe_values(field, value) -> None:
    values = dataclasses.asdict(GimbalUsbConfig(port="/dev/test"))
    values[field] = value
    with pytest.raises(GimbalUsbConfigError):
        GimbalUsbConfig(**values).validate()


def test_unknown_yaml_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "gimbal.yaml"
    path.write_text("port: /dev/test\nextra: unsafe\n", encoding="utf-8")
    with pytest.raises(GimbalUsbConfigError, match="unknown"):
        load_gimbal_usb_config(path)
```

- [ ] **Step 2: Run and confirm the package is absent**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_config.py
```

Expected: FAIL on missing config module.

- [ ] **Step 3: Implement strict separate configuration**

```python
class GimbalUsbConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GimbalUsbConfig:
    port: str
    baudrate: int = 115200
    output_hz: float = 50.0
    reconnect_interval_s: float = 1.0
    calibration_path: Path = Path("config/camera_calibration.yaml")
    max_calibration_rms_px: float = 0.5
    max_result_age_ms: float = 60.0
    predicted_control_max_frames: int = 3
    predicted_control_max_age_ms: float = 60.0
    predicted_max_angle_step_deg: float = 1.5
    yaw_sign: int = 1
    pitch_sign: int = 1

    def validate(self) -> None:
        if not isinstance(self.port, str) or not self.port.strip():
            raise GimbalUsbConfigError("port must be a non-empty string")
        integer_fields = {"baudrate": self.baudrate, "predicted_control_max_frames": self.predicted_control_max_frames}
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise GimbalUsbConfigError(f"{name} must be an integer")
        if self.baudrate <= 0 or self.predicted_control_max_frames < 1:
            raise GimbalUsbConfigError("baudrate must be positive and prediction frames must be at least one")
        for name in (
            "output_hz", "reconnect_interval_s", "max_calibration_rms_px",
            "max_result_age_ms", "predicted_control_max_age_ms", "predicted_max_angle_step_deg",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise GimbalUsbConfigError(f"{name} must be positive and finite")
        if self.yaw_sign not in (-1, 1) or self.pitch_sign not in (-1, 1):
            raise GimbalUsbConfigError("yaw_sign and pitch_sign must be -1 or 1")


def load_gimbal_usb_config(path: str | Path) -> GimbalUsbConfig:
    source = Path(path)
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise GimbalUsbConfigError("gimbal USB YAML must contain a mapping")
    allowed = {field.name for field in dataclasses.fields(GimbalUsbConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise GimbalUsbConfigError(f"unknown gimbal USB config keys: {', '.join(unknown)}")
    if "calibration_path" in values:
        values["calibration_path"] = Path(values["calibration_path"])
    try:
        config = GimbalUsbConfig(**values)
    except TypeError as exc:
        raise GimbalUsbConfigError(str(exc)) from exc
    config.validate()
    return config
```

Reject blank port, booleans used as integers, non-finite/non-positive rates/timeouts, prediction frames below one, signs outside `{-1, 1}`, and unknown YAML keys. Keep this separate from strict `AppConfig` because the CLI uses `--config` and `--gimbal-config` independently.

- [ ] **Step 4: Add exact deployment YAML files**

`config/gimbal_usb.yaml`:

```yaml
port: /dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00
baudrate: 115200
output_hz: 50.0
reconnect_interval_s: 1.0
calibration_path: config/camera_calibration.yaml
max_calibration_rms_px: 0.5
max_result_age_ms: 60.0
predicted_control_max_frames: 3
predicted_control_max_age_ms: 60.0
predicted_max_angle_step_deg: 1.5
yaw_sign: 1
pitch_sign: 1
```

Create `config/jetson-local.yaml` by copying all of `config/default.yaml`, changing only camera acquisition to 50, exposure to 15000, and gain to 14.0. Keep the complete classical detection mapping and keep its internal `predict_max_ms: 150.0`; the USB gate independently limits control prediction to 60 ms.

- [ ] **Step 5: Run and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_config.py tests/test_config.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add config/gimbal_usb.yaml config/jetson-local.yaml src/ev_vision/gimbal_usb tests/gimbal_usb
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: add Jetson gimbal runtime configuration'
```

## Task 5: Encode the independent A5 5A protocol

**Files:**
- Create: `src/ev_vision/gimbal_usb/protocol.py`
- Create: `tests/gimbal_usb/test_protocol.py`

- [ ] **Step 1: Write failing CRC and layout tests**

```python
import struct
from ev_vision.gimbal_usb.protocol import GimbalTargetCommand, crc16_modbus, encode_target_frame


def test_crc16_modbus_known_vector() -> None:
    assert crc16_modbus(b"123456789") == 0x4B37


def test_valid_target_frame_has_exact_26_byte_layout() -> None:
    frame = encode_target_frame(
        GimbalTargetCommand(yaw_deg=1.25, pitch_deg=-2.5, tracking=True),
        sequence=0x1234,
    )
    unpacked = struct.unpack("<2sBBBBHfffBBBBH", frame)
    assert len(frame) == 26
    assert unpacked[:6] == (b"\xA5\x5A", 1, 0x01, 16, 0, 0x1234)
    assert unpacked[6] == pytest.approx(1.25)
    assert unpacked[7] == pytest.approx(-2.5)
    assert unpacked[8:13] == (0.0, 1, 0, 0, 0)
    assert unpacked[13] == crc16_modbus(frame[:-2])


def test_invalid_command_forces_zero_payload() -> None:
    frame = encode_target_frame(
        GimbalTargetCommand(yaw_deg=99.0, pitch_deg=-88.0, tracking=False), 7
    )
    assert struct.unpack("<2sBBBBHfffBBBBH", frame)[6:13] == (
        0.0, 0.0, 0.0, 0, 0, 0, 0
    )


def test_sequence_is_uint16() -> None:
    assert struct.unpack("<H", encode_target_frame(GimbalTargetCommand.safe(), 65535)[6:8])[0] == 65535
    with pytest.raises(ValueError):
        encode_target_frame(GimbalTargetCommand.safe(), 65536)
```

- [ ] **Step 2: Run and confirm imports fail**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_protocol.py
```

Expected: FAIL because the protocol module is absent.

- [ ] **Step 3: Implement constants, command, CRC, and encoder**

```python
MAGIC = b"\xA5\x5A"
VERSION = 1
MESSAGE_TYPE_TARGET = 0x01
PAYLOAD_LENGTH = 16
FLAGS = 0
_HEADER = struct.Struct("<2sBBBBH")
_PAYLOAD = struct.Struct("<fffBBBB")
_CRC = struct.Struct("<H")


@dataclass(frozen=True)
class GimbalTargetCommand:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    tracking: bool = False

    @classmethod
    def safe(cls) -> "GimbalTargetCommand":
        return cls()


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def encode_target_frame(command: GimbalTargetCommand, sequence: int) -> bytes:
    if isinstance(sequence, bool) or not 0 <= sequence <= 0xFFFF:
        raise ValueError("sequence must be a uint16")
    tracking = bool(command.tracking)
    yaw = float(command.yaw_deg) if tracking else 0.0
    pitch = float(command.pitch_deg) if tracking else 0.0
    if tracking and (not math.isfinite(yaw) or not math.isfinite(pitch)):
        raise ValueError("valid target angles must be finite")
    body = _HEADER.pack(MAGIC, VERSION, MESSAGE_TYPE_TARGET, PAYLOAD_LENGTH, FLAGS, sequence)
    body += _PAYLOAD.pack(yaw, pitch, 0.0, int(tracking), 0, 0, 0)
    return body + _CRC.pack(crc16_modbus(body))
```

Do not import or modify `src/ev_vision/protocol.py`.

- [ ] **Step 4: Run new and legacy protocol tests, then commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_protocol.py tests/test_protocol.py tests/test_protocol_mapping.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add src/ev_vision/gimbal_usb/protocol.py tests/gimbal_usb/test_protocol.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: encode gimbal USB target frames'
```

Expected: all new and legacy protocol tests pass.

## Task 6: Convert target pixels to calibrated angles

**Files:**
- Create: `src/ev_vision/gimbal_usb/angles.py`
- Create: `tests/gimbal_usb/test_angles.py`

- [ ] **Step 1: Write failing validation and sign tests**

```python
from ev_vision.gimbal_usb.angles import TargetAngleConverter


def test_raw_signs_follow_image_axes(monkeypatch) -> None:
    converter = TargetAngleConverter(calibration(), yaw_sign=1, pitch_sign=1)
    monkeypatch.setattr(
        module.cv2, "undistortPoints",
        lambda points, matrix, distortion: np.array([[[0.1, -0.2]]], np.float64),
    )
    yaw, pitch = converter.convert((700.0, 400.0))
    assert yaw == pytest.approx(math.degrees(math.atan(0.1)))
    assert pitch == pytest.approx(math.degrees(math.atan(-0.2)))


def test_axis_signs_apply_after_raw_angle(monkeypatch) -> None:
    converter = TargetAngleConverter(calibration(), yaw_sign=-1, pitch_sign=-1)
    monkeypatch.setattr(module.cv2, "undistortPoints", lambda *a: np.array([[[0.1, 0.2]]]))
    yaw, pitch = converter.convert((700.0, 600.0))
    assert yaw < 0 and pitch < 0


@pytest.mark.parametrize("center", [None, (float("nan"), 1.0), (-1.0, 20.0), (1280.0, 20.0)])
def test_invalid_center_is_rejected(center) -> None:
    with pytest.raises(CalibrationError):
        TargetAngleConverter(calibration(), 1, 1).convert(center)


def test_from_file_rejects_wrong_size_or_rms(tmp_path) -> None:
    path = tmp_path / "camera.yaml"
    save_calibration(path, calibration(rms=0.7))
    with pytest.raises(CalibrationError, match="RMS"):
        TargetAngleConverter.from_file(
            path, image_size=(1280, 1024), max_rms_px=0.5, yaw_sign=1, pitch_sign=1
        )
```

- [ ] **Step 2: Run and confirm converter is missing**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_angles.py
```

Expected: FAIL on missing `TargetAngleConverter`.

- [ ] **Step 3: Implement single-point undistortion only**

```python
class AngleConverter(Protocol):
    def convert(self, center_px: tuple[float, float] | None) -> tuple[float, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class UnavailableTargetAngleConverter:
    reason: str

    def convert(self, center_px: tuple[float, float] | None) -> tuple[float, float]:
        del center_px
        raise CalibrationError(self.reason)


@dataclass(frozen=True)
class TargetAngleConverter:
    calibration: Calibration
    yaw_sign: int = 1
    pitch_sign: int = 1

    def __post_init__(self) -> None:
        self.calibration.validate()
        if self.yaw_sign not in (-1, 1) or self.pitch_sign not in (-1, 1):
            raise CalibrationError("yaw_sign and pitch_sign must be -1 or 1")

    @classmethod
    def from_file(cls, path, *, image_size, max_rms_px, yaw_sign, pitch_sign):
        calibration = load_calibration(path)
        calibration.validate(max_rms_px=max_rms_px)
        if calibration.image_size != image_size:
            raise CalibrationError(
                f"calibration image size {calibration.image_size} does not match runtime {image_size}"
            )
        return cls(calibration, yaw_sign, pitch_sign)

    def convert(self, center_px: tuple[float, float] | None) -> tuple[float, float]:
        if center_px is None or len(center_px) != 2:
            raise CalibrationError("target center is unavailable")
        x, y = map(float, center_px)
        width, height = self.calibration.image_size
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height):
            raise CalibrationError("target center is invalid or outside calibrated image")
        x_n, y_n = cv2.undistortPoints(
            np.array([[[x, y]]], np.float64),
            self.calibration.camera_matrix,
            self.calibration.distortion,
        ).reshape(2)
        yaw = self.yaw_sign * math.degrees(math.atan(float(x_n)))
        pitch = self.pitch_sign * math.degrees(math.atan(float(y_n)))
        if not math.isfinite(yaw) or not math.isfinite(pitch):
            raise CalibrationError("calibrated target angle is not finite")
        return yaw, pitch
```

Do not call `Calibration.maps()`, `Calibration.undistort()`, `cv2.remap()`, or remap the full frame.

- [ ] **Step 4: Run and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_angles.py tests/test_calibration.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add src/ev_vision/gimbal_usb/angles.py tests/gimbal_usb/test_angles.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: convert target pixels to calibrated angles'
```

## Task 7: Gate real and short-predicted observations

**Files:**
- Create: `src/ev_vision/gimbal_usb/gate.py`
- Create: `tests/gimbal_usb/test_gate.py`

- [ ] **Step 1: Write failing valid-source and fail-closed tests**

```python
REAL_SOURCES = ["FULL_BOARD", "CONCENTRIC_ARCS", "SINGLE_ARC", "WHITE_REGION", "FUSED_PARTIAL"]


@pytest.mark.parametrize("source", REAL_SOURCES)
def test_fresh_real_source_is_immediately_valid(source) -> None:
    gate = GimbalControlGate(FakeConverter((1.0, -2.0)), limits())
    decision = gate.evaluate(snapshot(observation_source=source, tracking_state="TRACKING"))
    assert decision.command == GimbalTargetCommand(1.0, -2.0, True)
    assert decision.reason == "real_observation"


@pytest.mark.parametrize("changes,reason", [
    ({"target_valid": False}, "target_invalid"),
    ({"tracking_state": "SEARCHING"}, "tracking_state"),
    ({"result_age_ms": 60.1}, "result_stale"),
    ({"center_px": None}, "center_invalid"),
    ({"failure_reason": "position_jump"}, "upstream_failure"),
    ({"error": "detector stopped"}, "detection_error"),
])
def test_invalid_observations_are_all_zero(changes, reason) -> None:
    decision = GimbalControlGate(FakeConverter((1.0, 2.0)), limits()).evaluate(snapshot(**changes))
    assert decision.command == GimbalTargetCommand.safe()
    assert decision.reason == reason


def test_predictions_one_to_three_are_valid_within_limits() -> None:
    gate = GimbalControlGate(
        SequenceConverter([(1.0, 1.0), (1.5, 1.4), (2.0, 1.8), (2.5, 2.2)]), limits()
    )
    assert gate.evaluate(snapshot()).command.tracking
    for frame in range(1, 4):
        assert gate.evaluate(snapshot(
            observation_source="PREDICTED", tracking_state="PREDICTING",
            predicted_frames=frame, source_age_us=frame * 20_000,
        )).command.tracking


def test_fourth_prediction_age_over_60ms_and_angle_jump_fail_closed() -> None:
    gate = GimbalControlGate(SequenceConverter([(0.0, 0.0), (2.0, 0.0)]), limits())
    gate.evaluate(snapshot())
    jump = gate.evaluate(snapshot(
        observation_source="PREDICTED", tracking_state="PREDICTING",
        predicted_frames=1, source_age_us=20_000,
    ))
    assert jump.reason == "prediction_angle_step" and not jump.command.tracking


def test_rejected_prediction_latches_until_real_observation() -> None:
    gate = GimbalControlGate(
        SequenceConverter([(0.0, 0.0), (3.0, 0.0), (0.5, 0.0), (20.0, 10.0)]), limits()
    )
    gate.evaluate(snapshot())
    assert not gate.evaluate(prediction()).command.tracking
    assert gate.evaluate(prediction(predicted_frames=2)).reason == "prediction_latched_invalid"
    recovered = gate.evaluate(snapshot(center_px=(700.0, 500.0)))
    assert recovered.command.tracking and recovered.reason == "real_observation"
```

Add explicit tests for prediction frame 4, `source_age_us=60001`, stale result, unknown source, out-of-image center, missing prior real observation, and converter exceptions.

- [ ] **Step 2: Run and confirm the gate is missing**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_gate.py
```

Expected: FAIL on missing gate module.

- [ ] **Step 3: Implement explicit state and decisions**

```python
REAL_OBSERVATION_SOURCES = frozenset({
    "FULL_BOARD", "CONCENTRIC_ARCS", "SINGLE_ARC", "WHITE_REGION", "FUSED_PARTIAL",
})


@dataclass(frozen=True)
class GateLimits:
    max_result_age_ms: float = 60.0
    predicted_control_max_frames: int = 3
    predicted_control_max_age_us: int = 60_000
    predicted_max_angle_step_deg: float = 1.5


@dataclass(frozen=True)
class GateDecision:
    command: GimbalTargetCommand
    reason: str
    observation_source: str
    result_age_ms: float | None
    source_age_us: int


class GimbalControlGate:
    def __init__(self, converter: AngleConverter, limits: GateLimits) -> None:
        self._converter = converter
        self._limits = limits
        self._previous_valid: GimbalTargetCommand | None = None
        self._prediction_latched_invalid = False

    def _reject(self, reason: str, detection: DetectionSnapshot | None) -> GateDecision:
        source = "NONE" if detection is None else str(detection.observation_source)
        age = None if detection is None else detection.result_age_ms
        source_age = 0 if detection is None else int(detection.source_age_us)
        return GateDecision(GimbalTargetCommand.safe(), reason, source, age, source_age)

    def evaluate(self, detection: DetectionSnapshot | None) -> GateDecision:
        if detection is None:
            return self._reject("detection_missing", detection)
        if not detection.enabled:
            return self._reject("detection_disabled", detection)
        if detection.error:
            return self._reject("detection_error", detection)
        if not detection.target_valid:
            return self._reject("target_invalid", detection)
        if detection.failure_reason:
            return self._reject("upstream_failure", detection)
        result_age = detection.result_age_ms
        if result_age is None or not math.isfinite(result_age) or result_age < 0 or result_age > self._limits.max_result_age_ms:
            return self._reject("result_stale", detection)
        center = detection.center_px
        if center is None or len(center) != 2 or any(not math.isfinite(float(value)) for value in center):
            return self._reject("center_invalid", detection)
        try:
            yaw, pitch = self._converter.convert(center)
        except Exception:
            return self._reject("angle_conversion_failed", detection)
        command = GimbalTargetCommand(yaw, pitch, True)
        source = str(detection.observation_source)
        if source in REAL_OBSERVATION_SOURCES:
            if detection.tracking_state != "TRACKING":
                return self._reject("tracking_state", detection)
            self._previous_valid = command
            self._prediction_latched_invalid = False
            return GateDecision(command, "real_observation", source, result_age, int(detection.source_age_us))
        if source != "PREDICTED":
            return self._reject("observation_source", detection)
        if self._prediction_latched_invalid:
            return self._reject("prediction_latched_invalid", detection)
        if detection.tracking_state != "PREDICTING":
            self._prediction_latched_invalid = True
            return self._reject("tracking_state", detection)
        if not 1 <= detection.predicted_frames <= self._limits.predicted_control_max_frames:
            self._prediction_latched_invalid = True
            return self._reject("prediction_frame_limit", detection)
        if not 0 <= detection.source_age_us <= self._limits.predicted_control_max_age_us:
            self._prediction_latched_invalid = True
            return self._reject("prediction_age_limit", detection)
        if self._previous_valid is None:
            self._prediction_latched_invalid = True
            return self._reject("prediction_without_real", detection)
        if (
            abs(command.yaw_deg - self._previous_valid.yaw_deg) > self._limits.predicted_max_angle_step_deg
            or abs(command.pitch_deg - self._previous_valid.pitch_deg) > self._limits.predicted_max_angle_step_deg
        ):
            self._prediction_latched_invalid = True
            return self._reject("prediction_angle_step", detection)
        self._previous_valid = command
        return GateDecision(command, "predicted_observation", source, result_age, int(detection.source_age_us))
```

Evaluation order: reject missing/disabled/error/invalid target; reject missing, negative, non-finite, or old `result_age_ms`; reject any `failure_reason`; reject invalid center before conversion. Real sources require `TRACKING`; success immediately updates the previous valid command and clears the prediction latch. `PREDICTED` requires `PREDICTING`, frames 1–3, age ≤60000 us, a prior valid real command, and both angular steps ≤1.5°. Any prediction failure latches later predictions invalid until a valid real observation. Unknown sources fail closed. Never clamp or retain non-zero angles in an invalid command. Do not compare recovered real data with predictions.

- [ ] **Step 4: Run and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_gate.py tests/tuning/test_models.py tests/tuning/test_service.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add src/ev_vision/gimbal_usb/gate.py tests/gimbal_usb/test_gate.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: gate gimbal commands against stale predictions'
```

## Task 8: Reconnect and send safely over CDC-ACM

**Files:**
- Create: `src/ev_vision/gimbal_usb/transport.py`
- Create: `tests/gimbal_usb/test_transport.py`

- [ ] **Step 1: Write failing transport tests**

Use fake serial handles, a sequence factory, and a fake monotonic clock:

```python
def test_first_frame_after_open_is_safe() -> None:
    serial = FakeSerial()
    transport = GimbalSerialTransport(
        "/dev/test", serial_factory=lambda **kwargs: serial, clock=FakeClock()
    )
    requested = GimbalTargetCommand(1.0, 2.0, True)
    assert transport.transmit(requested) is False
    assert decode(serial.writes[0]).tracking == 0
    assert transport.transmit(requested) is True
    assert decode(serial.writes[1]).tracking == 1


def test_open_failure_retries_after_one_second() -> None:
    factory = FailingThenWorkingFactory()
    clock = FakeClock(0.0)
    transport = GimbalSerialTransport(
        "/dev/test", reconnect_interval_s=1.0, serial_factory=factory, clock=clock
    )
    assert not transport.transmit(GimbalTargetCommand.safe())
    assert not transport.transmit(GimbalTargetCommand.safe())
    assert factory.calls == 1
    clock.advance(1.0)
    transport.transmit(GimbalTargetCommand.safe())
    assert factory.calls == 2


def test_short_write_disconnects_and_reconnect_starts_safe() -> None:
    first, second = FakeSerial(short_write=True), FakeSerial()
    transport = GimbalSerialTransport(
        "/dev/test", serial_factory=SequenceFactory([first, second]), clock=FakeClock()
    )
    transport.transmit(GimbalTargetCommand.safe())
    assert not transport.transmit(GimbalTargetCommand(1.0, 1.0, True))
    assert first.closed
    assert not transport.transmit(GimbalTargetCommand(1.0, 1.0, True))
    assert decode(second.writes[0]).tracking == 0


def test_sequence_wraps_and_survives_reconnect() -> None:
    transport = configured_transport(initial_sequence=0xFFFF)
    transport.transmit(GimbalTargetCommand.safe())
    transport.transmit(GimbalTargetCommand.safe())
    assert frame_sequence(transport.serial.writes[0]) == 0xFFFF
    assert frame_sequence(transport.serial.writes[1]) == 0


def test_close_attempts_five_safe_frames() -> None:
    transport = connected_transport()
    transport.close(safe_frames=5)
    assert [decode(frame).tracking for frame in transport.serial.writes[-5:]] == [0] * 5
    assert transport.serial.closed
```

- [ ] **Step 2: Run and confirm transport is missing**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_transport.py
```

Expected: FAIL on missing transport module.

- [ ] **Step 3: Implement lazy pyserial and bounded reconnects**

```python
@dataclass(frozen=True)
class SerialStats:
    connected: bool
    sent_frames: int
    connect_count: int
    reconnects: int
    open_errors: int
    write_errors: int
    sequence: int


class GimbalSerialTransport:
    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        reconnect_interval_s: float = 1.0,
        write_timeout_s: float = 0.2,
        serial_factory: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        initial_sequence: int = 0,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._reconnect_interval_s = reconnect_interval_s
        self._write_timeout_s = write_timeout_s
        self._serial_factory = serial_factory
        self._clock = clock
        self._serial = None
        self._next_retry_at = 0.0
        self._sequence = initial_sequence
        self._sent_frames = 0
        self._connect_count = 0
        self._reconnects = 0
        self._open_errors = 0
        self._write_errors = 0
        self._closed = False

    def _factory(self, **kwargs: Any) -> Any:
        if self._serial_factory is not None:
            return self._serial_factory(**kwargs)
        import serial
        return serial.Serial(
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            **kwargs,
        )

    def _write(self, command: GimbalTargetCommand) -> bool:
        frame = encode_target_frame(command, self._sequence)
        written = self._serial.write(frame)
        if written != len(frame):
            raise OSError(f"short serial write: {written}/{len(frame)}")
        self._sequence = (self._sequence + 1) & 0xFFFF
        self._sent_frames += 1
        return True

    def transmit(self, command: GimbalTargetCommand) -> bool:
        if self._closed:
            return False
        now = self._clock()
        if self._serial is None:
            if now < self._next_retry_at:
                return False
            try:
                previously_connected = self._connect_count > 0
                self._serial = self._factory(
                    port=self._port,
                    baudrate=self._baudrate,
                    timeout=0,
                    write_timeout=self._write_timeout_s,
                    xonxoff=False,
                    rtscts=False,
                    dsrdtr=False,
                )
                self._connect_count += 1
                self._reconnects += int(previously_connected)
                self._write(GimbalTargetCommand.safe())
                return False
            except Exception:
                self._open_errors += 1
                if self._serial is not None:
                    self._serial.close()
                self._serial = None
                self._next_retry_at = now + self._reconnect_interval_s
                return False
        try:
            return self._write(command)
        except Exception:
            self._write_errors += 1
            try:
                self._serial.close()
            finally:
                self._serial = None
                self._next_retry_at = now + self._reconnect_interval_s
            return False

    def snapshot(self) -> SerialStats:
        return SerialStats(
            self._serial is not None,
            self._sent_frames,
            self._connect_count,
            self._reconnects,
            self._open_errors,
            self._write_errors,
            self._sequence,
        )

    def close(self, *, safe_frames: int = 5) -> None:
        if self._closed:
            return
        self._closed = True
        serial_handle, self._serial = self._serial, None
        if serial_handle is None:
            return
        self._serial = serial_handle
        try:
            for _ in range(max(0, safe_frames)):
                try:
                    self._write(GimbalTargetCommand.safe())
                except Exception:
                    break
        finally:
            self._serial = None
            serial_handle.close()
```

The default factory imports `serial` only when opening and calls `serial.Serial(port=self._port, baudrate=self._baudrate, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0, write_timeout=self._write_timeout_s, xonxoff=False, rtscts=False, dsrdtr=False)`. On open, synchronously write one safe frame and return `False` for the requested command. Increment sequence only after a complete 26-byte write and wrap uint16. Short write/exception closes, counts an error, schedules retry, and returns false. Keep sequence across reconnect. `close()` is idempotent and best-effort sends five safe frames only when connected. Do not import pyserial at module import time.

- [ ] **Step 4: Run and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_transport.py tests/gimbal_usb/test_protocol.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add src/ev_vision/gimbal_usb/transport.py tests/gimbal_usb/test_transport.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: add reconnecting gimbal USB transport'
```

## Task 9: Run latest-only output at 50 Hz

**Files:**
- Create: `src/ev_vision/gimbal_usb/runtime.py`
- Create: `tests/gimbal_usb/test_runtime.py`

- [ ] **Step 1: Write failing cycle and scheduling tests**

```python
def test_cycle_reads_latest_once_and_transmits_decision() -> None:
    service = FakeService([snapshot()])
    gate = FakeGate(valid_decision())
    transport = FakeTransport([True])
    worker = GimbalOutputWorker(service, gate, transport, output_hz=50.0)
    worker.run_cycle()
    assert service.latest_detection_calls == 1
    assert transport.commands == [valid_decision().command]
    assert worker.snapshot().sent_valid == 1


def test_reconnect_deferred_command_is_not_counted_sent() -> None:
    worker = GimbalOutputWorker(
        FakeService([snapshot()]), FakeGate(valid_decision()), FakeTransport([False]), output_hz=50.0
    )
    worker.run_cycle()
    assert worker.snapshot().sent_valid == 0
    assert worker.snapshot().transport_deferred == 1


def test_scheduler_uses_deadlines_without_catchup_bursts() -> None:
    clock = FakeNanosecondClock()
    sleeper = AdvancingSleeper(clock)
    service = FakeService([snapshot(), snapshot(), snapshot(), snapshot()])
    worker = GimbalOutputWorker(
        service,
        FakeGate(valid_decision()),
        FakeTransport([True, True, True, True]),
        output_hz=50.0,
        clock_ns=clock,
        sleep=sleeper,
    )
    worker.run_for_test_cycles(4)
    assert sleeper.requested_delays == pytest.approx([0.02, 0.02, 0.02], abs=0.001)
    assert worker.service.latest_detection_calls == 4


def test_stop_closes_transport_with_five_safe_frames() -> None:
    worker = running_worker()
    worker.stop()
    assert worker.transport.close_calls == [5]
```

- [ ] **Step 2: Run and confirm worker is missing**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_runtime.py
```

Expected: FAIL on missing worker.

- [ ] **Step 3: Implement one output worker with bounded statistics**

```python
@dataclass(frozen=True)
class GimbalRuntimeSnapshot:
    running: bool
    cycles: int
    sent_valid: int
    sent_invalid: int
    transport_deferred: int
    real_observations: int
    predicted_observations: int
    reasons: Mapping[str, int]
    last_decision: GateDecision | None
    serial: SerialStats


class GimbalOutputWorker:
    def __init__(self, service, gate, transport, *, output_hz=50.0,
                 clock_ns=time.monotonic_ns, sleep=time.sleep) -> None:
        if not math.isfinite(output_hz) or output_hz <= 0:
            raise ValueError("output_hz must be positive and finite")
        self.service = service
        self.gate = gate
        self.transport = transport
        self._period_ns = round(1_000_000_000 / output_hz)
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cycles = self._sent_valid = self._sent_invalid = self._transport_deferred = 0
        self._real_observations = self._predicted_observations = 0
        self._reasons: Counter[str] = Counter()
        self._last_decision: GateDecision | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="gimbal-usb-output", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        deadline = self._clock_ns()
        while not self._stop_event.is_set():
            self.run_cycle()
            deadline += self._period_ns
            now = self._clock_ns()
            if now > deadline:
                deadline += ((now - deadline) // self._period_ns + 1) * self._period_ns
            delay_s = max(0.0, (deadline - self._clock_ns()) / 1_000_000_000)
            if delay_s:
                self._sleep(delay_s)

    def run_for_test_cycles(self, count: int) -> None:
        deadline = self._clock_ns()
        for index in range(count):
            self.run_cycle()
            if index + 1 == count:
                break
            deadline += self._period_ns
            now = self._clock_ns()
            if now > deadline:
                deadline += ((now - deadline) // self._period_ns + 1) * self._period_ns
            self._sleep(max(0.0, (deadline - self._clock_ns()) / 1_000_000_000))

    def run_cycle(self) -> None:
        detection = self.service.latest_detection()
        decision = self.gate.evaluate(detection)
        sent = self.transport.transmit(decision.command)
        self._cycles += 1
        self._last_decision = decision
        self._reasons[decision.reason] += 1
        self._real_observations += int(decision.reason == "real_observation")
        self._predicted_observations += int(decision.reason == "predicted_observation")
        if not sent:
            self._transport_deferred += 1
        elif decision.command.tracking:
            self._sent_valid += 1
        else:
            self._sent_invalid += 1

    def stop(self, *, safe_frames: int = 5) -> None:
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join()
        self.transport.close(safe_frames=safe_frames)

    def snapshot(self) -> GimbalRuntimeSnapshot:
        return GimbalRuntimeSnapshot(
            self._thread is not None and self._thread.is_alive(),
            self._cycles,
            self._sent_valid,
            self._sent_invalid,
            self._transport_deferred,
            self._real_observations,
            self._predicted_observations,
            dict(self._reasons),
            self._last_decision,
            self.transport.snapshot(),
        )
```

One daemon thread owns transport calls. Each tick calls `service.latest_detection()` exactly once, gates, and sends; no detection queue. Use accumulated monotonic nanosecond deadlines with period `round(1e9/output_hz)`. If late, skip whole expired periods rather than sending a burst. Count sent valid/invalid only when transport returns true; otherwise count deferred. Count reasons, real, and prediction decisions. Stop joins the thread and calls `transport.close(safe_frames=5)` once. Do not log every frame.

- [ ] **Step 4: Run and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_runtime.py tests/gimbal_usb/test_gate.py tests/gimbal_usb/test_transport.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add src/ev_vision/gimbal_usb/runtime.py tests/gimbal_usb/test_runtime.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: add 50 Hz gimbal output runtime'
```

## Task 10: Add the formal headless/preview CLI

**Files:**
- Create: `src/ev_vision/gimbal_usb/cli.py`
- Create: `tools/gimbal_vision.py`
- Create: `tests/gimbal_usb/test_cli.py`
- Modify: `pyproject.toml:19-25`

- [ ] **Step 1: Write failing parser and lifecycle tests**

```python
def test_parser_uses_runtime_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config == Path("config/jetson-local.yaml")
    assert args.gimbal_config == Path("config/gimbal_usb.yaml")
    assert args.serial == "00G02809155"
    assert not args.display
    assert (args.width, args.display_fps, args.detection_fps) == (640, 45.0, 50.0)


def test_only_display_mode_requires_display(monkeypatch, capsys) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert main(["--display"]) == 2
    assert "DISPLAY is not set" in capsys.readouterr().err


def test_missing_calibration_builds_safe_only_gate(tmp_path) -> None:
    config = replace(gimbal_config(), calibration_path=tmp_path / "missing.yaml")
    converter = build_angle_converter(config, image_size=(1280, 1024))
    assert isinstance(converter, UnavailableTargetAngleConverter)
    gate = GimbalControlGate(converter, limits())
    decision = gate.evaluate(snapshot())
    assert decision.command == GimbalTargetCommand.safe()
    assert decision.reason == "angle_conversion_failed"


def test_safe_only_worker_does_not_prevent_camera_start() -> None:
    events: list[str] = []
    assert run_application(
        FakeCameraRuntime(events),
        FakeWorker(events, command=GimbalTargetCommand.safe()),
        display=False,
        stop_event=AlreadySetEvent(),
    ) == 0
    assert events == ["service.start", "worker.start", "worker.stop:5", "service.stop"]


def test_cleanup_stops_worker_before_camera() -> None:
    events = []
    assert run_application(
        FakeCameraRuntime(events), FakeWorker(events), display=False,
        stop_event=AlreadySetEvent(),
    ) == 0
    assert events == ["service.start", "worker.start", "worker.stop:5", "service.stop"]
```

Assert `pyproject.toml` contains `ev-gimbal-vision = "ev_vision.gimbal_usb.cli:main"`.

- [ ] **Step 2: Run and confirm CLI is absent**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb/test_cli.py
```

Expected: FAIL on missing CLI.

- [ ] **Step 3: Implement assembly and lifecycle**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ev-gimbal-vision")
    parser.add_argument("--config", type=Path, default=Path("config/jetson-local.yaml"))
    parser.add_argument("--gimbal-config", type=Path, default=Path("config/gimbal_usb.yaml"))
    parser.add_argument("--serial", default="00G02809155")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--display-fps", type=float, default=45.0)
    parser.add_argument("--detection-fps", type=float, default=50.0)
    parser.add_argument("--timeout-ms", type=int, default=100)
    parser.add_argument("--shutdown-timeout", type=float, default=2.0)
    return parser


def build_angle_converter(
    config: GimbalUsbConfig,
    *,
    image_size: tuple[int, int],
) -> AngleConverter:
    try:
        return TargetAngleConverter.from_file(
            config.calibration_path,
            image_size=image_size,
            max_rms_px=config.max_calibration_rms_px,
            yaw_sign=config.yaw_sign,
            pitch_sign=config.pitch_sign,
        )
    except Exception as exc:
        return UnavailableTargetAngleConverter(str(exc))


def build_worker(camera_runtime: CameraRuntime, config: GimbalUsbConfig) -> GimbalOutputWorker:
    image_size = (camera_runtime.config.camera.width, camera_runtime.config.camera.height)
    converter = build_angle_converter(config, image_size=image_size)
    gate = GimbalControlGate(
        converter,
        GateLimits(
            max_result_age_ms=config.max_result_age_ms,
            predicted_control_max_frames=config.predicted_control_max_frames,
            predicted_control_max_age_us=round(config.predicted_control_max_age_ms * 1000),
            predicted_max_angle_step_deg=config.predicted_max_angle_step_deg,
        ),
    )
    transport = GimbalSerialTransport(
        config.port,
        baudrate=config.baudrate,
        reconnect_interval_s=config.reconnect_interval_s,
    )
    return GimbalOutputWorker(camera_runtime.service, gate, transport, output_hz=config.output_hz)


def run_application(camera_runtime, worker, *, display, width=640, display_fps=45.0,
                    stop_event: threading.Event | None = None) -> int:
    stop_event = stop_event or threading.Event()
    camera_runtime.service.start()
    worker.start()
    try:
        if display:
            run_local_preview(camera_runtime.service, max_width=width, display_fps=display_fps)
        else:
            while not stop_event.wait(1.0):
                print(format_runtime_snapshot(worker.snapshot()))
    finally:
        worker.stop(safe_frames=5)
        camera_runtime.service.stop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.display and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("gimbal vision startup failed: DISPLAY is not set", file=sys.stderr)
        return 2
    try:
        config = load_gimbal_usb_config(args.gimbal_config)
        camera_runtime = build_camera_runtime(
            config_path=args.config,
            serial=args.serial,
            read_timeout_ms=args.timeout_ms,
            detection_fps=args.detection_fps,
            diagnostics_fps=None,
            shutdown_timeout_s=args.shutdown_timeout,
        )
        worker = build_worker(camera_runtime, config)
        print_startup_summary(camera_runtime, config, worker)
        return run_application(
            camera_runtime,
            worker,
            display=args.display,
            width=args.width,
            display_fps=args.display_fps,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"gimbal vision failed: {exc}", file=sys.stderr)
        return 3
```

Parser includes `--config`, `--gimbal-config`, `--serial`, `--display`, `--width`, `--display-fps`, `--detection-fps`, `--timeout-ms`, and `--shutdown-timeout`. Build `CameraTuningService` through `build_camera_runtime(config_path=args.config, serial=args.serial, read_timeout_ms=args.timeout_ms, detection_fps=args.detection_fps, diagnostics_fps=None, shutdown_timeout_s=args.shutdown_timeout)`. Load and validate the gimbal YAML first, but treat calibration loading/validation separately: a missing, unreadable, wrong-size, or excessive-RMS calibration creates `UnavailableTargetAngleConverter`, prints the exact calibration error prominently, and does not abort startup. Camera/detection and the USB worker still start; every gate evaluation then returns `GimbalTargetCommand.safe()`, so the reconnecting transport continually sends all-zero `tracking=0` frames. `--display` runs existing `run_local_preview()` on the main thread and remains usable without calibration; headless waits for SIGINT/SIGTERM and prints one compact diagnostic snapshot per second. Finally stop worker first, then service. Do not import FastAPI, Uvicorn, or `ev_vision.web`. Serial open failure does not stop camera/detection.

Print actual camera parameters; calibration path, size, RMS, fx/fy/cx/cy; stable port; output/safety thresholds; signs; and the physical laser warning.

Create `tools/gimbal_vision.py` as the two-line wrapper used in Task 3.

- [ ] **Step 4: Run and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/gimbal_usb tests/preview
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add pyproject.toml src/ev_vision/gimbal_usb tools/gimbal_vision.py tests/gimbal_usb
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'feat: add Jetson gimbal vision command'
```

## Task 11: Document calibration and staged gimbal acceptance

**Files:**
- Create: `docs/runbooks/jetson-gimbal-usb.md`
- Modify: `docs/runbooks/jetson-local-preview.md`
- Create: `tests/test_gimbal_usb_documentation.py`

- [ ] **Step 1: Write a failing documentation contract**

```python
def test_gimbal_runbook_contains_safe_commands_and_limits() -> None:
    text = Path("docs/runbooks/jetson-gimbal-usb.md").read_text(encoding="utf-8")
    required = [
        "sudo systemctl stop ModemManager",
        "/dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00",
        "ev-camera-calibration-capture",
        "--columns 8", "--rows 5", "--square-mm 22",
        "tools/calibrate_camera.py", "--max-rms 0.5",
        "ev-gimbal-vision", "--detection-fps 50", "--display-fps 45",
        "3 frames", "60 ms", "fire=0", "physical",
    ]
    for item in required:
        assert item in text
```

- [ ] **Step 2: Run and confirm the runbook is missing**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/test_gimbal_usb_documentation.py
```

Expected: FAIL because the runbook is absent.

- [ ] **Step 3: Write exact Jetson commands and acceptance order**

The runbook must include these copy-paste sections in order:

1. Pull the feature branch:

```bash
cd ~/2025-E-Vision/2025-E-Vision
git fetch origin
git switch feature/classical-white-board-tracking
git pull --ff-only origin feature/classical-white-board-tracking
```

2. Activate/install and restore MVS paths:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 2025-e-vision
python -m pip install -e '.[vision,hardware]'
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

3. Stop ModemManager and verify the stable path:

```bash
sudo systemctl stop ModemManager
ls -l /dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00
```

4. Physically disconnect or reliably cover the always-on 405 nm laser before calibration or motor-disabled testing.

5. Capture 20–25 varied images:

```bash
ev-camera-calibration-capture \
  --config config/jetson-local.yaml \
  --serial 00G02809155 \
  --columns 8 \
  --rows 5 \
  --square-mm 22 \
  --output artifacts/calibration/images
```

6. Solve, requiring 10+ usable, recommending 15+, exact 1280×1024, RMS ≤0.5 px:

```bash
python tools/calibrate_camera.py \
  artifacts/calibration/images \
  --output config/camera_calibration.yaml \
  --columns 8 \
  --rows 5 \
  --square-mm 22 \
  --max-rms 0.5
```

7. Test software:

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONPYCACHEPREFIX="$(mktemp -d)" python -m compileall -q src tests tools
TEST_TMP="$(mktemp -d)"
python -m pytest -q -p no:cacheprovider --basetemp "$TEST_TMP"
```

8. With motors disabled or the mechanism restrained, run headless:

```bash
ev-gimbal-vision \
  --config config/jetson-local.yaml \
  --gimbal-config config/gimbal_usb.yaml \
  --serial 00G02809155 \
  --detection-fps 50
```

9. Optional local preview:

```bash
ev-gimbal-vision \
  --config config/jetson-local.yaml \
  --gimbal-config config/gimbal_usb.yaml \
  --serial 00G02809155 \
  --display \
  --width 640 \
  --display-fps 45 \
  --detection-fps 50
```

10. Acceptance matrix: no target/stale/invalid calibration/disconnected camera/fourth prediction means tracking zero and all numeric fields zero; real observation means immediate valid output; pure prediction only frames 1–3 and 60 ms; reconnect's first frame is safe; Ctrl+C attempts five safe frames; `fire=0` always but does not switch off the physical laser; gimbal firmware stops when `tracking=0` or no fresh frame for >100 ms.

Update `jetson-local-preview.md` to link to this runbook while retaining its preview-only instructions.

- [ ] **Step 4: Run and commit**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/test_gimbal_usb_documentation.py tests/test_classical_documentation.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add docs/runbooks/jetson-gimbal-usb.md docs/runbooks/jetson-local-preview.md tests/test_gimbal_usb_documentation.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'docs: add Jetson gimbal USB acceptance runbook'
```

## Task 12: Verify the branch and perform one review pass

**Files:**
- Modify only files identified by verification or the single review.

- [ ] **Step 1: Compile without repository caches**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$cache = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
New-Item -ItemType Directory -Force -Path $cache | Out-Null
$env:PYTHONPYCACHEPREFIX=$cache
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m compileall -q src tests tools
```

Expected: exit 0 and no output.

- [ ] **Step 2: Run the complete suite externally**

```powershell
$baseTemp = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
New-Item -ItemType Directory -Force -Path $baseTemp | Out-Null
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider --basetemp $baseTemp
```

Expected: all tests pass; existing platform skips are acceptable. Record exact passed/skipped counts from the output.

- [ ] **Step 3: Prove the legacy protocol stayed untouched**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pytest -q -p no:cacheprovider tests/test_protocol.py tests/test_protocol_mapping.py tests/gimbal_usb/test_protocol.py
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' diff origin/feature/classical-white-board-tracking -- src/ev_vision/protocol.py
```

Expected: tests pass and no diff for the legacy file.

- [ ] **Step 4: Perform the only review pass**

```powershell
Select-String -Path 'src\ev_vision\gimbal_usb\*.py','src\ev_vision\calibration_capture\*.py','docs\runbooks\jetson-gimbal-usb.md' -Pattern 'TBD|TODO|implement later|fill in details' -CaseSensitive:$false
Select-String -Path 'src\ev_vision\gimbal_usb\*.py' -Pattern 'fire\s*=\s*1|distance_m\s*=\s*[^0]|target_id\s*=\s*[^0]'
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' diff --check
```

Expected: no placeholder/unsafe-field matches and clean diff check. Review once for: complete design coverage; exact signature consistency; first-safe reconnect and five-safe shutdown; no valid prediction without prior real observation; no declined reacquisition rule; no web/YOLO/red/full-frame remap; no staged `.tmp/`. Fix concrete issues, rerun affected tests and the full suite once, but do not do a second review pass.

- [ ] **Step 5: Commit verification fixes only if needed**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' add src tests tools config docs pyproject.toml .gitignore
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' commit -m 'test: verify gimbal USB control integration'
```

Skip this commit when no tracked file changed; never create an empty commit.

- [ ] **Step 6: Show final branch state without merging**

```powershell
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' status --short --branch
& 'D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe' log --oneline --decorate -12
```

Expected: still on `feature/classical-white-board-tracking`, all implementation changes committed, and unrelated `.tmp/` remains unstaged.
