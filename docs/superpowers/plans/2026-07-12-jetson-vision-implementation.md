# Jetson Vision Aiming System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable Jetson Orin NX vision application that acquires the Hikrobot camera, locates the A4 target, detects the 405 nm laser spot, generates center/circle targets, and sends safe 100 Hz angular-rate commands to the gimbal.

**Architecture:** Use a Python package for orchestration and algorithm development, with hardware-independent typed models and dependency-injected camera, gimbal, chassis, and laser ports. Keep deterministic geometry, protocol, trajectory, control, and state-machine logic separate from OpenCV/TensorRT/MVS adapters so they can be exhaustively tested on Windows recordings before deployment to Jetson. Hardware adapters fail closed: loss of valid frames, feedback, calibration, or homography produces zero rates and laser off.

**Tech Stack:** Python 3.10 on JetPack, NumPy, OpenCV, PyYAML, pyserial, pytest; Hikrobot MVS SDK through a thin adapter; TensorRT FP16 for YOLO deployment; standard `logging`/CSV or JSONL for telemetry.

---

## File map

```text
pyproject.toml                         Packaging, dependencies, pytest settings, CLI entry point
config/default.yaml                   Non-secret runtime defaults
config/camera_calibration.example.yaml Calibration schema example
src/ev_vision/
  app.py                              Composition root and runtime loop
  cli.py                              validate-config, replay, run commands
  config.py                           YAML loading and strict validation
  models.py                           Shared immutable data models and enums
  geometry.py                         Target coordinates, homography, pixel/angle conversion
  trajectory.py                       Center and synchronized 6 cm circle targets
  control.py                          PD, prediction, deadband, velocity/acceleration limiting
  state_machine.py                    Safety-oriented operating state transitions
  protocol.py                         Frame encoding, CRC, streaming parser, payload codecs
  ports.py                            Camera/gimbal/chassis/laser protocols
  camera/latest_frame.py              Latest-only thread-safe frame buffer
  camera/hikrobot.py                  MVS SDK adapter
  camera/replay.py                    Image/video replay adapter
  detection/board_geometry.py         Black-border quadrilateral refinement
  detection/yolo_board.py             Search/reacquisition TensorRT adapter
  detection/laser_spot.py              Difference and continuous ROI spot detectors
  tracking/predictor.py               Timestamp-based constant-velocity prediction
  communication/serial_link.py         UART threads, feedback health, 100 Hz transmitter
  hardware/gpio_laser.py               Fail-closed Jetson GPIO laser backend
  hardware/mock.py                     Deterministic mock ports
  telemetry.py                         Non-blocking records and debug overlay
  runtime.py                           One-cycle pipeline and watchdog behavior
tests/                                Unit, integration, protocol-vector, replay tests
tools/calibrate_camera.py             Camera intrinsic calibration utility
tools/collect_dataset.py              Dataset image collection utility
tools/export_tensorrt.py              Documented model export wrapper
```

## Milestone 1 — Hardware-independent executable core

### Task 1: Package skeleton and validated configuration

**Files:**
- Create: `pyproject.toml`
- Create: `config/default.yaml`
- Create: `config/camera_calibration.example.yaml`
- Create: `src/ev_vision/__init__.py`
- Create: `src/ev_vision/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for loading defaults and rejecting unsafe null control limits**

```python
def test_default_config_is_valid_for_mock_mode(default_config_path):
    cfg = load_config(default_config_path, hardware_required=False)
    assert cfg.camera.width == 1280
    assert cfg.serial.baudrate == 921600
    assert cfg.control.command_hz == 100


def test_hardware_mode_rejects_missing_rate_limits(default_config_path):
    with pytest.raises(ConfigError, match="max_yaw_rate_deg_s"):
        load_config(default_config_path, hardware_required=True)
```

- [ ] **Step 2: Run `python -m pytest tests/test_config.py -v`; expect import failure for `ev_vision.config`.**
- [ ] **Step 3: Implement frozen dataclasses, YAML parsing, unknown-key rejection, ranges, and separate mock/hardware validation.**
- [ ] **Step 4: Run the test file; expect all tests to pass.**
- [ ] **Step 5: Commit `chore: scaffold vision package and validated config`.**

### Task 2: Shared models and latest-frame buffer

**Files:**
- Create: `src/ev_vision/models.py`
- Create: `src/ev_vision/camera/__init__.py`
- Create: `src/ev_vision/camera/latest_frame.py`
- Test: `tests/camera/test_latest_frame.py`

- [ ] **Step 1: Write tests proving overwrite semantics, monotonic sequence, and timeout behavior.**

```python
def test_latest_frame_overwrites_unconsumed_frame():
    buffer = LatestFrameBuffer()
    buffer.publish(Frame(sequence=1, captured_ns=10, image="old"))
    buffer.publish(Frame(sequence=2, captured_ns=20, image="new"))
    assert buffer.wait_next(after_sequence=0, timeout_s=0.01).sequence == 2
```

- [ ] **Step 2: Run the test and verify missing classes cause RED.**
- [ ] **Step 3: Implement `Frame`, observations, feedback models, operating enums, and a condition-variable latest-only buffer.**
- [ ] **Step 4: Run camera buffer tests and the full suite.**
- [ ] **Step 5: Commit `feat: add shared vision models and latest frame buffer`.**

### Task 3: Wire protocol and streaming parser

**Files:**
- Create: `src/ev_vision/protocol.py`
- Test: `tests/test_protocol.py`

- [ ] **Step 1: Add known CRC vector (`123456789` → `0x29B1`), frame round-trip, byte corruption, noise-prefix, partial-frame, and concatenated-frame tests.**
- [ ] **Step 2: Run tests; verify protocol imports fail.**
- [ ] **Step 3: Implement CRC-16/CCITT-FALSE, `AA 55` outer frame, little-endian headers, payload length cap, sequence handling, `VisionControlPayload`, `GimbalFeedbackPayload`, and `ChassisProgressPayload`.**
- [ ] **Step 4: Run `python -m pytest tests/test_protocol.py -v`; all vectors and parser recovery cases pass.**
- [ ] **Step 5: Commit `feat: implement gimbal serial protocol`.**

### Task 4: Target geometry and homography validation

**Files:**
- Create: `src/ev_vision/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write tests for A4 corners, target↔image round trip, 6 cm radius mapping, degenerate quadrilateral rejection, and pixel error to angular error.**

```python
def test_standard_target_center_and_circle_radius():
    geom = TargetGeometry(px_per_cm=40.0)
    assert geom.center_px == pytest.approx((420.0, 594.0))
    assert geom.cm_to_rectified_px((6.0, 0.0)) == pytest.approx((660.0, 594.0))
```

- [ ] **Step 2: Verify tests fail because geometry is absent.**
- [ ] **Step 3: Implement ordered corners, OpenCV homography creation, condition/area/orientation checks, transforms, and `atan2((u-cx), fx)` angular conversion.**
- [ ] **Step 4: Run geometry tests including noisy synthetic quadrilaterals.**
- [ ] **Step 5: Commit `feat: add target-plane geometry transforms`.**

### Task 5: Center and lap-synchronized circle trajectory

**Files:**
- Create: `src/ev_vision/trajectory.py`
- Test: `tests/test_trajectory.py`

- [ ] **Step 1: Test center target, cardinal circle positions, wraparound, stale chassis input, and bounded phase correction without target jumps.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement `CenterTarget` and `CircleTrajectory` with local monotonic phase plus a configurable, rate-limited correction toward `2π progress/1000 + theta0`.**
- [ ] **Step 4: Run tests and assert every generated point remains at 6.0 cm radius within floating-point tolerance.**
- [ ] **Step 5: Commit `feat: add center and synchronized circle targets`.**

### Task 6: Predictor, PD servo, deadband and rate limiter

**Files:**
- Create: `src/ev_vision/tracking/__init__.py`
- Create: `src/ev_vision/tracking/predictor.py`
- Create: `src/ev_vision/control.py`
- Test: `tests/test_control.py`

- [ ] **Step 1: Test proportional response, derivative filtering, hysteresis deadband, saturation, acceleration limits, axis mapping, stale-source zero output, and latency prediction.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement stateful controllers using explicit timestamps and no integral term; return zero plus `target_valid=False` on stale/invalid input.**
- [ ] **Step 4: Run control tests with deterministic time values.**
- [ ] **Step 5: Commit `feat: add safe visual servo controller`.**

### Task 7: Safety state machine

**Files:**
- Create: `src/ev_vision/state_machine.py`
- Test: `tests/test_state_machine.py`

- [ ] **Step 1: Encode transition-table tests for `BOOT→SAFE→SEARCH→ACQUIRE→CENTER→LASER_CALIBRATE→AIM→TRACK/CIRCLE`, short-loss `RECOVER`, and latched `FAULT`.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement explicit event-driven transitions, dwell counters, reset requirement, and output policy (`laser_allowed`, `motion_allowed`, `target_kind`).**
- [ ] **Step 4: Run all transition tests and property-check that FAULT never permits laser or motion.**
- [ ] **Step 5: Commit `feat: implement fail-closed operating state machine`.**

### Task 8: Ports, mock hardware, and one-cycle runtime

**Files:**
- Create: `src/ev_vision/ports.py`
- Create: `src/ev_vision/hardware/__init__.py`
- Create: `src/ev_vision/hardware/mock.py`
- Create: `src/ev_vision/runtime.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write integration tests for a valid board cycle, lost-board recovery, stale frame shutdown, gimbal fault shutdown, and CIRCLE rejection without homography.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Define injected port protocols and implement a deterministic `VisionRuntime.step(now_ns)` that produces exactly one control command and one laser decision.**
- [ ] **Step 4: Run integration tests and assert every unsafe case emits zero rates and laser off.**
- [ ] **Step 5: Commit `feat: add mockable fail-safe runtime pipeline`.**

### Task 9: CLI and replay smoke test

**Files:**
- Create: `src/ev_vision/cli.py`
- Create: `src/ev_vision/app.py`
- Create: `src/ev_vision/camera/replay.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Test `validate-config`, `protocol-selftest`, and a synthetic `mock-run --cycles 10`.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement CLI composition without importing MVS/TensorRT unless those backends are selected.**
- [ ] **Step 4: Run `python -m ev_vision.cli validate-config --config config/default.yaml` and `python -m ev_vision.cli mock-run --cycles 10`; expect exit code 0 and final laser OFF.**
- [ ] **Step 5: Commit `feat: add executable mock vision application`.**

## Milestone 2 — Classical precision vision

### Task 10: Black-border geometry detector

**Files:**
- Create: `src/ev_vision/detection/__init__.py`
- Create: `src/ev_vision/detection/board_geometry.py`
- Create: `tests/detection/test_board_geometry.py`
- Create: `tests/fixtures/synthetic_board.py`

- [ ] **Step 1:** Generate synthetic A4 boards with perspective, blur, illumination gradients, clipping, and distractor lines; define corner-error and rejection assertions.
- [ ] **Step 2:** Verify the tests fail before the detector exists.
- [ ] **Step 3:** Implement grayscale/contrast normalization, adaptive thresholding, morphology, contour/line candidates, RANSAC line fitting, floating-point intersections, corner ordering, and confidence metrics.
- [ ] **Step 4:** Run the synthetic matrix and require median corner error ≤1 px for clean images and deterministic rejection for invalid images.
- [ ] **Step 5:** Commit `feat: refine target board geometry from black border`.

### Task 11: Laser spot detection

**Files:**
- Create: `src/ev_vision/detection/laser_spot.py`
- Test: `tests/detection/test_laser_spot.py`

- [ ] **Step 1:** Test off/on registered difference, subpixel centroid, saturated spot, purple/blue response variation, lingering mark rejection, missing spot, and ROI boundary behavior.
- [ ] **Step 2:** Verify RED.
- [ ] **Step 3:** Implement robust channel/brightness difference, connected-component scoring, quadratic or weighted-centroid refinement, slow background model, and confidence output.
- [ ] **Step 4:** Run synthetic sequences and require correct current-spot selection over old marks.
- [ ] **Step 5:** Commit `feat: detect static and continuous 405nm laser spot`.

### Task 12: Debug overlay and replay evaluation

**Files:**
- Create: `src/ev_vision/telemetry.py`
- Create: `tools/evaluate_replay.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1:** Test bounded non-blocking queue, dropped-debug-frame counter, JSONL schema, and overlay creation.
- [ ] **Step 2:** Implement asynchronous writer and annotated frame rendering.
- [ ] **Step 3:** Run replay on fixture sequences and save metrics without blocking control.
- [ ] **Step 4:** Commit `feat: add replay metrics and debug telemetry`.

## Milestone 3 — Search model and camera hardware

### Task 13: YOLO search/reacquisition adapter

**Files:**
- Create: `src/ev_vision/detection/yolo_board.py`
- Create: `tools/collect_dataset.py`
- Create: `tools/export_tensorrt.py`
- Test: `tests/detection/test_yolo_adapter.py`

- [ ] **Step 1:** Test letterbox coordinate restoration, confidence filtering, best-candidate selection, and absence of TensorRT dependency in mock mode.
- [ ] **Step 2:** Implement an inference-port abstraction and TensorRT backend loaded only on Jetson.
- [ ] **Step 3:** Validate against an exported FP16 engine and recorded frames at 0.5–1.8 m.
- [ ] **Step 4:** Commit `feat: add YOLO board search TensorRT adapter`.

### Task 14: Hikrobot MVS latest-frame camera

**Files:**
- Create: `src/ev_vision/camera/hikrobot.py`
- Create: `tools/list_cameras.py`
- Test: `tests/camera/test_hikrobot_adapter.py`

- [ ] **Step 1:** Build tests around an injected fake MVS API for device selection, parameter application, timestamp propagation, timeout, disconnect, and buffer release.
- [ ] **Step 2:** Implement SDK lifecycle with context-managed acquisition and BayerRG8 conversion.
- [ ] **Step 3:** On Jetson, verify 1280×1024 fixed exposure, no old-frame queue, and sustained capture logging.
- [ ] **Step 4:** Commit `feat: integrate Hikrobot MVS camera`.

### Task 15: Camera calibration utility

**Files:**
- Create: `tools/calibrate_camera.py`
- Create: `src/ev_vision/calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1:** Test calibration-file schema, image-size mismatch rejection, and map caching.
- [ ] **Step 2:** Implement chessboard capture, solve, RMS report, YAML save, and undistortion map generation.
- [ ] **Step 3:** Collect 20–30 poses after final focus/aperture and accept only RMS ≤0.5 px.
- [ ] **Step 4:** Commit `feat: add camera intrinsic calibration workflow`.

## Milestone 4 — Real communication and laser hardware

### Task 16: Serial gimbal and chassis link

**Files:**
- Create: `src/ev_vision/communication/__init__.py`
- Create: `src/ev_vision/communication/serial_link.py`
- Test: `tests/communication/test_serial_link.py`

- [ ] **Step 1:** Test 100 Hz scheduling, latest-command use, ACK tracking, partial reads, CRC errors, 300 ms feedback timeout, reconnect, and shutdown zero command with an in-memory serial pair.
- [ ] **Step 2:** Implement independent RX/TX loops using the tested protocol parser.
- [ ] **Step 3:** Verify on a USB-UART loopback and then the real controller without enabling the laser.
- [ ] **Step 4:** Commit `feat: add resilient serial gimbal link`.

### Task 17: Fail-closed laser backend

**Files:**
- Create: `src/ev_vision/hardware/gpio_laser.py`
- Test: `tests/hardware/test_gpio_laser.py`

- [ ] **Step 1:** Test startup off, explicit enable, exception-triggered off, active-low mapping, unsupported PWM rejection, and context-manager cleanup.
- [ ] **Step 2:** Implement an injected GPIO driver so host tests do not require Jetson libraries.
- [ ] **Step 3:** Verify electrical level first with an LED/oscilloscope, then connect the laser driver; never drive a laser diode directly from Jetson GPIO.
- [ ] **Step 4:** Commit `feat: add fail-closed Jetson laser control`.

## Milestone 5 — Full-system tuning and acceptance

### Task 18: Closed-loop integration, calibration and acceptance tools

**Files:**
- Modify: `src/ev_vision/app.py`
- Modify: `src/ev_vision/runtime.py`
- Create: `tools/calibrate_axis_mapping.py`
- Create: `tools/calibrate_laser_model.py`
- Create: `tools/run_acceptance.py`
- Test: `tests/integration/test_full_pipeline.py`

- [ ] **Step 1:** Run recorded end-to-end fixtures through SEARCH, ACQUIRE, CENTER, AIM, TRACK, RECOVER, and CIRCLE with expected command envelopes.
- [ ] **Step 2:** Determine yaw/pitch signs and the 2×2 mapping matrix using small laser-safe motions.
- [ ] **Step 3:** Measure end-to-end latency from capture timestamp to observed response and configure prediction.
- [ ] **Step 4:** Tune P first, then filtered D, deadband, speed, and acceleration limits; keep I disabled.
- [ ] **Step 5:** Build the distance-related fallback laser model and verify seamless fallback/recovery.
- [ ] **Step 6:** Validate static 2 s/4 s aiming, moving center tracking, synchronized 6 cm circle, short occlusion recovery, and all fault injections.
- [ ] **Step 7:** Commit `feat: complete vision aiming integration and acceptance suite`.

## Global verification gates

After every task:

```powershell
python -m pytest -q
python -m compileall -q src tests tools
```

Before Jetson deployment:

```bash
python -m pytest -q
python -m ev_vision.cli validate-config --config config/jetson.yaml --hardware-required
python -m ev_vision.cli protocol-selftest
```

A milestone is accepted only if tests pass, configuration contains no required `null` hardware values, startup and shutdown leave the laser off, and every injected camera/serial/control fault produces zero gimbal rate.

## Explicit hardware-dependent limits

The repository can implement and test all deterministic software before hardware arrives. These items require the final physical system and therefore are planned calibration/acceptance work rather than guessed constants: camera intrinsics, exact 405 nm channel thresholds, laser GPIO line/electrical driver, yaw/pitch mapping, rate/acceleration limits, PD gains, end-to-end latency, and the distance-related laser reference model.
