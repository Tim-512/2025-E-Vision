# Ring-Only Motion Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active white-board pipeline with a fast ring-only detector that acquires strong rings quickly, confirms weaker two-ring targets, and tracks a continuously moving target with a velocity-predicted expanding ROI.

**Architecture:** Keep the existing public detector/tracker contracts and full-frame output coordinates. Add a fast normalization entry point, make ring geometry return structurally valid two-ring candidates for the quality classifier, and simplify `ClassicalBoardDetector` to a ring-only state path with staged ROI recovery. Extend compatible configuration/API schemas and preserve debug images with empty white-board planes.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, dataclasses, FastAPI/Pydantic, pytest, YAML.

---

## File map

- `src/ev_vision/config.py`: ring-only acquisition, threshold, and ROI configuration plus validation.
- `src/ev_vision/detection/image_normalization.py`: fast ring-only preprocessing.
- `src/ev_vision/detection/ring_geometry.py`: structural candidate validity independent of quality thresholds.
- `src/ev_vision/detection/ring_first.py`: strong/medium/rejected quality classification.
- `src/ev_vision/detection/classical_board.py`: ring-only state path, immediate strong acquisition, dynamic ROI, miss recovery, diagnostics.
- `src/ev_vision/tracking/board_tracker.py`: expose timestamp-aware predicted center/history required by ROI construction and configurable strong promotion.
- `src/ev_vision/web/camera_tuning_app.py`: request schema for new fields.
- `config/default.yaml`, `config/jetson-local.yaml`: defaults and field values including CLAHE 10.0.
- `tests/detection/*`, `tests/tracking/*`, `tests/web/*`, `tests/test_config.py`: regression and behavior coverage.

### Task 1: Configuration contract

**Files:**
- Modify: `src/ev_vision/config.py`
- Modify: `config/default.yaml`
- Modify: `config/jetson-local.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Assert defaults and YAML round-trip for `ring_only`, immediate strong acquisition, medium confirmation count, relaxed medium thresholds, and dynamic ROI fields. Assert invalid confirmation counts, safety factors, dimensions, and miss counts raise `ConfigError`. Update the Jetson-local assertion to CLAHE 10.0 while preserving exposure 10000, gain 5, and FPS 60.

- [ ] **Step 2: Run RED**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/test_config.py -k "ring_first or jetson_local" -q -p no:cacheprovider
```

Expected: failures for missing fields and old CLAHE 8.5.

- [ ] **Step 3: Implement config and YAML**

Add typed dataclass fields and range validation. Set field defaults from the approved design and update both YAML files without changing unrelated values.

- [ ] **Step 4: Run GREEN and commit**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/test_config.py -k "ring_first or jetson_local" -q -p no:cacheprovider
```

Commit: `feat: configure ring-only motion tracking`.

### Task 2: Fast normalization and relaxed structural geometry

**Files:**
- Modify: `src/ev_vision/detection/image_normalization.py`
- Modify: `src/ev_vision/detection/ring_geometry.py`
- Test: `tests/detection/test_image_normalization.py`
- Test: `tests/detection/test_ring_geometry.py`
- Test: `tests/detection/test_ring_first.py`

- [ ] **Step 1: Write failing fast-path tests**

Patch percentile, morphology, and Sobel calls to fail if invoked by `normalize_ring_frame()`. Assert output planes used by ring detection match image shape and white-board-only planes are zero-filled. Add a geometry test proving a structurally valid two-ring candidate is not rejected solely because ratio score is below the previous hard-coded 0.70 gate. Update quality tests for medium thresholds 0.60/0.65/0.12.

- [ ] **Step 2: Run RED**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection/test_image_normalization.py tests/detection/test_ring_geometry.py tests/detection/test_ring_first.py -q -p no:cacheprovider
```

Expected: missing fast function and threshold expectation failures.

- [ ] **Step 3: Implement minimal fast preprocessing and structural validity**

Factor shared grayscale/CLAHE/ring-mask work into a private helper. `normalize_ring_frame()` skips white percentile/morphology/Sobel and supplies zero arrays for compatibility. Change final ring `valid` to structural validity; keep scores in the result for `classify_ring()`.

- [ ] **Step 4: Run GREEN and commit**

Commit: `perf: add fast ring-only preprocessing`.

### Task 3: Strong and medium acquisition behavior

**Files:**
- Modify: `src/ev_vision/tracking/board_tracker.py`
- Modify: `src/ev_vision/detection/classical_board.py`
- Test: `tests/tracking/test_board_tracker.py`
- Test: `tests/detection/test_classical_board.py`

- [ ] **Step 1: Write failing acquisition tests**

Assert a strong three-ring observation enters `TRACKING` in one frame when `immediate_strong_acquisition` is enabled. Assert medium two-ring observations require exactly `medium_confirm_frames` consistent frames. Assert one ring remains in `SEARCHING`. Patch `find_white_board_candidates` to raise and prove it is never called in SEARCHING, CONFIRMING, TRACKING, PREDICTING, or LOST ring-only paths.

- [ ] **Step 2: Run RED**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/tracking/test_board_tracker.py tests/detection/test_classical_board.py -k "strong or medium or ring_only or white_board" -q -p no:cacheprovider
```

- [ ] **Step 3: Implement ring-only state path**

Use `normalize_ring_frame()` and a single ring detector pass. Build `CONCENTRIC_ARCS` observations with no corners/solution. Add an explicit tracker update option for trusted strong acquisition without weakening ordinary or medium confirmation. Preserve full-frame center and arc translation.

- [ ] **Step 4: Run GREEN and commit**

Commit: `feat: acquire targets from rings only`.

### Task 4: Velocity-predicted ROI and staged miss recovery

**Files:**
- Modify: `src/ev_vision/tracking/board_tracker.py`
- Modify: `src/ev_vision/detection/classical_board.py`
- Test: `tests/tracking/test_board_tracker.py`
- Test: `tests/detection/test_classical_board.py`

- [ ] **Step 1: Write failing ROI tests**

Seed two confirmed observations with known velocity. Assert the next ROI center advances by velocity times capture delta, fast motion widens the ROI, and ROI-local detections map to full coordinates. Count normalization calls and assert a miss performs one pass only; subsequent misses expand ROI; the configured miss level causes a full-frame pass. Assert a hit resets expansion.

- [ ] **Step 2: Run RED**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection/test_classical_board.py tests/tracking/test_board_tracker.py -k "roi or predict or miss" -q -p no:cacheprovider
```

- [ ] **Step 3: Implement ROI prediction and recovery**

Expose latest real timestamp, center, velocity, scale, and bounded timestamp prediction from the tracker. Build a square ROI with ring extent, axis motion displacement, prediction padding, safety factor, and miss expansion. Increment miss level only after an ROI miss; switch to full-frame on the configured later frame; reset on any accepted ring observation.

- [ ] **Step 4: Add diagnostics**

Record ROI width/height, predicted shift, miss level, fallback/full-frame flag, candidate count, and compatibility zeros for white-board timing flags.

- [ ] **Step 5: Run GREEN and commit**

Commit: `perf: track moving rings with dynamic ROI`.

### Task 5: Web tuning compatibility

**Files:**
- Modify: `src/ev_vision/web/camera_tuning_app.py`
- Modify only if required: `src/ev_vision/web/static/camera-tuning.js`
- Modify only if required: `src/ev_vision/web/static/camera-tuning.html`
- Test: `tests/web/test_detection_api.py`
- Test: `tests/tuning/test_service.py`

- [ ] **Step 1: Write failing API tests**

Assert GET exposes all ring-only fields and PATCH accepts them while preserving omitted values. Assert debug responses remain valid when white mask and gradient are black compatibility images.

- [ ] **Step 2: Run RED**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/web/test_detection_api.py tests/tuning/test_service.py -k "ring_first or detection_config or debug" -q -p no:cacheprovider
```

- [ ] **Step 3: Implement request/domain mapping**

Add a Pydantic ring-first request model and include it in `DetectionConfigRequest`. Reuse dataclass serialization for GET. Only change static controls if the current generic form does not expose nested fields.

- [ ] **Step 4: Run GREEN and commit**

Commit: `feat: expose ring-only tuning controls`.

### Task 6: Regression verification and publication

**Files:**
- Modify only for discovered regressions.

- [ ] **Step 1: Run focused suites**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection tests/tracking tests/web tests/tuning tests/gimbal_usb tests/test_config.py -q -p no:cacheprovider --basetemp=<writable-temp>
```

- [ ] **Step 2: Run full suite**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest -q -p no:cacheprovider --basetemp=<writable-temp>
```

Expected: all tests pass; no claim about Jetson FPS is made from desktop tests.

- [ ] **Step 3: Static verification**

Run compileall with a writable `PYTHONPYCACHEPREFIX`, `git diff --check`, inspect status, and confirm no `.tmp` or cache files are staged.

- [ ] **Step 4: Commit final fixes if needed**

Commit: `test: verify ring-only motion tracking` (skip if clean).

- [ ] **Step 5: Push current branch**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\Library\bin\git.exe -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push origin feature/ring-first-roi-tracking
```

Expected: Jetson can pull the new commits from the already checked-out branch.
