# Ring-First ROI Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase stable classical-detection FPS without reducing final ring-center resolution by allowing medium/strong ring-only acquisition, normalizing only a predicted ROI during tracking, and running white-board correction periodically.

**Architecture:** Add a focused ring-first policy/config module, keep the existing tracker as the consecutive-frame and motion safety gate, and refactor `ClassicalBoardDetector.detect()` so it chooses full-frame or ROI input before normalization. Ring-only observations use full-image coordinates; full white-board acquisition remains the preferred source whenever it succeeds.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, dataclasses, pytest, YAML configuration.

---

## File Map

- Create `src/ev_vision/detection/ring_first.py`: ring quality classification and ROI-coordinate translation helpers.
- Modify `src/ev_vision/config.py`: `RingFirstConfig`, parsing, and validation.
- Modify `src/ev_vision/detection/classical_board.py`: ring-only fallback, ROI-before-normalization path, white-board cadence, and stage timing.
- Modify `config/default.yaml` and `config/jetson-local.yaml`: expose ring-first defaults while preserving current tuning values.
- Modify `tests/detection/test_classical_board.py`: detector integration behavior and cadence/ROI tests.
- Create `tests/detection/test_ring_first.py`: policy and coordinate translation unit tests.
- Modify `tests/test_config.py` or the existing configuration test file found by search: parsing/validation coverage.

### Task 1: Ring-first policy and configuration

**Files:**
- Create: `src/ev_vision/detection/ring_first.py`
- Modify: `src/ev_vision/config.py`
- Create: `tests/detection/test_ring_first.py`
- Modify: matching configuration tests under `tests/`

- [ ] **Step 1: Write failing tests**

Add tests that construct `RingGeometryResult` values and assert:

```python
assert classify_ring(strong_result, config) is RingQuality.STRONG
assert classify_ring(medium_result, config) is RingQuality.MEDIUM
assert classify_ring(single_arc_result, config) is RingQuality.REJECTED
assert translate_ring_result(local_result, (100, 50)).center_px == (local_x + 100, local_y + 50)
```

Add configuration tests asserting the defaults, YAML parsing of `detection.ring_first`, rejection of zero cadence, and rejection of inconsistent strong/medium thresholds.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection/test_ring_first.py <config-test-path> -q
```

Expected: collection/import failure because `RingFirstConfig`, `RingQuality`, `classify_ring`, and `translate_ring_result` do not exist.

- [ ] **Step 3: Implement minimal policy/config**

Add frozen `RingFirstConfig` with the approved defaults. Implement enum `RingQuality` and pure functions:

```python
def classify_ring(result: RingGeometryResult, config: RingFirstConfig) -> RingQuality: ...
def translate_ring_result(result: RingGeometryResult, offset_xy: tuple[int, int]) -> RingGeometryResult: ...
```

Classification requires `result.valid`, finite center, positive finite scale, and threshold checks. Translation preserves all scores and failure state while offsetting the center and arc centers.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same targeted tests. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/ev_vision/config.py src/ev_vision/detection/ring_first.py tests/detection/test_ring_first.py <config-test-path>
git commit -m "feat: add ring-first quality policy"
```

### Task 2: Ring-only acquisition fallback

**Files:**
- Modify: `src/ev_vision/detection/classical_board.py`
- Modify: `tests/detection/test_classical_board.py`

- [ ] **Step 1: Write failing integration tests**

Use monkeypatches for deterministic evidence. Patch `find_white_board_candidates` to return `[]` and `detect_concentric_arcs` to return strong/medium results. Assert three consecutive medium observations transition from `CONFIRMING` to `TRACKING`, use `ObservationSource.CONCENTRIC_ARCS`, and preserve center/scale. Add a single-arc/rejected result test that remains invalid.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection/test_classical_board.py -k "ring_only or medium_ring or single_arc" -q
```

Expected: failure because `_acquire_full_board()` returns `NO_WHITE_CANDIDATE` before ring search.

- [ ] **Step 3: Implement ring-only fallback**

When no acceptable full-board observation exists, call `detect_concentric_arcs` over the supplied normalized frame/ROI. Convert medium/strong results to `TrackObservation` with source `CONCENTRIC_ARCS`, ring center, ring scale, empty corners, and confidence computed from center/ratio/coverage scores. Do not create an observation for rejected/single-arc results.

- [ ] **Step 4: Run detector and tracker tests**

Run:

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection/test_classical_board.py tests/tracking/test_board_tracker.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/ev_vision/detection/classical_board.py tests/detection/test_classical_board.py
git commit -m "feat: acquire target from concentric rings"
```

### Task 3: Crop before normalization and map ROI results

**Files:**
- Modify: `src/ev_vision/detection/classical_board.py`
- Modify: `tests/detection/test_classical_board.py`

- [ ] **Step 1: Write failing ROI tests**

Confirm a detector, monkeypatch `normalize_frame` to record input shapes, and patch ring detection to return an ROI-local center. Assert the next tracking frame normalizes a shape smaller than the original and returns `center_px` translated to full-image coordinates. Assert acquisition still normalizes the full frame.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection/test_classical_board.py -k "normalizes_tracking_roi or maps_roi" -q
```

Expected: failure because normalization currently occurs before `_tracking_roi()`.

- [ ] **Step 3: Refactor detector input selection**

Compute acquisition state and predicted ROI from `image.shape[:2]` before normalization. Crop with a safe helper, normalize the crop, and run ROI-local detection with no second ROI slice. Translate ring results and any ROI-local output back to full-image coordinates before tracker update. If no valid ROI exists, use the full frame.

- [ ] **Step 4: Preserve debug behavior**

When debug is requested for an ROI frame, return normalized ROI diagnostic images and ensure overlays use ROI-local render coordinates or explicitly label the mode in timings/metadata. Detection output coordinates must remain full-frame.

- [ ] **Step 5: Run relevant tests and commit**

Run:

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection/test_classical_board.py tests/detection/test_partial_board.py tests/detection/test_ring_first.py -q
```

Expected: PASS.

Commit:

```powershell
git add src/ev_vision/detection/classical_board.py tests/detection/test_classical_board.py
git commit -m "perf: normalize predicted tracking ROI"
```

### Task 4: White-board correction cadence and timings

**Files:**
- Modify: `src/ev_vision/detection/classical_board.py`
- Modify: `tests/detection/test_classical_board.py`
- Modify: `config/default.yaml`
- Modify: `config/jetson-local.yaml`

- [ ] **Step 1: Write failing cadence/timing tests**

Patch `find_white_board_candidates` with a counter. Assert it runs on every acquisition frame, then only every configured sixth stable-tracking frame while ring observations remain valid. Assert result timings contain non-negative `crop_ms`, `normalization_ms`, `ring_ms`, `white_board_ms`, `detection_ms`, and `total_ms`, plus numeric flags `roi_used` and `white_board_ran`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection/test_classical_board.py -k "white_board_cadence or stage_timings" -q
```

Expected: failure because white-board evaluation currently runs every frame and timing keys are absent.

- [ ] **Step 3: Implement cadence and timing**

Maintain a processed tracking-frame counter. Run white-board evaluation during acquisition/recovery, when history is absent, and at interval boundaries. On skipped frames run ring-only tracking. Reset cadence state in `reset()` and when applying configuration. Instrument crop, normalization, ring, white board, detection, and total stages with `perf_counter()`.

- [ ] **Step 4: Apply approved YAML defaults**

Add `detection.ring_first` to both YAML files. In `jetson-local.yaml`, preserve the user’s current persisted tuning values and set the approved ring-first cadence/thresholds.

- [ ] **Step 5: Run relevant tests and commit**

Run:

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection tests/tracking tests/test_config.py -q
```

Expected: PASS (adjust config test path to repository filename discovered during execution).

Commit:

```powershell
git add src/ev_vision/detection/classical_board.py config/default.yaml config/jetson-local.yaml tests/detection/test_classical_board.py
git commit -m "perf: decimate white-board correction"
```

### Task 5: Regression verification and branch publication

**Files:**
- Modify only if tests expose a regression.

- [ ] **Step 1: Run focused suite**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest tests/detection tests/tracking -q
```

Expected: all pass.

- [ ] **Step 2: Run full suite**

```powershell
D:\Coding\anaconda\envs\2025-e-vision\python.exe -m pytest -q
```

Expected: all pass; the known pytest cache permission warning is acceptable and unrelated.

- [ ] **Step 3: Review diff and configuration**

Verify no unrelated files changed, no user parameter regressed, all outputs remain full-frame coordinates, and no single-arc acquisition path exists.

- [ ] **Step 4: Commit any final test/docs adjustments**

```powershell
git add <reviewed-files>
git commit -m "test: verify ring-first ROI tracking"
```

Skip the commit if the tree is already clean.

- [ ] **Step 5: Push branch**

```powershell
git push -u origin feature/ring-first-roi-tracking
```

Expected: remote branch is created and upstream tracking is configured.
