# Ring-First ROI Tracking and Frame-Rate Optimization Design

**Date:** 2026-07-21  
**Branch:** `feature/ring-first-roi-tracking`  
**Target:** Jetson Orin NX, Hikrobot MV-CA013-21UC, 1280×1024 BayerRG8

## 1. Goals

1. Preserve the current center-position accuracy and the existing full-resolution ring fit.
2. Allow a valid concentric-ring observation to acquire and maintain the target even when the white-board detector fails.
3. Allow a **medium-quality ring observation** to start tracking because the competition scene is expected to contain almost no unrelated concentric-ring patterns.
4. Move expensive normalization from the full frame to the predicted ROI while tracking.
5. Run white-board detection as a low-frequency correction rather than a per-frame prerequisite.
6. Add stage timing so actual Jetson bottlenecks and FPS gains can be measured.

This change does not promise a fixed FPS before Jetson measurement. It aims to reduce the stable-tracking pixel workload substantially while keeping full-resolution localization inside the ROI.

## 2. Non-goals

- Do not replace the classical detector with a neural model.
- Do not reduce the camera output resolution used for final center fitting.
- Do not remove white-board geometry, homography, scale, or PnP support.
- Do not use a single circle or single arc to start tracking.
- Do not solve output interpolation in the USB worker in this change; higher measurement FPS is addressed first.

## 3. Detection Modes

### 3.1 Full-frame acquisition

When the tracker is `SEARCHING`, `CONFIRMING`, or `LOST`:

1. Normalize/search the full frame using the existing image pipeline for initial correctness.
2. Evaluate white-board candidates as before.
3. If no acceptable white board is found, run ring-only acquisition on the frame instead of immediately returning a miss.
4. A medium or strong multi-ring result may produce a `CONCENTRIC_ARCS` observation.
5. The existing tracker confirmation count remains the final protection against a one-frame false target.

A later optimization may add downsampled coarse search followed by original-resolution ROI refinement. It is not required for the first safe patch unless profiling shows it is necessary and tests cover coordinate remapping.

### 3.2 Stable ROI tracking

When the tracker is `TRACKING`:

1. Predict an ROI from confirmed history before normalization.
2. Crop the original input image.
3. Normalize only the cropped image.
4. Detect rings in ROI-local coordinates.
5. Translate the ring center, arcs, corners, and debug geometry back to full-image coordinates before tracker update and output.
6. If the ROI path fails, use the existing bounded prediction and then fall back to full-frame reacquisition according to tracker state.

The ROI must include margin around the remembered board corners and be clipped to all four image boundaries.

### 3.3 White-board correction cadence

During stable tracking, full white-board evaluation is not required every frame.

- Default cadence: once every **6 processed frames**.
- On correction frames, white-board geometry may refresh corners, scale, homography, and board-plane solution.
- Between correction frames, a valid medium/strong ring observation updates the center while previous board geometry remains available.
- White-board evaluation runs immediately if history is absent, the tracker is reacquiring, scale consistency fails, or the ring observation becomes weak/ambiguous.

The cadence is configurable so Jetson measurements can choose 5–10 frames without code changes.

## 4. Ring Quality Levels

### 4.1 Strong ring

A strong ring result satisfies all of:

- at least 3 distinct expected radii from the `1:2:3:4:5` family;
- `common_center_score >= 0.75`;
- `ratio_score >= 0.80`;
- `coverage_score >= 0.20`;
- finite center and positive, plausible scale.

It may acquire or maintain tracking without white-board evidence.

### 4.2 Medium ring

A medium ring result satisfies all of:

- at least 2 distinct expected radii;
- `common_center_score >= 0.65`;
- `ratio_score >= 0.70`;
- `coverage_score >= 0.15`;
- finite center and positive, plausible scale.

It may also start tracking, per the confirmed field assumption that unrelated concentric rings are extremely unlikely. Safety gates are:

- acquisition still requires the tracker’s normal consecutive confirmation frames;
- consecutive observations must satisfy the tracker center-jump, scale-jump, velocity, and acceleration limits;
- a single circle or one matched arc never starts tracking;
- ambiguity or invalid geometry produces a miss rather than a control target.

### 4.3 Weak/single-arc ring

A weak or single-arc result cannot start tracking. It may only support the existing short, bounded partial/predicted behavior after confirmed history exists.

## 5. Observation Selection

Priority for each processed frame:

1. White board + strong/medium ring: use full geometry and ring center.
2. No white board + strong ring: valid ring-only observation.
3. No white board + medium ring: valid ring-only observation, subject to normal multi-frame tracker confirmation.
4. White board without an acceptable ring: may refresh/rebuild ROI geometry but must not silently create a confident aiming center unless existing scoring rules accept it.
5. Weak/single arc with confirmed history: bounded partial assistance only.
6. Otherwise: miss/prediction according to the tracker state machine.

## 6. Coordinate Mapping

ROI processing uses `(x0, y0, x1, y1)` in full-image coordinates.

- Local center `(cx, cy)` maps to `(cx + x0, cy + y0)`.
- Every arc center and point used outside the ROI is translated by the same offset.
- Board corners and debug overlays must remain full-image coordinates.
- Reported `center_px`, yaw/pitch inputs, and history are always full-image coordinates.

Coordinate translation will be isolated in small helpers and directly unit tested.

## 7. Performance Instrumentation

Each result/debug payload should expose at least:

- `crop_ms`
- `normalization_ms`
- `ring_ms`
- `white_board_ms`
- `detection_ms`
- `total_ms`
- whether full-frame or ROI processing was used
- whether white-board correction ran on that frame

Timing uses `time.perf_counter()` and must not change detector decisions.

## 8. Configuration

Add configurable ring-first tracking fields with conservative defaults:

```yaml
detection:
  ring_first:
    enabled: true
    allow_medium_acquisition: true
    white_board_interval_frames: 6
    strong_min_arcs: 3
    strong_common_center_score: 0.75
    strong_ratio_score: 0.80
    strong_coverage_score: 0.20
    medium_min_arcs: 2
    medium_common_center_score: 0.65
    medium_ratio_score: 0.70
    medium_coverage_score: 0.15
```

Existing user parameters remain unchanged, including exposure `10000 us`, gain `5 dB`, acquisition `60 FPS`, CLAHE `8.5`, white occupancy `0.5`, ratio tolerance `0.18`, minimum arc coverage `0.18`, tracking threshold `0.52`, and acquisition threshold `0.66`.

## 9. Failure Handling

- Empty/invalid ROI: fall back safely; never index outside the image.
- Ring-only acquisition failure: return the existing structured miss.
- White-board correction failure: retain ring tracking if medium/strong ring evidence is valid.
- Sudden scale or center discontinuity: let tracker gates reject the observation.
- Tracking loss: bounded prediction, then full-frame reacquisition.
- Debug rendering failure must not crash detection.

## 10. Test-Driven Implementation

Tests are written and observed failing before production changes.

Required tests:

1. No white board + strong rings can enter confirmation and become tracked.
2. No white board + medium rings can enter confirmation and become tracked after consecutive valid frames.
3. Single/weak ring cannot start tracking.
4. ROI-local center and geometry map correctly to full-image coordinates.
5. Stable tracking normalizes the ROI rather than the full image.
6. White-board evaluation follows the configured cadence and runs immediately during acquisition.
7. White-board failure does not invalidate a medium/strong ring observation.
8. Timing fields are present and non-negative.
9. Existing classical detector, partial-board, tracker, configuration, and contract tests remain green.

## 11. Acceptance Criteria

### Functional

- Ring-only acquisition works for both strong and medium multi-ring observations.
- Medium-ring acquisition still needs consecutive tracker confirmation and continuity checks.
- Stable tracking uses pre-normalization ROI cropping.
- White-board detection is low-frequency during stable tracking and immediate during acquisition/recovery.
- Existing output coordinates remain in the original 1280×1024 frame.

### Accuracy and safety

- Existing synthetic/fixture center-error tests do not regress.
- Single arcs and invalid scale cannot start tracking.
- Existing tracker jump/velocity/acceleration gates remain active.

### Jetson measurement

After deployment, record at least 30 seconds for:

- acquisition FPS and stage timings;
- stable ROI tracking FPS and stage timings;
- target-valid rate;
- center jitter on a stationary target;
- loss/reacquisition behavior.

The change is accepted for competition use only after measured center jitter and valid-rate are not worse than the current baseline while stable-tracking FPS improves materially.
