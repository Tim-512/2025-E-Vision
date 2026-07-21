# Ring-Only Motion Tracking Design

**Date:** 2026-07-21  
**Branch:** `feature/ring-first-roi-tracking`  
**Target:** continuously moving vehicle on Jetson, full-resolution concentric-ring localization

## 1. Decision

The competition runtime switches from white-board-first detection to a **ring-only motion mode**. White-board contours, white occupancy, board corners, homography, and board-plane pose are removed from the active detection path. A target angle is produced from the detected concentric-ring center and camera intrinsics only.

This is an intentional field trade-off: the scene is expected to contain almost no unrelated concentric rings, so accepting a consistent two-ring observation is more valuable than retaining white-board verification. The change preserves the safety rule that a single circle or single arc cannot acquire the target.

## 2. Goals

1. Raise Jetson detection FPS without reducing image resolution or final ring-fitting precision.
2. Remove white-board processing and its periodic latency spikes from every runtime state.
3. Let strong rings lock immediately and medium/slightly weak two-ring observations lock after short temporal confirmation.
4. Keep a moving target inside the tracking ROI through velocity prediction and dynamic padding.
5. Avoid doing both ROI and full-frame processing on every ROI miss.
6. Keep web tuning/debug output usable and add enough timing data for Jetson profiling.

Actual FPS is an acceptance measurement, not a guaranteed number before testing on the Jetson.

## 3. Non-goals and accepted losses

- No white-board verification in the competition detection path.
- No board corners, homography, board-plane pose, or white-board scale refresh.
- No distance/pose correction derived from the white board.
- No neural detector and no camera-resolution reduction.
- No single-circle or single-arc acquisition.
- No artificial interpolation of missing yaw/pitch measurements; smoother output should come first from more frequent real detections.

## 4. Fast ring preprocessing

Introduce a ring-only normalization function that returns only data used by ring geometry:

- grayscale image;
- Gaussian-denoised image;
- CLAHE/illumination-normalized image;
- Canny edge mask;
- saturated-pixel mask;
- saturated-pixel-suppressed ring edge mask.

It must not calculate percentile white threshold, 9x9 white-mask morphology, Sobel X/Y, or gradient magnitude. The existing full `normalize_frame()` remains available for tools or tests that still require white-board diagnostics. Ring-only runtime calls the fast function. Debug rendering receives black `white_mask` and `gradient` planes of the correct shape so the web page never crashes.

## 5. Ring validity and quality

### 5.1 Geometry candidate

The geometry layer must return a finite concentric-ring candidate when at least two distinct plausible radii are present. It must not discard a candidate using the old hard-coded `common_center >= 0.60` and `ratio >= 0.70` gate before the quality classifier can inspect it. Structural minimums remain enforced: at least `min_multiple_arcs` (default 2), finite common center, positive finite scale, and non-ambiguous radius assignment.

### 5.2 Strong quality

Default requirements:

- at least 3 distinct rings;
- common-center score >= 0.75;
- radius-ratio score >= 0.80;
- coverage score >= 0.20.

A strong observation is acquisition-eligible immediately. The tracker may enter `TRACKING` on that frame when configured for fast strong acquisition.

### 5.3 Medium/slightly weak quality

Default field requirements:

- at least 2 distinct rings;
- common-center score >= 0.60;
- radius-ratio score >= 0.65;
- coverage score >= 0.12.

A medium observation is acquisition-eligible but requires 2 consecutive consistent observations. Existing center-jump, scale-jump, velocity, acceleration, freshness, and ambiguity protections continue to apply.

### 5.4 Rejected and single-arc observations

A result with fewer than two matched rings cannot start or re-acquire tracking. A single arc may only support the existing short bounded maintenance behavior after confirmed history exists. Invalid, non-finite, or ambiguous geometry produces a miss and no target command.

## 6. Runtime state machine

### SEARCHING / LOST

1. Process the full input frame with fast ring preprocessing.
2. Run concentric-ring geometry.
3. Strong ring: immediately enter tracking.
4. Medium ring: enter confirming; a second consistent frame enters tracking.
5. Rejected or single arc: remain searching.

### CONFIRMING

Use full-frame fast ring processing. Reset confirmation on an inconsistent or missing observation. Strong quality may promote immediately.

### TRACKING / PREDICTING

1. Predict the center for the current capture timestamp.
2. Crop a dynamic ROI from the original full-resolution frame.
3. Run fast preprocessing and full-resolution ring fitting inside that ROI.
4. Translate center and arcs back to full-frame coordinates.
5. On a valid observation, update velocity and scale history and reset miss expansion.
6. On a miss, do not repeat a full-frame pipeline in the same frame; expand the next frame's ROI according to the recovery policy.

## 7. Motion-predicted dynamic ROI

Let `c_latest` be the latest real center, `v` the filtered velocity in px/s, `dt` the clamped time to the current capture, `s` the latest ring scale, and `m` the consecutive ROI miss level.

```text
c_pred = c_latest + v * dt
ring_extent = max(min_half_extent_px, outer_ring_extent_per_scale * s)
motion_extent = |v| * dt
uncertainty = prediction_padding_px + miss_expand_px * m
half_extent = (ring_extent + motion_extent + uncertainty) * safety_factor
```

Defaults favor motion tolerance rather than a very tight crop:

- minimum half extent: 72 px;
- outer-ring extent coefficient: 115 px per scale unit;
- prediction padding: 24 px;
- safety factor: 1.30;
- bounded miss expansion levels.

The rectangle is clipped to the image. Prediction horizon remains bounded by tracking configuration so stale velocity cannot move the ROI indefinitely.

## 8. ROI miss recovery

Recovery is staged to avoid latency spikes caused by `ROI pass + full-frame pass` in one frame:

- miss 0: normal dynamic ROI;
- miss 1: expanded ROI on the next frame;
- miss 2: larger expanded ROI on the next frame;
- miss 3 or tracker becomes `LOST`: full-frame ring reacquisition.

If the predicted ROI already covers almost the complete image, it is treated as a full-frame pass. A valid observation immediately resets the ROI miss level. This introduces at most a small bounded reacquisition delay while making per-frame compute more uniform.

## 9. Output geometry and gimbal angles

Every public detection center remains in full-image pixel coordinates, even when processing a crop. Ring-only results intentionally carry no board corners or board-plane solution. The gimbal converter therefore uses its existing center/intrinsics path. Existing signs, axis swap, calibration offsets, controller filtering, rate limits, and serial scheduling remain unchanged.

## 10. Configuration

Extend `detection.ring_first` (retaining the name for compatibility) with:

```yaml
ring_first:
  enabled: true
  ring_only: true
  immediate_strong_acquisition: true
  medium_confirm_frames: 2
  strong_min_arcs: 3
  strong_common_center_score: 0.75
  strong_ratio_score: 0.80
  strong_coverage_score: 0.20
  medium_min_arcs: 2
  medium_common_center_score: 0.60
  medium_ratio_score: 0.65
  medium_coverage_score: 0.12
  roi_min_half_extent_px: 72.0
  roi_outer_extent_per_scale: 115.0
  roi_prediction_padding_px: 24.0
  roi_safety_factor: 1.30
  roi_miss_expand_px: 72.0
  roi_full_frame_after_misses: 3
```

`config/jetson-local.yaml` also sets `clahe_clip_limit: 10.0`, preserving exposure 10000 us, gain 5 dB, and acquisition FPS 60. The obsolete `white_board_interval_frames` field remains accepted during migration but is ignored when `ring_only` is true.

## 11. Diagnostics and web compatibility

Each result timing map includes:

- `crop_ms`, `normalization_ms`, `ring_ms`, `white_board_ms`, `detection_ms`, `total_ms`;
- `roi_used`, `roi_fallback`, `predicted_shift_px`, `roi_width`, `roi_height`, `roi_miss_level`;
- `ring_candidate_count`;
- `white_board_ran` (always 0 in ring-only mode).

The tuning API exposes the new configuration fields. Existing debug image endpoints continue to respond; white-board-only views display a black frame rather than failing.

## 12. Tests

Automated tests must prove:

1. fast normalization skips white-mask and Sobel work while preserving ring masks;
2. geometry candidates near ratio score 0.65 reach the quality classifier;
3. a strong multi-ring observation can acquire immediately;
4. medium two-ring observations require two consistent frames;
5. one ring or arc cannot acquire;
6. no white-board detector is called in any ring-only state;
7. velocity shifts the ROI forward and increases motion margin;
8. consecutive misses expand ROI and later trigger a full-frame pass without same-frame double processing;
9. ROI-local centers and arcs map correctly to full-frame coordinates;
10. diagnostics and web schemas include the new fields;
11. center-only angle conversion works without corners;
12. Jetson-local parameters include CLAHE 10.0 and all approved camera values.

## 13. Jetson acceptance procedure

After deployment, compare an identical scene and camera setup before and after:

1. warm the process for at least 30 seconds;
2. record detection FPS, normalization time, ring time, total time, and ROI dimensions for at least 60 seconds;
3. move the vehicle continuously through expected maximum angular speed;
4. verify no single-circle false lock;
5. verify medium rings re-acquire within the bounded recovery window;
6. inspect yaw/pitch update timestamps and curve shape on the gimbal board;
7. if misses occur at high speed, increase ROI padding before lowering ring-quality thresholds further.

Success means materially higher and more uniform detection FPS, zero white-board compute in timings, stable target retention during motion, and acceptable unchanged center accuracy. Exact FPS must be reported from Jetson measurements rather than inferred from desktop tests.
