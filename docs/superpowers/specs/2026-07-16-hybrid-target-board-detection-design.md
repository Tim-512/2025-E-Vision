# Hybrid Target-Board Detection Design

Date: 2026-07-16
Status: Approved design; pending written-spec review
Branch: `feature/classical-vision`

## 1. Purpose

Replace the fragile full-frame dark-threshold board search with a one-day, competition-oriented hybrid pipeline for the fixed 2025 E-problem target board:

1. a lightweight one-class YOLO detector finds the approximate board region in clutter;
2. classical geometry inside the detected ROI extracts and refines the four board corners;
3. temporal tracking and explicit validity gates suppress isolated misses and false positives;
4. the existing dashboard exposes detections, scores, failure reasons, and synchronized debug captures;
5. only validated, current tracking results may be sent to the gimbal, and no detector output directly enables the laser.

The immediate target is a first Jetson-testable system within one day. The user will collect and manually annotate 150-300 images. Training runs on a Windows RTX 4070 Laptop GPU with 8 GB VRAM; inference runs on Jetson Orin NX Super.

## 2. Confirmed constraints

- Camera: Hikrobot MV-CA013-21UC USB3, serial `00G02809155`.
- Lens: 8 mm, F/2.8, 1/1.8-inch.
- Acquisition format remains 1280 x 1024 BayerRG8 with stream buffer count 2.
- Target board pattern and physical size are fixed and match the current test board.
- There is one true board, but other rectangular boards or objects may be present.
- Desired result update rate is 15-20 Hz.
- Current static-test parameters of 50,000 microseconds and 14 dB produce about 20 acquisition FPS. Motion tests must reduce exposure to control blur.
- The gimbal is not yet available. Gimbal input data may be simulated while vision is developed.
- The 405 nm laser remains physically disconnected or OFF throughout collection, training, and detector testing.
- Existing branch and worktree remain in use; this design extends the current code rather than replacing it.

## 3. Evidence and problem diagnosis

The supplied real capture contains a complete target board, yet the current detector returns no detection without reporting a thread or camera error.

Capture diagnostics:

- exposure: 50,000 microseconds;
- gain: 14 dB;
- acquisition: about 19.86 FPS;
- detection: about 7.05 FPS;
- dark clipping: about 0.036 percent;
- bright clipping: about 0.318 percent;
- focus score: about 54.94;
- detector result: not detected, no detector exception.

The existing normalized dark mask marks about 43.9 percent of the image as foreground. With external-contour retrieval, background shadows, clutter, and the target frame merge into a full-image, non-convex contour with seven approximate vertices. The target outline therefore never becomes the independent convex quadrilateral expected by `BoardGeometryDetector`.

A Canny analysis of the same capture finds a plausible target quadrilateral occupying about 16.7 percent of the image, with approximate corners:

```text
(938, 315)
(384, 343)
(412, 728)
(962, 702)
```

The target is neither too small nor outside the frame. The primary failure is full-frame dark segmentation in a cluttered scene. Edge geometry remains useful when restricted to a model-proposed ROI, but changing only thresholds or contour retrieval is not a sufficient competition solution.

## 4. Considered approaches

### 4.1 Strengthen the classical full-frame detector

Possible changes include multi-threshold Canny, contour hierarchy, line intersections, and richer scoring.

Advantages:

- no data collection or training;
- simple deployment and low inference cost.

Disadvantages:

- the real capture already demonstrates sensitivity to clutter and illumination;
- full-frame edges introduce many rectangular distractors;
- tuning for the current room is unlikely to generalize safely to the competition environment.

Use: retain as a diagnostic and possible fallback signal, not as the primary competition-valid detector.

### 4.2 One-class detector plus ROI geometry

A lightweight detector finds the board bounding box. Classical geometry then computes precise corners inside an expanded ROI.

Advantages:

- bounding-box annotation is feasible in the available day;
- the model handles background variation while geometry preserves precision;
- integrates with the existing YOLO adapter and geometry code;
- suitable for Jetson deployment.

Disadvantages:

- requires representative real images and negative samples;
- the model box cannot itself be treated as an accurate board boundary;
- Jetson export and backend compatibility require validation.

Decision: implement this approach now.

### 4.3 Keypoint or instance-segmentation model plus geometry

A model directly predicts four corners or a board mask, followed by geometric refinement.

Advantages:

- best long-term match for homography and difficult backgrounds;
- less dependence on ROI contour completeness.

Disadvantages:

- more expensive annotation and model integration;
- too risky for the current one-day deadline.

Use: preserve interfaces so this can replace the bounding-box front end later.

## 5. System architecture

```text
Hikrobot camera acquisition
        |
        +--> latest full-resolution frame --> preview / diagnostics / capture
        |
        v
lightweight one-class YOLO board detector
        |
        +--> no candidate or model error --> invalid result, laser permission false
        |
        v
candidate ranking and top-N limit
        |
        v
expanded ROI geometry refinement
        |-- multi-threshold edges
        |-- contour hierarchy / quadrilateral extraction
        |-- convexity, side, area, aspect, boundary checks
        |-- edge-support and internal-frame structure scores
        |-- subpixel corner refinement
        |
        +--> no valid unique quadrilateral --> invalid result with reason
        |
        v
temporal tracker and freshness gate
        |
        v
ordered corners, board center, homography, target-plane coordinates
        |
        +--> dashboard status and debug overlay
        +--> gimbal result message
        +--> safety logic input; never a direct laser command
```

Processing remains latest-only. Detection skips obsolete frames instead of building a queue. Camera acquisition, diagnostics, detection, and browser preview remain independently rate-limited.

## 6. Detection data model

### 6.1 Model candidate

Each model candidate contains:

```text
xyxy_px
confidence
class_id
```

`YoloBoardDetector` will expose all valid candidates through `detect_candidates(image)`. The existing single-result `detect(image)` behavior may remain as a compatibility wrapper selecting the first candidate.

### 6.2 Geometry result

ROI refinement returns:

```text
corners_px               ordered TL, TR, BR, BL in full-image coordinates
center_px
geometry_score
edge_support_score
structure_score
roi_xyxy_px
failure_reason
optional debug images
```

Debug images are generated or encoded only when requested for the latest frame or a capture; normal operation must not continuously write them to disk.

### 6.3 Hybrid result

The hybrid detector publishes:

```text
timestamp_ns
source_sequence
detected
target_valid
tracking_state
model_backend
model_confidence
geometry_score
edge_support_score
structure_score
combined_score
candidate_count
corners_px
center_px
failure_reason
inference_ms
geometry_ms
total_ms
```

A model box alone never produces `target_valid = true`.

## 7. Model candidate handling

The one-class model uses class ID 0, named `target_board`. Initial input size is 640 x 640 with letterboxing and mapping back to the 1280 x 1024 source image.

Candidate processing:

1. discard wrong-class, non-finite, degenerate, or below-threshold boxes;
2. sort by model confidence;
3. retain at most the configured top three candidates;
4. expand each box by an initial 8 percent on each dimension and clip to image boundaries;
5. run ROI geometry on each retained candidate;
6. reject candidates that fail hard geometry gates;
7. rank survivors using explicit model, geometry, structure, and temporal scores;
8. reject the frame as ambiguous if the best and second-best combined scores differ by less than the ambiguity margin.

Initial combined score:

```text
0.45 * model confidence
+ 0.30 * geometry score
+ 0.15 * internal structure score
+ 0.10 * temporal consistency score
```

Weights are configuration values. They must be finite, non-negative, and normalized by the configuration layer. Aspect ratio is a soft geometric feature because perspective changes the image-space ratio.

## 8. ROI geometry refinement

Add `src/ev_vision/detection/roi_board_geometry.py`. It receives the original BGR image and one model box. It does not search outside the expanded ROI.

First-release sequence:

1. clip and extract the expanded ROI;
2. convert to gray and apply light denoising;
3. compute at least two edge maps around configured Canny thresholds;
4. retrieve contours with hierarchy instead of external contours only;
5. approximate quadrilaterals and optionally derive a quadrilateral from strong boundary lines when contours are locally broken;
6. reject non-convex, undersized, degenerate, or boundary-truncated candidates;
7. score area agreement with the model box, side lengths, opposite-side consistency, perspective-tolerant aspect, and edge support;
8. evaluate the expected dark rectangular-frame structure inside the candidate;
9. choose the best unique quadrilateral;
10. refine corners with a subpixel method when local gradients support it;
11. order corners TL, TR, BR, BL and map them to source coordinates.

Hard rejection conditions include:

- invalid or empty ROI;
- no complete convex quadrilateral;
- side length below the configured minimum;
- candidate touches the expanded ROI boundary in a way that indicates truncation;
- quadrilateral area is inconsistent with the model box;
- edge support or geometry score is below its minimum;
- corners cannot be ordered consistently.

The internal dark-frame check is a score rather than a brittle exact-template match. It must tolerate exposure, perspective, print variation, and partial shadow.

## 9. Temporal tracking and validity

Add `src/ev_vision/tracking/board_tracker.py` with these states:

```text
SEARCHING
CONFIRMING
TRACKING
PREDICTING
LOST
```

Initial state behavior:

- first accepted hybrid observation: `SEARCHING -> CONFIRMING`;
- three consecutive accepted observations: `CONFIRMING -> TRACKING`;
- any confirmation failure: return to `SEARCHING`;
- one or two missed observations while tracking: enter `PREDICTING`;
- third consecutive miss: enter `LOST`;
- new accepted observation after loss: enter `CONFIRMING`;
- excessive center or corner jump: immediately invalidate and reconfirm.

Validity rules:

| State | Position may be reported | `target_valid` | Eligible for laser safety input |
|---|---|---:|---:|
| SEARCHING | no | false | no |
| CONFIRMING | diagnostics only | false | no |
| TRACKING | yes | true | yes, subject to every other safety condition |
| PREDICTING | marked prediction only | false | no |
| LOST | no | false | no |

Every published result is freshness-checked. An observation older than the configured maximum age is invalid even if the tracker was previously in `TRACKING`. Old coordinates are never silently reused as valid coordinates.

## 10. Data collection and annotation

### 10.1 Dataset size

Collect about 240-280 images, stopping near 250 if time is tight:

| Group | Approximate count |
|---|---:|
| Normal positive images | 150 |
| Difficult positive images | 60-80 |
| Negative images | 30-50 |

Only `original.png` from dashboard captures is used for training. Overlay and mask images are excluded.

### 10.2 Positive coverage

The positive set covers:

- near, medium, and far working distances;
- all nine broad image regions, not only the center;
- frontal, left, right, upper, lower, and moderate roll/perspective views;
- bright, dim, uneven, shadowed, and mildly reflective scenes;
- mild defocus, sensor noise, hand movement, and motion blur;
- exposure groups spanning approximately 3,000-50,000 microseconds and gain approximately 6-16 dB.

Images that are almost black, severely overexposed, severely blurred, more than about one-third occluded, mostly outside the image, or unreadable to a person are excluded from the first positive set. The first release intentionally fails safe on such observations.

### 10.3 Negative coverage

Negative samples include the real scene without the target and hard distractors such as:

- blank white paper;
- black rectangular frames;
- boxes, displays, doors, windows, wall lines, and rectangular boards;
- objects with a similar overall aspect ratio;
- partial rectangles at image boundaries;
- strong shadows and dark clutter.

When the first model produces a false positive, that exact scene becomes a high-priority negative for the second training round.

### 10.4 Annotation

There is one class:

```text
0 target_board
```

Each positive image gets one horizontal bounding box tightly enclosing the complete physical board boundary. Annotators do not label only the printed black frame and do not include unnecessary background. Negative images have empty label files or no objects according to the selected YOLO dataset tooling.

### 10.5 Split

Split by capture scene, not by random individual image:

```text
train 70 percent
validation 20 percent
test 10 percent
```

Near-duplicate frames from one position and lighting setup stay in the same split. The test split explicitly includes clutter, oblique views, far views, negatives, and rectangular distractors.

## 11. Training and evaluation

Train a pretrained Nano-class YOLO detection model at 640 x 640 with one class. Larger models are not part of the first release because the dataset is small and Jetson deployment time is limited.

Permitted augmentation:

- moderate brightness, contrast, and color variation;
- small rotation, scale, translation, and perspective changes;
- light Gaussian noise, defocus, and motion blur;
- horizontal flip when physically representative.

Avoid vertical flip, extreme rotation, severe crop, unrealistic deformation, and color changes that cannot occur in the real system.

Training uses two short iterations:

1. train on the first manually annotated dataset;
2. inspect unseen real scenes, collect 30-60 targeted misses and hard negatives, correct labels, and retrain.

Evaluation is not reduced to mAP because the test set is small. First-release acceptance targets are:

- at least 95 percent detection on complete, clear target images;
- at least 85 percent on the defined difficult positives;
- no sustained high-confidence target track on negative scenes;
- boxes contain the whole board sufficiently for ROI expansion and geometry;
- the hybrid detector, not the model alone, rejects unverified rectangles.

Training runs on the Windows RTX 4070 Laptop GPU. The portable artifacts are `.pt` and `.onnx`. A TensorRT `.engine` is generated on the target Jetson and is not treated as portable across TensorRT, CUDA, JetPack, or device changes.

## 12. Project structure

Planned additions and extensions:

```text
src/ev_vision/detection/yolo_board.py            extend to multi-candidate output and portable backends
src/ev_vision/detection/roi_board_geometry.py     ROI corner extraction and scoring
src/ev_vision/detection/hybrid_board.py           model + geometry orchestration
src/ev_vision/tracking/board_tracker.py           temporal state and validity
tests/detection/test_yolo_adapter.py               extend candidate tests
tests/detection/test_roi_board_geometry.py
tests/detection/test_hybrid_board.py
tests/tracking/test_board_tracker.py
tests/web/test_detection_api.py

datasets/target_board/
  dataset.yaml
  README.md
  images/{train,val,test}/
  labels/{train,val,test}/

models/
  README.md
  target-board.pt
  target-board.onnx
  target-board.engine

tools/
  prepare_target_dataset.py
  validate_target_dataset.py
  train_target_detector.py
  evaluate_target_detector.py
  export_target_detector.py
  inspect_target_predictions.py
```

Large raw data, generated runs, caches, and temporary artifacts remain Git-ignored. Repository policy determines whether the final `.pt` or `.onnx` is committed or released separately. TensorRT engines are generated on Jetson.

## 13. Configuration

Add a `detection` section to `config/default.yaml` following this initial structure:

```yaml
detection:
  backend: hybrid

  model:
    path: models/target-board.engine
    fallback_path: models/target-board.onnx
    input_width: 640
    input_height: 640
    confidence_threshold: 0.45
    max_candidates: 3
    device: 0

  roi_geometry:
    padding_fraction: 0.08
    canny_low: 60
    canny_high: 180
    min_edge_support: 0.45
    min_geometry_score: 0.55
    expected_aspect_ratio: 0.707
    aspect_ratio_tolerance: 0.35

  candidate_scoring:
    model_weight: 0.45
    geometry_weight: 0.30
    structure_weight: 0.15
    temporal_weight: 0.10
    ambiguity_margin: 0.08

  tracking:
    confirm_frames: 3
    predict_frames: 2
    lost_frames: 3
    max_center_jump_px: 160
    max_result_age_ms: 100
```

All values receive schema and range validation. Detection configuration changes do not reopen the camera. Applying camera parameters and applying detection parameters remain separate transactions.

Backend startup order is:

1. configured TensorRT engine;
2. configured ONNX fallback;
3. diagnostic classical mode if neither model loads.

Diagnostic classical mode does not grant competition-valid tracking. If the learned model is unavailable, `target_valid` and laser permission remain false even if the classical diagnostic detector finds a shape.

## 14. Dashboard and API

Extend the existing FastAPI dashboard rather than introducing another web application.

### 14.1 Overlay

The live overlay shows:

- yellow rectangle: model candidate;
- green quadrilateral: validated refined board boundary;
- red rectangle: model candidate rejected by geometry;
- refined board center and image center;
- center-offset line;
- textual state and scores so meaning does not depend on color alone.

### 14.2 Status

Expose:

```text
model state and backend
model path
model inference FPS and latency
candidate count
model confidence
geometry, edge, structure, and combined scores
tracking state
confirmation and miss counters
target_valid
failure_reason
result age
```

Failure reasons use stable codes including:

```text
NO_MODEL_CANDIDATE
LOW_MODEL_CONFIDENCE
NO_VALID_QUADRILATERAL
INVALID_ASPECT_RATIO
LOW_EDGE_SUPPORT
LOW_INTERNAL_STRUCTURE
AMBIGUOUS_CANDIDATES
EXCESSIVE_POSITION_JUMP
STALE_FRAME
MODEL_UNAVAILABLE
MODEL_ERROR
```

### 14.3 Debug views and captures

The dashboard may display the latest:

```text
original image
model candidates
expanded ROI
ROI edges
ROI geometry candidates
final overlay
```

A synchronized capture includes:

```text
metadata.yaml
original.png
overlay.png
yolo-candidates.png
roi.png
roi-edges.png
roi-geometry.png
```

Only the latest in-memory debug products are retained. Encoding and disk writes occur on explicit capture, not continuously.

### 14.4 Detection controls

The first-release page may adjust:

- model confidence threshold;
- maximum candidates;
- ROI padding;
- Canny thresholds;
- minimum geometry and edge-support scores;
- confirmation and short-miss frame counts.

Detection settings have separate apply, restore-default, and save actions. They never trigger camera close or reopen.

### 14.5 API

Planned endpoints:

```text
GET  /api/detection/config
PUT  /api/detection/config
GET  /api/detection/status
GET  /api/detection/debug/{image_name}
POST /api/detection/model/reload
```

Model reload sets the result invalid before loading. Failure returns a specific error while camera acquisition and preview continue.

## 15. Runtime and performance

Reuse the current latest-frame architecture:

```text
acquisition thread --> latest immutable frame
                       |--> detection worker
                       |--> diagnostics worker
                       |--> preview/capture readers
```

Requirements:

- no unbounded frame queue;
- obsolete frames may be skipped;
- model loading and reload never execute in the acquisition thread;
- a per-frame model or geometry exception invalidates that result but does not stop acquisition;
- repeated model exceptions set `MODEL_ERROR`;
- closing a browser does not stop background detection;
- capture generation minimizes time holding shared locks.

Indicative performance budget:

| Stage | Target |
|---|---:|
| Letterbox and tensor preparation | 2-4 ms |
| TensorRT model inference | 5-15 ms |
| Candidate parsing | 1-2 ms |
| ROI geometry | 5-15 ms |
| Tracking and overlay | 1-3 ms |
| Total | about 15-35 ms |

The acceptance goal is an effective 15-20 Hz target-result stream. Browser preview may run slower without lowering the detector or gimbal result rate.

## 16. Gimbal interface boundary

The vision result sent to the gimbal side includes:

```text
timestamp_ms
frame_sequence
target_valid
tracking_state
confidence
center_x_px
center_y_px
offset_x_px
offset_y_px
target_x_mm
target_y_mm
corners[4]
frame_age_ms
```

The gimbal side is expected to provide:

```text
timestamp_ms
gimbal_ready
yaw_angle
pitch_angle
yaw_rate
pitch_rate
motion_state
```

The exact transport and packing may be completed with the gimbal team later. These semantic fields define what vision requires and provides. During vision-only work, gimbal inputs use safe simulated values.

## 17. Safety behavior

The detector produces observations and validity; it never directly toggles the 405 nm laser.

Laser eligibility requires all of the following outside this detector:

- `target_valid = true` and tracker state `TRACKING`;
- geometry validation passed on a fresh observation;
- gimbal feedback is fresh and reports ready/stable;
- aim error is within the configured tolerance for the configured dwell time;
- camera, model, detection, and communication are healthy;
- the global state machine explicitly allows the laser action.

Any missing, ambiguous, stale, predicted, disconnected, or error state forces laser permission false. During this implementation and its tests, the laser remains physically disconnected or OFF.

## 18. Fault handling

| Fault | Required behavior |
|---|---|
| Model file missing | preview continues; `MODEL_UNAVAILABLE`; target invalid |
| TensorRT incompatible | try ONNX fallback; otherwise target invalid |
| Model reload fails | keep acquisition alive; publish specific error; target invalid |
| Single inference error | discard frame and invalidate result |
| Repeated inference errors | enter `MODEL_ERROR` |
| Model detects but geometry fails | show rejected candidate; no valid coordinates |
| Multiple indistinguishable candidates | `AMBIGUOUS_CANDIDATES` |
| Result exceeds age limit | `STALE_FRAME`; invalidate immediately |
| Excessive target jump | invalidate and require reconfirmation |
| Camera disconnect | clear tracker and force target invalid |
| Browser disconnect | acquisition and detection continue |
| Gimbal feedback timeout | detector may continue; laser permission false |
| Process restart | start from a safe invalid state |

The governing rule is fail-safe invalidation, never continued use of an old coordinate.

## 19. Verification

### 19.1 Unit and integration tests

Add coverage for:

- multi-candidate filtering and coordinate mapping;
- ROI expansion, clipping, and coordinate restoration;
- TL/TR/BR/BL ordering;
- valid quadrilateral acceptance;
- non-convex, truncated, undersized, and low-edge candidates;
- candidate scoring and ambiguity rejection;
- three-frame confirmation;
- short prediction window and eventual loss;
- excessive jump rejection;
- stale result invalidation;
- model error and reload failure safety;
- API schema and range validation;
- model failure not stopping camera service.

### 19.2 Real-capture regression

Use the supplied real capture as a regression fixture or documented offline acceptance input. The old full-frame dark-mask path is expected to fail, while geometry given the correct ROI must recover a quadrilateral near the analyzed corners within an explicit tolerance.

### 19.3 Jetson acceptance

The Jetson test passes when:

1. the selected model backend loads;
2. camera preview remains normal;
3. a yellow model box appears on the true board;
4. successful refinement shows a green quadrilateral and center;
5. three accepted frames enter `TRACKING`;
6. removing the board leads to `LOST` without stale valid coordinates;
7. rectangular distractors do not sustain `TRACKING`;
8. effective result rate is approximately 15-20 Hz under the selected exposure;
9. the service runs for at least 20 minutes without camera loss, worker exit, or continuously growing memory;
10. the laser remains physically disconnected or OFF.

## 20. One-day delivery sequence

| Time | Activity |
|---|---|
| 0-1.5 h | collect about 250 varied positive and negative frames on Jetson |
| 1.5-3 h | organize and manually annotate on Windows |
| 3-4 h | first Nano-model training and unseen-scene validation |
| 4-5 h | collect hard negatives and failure cases |
| 5-6 h | second training iteration and export |
| 6-8 h | integrate candidates, ROI geometry, tracker, and dashboard state |
| 8-10 h | Jetson backend deployment and live debugging |
| remaining | preserve final artifacts, configuration, acceptance evidence, and fallback version |

Work can overlap: training may run while additional captures are organized or the gimbal message semantics are prepared.

## 21. Scope priority

Must complete for the first competition-oriented release:

- dataset structure and validation;
- Nano-class detector training and portable model artifact;
- multi-candidate model adapter;
- ROI geometry refinement;
- hybrid scoring and explicit failure reasons;
- temporal confirmation, loss, jump, and freshness gates;
- dashboard overlay and status essentials;
- Jetson backend loading and live validation;
- gimbal input/output semantic definitions;
- fail-safe invalidation.

May be deferred if the one-day deadline is threatened:

- exhaustive online tuning of every score weight;
- complex line-intersection fallback;
- automatic pre-labeling UI;
- full performance-history charts;
- maximum TensorRT optimization;
- learned keypoint or instance-segmentation replacement.

Geometry validation, freshness, temporal confirmation, `target_valid`, model-failure invalidation, and default laser-off behavior are not deferrable.

## 22. Future upgrade path

The front-end model interface is intentionally bounded. A future four-corner keypoint or instance-segmentation backend can replace bounding-box proposal while preserving:

- geometry refinement;
- hybrid observation structure;
- temporal tracker;
- homography and physical-coordinate calculation;
- dashboard and fault codes;
- gimbal protocol and safety gates.

This preserves the one-day MVP investment while providing a path to a more robust competition model.
