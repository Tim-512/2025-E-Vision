# Camera Tuning Dashboard Design

Date: 2026-07-14
Status: Proposed for implementation
Branch: `feature/classical-vision`

## 1. Purpose

Build a safe Jetson-hosted web dashboard for handheld testing of the Hikrobot MV-CA013-21UC. It provides live preview, explicit camera-parameter application, image-quality diagnostics, target-board overlays, synchronized captures, and named profiles. The same page works in a Jetson browser and from Windows on the same trusted LAN.

## 2. Hardware and safety boundary

- Jetson Orin NX Super; Hikrobot MV-CA013-21UC USB3, serial `00G02809155`; 8 mm F/2.8 1/1.8-inch lens.
- Fixed first-release acquisition format: 1280 x 1024, BayerRG8, stream buffer count 2.
- No gimbal-control or laser-control UI/API. During these tests the 405 nm laser stays disconnected or OFF.
- The system invariant remains: camera/serial/control fault means yaw rate 0, pitch rate 0, and laser OFF.

## 3. Scope

Included:

- Live single-stream preview, usable locally and over a trusted LAN.
- Read-only camera identity and fixed-format display.
- Draft/edit exposure, gain, acquisition FPS, auto exposure, auto gain, and auto white balance; changes require `Apply to camera`.
- Revert draft to last-good values and restore project defaults to the form without auto-applying.
- Acquisition, preview, detection, and diagnostics rates displayed separately.
- Gray and B/G/R histograms, dark/bright clipping percentages, central-ROI Laplacian focus score, and center ROI view.
- Independently switchable board outline, four corners, board center, image-center crosshair, ROI, and detection-text overlays.
- Board detection enabled by default with its own switch and rate limit.
- Browser-only preview pause; camera acquisition continues.
- Full-resolution original/overlay capture plus YAML metadata.
- Safe named parameter-profile save/load.

Excluded:

- Gimbal/laser commands; editable resolution, pixel format, offset, ROI acquisition, or buffer count.
- Burst/timed datasets, experiment sessions, and automatic reports.
- Public-internet exposure, authentication, or automatic edits to `config/default.yaml`.
- Replacing the main competition runtime.

## 4. Chosen approach

Use FastAPI plus Uvicorn on Jetson, with package-included plain HTML/CSS/JavaScript and no Node build. Add a `tuning` optional dependency group. Installation:

```bash
python -m pip install -e '.[vision,tuning]'
```

Default binding is `127.0.0.1`. LAN access requires explicit `--host 0.0.0.0`; documentation warns against public exposure.

## 5. Architecture

```text
MVS SDK -> Hikrobot camera controller -> dedicated acquisition thread
                                      -> latest full-resolution frame
                                           | preview worker <=20 FPS
                                           | diagnostics <=10 FPS
                                           | board detection <=15 FPS
                                      -> FastAPI REST + MJPEG
                                      -> Jetson/Windows browser
```

One controller owns and serializes MVS access. Downstream work is latest-only: it may skip frames and never blocks acquisition or builds an unbounded queue.

Parameter application uses reopen-on-apply for the first release. It avoids concurrent MVS writes and reuses the proven `HikrobotCamera.open()` path. A short preview interruption is acceptable.

Default limits are camera target FPS from configuration (initially 120), preview 20 FPS, board detection 15 FPS, and diagnostics 10 FPS. Server CLI options may change processing limits; the page does not change acquisition format.

The existing classical board-geometry detector is called through a small adapter. Disabling detection clears its latest result. A per-frame miss is `not detected`; an unexpected detector error is reported without stopping acquisition.

## 6. Parameter transaction

Editable values are finite, validated exposure microseconds, gain dB, acquisition FPS, and three automatic-mode booleans. Conservative server limits apply, augmented by device limits when available. Enabling auto exposure/gain disables the corresponding manual input in the browser.

Draft and applied values are distinct. Applying performs:

1. Validate the complete candidate before camera access.
2. Preserve last-good configuration.
3. Pause acquisition at a controlled boundary and close the camera.
4. Reopen with the complete candidate.
5. Require a valid frame within a bounded recovery period.
6. Publish candidate as applied only after success.
7. On failure, reopen using last-good configuration.
8. Report both apply and rollback outcomes.

`Restore project defaults` and `Revert draft` only change form values; neither silently applies settings.

## 7. Interface

Desktop layout:

```text
+------------------------------------------------------------------+
| camera state | acquisition FPS | preview FPS | detection FPS     |
+---------------------------------------------+--------------------+
|                                             | camera parameters  |
|             live preview                    | draft/applied state|
|       optional overlays and center ROI      | apply/revert       |
+---------------------------------------------+--------------------+
| histograms, clipping, focus, board quality  | capture/profiles   |
+---------------------------------------------+--------------------+
```

Panels stack on narrow screens. Preview keeps aspect ratio and is downscaled for transport; captures remain 1280 x 1024.

Pause freezes only the browser image. Status values are `Starting`, `Connected`, `Applying`, `Recovering`, `Disconnected`, and `Stopped`. Persistent messages show camera timeouts, apply/rollback errors, worker exceptions, capture failures, and profile validation failures. The UI never displays draft values as active before server confirmation.

## 8. HTTP interface

- `GET /`: dashboard.
- `GET /api/status`: connection, identity, fixed format, rates/counters, last error.
- `GET /api/parameters`: applied parameters, project defaults, validation bounds.
- `PUT /api/parameters`: validate and transactionally apply a complete set.
- `GET /api/diagnostics`: histograms, clipping, focus, detection summary.
- `GET /api/preview.mjpg`: latest-only MJPEG with overlay query flags.
- `POST /api/captures`: save latest frame pair and metadata.
- `GET /api/profiles`: list profiles.
- `GET /api/profiles/{name}`: load profile into browser draft.
- `PUT /api/profiles/{name}`: validate/save profile.
- `DELETE /api/profiles/{name}`: delete after browser confirmation.

Mutations return structured JSON. Profile names use a restricted slug and resolve strictly under the configured profile directory.

## 9. Diagnostics and overlays

At no more than 10 FPS compute 256-bin gray and B/G/R histograms; grayscale clipping at default thresholds <=5 and >=250; and Laplacian variance over the central ROI. Record ROI dimensions.

Detection data includes enabled/detected state, ordered corners, center, detector confidence/quality fields, source sequence, and result age. These metrics are observational and never auto-adjust camera parameters.

Overlay rendering is a pure operation over a frame snapshot, flags, and a compatible detection result. Stale geometry is not drawn as current; its age/source are still reported.

## 10. Captures and profiles

```text
artifacts/camera-tuning/
  captures/YYYYMMDD_HHMMSS_mmm/
    original.png
    overlay.png
    metadata.yaml
  profiles/indoor-normal.yaml
```

A capture atomically snapshots the latest full-resolution frame, diagnostics, applied parameters, and detection state. If detection came from another frame, metadata records sequence and age; stale geometry is omitted from the saved overlay.

Metadata includes timestamps, model/serial, frame sequence/timestamp, fixed format, applied parameters, rates, timeout/gap counters, clipping/focus, detection data/age, and overlay flags.

Profiles contain schema version, optional display name, and editable camera parameters only. Loading never auto-applies; saving never alters project defaults. Millisecond capture directories prevent overwrites.

## 11. Lifecycle and fault behavior

Startup validates configuration, creates the native MVS adapter/controller, opens and confirms one frame, starts latest-only workers, then serves requests.

Shutdown rejects new mutations, signals and joins workers with bounded timeouts, then stops grabbing and closes the camera exactly once.

Repeated timeouts transition to `Disconnected` after a configured threshold. A bounded reconnect loop may reopen only with last-good settings. No fault or reconnect path introduces gimbal or laser behavior.

## 12. Implementation boundaries

Expected focused units:

```text
src/ev_vision/camera/tuning.py
src/ev_vision/diagnostics.py
src/ev_vision/web/camera_tuning_app.py
src/ev_vision/web/static/camera-tuning.{html,css,js}
tools/camera_tuning_server.py
tests/camera/test_camera_tuning.py
tests/test_diagnostics.py
tests/web/test_camera_tuning_api.py
```

Exact boundaries may be refined in the plan, but camera ownership, diagnostics, persistence, HTTP transport, and frontend remain separate and testable.

## 13. Testing

Use TDD for parameter validation; successful apply; apply failure with successful/failed rollback; acquisition independence from slow consumers; diagnostics on synthetic images; overlay and stale-result behavior; safe profile paths and serialization; capture metadata; API contracts; fake-camera startup/shutdown; and package inclusion of static assets.

Jetson acceptance checks MVS serial `00G02809155`, continuous fixed-format acquisition, repeated apply/rollback, local and Windows LAN access, rate/latency counters, persistence, and handheld board tests under varied distance and lighting.

## 14. Acceptance criteria

1. One command starts the dashboard with the installed MVS SDK.
2. It works locally and, only after explicit LAN binding, from Windows.
3. Slow/disconnected browsers cannot block acquisition or grow queues.
4. Supported parameters apply explicitly and roll back reliably on failure.
5. Fixed format cannot be edited.
6. Diagnostics and rate-limited detection are visible.
7. Overlay switches and browser-only pause work.
8. Captures write non-overwriting full-resolution original/overlay/metadata files.
9. Profiles are path-safe, load without applying, and never overwrite defaults.
10. No route or UI element controls the gimbal or 405 nm laser.
11. Automated tests pass before documented Jetson hardware acceptance.
