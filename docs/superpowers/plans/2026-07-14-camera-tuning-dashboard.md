# Camera Tuning Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Jetson-hosted browser dashboard that safely previews the Hikrobot camera, explicitly applies camera parameters, reports image/board diagnostics, and saves full-resolution captures and named profiles without exposing gimbal or laser controls.

**Architecture:** Keep the existing `HikrobotCamera` as the hardware adapter and add a tuning service that exclusively owns camera lifecycle, acquisition, and rollback. Pure diagnostic/overlay and persistence modules feed a FastAPI REST/MJPEG application with package-included plain HTML/CSS/JavaScript. All consumers use latest-only snapshots so browser, encoding, and detection work cannot block acquisition.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PyYAML, FastAPI, Uvicorn, pytest, HTTPX/TestClient, Hikrobot MVS Python SDK.

---

## File map

### New production files

- `src/ev_vision/tuning/__init__.py` — public tuning-domain exports.
- `src/ev_vision/tuning/models.py` — immutable parameter, status, diagnostic, detection, overlay, and snapshot data models.
- `src/ev_vision/tuning/diagnostics.py` — pure histogram, clipping, focus, and overlay functions.
- `src/ev_vision/tuning/storage.py` — path-safe YAML profiles and atomic capture persistence.
- `src/ev_vision/tuning/service.py` — camera lifecycle, latest-only acquisition/analysis, transactional apply/rollback, counters, and snapshots.
- `src/ev_vision/web/__init__.py` — web package marker.
- `src/ev_vision/web/camera_tuning_app.py` — FastAPI application, REST routes, MJPEG stream, lifecycle.
- `src/ev_vision/web/camera_tuning_server.py` — installed Jetson command entry point.
- `src/ev_vision/web/static/camera-tuning.html` — responsive dashboard structure.
- `src/ev_vision/web/static/camera-tuning.css` — dashboard layout and state styling.
- `src/ev_vision/web/static/camera-tuning.js` — polling, draft/applied state, MJPEG reload, profile and capture actions.
- `tools/camera_tuning_server.py` — thin source-tree command wrapper.

### New tests

- `tests/tuning/test_diagnostics.py` — synthetic image metrics and overlay behavior.
- `tests/tuning/test_storage.py` — profile path safety, YAML round-trip, and capture persistence.
- `tests/tuning/test_service.py` — fake-camera acquisition, apply, rollback, counters, and shutdown.
- `tests/web/test_camera_tuning_api.py` — API contracts, validation, lifecycle, and static assets.
- `tests/web/test_camera_tuning_tool.py` — parser defaults and LAN-binding behavior.
- `tests/test_tuning_packaging.py` — optional dependencies and package-data declaration.

### Modified files

- `pyproject.toml` — add `tuning` optional dependencies, command, and static package data.
- `.gitignore` — ignore local `artifacts/` and resolve existing conflict-marker damage.
- `README.md` — add install, startup, LAN, safety, and acceptance instructions.

## Fixed first-release invariants

- Acquisition format is exactly `1280 x 1024`, `BayerRG8`, stream buffer count `2`; the web page exposes these as read-only values.
- Default camera target is `120 FPS`; preview/detection/diagnostics limits are `20/15/10 FPS`.
- Every production task follows TDD: failing focused test, observed failure, minimal implementation, observed pass, then commit.
- No implementation file, route, or UI control may actuate the gimbal or 405 nm laser.
## Shared contracts

`EditableCameraParameters(exposure_us, gain_db, acquisition_fps, auto_exposure, auto_gain, auto_white_balance)` converts to a new `CameraConfig` while preserving fixed width, height, pixel format, and buffer size. `ParameterBounds` validates finite/ranged numeric inputs. `ImageDiagnostics`, `DetectionSnapshot`, `OverlayOptions`, `CaptureSnapshot`, and `RuntimeSnapshot` are frozen dataclasses. API serializers convert tuples/NumPy values to built-in JSON values.

---

### Task 1: Package skeleton, dependencies, and immutable domain models

**Files:** Create `src/ev_vision/tuning/__init__.py`, `src/ev_vision/tuning/models.py`, `tests/test_tuning_packaging.py`, `tests/tuning/__init__.py`; modify `pyproject.toml`, `.gitignore`.

- [ ] **Step 1: Write failing tests** covering `tuning = ["fastapi>=0.110", "uvicorn>=0.27"]`, `"ev_vision.web" = ["static/*"]`, fixed-format preservation with `dataclasses.replace`, and rejection of NaN/out-of-range exposure/gain/FPS.
- [ ] **Step 2: Run** `python -m pytest tests/test_tuning_packaging.py -q`; expect `ModuleNotFoundError: ev_vision.tuning`.
- [ ] **Step 3: Add package configuration:** add HTTPX to `dev`, FastAPI/Uvicorn to `tuning`, and `[tool.setuptools.package-data]`. Resolve `.gitignore` conflict markers and add `artifacts/`.
- [ ] **Step 4: Implement models:** frozen dataclasses named above; `from_camera_config`, `to_camera_config`, `to_dict`; ranges exposure 20–1,000,000 us, gain 0–24 dB, FPS 1–120; finite checks via `math.isfinite`.
- [ ] **Step 5: Run** `python -m pytest tests/test_tuning_packaging.py tests/test_config.py -q`; expect pass.
- [ ] **Step 6: Commit** `git commit -m "feat: add camera tuning domain models"`.

Implementation core:

```python
@dataclass(frozen=True)
class EditableCameraParameters:
    exposure_us: float
    gain_db: float
    acquisition_fps: float
    auto_exposure: bool
    auto_gain: bool
    auto_white_balance: bool

    def to_camera_config(self, base: CameraConfig) -> CameraConfig:
        return replace(base, exposure_us=self.exposure_us, gain_db=self.gain_db,
                       acquisition_fps=self.acquisition_fps,
                       auto_exposure=self.auto_exposure, auto_gain=self.auto_gain,
                       auto_white_balance=self.auto_white_balance)
```

### Task 2: Pure diagnostics and overlay rendering

**Files:** Create `src/ev_vision/tuning/diagnostics.py`, `tests/tuning/test_diagnostics.py`.

- [ ] **Step 1: Write failing tests** using synthetic half-dark/half-bright checkerboard images. Assert 256 bins, histogram sum equals pixels, clipping percentages, positive Laplacian variance, central ROI coordinates, compatible detection drawing, stale geometry omission, and input image immutability.
- [ ] **Step 2: Run** `python -m pytest tests/tuning/test_diagnostics.py -q`; expect missing module.
- [ ] **Step 3: Implement `compute_diagnostics`:** BGR-to-gray, centered ROI, `np.bincount(..., minlength=256)`, thresholds <=5 and >=250, `cv2.Laplacian(...).var()`.
- [ ] **Step 4: Implement `render_overlay`:** copy image; draw compatible sequence only using `cv2.polylines/circle/drawMarker/rectangle/putText`; fixed colors; show detected/not-detected/error text; never mutate source.
- [ ] **Step 5: Run** `python -m pytest tests/tuning/test_diagnostics.py tests/detection/test_board_geometry.py -q`; expect pass.
- [ ] **Step 6: Commit** `git commit -m "feat: add camera image diagnostics and overlays"`.

### Task 3: Safe named profiles and capture storage

**Files:** Create `src/ev_vision/tuning/storage.py`, `tests/tuning/test_storage.py`.

- [ ] **Step 1: Write failing tests** for profile round trip, sorted listing, `../outside` rejection, schema validation, PNG creation, YAML frame sequence/parameters, and unique capture directories.
- [ ] **Step 2: Run** `python -m pytest tests/tuning/test_storage.py -q`; expect missing module.
- [ ] **Step 3: Implement profiles** with `^[a-z0-9][a-z0-9-]{0,63}$`, schema version 1, exact parameter keys, bounds validation, `.tmp` plus atomic replace.
- [ ] **Step 4: Implement captures** under `captures/YYYYMMDD_HHMMSS_mmm[-N]`; temporary PNG/YAML writes; cleanup on error; metadata keys `schema_version`, `created_utc`, `frame`, `fixed_format`, `camera_parameters`, `runtime`, `diagnostics`, `detection`, `overlay_options`; missing values remain null.
- [ ] **Step 5: Run** `python -m pytest tests/tuning/test_storage.py -q`; expect pass.
- [ ] **Step 6: Commit** `git commit -m "feat: persist camera profiles and captures"`.

Profile schema:

```yaml
schema_version: 1
display_name: 室内普通光
parameters:
  exposure_us: 800.0
  gain_db: 6.0
  acquisition_fps: 120.0
  auto_exposure: false
  auto_gain: false
  auto_white_balance: false
```

### Task 4: Latest-only tuning service and rollback

**Files:** Create `src/ev_vision/tuning/service.py`, `tests/tuning/test_service.py`.

- [ ] **Step 1: Write fake-camera tests** for latest frame, sequence gaps, timeouts, detection disable/clear, validation before closure, successful apply, apply failure plus successful rollback, rollback failure, and idempotent stop/one close.
- [ ] **Step 2: Run** `python -m pytest tests/tuning/test_service.py -q`; expect missing service.
- [ ] **Step 3: Define protocols:** `CameraPort.open/close/read`, `CameraFactory = Callable[[CameraConfig], CameraPort]`, and detector `detect(image, captured_ns)`.
- [ ] **Step 4: Implement service state** under `threading.RLock`, operation mutex, stop events, acquisition/analysis threads, copied immutable frame snapshots, monotonic rate windows, counts, and states Starting/Connected/Applying/Recovering/Disconnected/Stopped.
- [ ] **Step 5: Implement loops:** acquisition never waits for consumers; diagnostics/detection operate on latest frame at separate 10/15 FPS limits; exceptions are reported without killing acquisition; disabling detection clears result.
- [ ] **Step 6: Implement apply transaction:** validate full candidate; stop/close; open candidate and require confirming frame; on error open/confirm last-good; publish candidate only after success; preserve both errors if rollback fails.
- [ ] **Step 7: Implement snapshots:** `runtime_snapshot`, `applied_parameters`, `latest_frame`, `latest_diagnostics`, `latest_detection`, `capture_snapshot`, `record_preview_frame`, `start`, `stop`.
- [ ] **Step 8: Run** `python -m pytest tests/tuning/test_service.py tests/camera/test_hikrobot_adapter.py tests/camera/test_latest_frame.py -q`; expect pass.
- [ ] **Step 9: Commit** `git commit -m "feat: add transactional camera tuning service"`.

Apply skeleton:

```python
candidate = self._bounds.validate(candidate)
last_good = self._applied
self._set_state("Applying")
self._stop_camera_threads(); self._close_camera()
try:
    self._open_and_confirm(candidate.to_camera_config(self._base_config))
except Exception as apply_error:
    self._set_state("Recovering", str(apply_error))
    try:
        self._open_and_confirm(last_good.to_camera_config(self._base_config))
    except Exception as rollback_error:
        self._set_state("Disconnected", f"apply failed: {apply_error}; rollback failed: {rollback_error}")
        raise ParameterApplyError(...) from apply_error
    self._set_state("Connected", f"apply failed and rolled back: {apply_error}")
    raise ParameterApplyError(...) from apply_error
self._applied = candidate
self._set_state("Connected", None)
```

### Task 5: FastAPI REST and MJPEG

**Files:** Create `src/ev_vision/web/__init__.py`, `src/ev_vision/web/camera_tuning_app.py`, `tests/web/__init__.py`, `tests/web/test_camera_tuning_api.py`.

- [ ] **Step 1: Write fake-service API tests** for `/`, status fixed format/no laser/no gimbal, complete parameter PUT, incomplete/invalid 422 without apply, apply failure 409, diagnostics serialization, profile CRUD/path errors, capture response, static assets, lifespan start/stop, and first MJPEG chunk.
- [ ] **Step 2: Run** `python -m pytest tests/web/test_camera_tuning_api.py -q`; expect missing module.
- [ ] **Step 3: Implement Pydantic request models** requiring all six parameters and optional profile display name; explicit serializers for dataclasses/tuples/NumPy values.
- [ ] **Step 4: Implement `create_camera_tuning_app(service, storage, project_defaults, preview_fps=20.0)`:** lifespan start/stop, static mount, HTML route, status/parameters/diagnostics/capture/profile endpoints. Map ValueError 422, apply error 409, missing profile 404, storage failure 500.
- [ ] **Step 5: Implement latest-only MJPEG:** fetch latest snapshot per interval, overlay, optional max-width resize, JPEG quality 85, yield multipart frame, call preview counter, sleep/retry when no frame, no queue.
- [ ] **Step 6: Run** `python -m pytest tests/web/test_camera_tuning_api.py -q`; expect pass.
- [ ] **Step 7: Commit** `git commit -m "feat: expose camera tuning web API"`.

Required endpoints: `GET /`, `/api/status`, `/api/parameters`, `/api/diagnostics`, `/api/preview.mjpg`, `/api/profiles`; `PUT /api/parameters`, `/api/profiles/{name}`; `GET/DELETE /api/profiles/{name}`; `POST /api/captures`. There are no gimbal or laser routes.

### Task 6: Responsive dashboard frontend

**Files:** Create `src/ev_vision/web/static/camera-tuning.html`, `.css`, `.js`; modify API tests.

- [ ] **Step 1: Add failing HTML test** requiring IDs `preview`, `exposure-us`, `gain-db`, `acquisition-fps`, three auto controls, apply/revert/defaults, detection/pause, capture/profile controls, gray/RGB histogram canvases, and center ROI; assert no laser/gimbal text.
- [ ] **Step 2: Run focused static test**; expect missing controls.
- [ ] **Step 3: Build Chinese semantic HTML:** top state/rates, preview and overlay switches, parameter draft/applied panel, histogram/metric cards, ROI canvas, capture/profile controls, persistent `role=status` messages. Labels use `for`; buttons use `type=button`; no inline JS.
- [ ] **Step 4: Add CSS:** two columns above 1000 px, stacked below, dark preview, green/amber/red state classes, visible focus rings, no horizontal scroll at 1366x768, system fonts.
- [ ] **Step 5: Add JS:** 500-ms status/diagnostic polling; normalized draft/applied comparison; MJPEG URL from flags; browser-only pause/resume; auto-mode manual input disabling; canvas histograms; ROI preview/fallback text; explicit apply; profile loads draft only; confirmed delete; centralized error helper.
- [ ] **Step 6: Run** `python -m pytest tests/web/test_camera_tuning_api.py -q` and `python -m compileall -q src tools`; expect pass/silence.
- [ ] **Step 7: Commit** `git commit -m "feat: add camera tuning dashboard UI"`.

### Task 7: Jetson command and native MVS wiring

**Files:** Create `src/ev_vision/web/camera_tuning_server.py`, `tools/camera_tuning_server.py`, `tests/web/test_camera_tuning_tool.py`; modify `pyproject.toml`.

- [ ] **Step 1: Write parser tests:** defaults config, serial, host 127.0.0.1, port 8000, output, rates 20/15/10, timeout 100; help says 0.0.0.0 is trusted-LAN only and laser stays OFF.
- [ ] **Step 2: Run** `python -m pytest tests/web/test_camera_tuning_tool.py -q`; expect missing module.
- [ ] **Step 3: Implement parser/main:** load config, reject non-fixed format, create one native MVS API and camera factory, detector, service, storage, app, print URL/safety warning, call Uvicorn.
- [ ] **Step 4: Add** `ev-camera-tuning = "ev_vision.web.camera_tuning_server:main"`; keep `tools/camera_tuning_server.py` as thin delegating wrapper.
- [ ] **Step 5: Run** `python -m pytest tests/web/test_camera_tuning_tool.py tests/test_tuning_packaging.py -q` and `python -m ev_vision.web.camera_tuning_server --help`; expect pass/help exit 0.
- [ ] **Step 6: Commit** `git commit -m "feat: add Jetson camera tuning server command"`.

CLI options: `--config`, `--serial`, `--host`, `--port`, `--output`, `--preview-fps`, `--detection-fps`, `--diagnostic-fps`, `--timeout-ms`, `--log-level`.

### Task 8: Documentation and Jetson acceptance

**Files:** Modify `README.md`; create `docs/camera-tuning-acceptance.md`.

- [ ] **Step 1: Document install/start:** MVS `PYTHONPATH`/`LD_LIBRARY_PATH`, `python -m pip install -e '.[vision,tuning]'`, safe local command, local URL.
- [ ] **Step 2: Document LAN:** `hostname -I`, explicit `--host 0.0.0.0`, Windows URL, no public exposure/port forwarding, 405 nm laser disconnected/OFF.
- [ ] **Step 3: Write acceptance checklist:** close MVS viewer; enumerate serial; connected/fixed-format check; draft vs apply; invalid 422; revert/default/profile; overlays/detection; capture PNG/YAML; Windows LAN; 10-minute tegrastats/dmesg; camera unplug Disconnected/no motion/no laser; recovery. Include observed/pass/notes table.
- [ ] **Step 4: Scan docs** with `grep -RniE "laser.*on|0\.0\.0\.0.*public|config/default\.yaml.*overwrite" README.md docs/camera-tuning-acceptance.md || true`; ensure no unsafe instruction.
- [ ] **Step 5: Commit** `git commit -m "docs: add camera tuning dashboard operation guide"`.

Exact Jetson commands:

```bash
cd ~/2025-E-Vision/2025-E-Vision
conda activate 2025-e-vision
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install -e '.[vision,tuning]'
ev-camera-tuning --config config/default.yaml --serial 00G02809155
```

### Task 9: Full verification and checkpoint

**Files:** Modify only files required by verified failures.

- [ ] **Step 1: Install Windows dev extras:** `& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m pip install -e '.[dev,vision,tuning]'`.
- [ ] **Step 2: Compile:** `& 'D:\Coding\anaconda\envs\2025-e-vision\python.exe' -m compileall -q src tests tools`; expect silence.
- [ ] **Step 3: Run complete isolated tests:** set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, create GUID temp path, run `python -m pytest -q -p no:cacheprovider --basetemp $testTmp`; expect all pass.
- [ ] **Step 4: Verify installed static assets** with `importlib.resources.files("ev_vision.web").joinpath("static", name).is_file()` for all three assets; run `ev-camera-tuning --help`.
- [ ] **Step 5: Run fake-service web smoke:** request root, status, diagnostics, and one MJPEG chunk; assert shutdown once.
- [ ] **Step 6: Inspect:** `git diff --check`, `git status --short`, and grep web/tuning source for laser/gimbal routes; expect none.
- [ ] **Step 7: Commit only necessary fixes** as `fix: harden camera tuning dashboard verification`; no empty commit.
- [ ] **Step 8: Record** `git status --short --branch` and `git log --oneline -10`; expect clean worktree and feature commits ahead of remote until push.

## Plan self-review result

- Every design requirement maps to Tasks 1–9: hardware/safety boundary, fixed format, explicit apply/rollback, latest-only processing, diagnostics, rate-limited detection, overlays, profiles, captures, local/LAN serving, packaging, lifecycle, testing, and Jetson acceptance.
- Scope stays limited: no gimbal, laser, dataset sessions, editable acquisition ROI, public deployment, authentication, or default-config overwrite.
- Shared types and names are consistent across domain, service, storage, API, frontend, and tests.
- Each behavior starts with a failing focused test, followed by minimal implementation, verification, and a small commit.
