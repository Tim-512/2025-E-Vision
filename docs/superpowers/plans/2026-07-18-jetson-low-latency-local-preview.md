# Jetson Low-Latency Local Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared and standalone Jetson-local OpenCV previews that draw only the confirmed target outline and center with minimal latency and memory use.

**Architecture:** Extract camera/detector/service construction into a web-independent runtime module. Add a minimal local renderer and latest-only GUI loop, expose a lightweight `ev-camera-preview` CLI, and optionally attach that loop to the existing dashboard command while keeping one service lifecycle.

**Tech Stack:** Python 3.10, OpenCV, NumPy, Hikrobot MVS adapter, existing `CameraTuningService`, argparse, Uvicorn for combined mode only, pytest.

---

### Task 1: Minimal local overlay and GUI loop

**Files:**
- Create: `src/ev_vision/preview/local.py`
- Create: `src/ev_vision/preview/__init__.py`
- Create: `tests/preview/test_local_preview.py`

- [ ] Write tests proving invalid/stale detections draw nothing, valid detections draw a green outline and center, resizing preserves aspect ratio, and the loop consumes only the newest sequence-matched detection frame.
- [ ] Run `python -m pytest -q -p no:cacheprovider tests/preview/test_local_preview.py` and confirm failure because the module does not exist.
- [ ] Implement `render_local_overlay`, `resize_preview`, and `run_local_preview` with injected OpenCV/time functions for headless tests.
- [ ] Re-run the test file and confirm it passes.

### Task 2: Web-independent runtime construction and diagnostics disable

**Files:**
- Create: `src/ev_vision/tuning/runtime.py`
- Modify: `src/ev_vision/tuning/service.py`
- Modify: `src/ev_vision/web/camera_tuning_server.py`
- Modify: `tests/tuning/test_service.py`
- Modify: `tests/web/test_camera_tuning_tool.py`

- [ ] Write tests proving `diagnostics_fps=None` creates no diagnostics worker while detection still runs, and proving the runtime builder uses YAML camera parameters and one native API.
- [ ] Run the focused tests and confirm the new expectations fail.
- [ ] Allow diagnostics to be disabled, extract fixed-format/detector/service construction, and keep compatibility exports in the web server module.
- [ ] Re-run focused service and web command tests.

### Task 3: Standalone low-memory command

**Files:**
- Create: `src/ev_vision/preview/cli.py`
- Modify: `pyproject.toml`
- Create: `tools/camera_local_preview.py`
- Create: `tests/preview/test_local_preview_cli.py`

- [ ] Write parser, lifecycle, configuration, and error-path tests for `ev-camera-preview`.
- [ ] Verify tests fail before implementation.
- [ ] Implement the CLI so it loads `--config`, builds a service with diagnostics disabled, prints effective camera parameters and safety text, and runs the local GUI without importing web modules.
- [ ] Re-run the focused CLI tests.

### Task 4: Optional local window on dashboard command

**Files:**
- Modify: `src/ev_vision/web/camera_tuning_app.py`
- Modify: `src/ev_vision/web/camera_tuning_server.py`
- Modify: `tests/web/test_camera_tuning_api.py`
- Modify: `tests/web/test_camera_tuning_tool.py`

- [ ] Write tests for optional FastAPI lifecycle ownership and `--local-preview` parser/main behavior.
- [ ] Confirm the tests fail.
- [ ] Add `manage_service_lifecycle`, start Uvicorn in a background thread only for combined mode, run OpenCV in the main thread, and stop both cleanly.
- [ ] Re-run focused web tests.

### Task 5: Documentation and verification

**Files:**
- Modify: `README.md`
- Create: `docs/runbooks/jetson-local-preview.md`

- [ ] Document how to save tested exposure/gain/FPS into `config/default.yaml`, standalone and combined commands, key controls, `DISPLAY` troubleshooting, process checks, and laser safety.
- [ ] Run `python -m compileall -q src tests tools` with a temporary pycache prefix.
- [ ] Run all tests with plugin autoload disabled and a temporary base directory.
- [ ] Inspect git diff/status, commit on `feature/classical-white-board-tracking`, and provide exact push/pull/Jetson commands.