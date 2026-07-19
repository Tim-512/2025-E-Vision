from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_PATH = ROOT / "docs" / "runbooks" / "jetson-gimbal-vision.md"
LOCAL_PREVIEW_PATH = ROOT / "docs" / "runbooks" / "jetson-local-preview.md"
README_PATH = ROOT / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_gimbal_runbook_contains_copy_paste_environment_commands() -> None:
    text = _read(RUNBOOK_PATH)
    required = (
        "cd ~/2025-E-Vision/2025-E-Vision",
        "source ~/anaconda3/etc/profile.d/conda.sh",
        "conda activate 2025-e-vision",
        "python -m pip install -e '.[vision,hardware]'",
        'export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"',
        'export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
        "sudo systemctl stop ModemManager",
        "sudo systemctl disable ModemManager",
        "/dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00",
    )
    for item in required:
        assert item in text


def test_gimbal_runbook_pins_calibration_board_and_quality_contract() -> None:
    text = _read(RUNBOOK_PATH)
    required = (
        "6 x 9 physical squares",
        "8 x 5 inner corners",
        "22 mm",
        "20-25 images",
        "at least 15 usable",
        "1280 x 1024",
        "ev-camera-calibration-capture",
        "--columns 8",
        "--rows 5",
        "--square-mm 22",
        "tools/calibrate_camera.py",
        "--max-rms 0.5",
        "config/camera_calibration.yaml",
        "rms_px",
        "camera_matrix",
        "distortion",
    )
    for item in required:
        assert item in text
    assert "dist_coeffs" not in text


def test_gimbal_runbook_contains_runtime_commands_and_tuned_values() -> None:
    text = _read(RUNBOOK_PATH)
    required = (
        "ev-gimbal-vision",
        "--display",
        "--detection-fps 50",
        "--display-fps 45",
        "exposure_us: 15000",
        "gain_db: 14.0",
        "acquisition_fps: 50",
        "fire=0",
        "physical laser",
    )
    for item in required:
        assert item in text


def test_gimbal_runbook_acceptance_stages_are_in_safe_order() -> None:
    text = _read(RUNBOOK_PATH)
    ordered_markers = (
        "Stage 0 - physical laser safety",
        "Stage 1 - software and camera verification",
        "Stage 2 - no gimbal and no calibration",
        "Stage 3 - valid calibration with motors disabled",
        "Stage 4 - static target and axis signs",
        "Stage 5 - prediction limit",
        "Stage 6 - disconnect and reconnect",
        "Stage 7 - controlled motor movement",
        "Stage 8 - shutdown safety",
    )
    positions = [text.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)


def test_gimbal_runbook_pins_fail_closed_usb_behavior() -> None:
    text = _read(RUNBOOK_PATH)
    required = (
        "all-zero",
        "tracking=0",
        "3 frames",
        "60 ms",
        "fourth predicted frame",
        "first frame after reconnect",
        "5 safe frames",
        "more than 100 ms",
        "fire=0 does not turn off the physical laser",
    )
    for item in required:
        assert item in text


def test_gimbal_runbook_contains_diagnostics_and_troubleshooting_commands() -> None:
    text = _read(RUNBOOK_PATH)
    required = (
        "lsusb",
        "ls -l /dev/serial/by-id/",
        "sudo fuser -v",
        "dmesg --follow",
        "tegrastats",
        "ps -ef | grep",
        "journalctl",
        "PYTHONPYCACHEPREFIX=\"$(mktemp -d)\"",
        "--basetemp \"$TEST_TMP\"",
    )
    for item in required:
        assert item in text


def test_runbook_is_linked_from_preview_and_readme() -> None:
    link = "docs/runbooks/jetson-gimbal-vision.md"
    assert "jetson-gimbal-vision.md" in _read(LOCAL_PREVIEW_PATH)
    assert link in _read(README_PATH)