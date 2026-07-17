from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text("utf-8")
ACCEPTANCE_PATH = ROOT / "docs" / "jetson-classical-vision-acceptance.md"
ACCEPTANCE = ACCEPTANCE_PATH.read_text("utf-8") if ACCEPTANCE_PATH.exists() else ""


def test_readme_does_not_claim_jetson_controls_laser() -> None:
    assert "由 Jetson 控制开关" not in README
    assert "激光" in README
    assert "上电" in README
    assert "V2" in README
    assert "软件不能" in README or "software cannot" in README


def test_acceptance_contains_exact_environment_and_launch_commands() -> None:
    for command in (
        "cd ~/2025-E-Vision/2025-E-Vision",
        "conda activate 2025-e-vision",
        'export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"',
        'export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
        "export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "python tools/camera_tuning_server.py --config config/default.yaml --host 0.0.0.0 --port 8000",
    ):
        assert command in ACCEPTANCE


def test_acceptance_contains_full_clean_test_commands() -> None:
    for text in (
        'PYTHONPYCACHEPREFIX="$(mktemp -d)"',
        "python -m compileall -q src tests tools",
        'TEST_TMP="$(mktemp -d)"',
        "-p no:cacheprovider",
        '--basetemp "$TEST_TMP"',
    ):
        assert text in ACCEPTANCE


def test_acceptance_pins_tracking_and_safety_checks() -> None:
    for text in (
        "3 frames",
        "150 ms",
        "LOST",
        "FUSED_PARTIAL",
        "FULL_BOARD",
        "five classical debug images",
        "measured_detection_fps",
        "software cannot make it safe",
    ):
        assert text in ACCEPTANCE


def test_acceptance_requires_measured_results_and_single_camera_owner() -> None:
    for text in (
        "PUT /api/detection/config",
        "must not restart or close MVS acquisition",
        "10, 15, 20, 30 and 50 ms",
        "Do not claim the 15 FPS design target unless measured on Jetson",
        "pkill -f MvViewer || true",
    ):
        assert text in ACCEPTANCE


def test_acceptance_names_all_five_debug_products() -> None:
    for filename in (
        "normalized-gray.png",
        "white-mask.png",
        "edge-mask.png",
        "ring-arcs.png",
        "candidate-scores.png",
    ):
        assert filename in ACCEPTANCE
