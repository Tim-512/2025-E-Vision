from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ev_vision.calibration import Calibration, ChessboardCalibrationResult


TOOL_PATH = Path(__file__).parents[2] / "tools" / "calibrate_camera.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("calibrate_camera", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calibration(rms: float = 0.31) -> Calibration:
    return Calibration(
        image_size=(1280, 1024),
        camera_matrix=np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 512.0], [0.0, 0.0, 1.0]]),
        distortion=np.zeros(5),
        rms_px=rms,
    )


def test_cli_prints_loaded_and_usable_counts(tmp_path, monkeypatch, capsys) -> None:
    module = load_tool()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(12):
        (image_dir / f"capture-{index:03d}.png").write_bytes(b"image")
    monkeypatch.setattr(module.cv2, "imread", lambda *a: np.zeros((1024, 1280, 3), np.uint8))
    solve_args = {}

    def solve(frames, **kwargs):
        solve_args.update(kwargs)
        return ChessboardCalibrationResult(calibration(), 12, 10, 2)

    monkeypatch.setattr(module, "solve_chessboard_with_report", solve)
    monkeypatch.setattr(module, "save_calibration", lambda path, value: Path(path).write_text("saved"))
    output = tmp_path / "camera.yaml"

    assert module.main([
        str(image_dir), "--output", str(output),
        "--square-mm", "22", "--max-rms", "0.5",
    ]) == 0
    assert solve_args == {"pattern_size": (8, 5), "square_size_mm": 22.0}
    assert capsys.readouterr().out == (
        f"images=12 usable_poses=10 rejected=2 rms_px=0.3100 output={output}\n"
    )
