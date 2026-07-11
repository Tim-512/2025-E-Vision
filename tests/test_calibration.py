from __future__ import annotations

import numpy as np
import pytest

from ev_vision.calibration import Calibration, CalibrationError, load_calibration, save_calibration


def calibration(rms: float = 0.31) -> Calibration:
    return Calibration(
        image_size=(1280, 1024),
        camera_matrix=np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 512.0], [0.0, 0.0, 1.0]]),
        distortion=np.array([-0.12, 0.04, 0.001, -0.002, 0.0]),
        rms_px=rms,
    )


def test_yaml_schema_round_trip(tmp_path) -> None:
    path = tmp_path / "camera.yaml"
    save_calibration(path, calibration())

    loaded = load_calibration(path)

    assert loaded.image_size == (1280, 1024)
    assert loaded.rms_px == pytest.approx(0.31)
    assert np.allclose(loaded.camera_matrix, calibration().camera_matrix)
    assert np.allclose(loaded.distortion, calibration().distortion)


def test_invalid_schema_and_unsafe_rms_are_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("image_size: [1280, 1024]\n", encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_calibration(path)
    with pytest.raises(CalibrationError, match="RMS"):
        calibration(rms=0.7).validate(max_rms_px=0.5)


def test_image_size_mismatch_is_rejected() -> None:
    with pytest.raises(CalibrationError, match="image size"):
        calibration().undistort(np.zeros((720, 1280, 3), np.uint8))


def test_undistortion_maps_are_cached(monkeypatch) -> None:
    item = calibration()
    image = np.zeros((1024, 1280, 3), np.uint8)
    calls = 0

    import ev_vision.calibration as module

    original = module.cv2.initUndistortRectifyMap

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module.cv2, "initUndistortRectifyMap", counted)
    item.undistort(image)
    item.undistort(image)
    assert calls == 1


def test_object_points_use_requested_chessboard_geometry() -> None:
    points = Calibration.chessboard_object_points((3, 2), square_size_mm=15.0)
    assert points.shape == (6, 3)
    assert np.allclose(points[:, 2], 0.0)
    assert np.allclose(points[-1], (30.0, 15.0, 0.0))
