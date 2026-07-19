from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest

import ev_vision.calibration_capture.local as module
from ev_vision.calibration_capture.local import (
    ChessboardFrameAnalysis,
    analyze_chessboard_frame,
    calibration_image_path,
    poses_are_similar,
    remove_last_saved,
    render_capture_overlay,
    save_original_frame,
)


PATTERN_SIZE = (8, 5)
IMAGE_SHAPE = (1024, 1280)


def complete_corners(
    *,
    x_range: tuple[float, float] = (280.0, 1000.0),
    y_range: tuple[float, float] = (260.0, 760.0),
    angle_deg: float = 0.0,
) -> np.ndarray:
    xs, ys = np.meshgrid(
        np.linspace(x_range[0], x_range[1], PATTERN_SIZE[0]),
        np.linspace(y_range[0], y_range[1], PATTERN_SIZE[1]),
    )
    points = np.stack((xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)
    if angle_deg:
        center = points.mean(axis=0)
        radians = np.deg2rad(angle_deg)
        rotation = np.array(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
            dtype=np.float32,
        )
        points = (points - center) @ rotation.T + center
    return points.reshape(-1, 1, 2)


def patch_detection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    found: bool = True,
    corners: np.ndarray | None = None,
    laplacian: np.ndarray | None = None,
) -> None:
    detected = complete_corners() if corners is None else corners
    monkeypatch.setattr(
        module.cv2,
        "findChessboardCorners",
        lambda *args, **kwargs: (found, detected if found else None),
    )
    monkeypatch.setattr(
        module.cv2,
        "cornerSubPix",
        lambda gray, input_corners, *args: input_corners,
    )
    response = np.array([0.0, 100.0]) if laplacian is None else laplacian
    monkeypatch.setattr(module.cv2, "Laplacian", lambda *args, **kwargs: response)


def test_analysis_is_immutable() -> None:
    analysis = ChessboardFrameAnalysis(
        False, None, 0.0, 0.0, False, None, False, "corners_not_found"
    )
    with pytest.raises(FrozenInstanceError):
        analysis.reason = "ready"  # type: ignore[misc]


def test_complete_sharp_centered_board_is_saveable(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_detection(monkeypatch)

    result = analyze_chessboard_frame(
        np.zeros((*IMAGE_SHAPE, 3), np.uint8), pattern_size=PATTERN_SIZE
    )

    assert result.found
    assert result.corners is not None and result.corners.shape == (40, 2)
    assert result.focus_score == pytest.approx(2500.0)
    assert result.coverage_fraction > 0.02
    assert result.edge_margin_ok
    assert result.pose_signature is not None
    assert result.pose_signature[:2] == pytest.approx((0.5, 0.498046875))
    assert result.pose_signature[2] == pytest.approx(result.coverage_fraction)
    assert result.pose_signature[3] == pytest.approx(0.0)
    assert result.save_allowed
    assert result.reason == "ready"


def test_analysis_accepts_grayscale_and_uses_required_chessboard_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def find(gray: np.ndarray, pattern_size: tuple[int, int], flags: int):
        calls.update(gray=gray, pattern_size=pattern_size, flags=flags)
        return True, complete_corners()

    monkeypatch.setattr(module.cv2, "findChessboardCorners", find)
    monkeypatch.setattr(module.cv2, "cornerSubPix", lambda gray, corners, *args: corners)
    monkeypatch.setattr(module.cv2, "Laplacian", lambda *args, **kwargs: np.array([0.0, 100.0]))
    image = np.zeros(IMAGE_SHAPE, np.uint8)

    result = analyze_chessboard_frame(image, pattern_size=PATTERN_SIZE)

    assert result.save_allowed
    assert calls["gray"] is image
    assert calls["pattern_size"] == PATTERN_SIZE
    assert calls["flags"] == (
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )


@pytest.mark.parametrize(
    ("found", "corners"),
    [
        (False, None),
        (True, complete_corners()[:-1]),
    ],
    ids=["missing", "wrong-count"],
)
def test_missing_or_wrong_corner_count_is_not_saveable(
    monkeypatch: pytest.MonkeyPatch,
    found: bool,
    corners: np.ndarray | None,
) -> None:
    patch_detection(monkeypatch, found=found, corners=corners)

    result = analyze_chessboard_frame(
        np.zeros((*IMAGE_SHAPE, 3), np.uint8), pattern_size=PATTERN_SIZE
    )

    assert not result.found
    assert result.corners is None
    assert not result.save_allowed
    assert result.reason == "corners_not_found"


def test_focus_below_40_is_not_saveable(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_detection(monkeypatch, laplacian=np.array([0.0, 10.0]))

    result = analyze_chessboard_frame(
        np.zeros((*IMAGE_SHAPE, 3), np.uint8), pattern_size=PATTERN_SIZE
    )

    assert result.focus_score == pytest.approx(25.0)
    assert not result.save_allowed
    assert result.reason == "focus_too_low"


def test_coverage_below_two_percent_is_not_saveable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_detection(
        monkeypatch,
        corners=complete_corners(x_range=(590.0, 690.0), y_range=(470.0, 550.0)),
    )

    result = analyze_chessboard_frame(
        np.zeros((*IMAGE_SHAPE, 3), np.uint8), pattern_size=PATTERN_SIZE
    )

    assert result.coverage_fraction < 0.02
    assert result.edge_margin_ok
    assert not result.save_allowed
    assert result.reason == "coverage_too_small"


def test_corner_within_three_percent_of_edge_is_not_saveable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_detection(
        monkeypatch,
        corners=complete_corners(x_range=(20.0, 500.0), y_range=(250.0, 750.0)),
    )

    result = analyze_chessboard_frame(
        np.zeros((*IMAGE_SHAPE, 3), np.uint8), pattern_size=PATTERN_SIZE
    )

    assert result.coverage_fraction > 0.02
    assert not result.edge_margin_ok
    assert not result.save_allowed
    assert result.reason == "corners_too_close_to_edge"


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((1280,), np.uint8),
        np.zeros((1, 1024, 1280, 3), np.uint8),
        np.zeros((1024, 1280, 4), np.uint8),
        np.zeros((0, 1280), np.uint8),
    ],
)
def test_invalid_image_shape_is_rejected(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="grayscale or BGR"):
        analyze_chessboard_frame(image, pattern_size=PATTERN_SIZE)


def test_pose_similarity_uses_center_area_and_wrapped_rotation() -> None:
    baseline = (0.50, 0.50, 0.20, 175.0)
    assert poses_are_similar(baseline, (0.54, 0.53, 0.22, -175.0))
    assert not poses_are_similar(baseline, (0.59, 0.50, 0.20, 175.0))
    assert not poses_are_similar(baseline, (0.50, 0.50, 0.26, 175.0))
    assert not poses_are_similar(baseline, (0.50, 0.50, 0.20, -160.0))


def test_calibration_image_path_uses_zero_padded_name(tmp_path: Path) -> None:
    assert calibration_image_path(tmp_path, 7) == tmp_path / "calibration-007.png"


def test_saved_image_keeps_original_pixels_and_remove_is_session_scoped(
    tmp_path: Path,
) -> None:
    rows = np.arange(IMAGE_SHAPE[0], dtype=np.uint16)[:, None]
    cols = np.arange(IMAGE_SHAPE[1], dtype=np.uint16)[None, :]
    image = np.stack(
        (
            np.broadcast_to(cols % 256, IMAGE_SHAPE),
            np.broadcast_to(rows % 256, IMAGE_SHAPE),
            (rows + cols) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    unrelated = tmp_path / "preexisting.png"
    unrelated.write_bytes(b"keep")

    path = save_original_frame(tmp_path / "captures", image, index=1)
    session_paths = [path]

    loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert loaded is not None and loaded.shape == (1024, 1280, 3)
    assert np.array_equal(loaded, image)
    assert remove_last_saved(session_paths) == path
    assert session_paths == []
    assert not path.exists()
    assert unrelated.read_bytes() == b"keep"
    assert remove_last_saved(session_paths) is None


def test_failed_imwrite_raises_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = np.zeros((20, 30, 3), np.uint8)
    calls: list[tuple[str, np.ndarray]] = []

    def imwrite(path: str, candidate: np.ndarray) -> bool:
        calls.append((path, candidate))
        return False

    monkeypatch.setattr(module.cv2, "imwrite", imwrite)

    with pytest.raises(OSError, match="failed to write calibration image"):
        save_original_frame(tmp_path / "captures", image, index=12)

    assert (tmp_path / "captures").is_dir()
    assert calls == [(str(tmp_path / "captures" / "calibration-012.png"), image)]


def test_render_overlay_returns_copy_and_reports_capture_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((200, 300, 3), np.uint8)
    corners = complete_corners(x_range=(40.0, 260.0), y_range=(50.0, 150.0)).reshape(-1, 2)
    analysis = ChessboardFrameAnalysis(
        True,
        corners,
        125.5,
        0.18,
        True,
        (0.5, 0.5, 0.18, 0.0),
        True,
        "ready",
    )
    draw_calls: list[tuple[tuple[int, int], tuple[int, ...], bool]] = []
    text_lines: list[str] = []

    def draw(rendered, pattern_size, drawn_corners, found):
        draw_calls.append((pattern_size, drawn_corners.shape, found))

    def put_text(rendered, text, *args, **kwargs):
        text_lines.append(text)
        return rendered

    monkeypatch.setattr(module.cv2, "drawChessboardCorners", draw)
    monkeypatch.setattr(module.cv2, "putText", put_text)

    rendered = render_capture_overlay(image, analysis, saved_count=3, duplicate_warning=True)

    assert rendered is not image
    assert np.array_equal(rendered, image)
    assert draw_calls == [((8, 5), (40, 1, 2), True)]
    combined = "\n".join(text_lines)
    assert "saved=3" in combined
    assert "focus=125.5" in combined
    assert "coverage=0.180" in combined
    assert "status=ready" in combined
    assert "SPACE save" in combined and "R remove" in combined and "Q/ESC quit" in combined
    assert "similar" in combined
