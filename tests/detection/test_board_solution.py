import pytest

from ev_vision.config import BoardConfig
from ev_vision.detection.board_solution import solve_board_plane


def test_board_solution_maps_ordered_corners_to_physical_board_mm() -> None:
    corners = ((100.0, 80.0), (520.0, 100.0), (500.0, 694.0), (120.0, 674.0))
    solution = solve_board_plane(
        corners,
        board=BoardConfig(
            width_cm=21.0,
            height_cm=29.7,
            rectified_px_per_cm=40.0,
        ),
    )

    assert solution.homography_valid is True
    # The physical center maps to the projective diagonal intersection, not
    # the arithmetic mean of perspective-distorted image corners.
    assert solution.center_px == pytest.approx((309.6633, 401.8332), abs=1e-3)
    assert solution.image_to_target_mm(corners[0]) == pytest.approx((-105.0, 148.5))
    assert solution.image_to_target_mm(corners[2]) == pytest.approx((105.0, -148.5))
    assert solution.center_px is not None
    assert solution.image_to_target_mm(solution.center_px) == pytest.approx(
        (0.0, 0.0), abs=1e-5
    )


def test_degenerate_corners_return_invalid_solution_without_coordinates() -> None:
    solution = solve_board_plane(
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)),
        board=BoardConfig(),
    )

    assert solution.homography_valid is False
    assert solution.target_center_mm is None
