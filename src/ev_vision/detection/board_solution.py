from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ev_vision.config import BoardConfig
from ev_vision.geometry import HomographyError, TargetGeometry


@dataclass(frozen=True)
class BoardPlaneSolution:
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float] | None
    homography_valid: bool
    target_center_mm: tuple[float, float] | None
    _geometry: TargetGeometry | None = field(default=None, repr=False, compare=False)

    def image_to_target_mm(
        self,
        point_px: tuple[float, float],
    ) -> tuple[float, float]:
        if self._geometry is None:
            raise HomographyError("board homography is invalid")
        x_cm, y_cm = self._geometry.image_to_target_cm(point_px)
        return x_cm * 10.0, y_cm * 10.0


def solve_board_plane(
    corners_px: Sequence[Sequence[float]],
    *,
    board: BoardConfig,
) -> BoardPlaneSolution:
    try:
        corners = tuple((float(x), float(y)) for x, y in corners_px)
    except (TypeError, ValueError):
        return BoardPlaneSolution((), None, False, None, None)

    try:
        geometry = TargetGeometry.from_image_corners(
            corners,
            px_per_cm=board.rectified_px_per_cm,
            width_cm=board.width_cm,
            height_cm=board.height_cm,
        )
        center = geometry.target_cm_to_image((0.0, 0.0))
        return BoardPlaneSolution(corners, center, True, (0.0, 0.0), geometry)
    except (HomographyError, TypeError, ValueError):
        return BoardPlaneSolution(corners, None, False, None, None)
