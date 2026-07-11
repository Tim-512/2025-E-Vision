"""Classical image detectors used by the precision tracking pipeline."""

from ev_vision.detection.board_geometry import BoardGeometryDetector
from ev_vision.detection.laser_spot import LaserSpotDetector

__all__ = ["BoardGeometryDetector", "LaserSpotDetector"]
