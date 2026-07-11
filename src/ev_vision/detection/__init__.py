"""Classical image detectors used by the precision tracking pipeline."""

from ev_vision.detection.board_geometry import BoardGeometryDetector
from ev_vision.detection.laser_spot import LaserSpotDetector
from ev_vision.detection.yolo_board import BoardSearchResult, YoloBoardDetector

__all__ = ["BoardGeometryDetector", "BoardSearchResult", "LaserSpotDetector", "YoloBoardDetector"]
