from __future__ import annotations

import sys

import numpy as np

from ev_vision.detection.yolo_board import RawDetection, YoloBoardDetector, letterbox


class FakeInference:
    input_size = (640, 640)

    def __init__(self, detections):
        self.detections = detections
        self.last_tensor = None

    def infer(self, tensor: np.ndarray):
        self.last_tensor = tensor
        return self.detections


def test_letterbox_restores_coordinates_to_original_image() -> None:
    image = np.zeros((1024, 1280, 3), np.uint8)
    _, transform = letterbox(image, (640, 640))
    original_box = np.array([256.0, 102.0, 1024.0, 922.0])
    model_box = transform.to_model_box(original_box)
    backend = FakeInference([RawDetection(tuple(model_box), confidence=0.9, class_id=0)])

    result = YoloBoardDetector(backend, confidence_threshold=0.5).detect(image)

    assert result is not None
    assert np.allclose(result.xyxy_px, original_box, atol=1e-4)
    assert backend.last_tensor.shape == (1, 3, 640, 640)
    assert backend.last_tensor.dtype == np.float32


def test_confidence_filtering_and_best_candidate_selection() -> None:
    image = np.zeros((480, 640, 3), np.uint8)
    backend = FakeInference(
        [
            RawDetection((10, 10, 100, 100), confidence=0.49, class_id=0),
            RawDetection((20, 20, 220, 200), confidence=0.76, class_id=0),
            RawDetection((30, 30, 180, 170), confidence=0.92, class_id=1),
            RawDetection((40, 40, 300, 260), confidence=0.88, class_id=0),
        ]
    )

    result = YoloBoardDetector(backend, confidence_threshold=0.5, board_class_id=0).detect(image)

    assert result is not None
    assert result.confidence == 0.88


def test_detection_is_clipped_and_degenerate_box_is_rejected() -> None:
    image = np.zeros((100, 200, 3), np.uint8)
    _, transform = letterbox(image, (640, 640))
    outside = transform.to_model_box(np.array([-20, -10, 230, 120], dtype=float))
    backend = FakeInference([RawDetection(tuple(outside), confidence=0.9, class_id=0)])
    result = YoloBoardDetector(backend).detect(image)
    assert result is not None
    assert result.xyxy_px == (0.0, 0.0, 199.0, 99.0)

    point = transform.to_model_box(np.array([30, 30, 30, 30], dtype=float))
    backend = FakeInference([RawDetection(tuple(point), confidence=0.99, class_id=0)])
    assert YoloBoardDetector(backend).detect(image) is None


def test_empty_detections_return_none_without_tensorrt_import() -> None:
    sys.modules.pop("tensorrt", None)
    detector = YoloBoardDetector(FakeInference([]))

    assert detector.detect(np.zeros((240, 320, 3), np.uint8)) is None
    assert "tensorrt" not in sys.modules

def test_tensorrt_backend_lazily_loads_engine_and_converts_results(monkeypatch) -> None:
    import types

    class ArrayValue:
        def __init__(self, values):
            self.values = np.asarray(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Boxes:
        xyxy = ArrayValue([[11.0, 22.0, 111.0, 222.0]])
        conf = ArrayValue([0.87])
        cls = ArrayValue([0])

    class FakeModel:
        def __init__(self, path, task):
            assert path == "board.engine"
            assert task == "detect"

        def predict(self, tensor, verbose, device):
            assert tensor.shape == (1, 3, 640, 640)
            assert verbose is False
            assert device == 0
            return [types.SimpleNamespace(boxes=Boxes())]

    module = types.SimpleNamespace(YOLO=FakeModel)
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    from ev_vision.detection.yolo_board import TensorRTBackend

    backend = TensorRTBackend("board.engine", device=0)
    detections = list(backend.infer(np.zeros((1, 3, 640, 640), np.float32)))

    assert detections == [RawDetection((11.0, 22.0, 111.0, 222.0), 0.87, 0)]
