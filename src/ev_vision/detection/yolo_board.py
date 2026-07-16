from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class RawDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


@dataclass(frozen=True)
class BoardSearchResult:
    xyxy_px: tuple[float, float, float, float]
    confidence: float

    @property
    def center_px(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy_px
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int

    def to_model_box(self, xyxy: np.ndarray) -> np.ndarray:
        box = np.asarray(xyxy, dtype=np.float64).copy()
        box[[0, 2]] = box[[0, 2]] * self.scale + self.pad_x
        box[[1, 3]] = box[[1, 3]] * self.scale + self.pad_y
        return box

    def to_original_box(self, xyxy: np.ndarray) -> np.ndarray:
        box = np.asarray(xyxy, dtype=np.float64).copy()
        box[[0, 2]] = (box[[0, 2]] - self.pad_x) / self.scale
        box[[1, 3]] = (box[[1, 3]] - self.pad_y) / self.scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0.0, self.original_width - 1.0)
        box[[1, 3]] = np.clip(box[[1, 3]], 0.0, self.original_height - 1.0)
        return box


class InferencePort(Protocol):
    input_size: tuple[int, int]

    def infer(self, tensor: np.ndarray) -> Iterable[RawDetection]: ...


def letterbox(
    image: np.ndarray,
    size: tuple[int, int],
    *,
    fill: int = 114,
) -> tuple[np.ndarray, LetterboxTransform]:
    target_width, target_height = size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    output = np.full((target_height, target_width, image.shape[2]), fill, dtype=image.dtype)
    output[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    transform = LetterboxTransform(scale, float(pad_left), float(pad_top), width, height)
    return output, transform


class YoloBoardDetector:
    def __init__(
        self,
        backend: InferencePort,
        *,
        confidence_threshold: float = 0.5,
        board_class_id: int = 0,
        max_candidates: int = 3,
    ) -> None:
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.board_class_id = board_class_id
        self.max_candidates = max_candidates

    @staticmethod
    def _tensor(image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    def detect_candidates(self, image: np.ndarray) -> tuple[BoardSearchResult, ...]:
        prepared, transform = letterbox(image, self.backend.input_size)
        valid: list[BoardSearchResult] = []
        for detection in self.backend.infer(self._tensor(prepared)):
            if detection.class_id != self.board_class_id:
                continue
            if not math.isfinite(detection.confidence):
                continue
            if detection.confidence < self.confidence_threshold:
                continue
            model_box = np.asarray(detection.xyxy, dtype=np.float64)
            if not np.isfinite(model_box).all():
                continue
            box = transform.to_original_box(model_box)
            if not np.isfinite(box).all():
                continue
            x0, y0, x1, y1 = map(float, box)
            if x1 <= x0 or y1 <= y0:
                continue
            valid.append(BoardSearchResult((x0, y0, x1, y1), float(detection.confidence)))
        valid.sort(key=lambda item: item.confidence, reverse=True)
        return tuple(valid[: self.max_candidates])

    def detect(self, image: np.ndarray) -> BoardSearchResult | None:
        candidates = self.detect_candidates(image)
        return candidates[0] if candidates else None


class UltralyticsBackend:
    """Lazy Ultralytics adapter for portable YOLO model artifacts."""

    _SUPPORTED_ARTIFACTS = frozenset({"pt", "onnx", "engine"})

    def __init__(
        self,
        artifact_path: str,
        *,
        input_size: tuple[int, int] = (640, 640),
        device: int | str = 0,
    ) -> None:
        artifact_kind = Path(artifact_path).suffix.lower().lstrip(".")
        if artifact_kind not in self._SUPPORTED_ARTIFACTS:
            raise ValueError("YOLO artifact must use a .pt, .onnx, or .engine suffix")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required to load YOLO model artifacts") from exc

        self.input_size = input_size
        self.device = device
        self.artifact_kind = artifact_kind
        self._model = YOLO(artifact_path, task="detect")

    def infer(self, tensor: np.ndarray) -> Iterable[RawDetection]:
        results = self._model.predict(tensor, verbose=False, device=self.device)
        detections: list[RawDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confidence = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for box, score, class_id in zip(xyxy, confidence, classes):
                detections.append(
                    RawDetection(
                        tuple(float(value) for value in box),
                        float(score),
                        int(class_id),
                    )
                )
        return detections


TensorRTBackend = UltralyticsBackend
