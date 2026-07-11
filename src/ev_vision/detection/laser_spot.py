from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ev_vision.models import LaserObservation


@dataclass
class LaserSpotDetector:
    background_alpha: float = 0.01
    minimum_peak: float = 28.0
    minimum_area: int = 2
    maximum_area: int = 500
    _background: np.ndarray | None = field(default=None, init=False, repr=False)

    def update_background(self, image: np.ndarray) -> None:
        frame = np.asarray(image, dtype=np.float32)
        if self._background is None:
            self._background = frame.copy()
            return
        cv2.accumulateWeighted(frame, self._background, self.background_alpha)

    @staticmethod
    def _signal(image: np.ndarray, background: np.ndarray) -> np.ndarray:
        current = image.astype(np.float32)
        difference = np.maximum(current - background, 0.0)
        brightness = difference.max(axis=2)
        blue = difference[:, :, 0]
        red = difference[:, :, 2]
        chromatic = np.maximum(blue, red)
        return 0.65 * brightness + 0.35 * chromatic

    def detect(
        self,
        image: np.ndarray,
        *,
        captured_ns: int,
        roi: tuple[int, int, int, int] | None = None,
    ) -> LaserObservation | None:
        if self._background is None:
            return None
        if image.shape != self._background.shape:
            raise ValueError("image shape must match the registered background")

        height, width = image.shape[:2]
        if roi is None:
            x0, y0, x1, y1 = 0, 0, width, height
        else:
            x0, y0, x1, y1 = roi
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(width, int(x1)), min(height, int(y1))
            if x1 <= x0 or y1 <= y0:
                return None

        signal = self._signal(image[y0:y1, x0:x1], self._background[y0:y1, x0:x1])
        peak = float(signal.max(initial=0.0))
        if peak < self.minimum_peak:
            return None
        threshold = max(self.minimum_peak, peak * 0.28)
        mask = (signal >= threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        best: tuple[float, float, float, float] | None = None
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not self.minimum_area <= area <= self.maximum_area:
                continue
            component = labels == label
            values = np.where(component, signal, 0.0)
            component_peak = float(values.max(initial=0.0))
            total = float(values.sum())
            if total <= 0.0:
                continue
            yy, xx = np.nonzero(component)
            weights = values[yy, xx].astype(np.float64) ** 1.5
            weight_sum = float(weights.sum())
            cx = float((xx * weights).sum() / weight_sum + x0)
            cy = float((yy * weights).sum() / weight_sum + y0)
            compactness = min(1.0, area / 8.0)
            score = component_peak * (0.75 + 0.25 * compactness)
            confidence = float(np.clip((component_peak - self.minimum_peak) / 80.0, 0.0, 1.0))
            candidate = (score, cx, cy, confidence)
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None:
            return None
        _, cx, cy, confidence = best
        return LaserObservation(captured_ns=captured_ns, position_px=(cx, cy), confidence=confidence)
