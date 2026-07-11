from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect labeled-board candidate frames from an OpenCV camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=250)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera}")
    try:
        index = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("camera frame read failed")
            cv2.imshow("dataset collector - SPACE save, ESC quit", frame)
            key = cv2.waitKey(args.interval_ms) & 0xFF
            if key == 27:
                break
            if key == 32:
                path = args.output / f"board-{time.time_ns()}-{index:05d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"failed to save {path}")
                print(path)
                index += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
