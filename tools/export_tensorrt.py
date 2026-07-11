from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a YOLO model to an FP16 TensorRT engine on Jetson")
    parser.add_argument("model", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace", type=float, default=2.0)
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install ultralytics in the Jetson export environment") from exc
    model = YOLO(str(args.model))
    result = model.export(format="engine", half=True, imgsz=args.imgsz, workspace=args.workspace)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
