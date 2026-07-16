from __future__ import annotations
import argparse
import sys
from collections.abc import Sequence

_HINT = "ultralytics is required; install with: pip install -e .[training]"

def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(_HINT) from exc
    return YOLO

def _parser():
    p = argparse.ArgumentParser(description="Train the target-board YOLO detector.")
    p.add_argument("--data", default="datasets/target_board/dataset.yaml")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0"); p.add_argument("--project", default="runs/target-board")
    p.add_argument("--name", default="target-board-v1")
    p.add_argument("--single-cls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=20250716)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--patience", type=int, default=20); p.add_argument("--batch", type=int, default=-1)
    for name, default in (("hsv-h", .015), ("hsv-s", .50), ("hsv-v", .40), ("degrees", 7.0),
        ("translate", .08), ("scale", .25), ("shear", 0.0), ("perspective", .0005),
        ("flipud", 0.0), ("fliplr", .50), ("mosaic", .20), ("mixup", 0.0),
        ("cutmix", 0.0), ("copy-paste", 0.0)):
        p.add_argument(f"--{name}", type=float, default=default)
    p.add_argument("--close-mosaic", type=int, default=10)
    return p

def main(argv: Sequence[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try: YOLO = load_yolo()
    except RuntimeError as exc:
        print(exc, file=sys.stderr); return 2
    YOLO(a.model).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, device=a.device,
        project=a.project, name=a.name, single_cls=a.single_cls, seed=a.seed,
        deterministic=a.deterministic, patience=a.patience, batch=a.batch,
        hsv_h=a.hsv_h, hsv_s=a.hsv_s, hsv_v=a.hsv_v, degrees=a.degrees,
        translate=a.translate, scale=a.scale, shear=a.shear, perspective=a.perspective,
        flipud=a.flipud, fliplr=a.fliplr, mosaic=a.mosaic, close_mosaic=a.close_mosaic,
        mixup=a.mixup, cutmix=a.cutmix, copy_paste=a.copy_paste)
    return 0

if __name__ == "__main__": raise SystemExit(main())
