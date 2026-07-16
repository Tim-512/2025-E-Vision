from __future__ import annotations
import argparse, csv, sys
from collections.abc import Sequence
from pathlib import Path
try:
    from tools.evaluate_target_detector import box_iou, detections, labels, shape
except ModuleNotFoundError:
    from evaluate_target_detector import box_iou, detections, labels, shape

_HINT = "ultralytics is required; install with: pip install -e .[training]"
_FIELDS = ("image", "difficulty", "predicted_confidence", "matched_iou", "full_board_contained", "notes")

def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc: raise RuntimeError(_HINT) from exc
    return YOLO

def _parser():
    p = argparse.ArgumentParser(description="Save annotated predictions and a review CSV.")
    p.add_argument("--model", default="models/target-board.pt"); p.add_argument("--source", required=True); p.add_argument("--output", required=True)
    p.add_argument("--manifest"); p.add_argument("--labels"); p.add_argument("--confidence", type=float, default=.45); p.add_argument("--iou", type=float, default=.5)
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--device", default="0")
    return p

def main(argv: Sequence[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try: YOLO = load_yolo()
    except RuntimeError as exc: print(exc, file=sys.stderr); return 2
    source, output = Path(a.source), Path(a.output); output.mkdir(parents=True, exist_ok=True)
    difficulty = {}
    if a.manifest:
        with Path(a.manifest).open(encoding="utf-8-sig", newline="") as f: difficulty = {Path(r["image"]).as_posix(): r.get("difficulty", "") for r in csv.DictReader(f)}
    results = YOLO(a.model).predict(source=a.source, conf=a.confidence, iou=a.iou, imgsz=a.imgsz, device=a.device, save=False, verbose=False)
    rows = []
    for index, result in enumerate(results):
        image = Path(getattr(result, "path", "") or (source if source.is_file() else source / f"prediction-{index:06d}.png"))
        try: relative = image.resolve().relative_to(source.resolve()) if source.is_dir() else Path(image.name)
        except ValueError: relative = Path(image.name)
        destination = output / relative; destination.parent.mkdir(parents=True, exist_ok=True); result.save(filename=str(destination))
        ds = [d for d in detections(result) if d[0] == 0 and d[1] >= a.confidence]
        confidence = max((d[1] for d in ds), default=0.0); matched = 0.0
        if a.labels:
            h, w = shape(result, a.imgsz); truth = labels(Path(a.labels) / f"{relative.stem}.txt", h, w)
            matched = max((box_iou(d[2], t) for d in ds for t in truth), default=0.0)
        rows.append({"image": relative.as_posix(), "difficulty": difficulty.get(relative.as_posix(), difficulty.get(relative.name, "")),
            "predicted_confidence": f"{confidence:.6f}", "matched_iou": f"{matched:.6f}", "full_board_contained": "no", "notes": ""})
    with (output / "review.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS); writer.writeheader(); writer.writerows(rows)
    return 0

if __name__ == "__main__": raise SystemExit(main())
