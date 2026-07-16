from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

_HINT = "ultralytics is required; install with: pip install -e .[training]"

def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc: raise RuntimeError(_HINT) from exc
    return YOLO

def _list(value):
    if value is None: return []
    for method in ("detach", "cpu"):
        if hasattr(value, method): value = getattr(value, method)()
    if hasattr(value, "tolist"): value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]

def _metric(value):
    values = _list(value); flat = []
    for item in values: flat.extend(item if isinstance(item, (list, tuple)) else [item])
    return sum(map(float, flat)) / len(flat) if flat else 0.0

def detections(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None: return []
    return [(int(c), float(q), tuple(map(float, xy))) for c, q, xy in zip(_list(boxes.cls), _list(boxes.conf), _list(boxes.xyxy))]

def shape(result, imgsz):
    value = getattr(result, "orig_shape", None)
    return (int(value[0]), int(value[1])) if value is not None else (imgsz, imgsz)

def labels(path: Path, height: int, width: int):
    if not path.is_file(): return []
    answer = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or int(fields[0]) != 0: continue
        x, y, w, h = map(float, fields[1:])
        answer.append(((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height))
    return answer

def box_iou(a, b):
    inter = max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(0.0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]); area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0

def _parser():
    p = argparse.ArgumentParser(description="Evaluate a target-board detector and write JSON.")
    p.add_argument("--model", default="models/target-board.pt"); p.add_argument("--data", default="datasets/target_board/dataset.yaml")
    p.add_argument("--manifest", default="datasets/target_board/split-manifest.csv"); p.add_argument("--output", default="models/target-board-metrics.json")
    p.add_argument("--confidence", type=float, default=.45); p.add_argument("--iou", type=float, default=.5)
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--device", default="0")
    return p

def main(argv: Sequence[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try: YOLO = load_yolo()
    except RuntimeError as exc: print(exc, file=sys.stderr); return 2
    model = YOLO(a.model); metrics = model.val(data=a.data, split="test", imgsz=a.imgsz, device=a.device)
    with Path(a.manifest).open(encoding="utf-8-sig", newline="") as f: rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    root = Path(a.data).parent; paths = [root / "images/test" / r["image"] for r in rows]
    results = model.predict(source=[str(p) for p in paths], conf=a.confidence, iou=a.iou, imgsz=a.imgsz, device=a.device, save=False, verbose=False)
    found = defaultdict(list); negatives = []
    for row, image, result in zip(rows, paths, results):
        ds = [d for d in detections(result) if d[0] == 0 and d[1] >= a.confidence]
        if row["difficulty"] == "negative": negatives.append((row["scene"], row["image"], bool(ds))); continue
        h, w = shape(result, a.imgsz); truth = labels(root / "labels/test" / f"{image.stem}.txt", h, w)
        found[row["difficulty"]].append(any(box_iou(d[2], t) >= a.iou for d in ds for t in truth))
    streak = maximum = 0; scene = None
    for next_scene, _, hit in sorted(negatives):
        if next_scene != scene: scene, streak = next_scene, 0
        streak = streak + 1 if hit else 0; maximum = max(maximum, streak)
    recall = lambda key: sum(found[key]) / len(found[key]) if found[key] else 0.0
    report = {"model": a.model, "data": a.data, "manifest": a.manifest, "confidence_threshold": a.confidence, "iou_threshold": a.iou,
        "precision": _metric(metrics.box.p), "recall": _metric(metrics.box.r), "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "speed": {k: float(v) for k, v in metrics.speed.items()}, "clear_recall": recall("clear"), "difficult_recall": recall("difficult"),
        "negative_false_positive_images": sum(hit for _, _, hit in negatives), "max_negative_high_confidence_streak": maximum}
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return 0

if __name__ == "__main__": raise SystemExit(main())
