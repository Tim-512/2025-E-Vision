from __future__ import annotations
import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

_HINT = "ultralytics is required; install with: pip install -e .[training]"

def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(_HINT) from exc
    return YOLO

def is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").is_file(): return True
    model = Path("/proc/device-tree/model")
    if model.is_file():
        try: return "jetson" in model.read_text(errors="ignore").lower()
        except OSError: return False
    return sys.platform.startswith("linux") and platform.machine().lower() in {"aarch64", "arm64"}

def _parser():
    p = argparse.ArgumentParser(description="Export a target-board detector artifact.")
    p.add_argument("--model", default="models/target-board.pt")
    p.add_argument("--format", default="onnx")
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--device", default="0")
    return p

def main(argv: Sequence[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try: YOLO = load_yolo()
    except RuntimeError as exc:
        print(exc, file=sys.stderr); return 2
    fmt = a.format.lower()
    if fmt == "engine" and not is_jetson():
        print("TensorRT engine export must be run locally on Jetson; never copy an engine from Windows.", file=sys.stderr); return 2
    kwargs = {"format": fmt, "imgsz": a.imgsz, "device": a.device}
    if fmt == "onnx": kwargs.update(dynamic=False, simplify=True, opset=12)
    elif fmt == "engine": kwargs.update(half=True)
    YOLO(a.model).export(**kwargs)
    return 0

if __name__ == "__main__": raise SystemExit(main())
