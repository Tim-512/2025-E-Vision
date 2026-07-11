from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ev_vision.calibration import save_calibration, solve_chessboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the fixed Hikrobot camera/lens using chessboard images")
    parser.add_argument("images", type=Path, help="directory containing calibration images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=9, help="inner chessboard corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner chessboard corners per column")
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--max-rms", type=float, default=0.5)
    args = parser.parse_args()
    frames = []
    for path in sorted(args.images.iterdir()):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
    calibration = solve_chessboard(frames, pattern_size=(args.columns, args.rows), square_size_mm=args.square_mm)
    calibration.validate(max_rms_px=args.max_rms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(args.output, calibration)
    print(f"poses={len(frames)} rms_px={calibration.rms_px:.4f} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
