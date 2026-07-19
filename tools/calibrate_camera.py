from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2

from ev_vision.calibration import save_calibration, solve_chessboard_with_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the fixed Hikrobot camera/lens using chessboard images")
    parser.add_argument("images", type=Path, help="directory containing calibration images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=9, help="inner chessboard corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner chessboard corners per column")
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--max-rms", type=float, default=0.5)
    args = parser.parse_args(argv)
    frames = []
    for path in sorted(args.images.iterdir()):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
    report = solve_chessboard_with_report(
        frames,
        pattern_size=(args.columns, args.rows),
        square_size_mm=args.square_mm,
    )
    report.calibration.validate(max_rms_px=args.max_rms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(args.output, report.calibration)
    print(
        f"images={report.input_images} usable_poses={report.usable_poses} "
        f"rejected={report.rejected_images} rms_px={report.calibration.rms_px:.4f} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
