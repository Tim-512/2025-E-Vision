from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from ev_vision.config import load_config
from ev_vision.detection.classical_board import ClassicalBoardDetector
from ev_vision.detection.debug_rendering import DEBUG_IMAGE_NAMES


_IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz")
_PARTIAL_SOURCES = {
    "CONCENTRIC_ARCS",
    "SINGLE_ARC",
    "WHITE_REGION",
    "FUSED_PARTIAL",
}


@dataclass(frozen=True)
class ReplayFrame:
    sequence: int
    captured_ns: int
    image: np.ndarray
    source_name: str


def _iter_image_paths(root: Path) -> Iterator[Path]:
    yield from sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _safe_member_parts(name: str) -> tuple[str, ...]:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or normalized.startswith("//")
        or re.match(r"^[A-Za-z]:", normalized) is not None
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe archive member: {name}")
    return path.parts


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive:
            parts = _safe_member_parts(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsafe archive member: {member.name}")
            target = destination.joinpath(*parts)
            try:
                target.resolve().relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"unsafe archive member: {member.name}") from exc
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsafe archive member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _is_archive(path: Path) -> bool:
    return path.name.lower().endswith(_ARCHIVE_SUFFIXES)


def _image_frames(
    paths: Sequence[Path], *, fps: float, max_frames: int | None
) -> Iterator[ReplayFrame]:
    if fps <= 0:
        raise ValueError("fps must be positive for image input")
    produced = 0
    for path in paths:
        if max_frames is not None and produced >= max_frames:
            break
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        yield ReplayFrame(
            sequence=produced,
            captured_ns=round(produced * 1_000_000_000 / fps),
            image=image,
            source_name=path.name,
        )
        produced += 1


def iter_replay_frames(
    input_path: Path, *, fps: float, max_frames: int | None
) -> Iterator[ReplayFrame]:
    input_path = Path(input_path)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    if input_path.is_dir():
        yield from _image_frames(
            list(_iter_image_paths(input_path)), fps=fps, max_frames=max_frames
        )
        return
    if input_path.is_file() and input_path.suffix.lower() in _IMAGE_EXTENSIONS:
        yield from _image_frames([input_path], fps=fps, max_frames=max_frames)
        return
    if input_path.is_file() and _is_archive(input_path):
        if fps <= 0:
            raise ValueError("fps must be positive for archive input")
        with tempfile.TemporaryDirectory(prefix="ev-classical-replay-") as temporary:
            root = Path(temporary)
            _safe_extract(input_path, root)
            yield from _image_frames(
                list(_iter_image_paths(root)), fps=fps, max_frames=max_frames
            )
        return

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"unsupported or unreadable input: {input_path}")
    source_fps = fps if fps > 0 else float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError("video FPS is unavailable; pass --fps")
    sequence = 0
    try:
        while max_frames is None or sequence < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            yield ReplayFrame(
                sequence=sequence,
                captured_ns=round(sequence * 1_000_000_000 / source_fps),
                image=image,
                source_name=f"frame-{sequence:06d}",
            )
            sequence += 1
    finally:
        capture.release()


def _enum_text(value: object) -> str:
    return str(getattr(value, "value", value))


def _render_overlay(frame: ReplayFrame, result: object) -> np.ndarray:
    overlay = frame.image.copy()
    corners = np.asarray(getattr(result, "corners_px", ()), dtype=np.float32)
    if corners.size >= 8:
        polygon = np.rint(corners.reshape((-1, 2))).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
    center = getattr(result, "center_px", None)
    if center is not None:
        point = tuple(int(round(value)) for value in center)
        cv2.drawMarker(overlay, point, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    source = _enum_text(getattr(result, "observation_source", "NONE"))
    state = _enum_text(getattr(result, "tracking_state", "SEARCHING"))
    cv2.putText(
        overlay,
        f"{state} / {source}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay


def save_frame_debug(root: Path, frame: ReplayFrame, result: object) -> None:
    destination = Path(root) / f"{frame.sequence:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    debug_images = dict(getattr(result, "debug_images", {}))
    products: dict[str, np.ndarray] = {"overlay": _render_overlay(frame, result)}
    for name in DEBUG_IMAGE_NAMES:
        if name not in debug_images:
            raise ValueError(f"frame {frame.sequence} is missing debug image {name}")
        products[name] = debug_images[name]
    for name, image in products.items():
        if not cv2.imwrite(str(destination / f"{name}.png"), image):
            raise OSError(f"cannot write debug image {name} for frame {frame.sequence}")


def evaluate(
    *,
    input_path: Path,
    output_path: Path,
    config_path: Path,
    fps: float,
    save_debug: Path | None,
    max_frames: int | None,
) -> dict[str, object]:
    config = load_config(config_path)
    detector = ClassicalBoardDetector(config.detection)
    durations_ms: list[float] = []
    source_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    valid_count = 0
    full_count = 0
    partial_count = 0
    predicted_count = 0
    lost_count = 0
    prediction_streak = 0
    max_prediction_streak = 0
    frame_count = 0

    for frame in iter_replay_frames(input_path, fps=fps, max_frames=max_frames):
        started_ns = time.perf_counter_ns()
        result = detector.detect(
            frame.image,
            captured_ns=frame.captured_ns,
            source_sequence=frame.sequence,
            include_debug=save_debug is not None,
        )
        durations_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
        source = _enum_text(result.observation_source)
        state = _enum_text(result.tracking_state)
        source_counts[source] += 1
        state_counts[state] += 1
        frame_count += 1
        valid_count += int(bool(result.target_valid))
        full_count += int(source == "FULL_BOARD")
        partial_count += int(source in _PARTIAL_SOURCES)
        predicted_count += int(source == "PREDICTED")
        lost_count += int(state == "LOST")
        prediction_streak = prediction_streak + 1 if source == "PREDICTED" else 0
        max_prediction_streak = max(max_prediction_streak, prediction_streak)
        if save_debug is not None:
            save_frame_debug(save_debug, frame, result)

    mean_ms = float(np.mean(durations_ms)) if durations_ms else 0.0
    p95_ms = float(np.percentile(durations_ms, 95)) if durations_ms else 0.0
    report: dict[str, object] = {
        "input": str(input_path),
        "config": str(config_path),
        "frames": frame_count,
        "valid_frames": valid_count,
        "full_board_frames": full_count,
        "partial_frames": partial_count,
        "predicted_frames": predicted_count,
        "lost_frames": lost_count,
        "source_counts": dict(sorted(source_counts.items())),
        "state_counts": dict(sorted(state_counts.items())),
        "max_prediction_streak": max_prediction_streak,
        "mean_processing_ms": mean_ms,
        "p95_processing_ms": p95_ms,
        "measured_detection_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the classical white-board detector without camera hardware"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--save-debug", type=Path)
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = evaluate(
            input_path=arguments.input,
            output_path=arguments.output,
            config_path=arguments.config,
            fps=arguments.fps,
            save_debug=arguments.save_debug,
            max_frames=arguments.max_frames,
        )
    except (OSError, ValueError) as exc:
        build_parser().error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
