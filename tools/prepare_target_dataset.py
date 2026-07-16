from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Split = Literal["train", "val", "test"]
Difficulty = Literal["clear", "difficult", "negative"]
_VALID_SPLITS = ("train", "val", "test")
_VALID_DIFFICULTIES = ("clear", "difficult", "negative")


@dataclass(frozen=True)
class PreparedImage:
    source: Path
    destination: Path
    scene: str
    split: str
    difficulty: str


def prepare_dataset(
    *,
    sources: Sequence[Path],
    output: Path,
    scene: str,
    split: Split,
    difficulty: Difficulty,
    copy_file: Callable[[Path, Path], None] = shutil.copy2,
) -> tuple[PreparedImage, ...]:
    """Copy capture ``original.png`` files into a deterministic staging set."""
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of: {', '.join(_VALID_SPLITS)}")
    if difficulty not in _VALID_DIFFICULTIES:
        raise ValueError(
            f"difficulty must be one of: {', '.join(_VALID_DIFFICULTIES)}"
        )
    if not scene.strip():
        raise ValueError("scene must not be empty")
    if Path(scene).name != scene or "/" in scene or "\\" in scene:
        raise ValueError("scene must be a filename-safe name, not a path")

    originals = sorted(
        (path for root in sources for path in Path(root).rglob("original.png")),
        key=lambda path: path.as_posix(),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    prefix = f"{scene}-"
    existing_indexes = []
    for path in output.iterdir():
        if not path.is_file():
            continue
        suffix = path.name[len(prefix) :] if path.name.startswith(prefix) else ""
        index_text, separator, _ = suffix.partition(".")
        if separator and len(index_text) == 6 and index_text.isdigit():
            existing_indexes.append(int(index_text))
    next_index = max(existing_indexes, default=0) + 1

    prepared: list[PreparedImage] = []
    for index, source in enumerate(originals, start=next_index):
        target = output / f"{scene}-{index:06d}.png"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing target: {target}")
        copy_file(source, target)
        prepared.append(
            PreparedImage(
                source=source,
                destination=target,
                scene=scene,
                split=split,
                difficulty=difficulty,
            )
        )
    return tuple(prepared)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect original.png captures into deterministic target-board staging names."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="capture directories to search recursively",
    )
    parser.add_argument("--output", type=Path, required=True, help="staging directory")
    parser.add_argument("--scene", required=True, help="scene/group identifier")
    parser.add_argument("--split", choices=_VALID_SPLITS, required=True)
    parser.add_argument("--difficulty", choices=_VALID_DIFFICULTIES, required=True)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="write manifest CSV as UTF-8 instead of writing it to stdout",
    )
    return parser


def _write_manifest(prepared: Sequence[PreparedImage], stream) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("image", "scene", "split", "difficulty"))
    for item in prepared:
        writer.writerow(
            (item.destination.name, item.scene, item.split, item.difficulty)
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = prepare_dataset(
        sources=args.sources,
        output=args.output,
        scene=args.scene,
        split=args.split,
        difficulty=args.difficulty,
    )
    if args.manifest_output is None:
        _write_manifest(prepared, sys.stdout)
    else:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_output.open("w", newline="", encoding="utf-8") as stream:
            _write_manifest(prepared, stream)
    print(f"prepared images: {len(prepared)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
