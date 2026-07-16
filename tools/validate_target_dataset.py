from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


_SPLITS = ("train", "val", "test")
_DIFFICULTIES = ("clear", "difficult", "negative")
_MANIFEST_COLUMNS = ("image", "scene", "split", "difficulty")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class DatasetReport:
    images_by_split: Mapping[str, int]
    positive_by_split: Mapping[str, int]
    negative_by_split: Mapping[str, int]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    clear_positive_count: int
    difficult_positive_count: int
    negative_count: int

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class _ManifestRow:
    image: str
    scene: str
    split: str
    difficulty: str


@dataclass(frozen=True)
class _ImageRecord:
    path: Path
    split: str
    relative: str
    label_path: Path | None
    positive: bool


def _empty_counts() -> dict[str, int]:
    return {split: 0 for split in _SPLITS}


def _readable_image(path: Path) -> bool:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        if encoded.size == 0:
            return False
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError, cv2.error):
        return False
    return image is not None and image.size > 0


def _validate_label(path: Path, errors: list[str]) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        errors.append(f"{path.name}: label is not readable: {exc}")
        return False

    positive = any(line.strip() for line in lines)
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        prefix = f"{path.name}:{line_number}"
        if len(fields) != 5:
            errors.append(f"{prefix}: label line must contain five fields")
            continue
        class_id, *coordinate_fields = fields
        if class_id != "0":
            errors.append(f"{prefix}: class ID must be 0")
        try:
            coordinates = tuple(float(field) for field in coordinate_fields)
        except ValueError:
            errors.append(f"{prefix}: coordinates must be numeric")
            continue
        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"{prefix}: coordinates must be finite")
            continue
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            errors.append(
                f"{path.name}: normalized coordinates must be within [0, 1]"
            )
        width, height = coordinates[2], coordinates[3]
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{prefix}: width and height must be greater than zero")
    return positive


def _read_manifest(path: Path, errors: list[str]) -> tuple[_ManifestRow, ...]:
    if not path.is_file():
        errors.append("split-manifest.csv: file is missing")
        return ()
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _MANIFEST_COLUMNS:
                errors.append(
                    "split-manifest.csv: columns must be exactly "
                    "image,scene,split,difficulty"
                )
                return ()
            rows = tuple(
                _ManifestRow(
                    image=(row.get("image") or "").strip(),
                    scene=(row.get("scene") or "").strip(),
                    split=(row.get("split") or "").strip(),
                    difficulty=(row.get("difficulty") or "").strip(),
                )
                for row in reader
            )
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"split-manifest.csv: cannot be read: {exc}")
        return ()
    return rows


def _find_images(dataset: Path) -> tuple[tuple[str, Path], ...]:
    found: list[tuple[str, Path]] = []
    for split in _SPLITS:
        directory = dataset / "images" / split
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                found.append((split, path))
    return tuple(sorted(found, key=lambda item: (item[0], item[1].as_posix())))


def _find_labels(dataset: Path) -> tuple[tuple[str, Path], ...]:
    found: list[tuple[str, Path]] = []
    for split in _SPLITS:
        directory = dataset / "labels" / split
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".txt":
                found.append((split, path))
    return tuple(sorted(found, key=lambda item: (item[0], item[1].as_posix())))


def _relative_stem(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


def validate_dataset(dataset: Path) -> DatasetReport:
    dataset = Path(dataset)
    errors: list[str] = []
    warnings: list[str] = []
    images_by_split = _empty_counts()
    positive_by_split = _empty_counts()
    negative_by_split = _empty_counts()
    records: dict[tuple[str, str], _ImageRecord] = {}
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    images = _find_images(dataset)
    labels = _find_labels(dataset)
    images_by_stem: dict[tuple[str, str], list[Path]] = defaultdict(list)
    labels_by_stem: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for split, image_path in images:
        image_root = dataset / "images" / split
        images_by_stem[(split, _relative_stem(image_path, image_root))].append(
            image_path
        )
    for split, label_path in labels:
        label_root = dataset / "labels" / split
        labels_by_stem[(split, _relative_stem(label_path, label_root))].append(
            label_path
        )

    for (split, stem), paths in sorted(images_by_stem.items()):
        if len(paths) > 1:
            image_root = dataset / "images" / split
            names = ", ".join(
                sorted(path.relative_to(image_root).as_posix() for path in paths)
            )
            errors.append(
                f"{split}: image stem {stem} is used by multiple files: {names}"
            )
    for (split, stem), paths in sorted(labels_by_stem.items()):
        if len(paths) > 1:
            label_root = dataset / "labels" / split
            names = ", ".join(
                sorted(path.relative_to(label_root).as_posix() for path in paths)
            )
            errors.append(
                f"{split}: label stem {stem} is used by multiple files: {names}"
            )
        if (split, stem) not in images_by_stem:
            for label_path in paths:
                display_path = label_path.relative_to(dataset).as_posix()
                errors.append(f"{display_path}: label has no matching image")

    for split, image_path in images:
        image_root = dataset / "images" / split
        image_relative = image_path.relative_to(image_root).as_posix()
        display_path = image_path.relative_to(dataset).as_posix()
        images_by_split[split] += 1
        if not _readable_image(image_path):
            errors.append(f"{display_path}: image is not readable")

        stem = _relative_stem(image_path, image_root)
        matching_labels = labels_by_stem.get((split, stem), [])
        if len(matching_labels) != 1:
            errors.append(f"{display_path}: expected exactly one label file")
            label_path = None
            positive = False
        else:
            label_path = matching_labels[0]
            positive = _validate_label(label_path, errors)

        if positive:
            positive_by_split[split] += 1
        else:
            negative_by_split[split] += 1
        records[(split, image_relative)] = _ImageRecord(
            path=image_path,
            split=split,
            relative=image_relative,
            label_path=label_path,
            positive=positive,
        )
        try:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        except OSError:
            continue
        hashes[digest].append((split, display_path))

    for duplicates in hashes.values():
        duplicate_splits = sorted(
            {split for split, _ in duplicates}, key=_SPLITS.index
        )
        if len(duplicate_splits) > 1:
            files = ", ".join(path for _, path in duplicates)
            errors.append(
                "duplicate image content appears across splits: "
                f"{', '.join(duplicate_splits)} ({files})"
            )

    rows = _read_manifest(dataset / "split-manifest.csv", errors)
    seen_rows: Counter[tuple[str, str]] = Counter()
    scene_splits: dict[str, set[str]] = defaultdict(set)
    clear_positive_count = 0
    difficult_positive_count = 0
    negative_count = 0

    for row_number, row in enumerate(rows, start=2):
        if not row.image:
            errors.append(f"split-manifest.csv:{row_number}: image must not be empty")
        if not row.scene:
            errors.append(f"split-manifest.csv:{row_number}: scene must not be empty")
        if row.split not in _SPLITS:
            errors.append(
                f"{row.image or f'row {row_number}'}: split must be train, val, or test"
            )
        if row.difficulty not in _DIFFICULTIES:
            errors.append(
                f"{row.image or f'row {row_number}'}: difficulty must be clear, difficult, or negative"
            )

        key = (row.split, Path(row.image).as_posix())
        seen_rows[key] += 1
        if row.scene and row.split in _SPLITS:
            scene_splits[row.scene].add(row.split)

        record = records.get(key)
        if record is None:
            errors.append(
                f"{row.image or f'row {row_number}'}: manifest image does not exist in {row.split or 'unknown'} split"
            )
            continue
        if record.positive:
            if row.difficulty == "clear":
                clear_positive_count += 1
            elif row.difficulty == "difficult":
                difficult_positive_count += 1
            elif row.difficulty in _DIFFICULTIES:
                errors.append(
                    f"{row.image}: positive labels require clear or difficult difficulty"
                )
        else:
            if row.difficulty == "negative":
                negative_count += 1
            elif row.difficulty in _DIFFICULTIES:
                errors.append(f"{row.image}: empty labels require negative difficulty")

    for key, count in seen_rows.items():
        if count > 1:
            split, image = key
            errors.append(f"{image}: appears {count} times in manifest split {split}")
    for split, image in records:
        count = seen_rows[(split, image)]
        if count == 0:
            errors.append(f"{image}: missing from manifest split {split}")

    for scene, splits in sorted(scene_splits.items()):
        if len(splits) > 1:
            ordered = sorted(splits, key=_SPLITS.index)
            errors.append(f"{scene} appears in multiple splits: {', '.join(ordered)}")

    return DatasetReport(
        images_by_split=images_by_split,
        positive_by_split=positive_by_split,
        negative_by_split=negative_by_split,
        errors=tuple(errors),
        warnings=tuple(warnings),
        clear_positive_count=clear_positive_count,
        difficult_positive_count=difficult_positive_count,
        negative_count=negative_count,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the target-board YOLO dataset contract."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("datasets/target_board"),
        help="dataset root (default: datasets/target_board)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_dataset(args.dataset)
    print(f"clear positives: {report.clear_positive_count}")
    print(f"difficult positives: {report.difficult_positive_count}")
    print(f"negatives: {report.negative_count}")
    for split in _SPLITS:
        print(
            f"{split}: images={report.images_by_split[split]}, "
            f"positive={report.positive_by_split[split]}, "
            f"negative={report.negative_by_split[split]}"
        )
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print("dataset valid" if report.valid else "dataset invalid")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
