from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from tools.prepare_target_dataset import prepare_dataset
from tools.validate_target_dataset import main as validate_main
from tools.validate_target_dataset import validate_dataset


MANIFEST_HEADER = ("image", "scene", "split", "difficulty")


def write_png(path: Path, value: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 10, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def write_label(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def write_manifest(dataset: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path = dataset / "split-manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(rows)


def append_manifest(dataset: Path, rows: str) -> None:
    with (dataset / "split-manifest.csv").open("a", encoding="utf-8", newline="") as stream:
        stream.write(rows)


def build_minimal_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    write_png(dataset / "images/train/a.png", 32)
    write_label(dataset / "labels/train/a.txt", "0 0.50 0.50 0.40 0.30\n")
    write_png(dataset / "images/val/b.png", 96)
    write_label(dataset / "labels/val/b.txt", "0 0.45 0.55 0.30 0.20\n")
    write_png(dataset / "images/test/n.png", 160)
    write_label(dataset / "labels/test/n.txt", "")
    write_manifest(
        dataset,
        [
            ("a.png", "scene-train", "train", "clear"),
            ("b.png", "scene-val", "val", "difficult"),
            ("n.png", "scene-test", "test", "negative"),
        ],
    )
    return dataset


def test_prepare_uses_only_original_png_and_deterministic_names(tmp_path: Path) -> None:
    capture_a = tmp_path / "captures" / "20260716T100000Z"
    capture_b = tmp_path / "captures" / "20260716T100100Z"
    capture_a.mkdir(parents=True)
    capture_b.mkdir(parents=True)
    (capture_a / "original.png").write_bytes(b"first")
    (capture_a / "overlay.png").write_bytes(b"must-not-copy")
    (capture_b / "original.png").write_bytes(b"second")

    manifest = prepare_dataset(
        sources=[capture_a.parent],
        output=tmp_path / "staging",
        scene="desk-left",
        split="train",
        difficulty="clear",
        copy_file=lambda source, target: target.write_bytes(source.read_bytes()),
    )

    assert [item.destination.name for item in manifest] == [
        "desk-left-000001.png",
        "desk-left-000002.png",
    ]
    assert [item.destination.read_bytes() for item in manifest] == [b"first", b"second"]
    assert all(item.scene == "desk-left" for item in manifest)
    assert all(item.split == "train" for item in manifest)
    assert all(item.difficulty == "clear" for item in manifest)
    assert not (tmp_path / "staging" / "overlay.png").exists()


def test_validator_accepts_contract_and_reports_counts(tmp_path: Path) -> None:
    dataset = build_minimal_dataset(tmp_path)

    report = validate_dataset(dataset)

    assert report.valid
    assert report.errors == ()
    assert report.images_by_split == {"train": 1, "val": 1, "test": 1}
    assert report.positive_by_split == {"train": 1, "val": 1, "test": 0}
    assert report.negative_by_split == {"train": 0, "val": 0, "test": 1}
    assert report.clear_positive_count == 1
    assert report.difficult_positive_count == 1
    assert report.negative_count == 1


def test_validator_rejects_out_of_bounds_and_scene_leakage(tmp_path: Path) -> None:
    dataset = build_minimal_dataset(tmp_path)
    write_label(dataset / "labels/train/a.txt", "0 1.10 0.50 0.20 0.20\n")
    write_png(dataset / "images/val/a2.png", 220)
    write_label(dataset / "labels/val/a2.txt", "0 0.50 0.50 0.20 0.20\n")
    append_manifest(dataset, "a2.png,scene-train,val,clear\n")

    report = validate_dataset(dataset)

    assert "a.txt: normalized coordinates must be within [0, 1]" in report.errors
    assert "scene-train appears in multiple splits: train, val" in report.errors


def test_validator_rejects_bad_images_and_missing_labels(tmp_path: Path) -> None:
    dataset = build_minimal_dataset(tmp_path)
    (dataset / "images/train/a.png").write_bytes(b"not-an-image")
    (dataset / "labels/val/b.txt").unlink()

    report = validate_dataset(dataset)

    assert "images/train/a.png: image is not readable" in report.errors
    assert "images/val/b.png: expected exactly one label file" in report.errors


def test_validator_rejects_malformed_class_nonfinite_and_nonpositive_boxes(
    tmp_path: Path,
) -> None:
    dataset = build_minimal_dataset(tmp_path)
    write_label(
        dataset / "labels/train/a.txt",
        "1 0.5 0.5 0.2 0.2\n"
        "0 0.5 0.5 0.2\n"
        "0 nan 0.5 0.2 0.2\n"
        "0 0.5 0.5 0.0 0.2\n",
    )

    report = validate_dataset(dataset)

    assert "a.txt:1: class ID must be 0" in report.errors
    assert "a.txt:2: label line must contain five fields" in report.errors
    assert "a.txt:3: coordinates must be finite" in report.errors
    assert "a.txt:4: width and height must be greater than zero" in report.errors


def test_validator_rejects_manifest_schema_and_difficulty_mismatch(tmp_path: Path) -> None:
    dataset = build_minimal_dataset(tmp_path)
    (dataset / "split-manifest.csv").write_text(
        "image,split,scene,difficulty\n"
        "a.png,train,scene-train,negative\n",
        encoding="utf-8",
    )

    schema_report = validate_dataset(dataset)

    assert (
        "split-manifest.csv: columns must be exactly "
        "image,scene,split,difficulty"
    ) in schema_report.errors

    write_manifest(
        dataset,
        [
            ("a.png", "scene-train", "train", "negative"),
            ("b.png", "scene-val", "val", "unknown"),
            ("n.png", "scene-test", "test", "clear"),
        ],
    )
    mismatch_report = validate_dataset(dataset)

    assert "a.png: positive labels require clear or difficult difficulty" in mismatch_report.errors
    assert "b.png: difficulty must be clear, difficult, or negative" in mismatch_report.errors
    assert "n.png: empty labels require negative difficulty" in mismatch_report.errors


def test_validator_reports_duplicate_hashes_across_splits(tmp_path: Path) -> None:
    dataset = build_minimal_dataset(tmp_path)
    duplicate = dataset / "images/val/b.png"
    duplicate.write_bytes((dataset / "images/train/a.png").read_bytes())

    report = validate_dataset(dataset)

    assert any(
        "duplicate image content appears across splits: train, val" in error
        for error in report.errors
    )


def test_validator_cli_prints_counts_and_returns_nonzero_for_errors(
    tmp_path: Path, capsys
) -> None:
    dataset = build_minimal_dataset(tmp_path)
    write_label(dataset / "labels/train/a.txt", "0 1.10 0.50 0.20 0.20\n")

    exit_code = validate_main([str(dataset)])
    output = capsys.readouterr().out

    assert exit_code != 0
    assert "clear positives: 1" in output
    assert "difficult positives: 1" in output
    assert "negatives: 1" in output
