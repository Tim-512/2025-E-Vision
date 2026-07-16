from __future__ import annotations

import builtins
import csv
import json
import sys
import types
from pathlib import Path

import pytest

from tools.evaluate_target_detector import main as evaluate_main
from tools.export_target_detector import main as export_main
from tools.inspect_target_predictions import main as inspect_main
from tools.train_target_detector import main as train_main


class FakeBoxes:
    def __init__(self, rows: list[tuple[int, float, tuple[float, float, float, float]]]):
        self.cls = [row[0] for row in rows]
        self.conf = [row[1] for row in rows]
        self.xyxy = [row[2] for row in rows]


class FakeResult:
    def __init__(self, rows, *, path: Path | None = None) -> None:
        self.boxes = FakeBoxes(rows)
        self.path = str(path) if path is not None else None
        self.save_calls: list[str] = []

    def save(self, filename: str) -> str:
        self.save_calls.append(filename)
        output = Path(filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"annotated")
        return filename


class FakeMetricBox:
    p = [0.91]
    r = [0.82]
    map50 = 0.88
    map = 0.67


class FakeMetrics:
    box = FakeMetricBox()
    speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}


class FakeYOLOInstance:
    def __init__(self, owner: "FakeUltralytics", model: str) -> None:
        self.owner = owner
        self.model = model

    def train(self, **kwargs):
        self.owner.train_calls.append(kwargs)
        return object()

    def val(self, **kwargs):
        self.owner.val_calls.append(kwargs)
        return FakeMetrics()

    def predict(self, **kwargs):
        self.owner.predict_calls.append(kwargs)
        source = kwargs["source"]
        if isinstance(source, (list, tuple)):
            return [self.owner.result_for(Path(item)) for item in source]
        source_path = Path(source)
        if source_path.is_dir():
            return [self.owner.result_for(path) for path in sorted(source_path.glob("*.png"))]
        return [self.owner.result_for(source_path)]

    def export(self, **kwargs):
        self.owner.export_calls.append(kwargs)
        return "exported"


class FakeUltralytics:
    def __init__(self) -> None:
        self.models: list[str] = []
        self.train_calls: list[dict[str, object]] = []
        self.val_calls: list[dict[str, object]] = []
        self.predict_calls: list[dict[str, object]] = []
        self.export_calls: list[dict[str, object]] = []
        self.results: dict[str, FakeResult] = {}

    def yolo(self, model: str) -> FakeYOLOInstance:
        self.models.append(model)
        return FakeYOLOInstance(self, model)

    def result_for(self, path: Path) -> FakeResult:
        return self.results.get(path.name, FakeResult([], path=path))


def install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> FakeUltralytics:
    fake = FakeUltralytics()
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=fake.yolo))
    return fake


def block_import(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module or name.startswith(f"{module}."):
            raise ImportError(f"blocked {module}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)


def write_manifest(dataset: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    manifest = dataset / "split-manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "scene", "split", "difficulty"))
        writer.writerows(rows)
    return manifest


def write_test_image_and_label(dataset: Path, image: str, label: str) -> Path:
    image_path = dataset / "images" / "test" / image
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"image")
    label_path = dataset / "labels" / "test" / f"{Path(image).stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(label, encoding="utf-8")
    return image_path


def test_train_forwards_reproducible_nano_settings(monkeypatch, tmp_path: Path) -> None:
    fake = install_fake_ultralytics(monkeypatch)
    exit_code = train_main([
        "--data", str(tmp_path / "dataset.yaml"), "--model", "yolo11n.pt",
        "--epochs", "80", "--imgsz", "640", "--device", "0",
        "--project", str(tmp_path / "runs"), "--name", "target-board-v1",
    ])

    assert exit_code == 0
    assert fake.models == ["yolo11n.pt"]
    assert fake.train_calls == [{
        "data": str(tmp_path / "dataset.yaml"), "epochs": 80, "imgsz": 640,
        "device": "0", "project": str(tmp_path / "runs"), "name": "target-board-v1",
        "single_cls": True, "seed": 20250716, "deterministic": True,
        "patience": 20, "batch": -1, "hsv_h": 0.015, "hsv_s": 0.50,
        "hsv_v": 0.40, "degrees": 7.0, "translate": 0.08, "scale": 0.25,
        "shear": 0.0, "perspective": 0.0005, "flipud": 0.0, "fliplr": 0.50,
        "mosaic": 0.20, "close_mosaic": 10, "mixup": 0.0, "cutmix": 0.0,
        "copy_paste": 0.0,
    }]


def test_train_all_fixed_settings_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    fake = install_fake_ultralytics(monkeypatch)
    exit_code = train_main([
        "--data", "custom.yaml", "--model", "custom.pt", "--epochs", "3",
        "--imgsz", "320", "--device", "cpu", "--project", str(tmp_path),
        "--name", "custom", "--no-single-cls", "--seed", "7",
        "--no-deterministic", "--patience", "4", "--batch", "2",
        "--hsv-h", "0.1", "--hsv-s", "0.2", "--hsv-v", "0.3",
        "--degrees", "1.0", "--translate", "0.02", "--scale", "0.1",
        "--shear", "2.0", "--perspective", "0.001", "--flipud", "0.1",
        "--fliplr", "0.2", "--mosaic", "0.3", "--close-mosaic", "2",
        "--mixup", "0.1", "--cutmix", "0.2", "--copy-paste", "0.3",
    ])

    assert exit_code == 0
    assert fake.train_calls[0]["single_cls"] is False
    assert fake.train_calls[0]["deterministic"] is False
    assert fake.train_calls[0]["seed"] == 7
    assert fake.train_calls[0]["copy_paste"] == 0.3


def test_missing_ultralytics_prints_install_command(monkeypatch, capsys) -> None:
    block_import(monkeypatch, "ultralytics")
    assert export_main(["--model", "best.pt", "--format", "onnx"]) == 2
    assert "pip install -e .[training]" in capsys.readouterr().err


def test_evaluation_writes_metrics_and_manifest_breakdown(monkeypatch, tmp_path: Path) -> None:
    fake = install_fake_ultralytics(monkeypatch)
    dataset = tmp_path / "dataset"
    data = dataset / "dataset.yaml"
    data.parent.mkdir(parents=True)
    data.write_text("test: images/test\n", encoding="utf-8")
    write_test_image_and_label(dataset, "clear.png", "0 0.5 0.5 0.4 0.4\n")
    write_test_image_and_label(dataset, "difficult.png", "0 0.5 0.5 0.4 0.4\n")
    write_test_image_and_label(dataset, "neg-1.png", "")
    write_test_image_and_label(dataset, "neg-2.png", "")
    write_test_image_and_label(dataset, "neg-3.png", "")
    manifest = write_manifest(dataset, [
        ("clear.png", "clear-scene", "test", "clear"),
        ("difficult.png", "difficult-scene", "test", "difficult"),
        ("neg-2.png", "negative-scene", "test", "negative"),
        ("neg-1.png", "negative-scene", "test", "negative"),
        ("neg-3.png", "negative-scene", "test", "negative"),
    ])
    fake.results = {
        "clear.png": FakeResult([(0, 0.9, (30.0, 30.0, 70.0, 70.0))]),
        "difficult.png": FakeResult([(0, 0.9, (0.0, 0.0, 10.0, 10.0))]),
        "neg-1.png": FakeResult([(0, 0.8, (1.0, 1.0, 5.0, 5.0))]),
        "neg-2.png": FakeResult([(0, 0.7, (1.0, 1.0, 5.0, 5.0))]),
        "neg-3.png": FakeResult([]),
    }
    output = tmp_path / "metrics.json"

    exit_code = evaluate_main([
        "--model", "best.pt", "--data", str(data), "--manifest", str(manifest),
        "--output", str(output), "--confidence", "0.45", "--iou", "0.5",
        "--imgsz", "100", "--device", "cpu",
    ])

    assert exit_code == 0
    assert fake.val_calls == [{"data": str(data), "split": "test", "imgsz": 100, "device": "cpu"}]
    assert all(call["save"] is False for call in fake.predict_calls)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == {
        "model": "best.pt", "data": str(data), "manifest": str(manifest),
        "confidence_threshold": 0.45, "iou_threshold": 0.5,
        "precision": 0.91, "recall": 0.82, "map50": 0.88, "map50_95": 0.67,
        "speed": {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5},
        "clear_recall": 1.0, "difficult_recall": 0.0,
        "negative_false_positive_images": 2, "max_negative_high_confidence_streak": 2,
    }


def test_export_forwards_onnx_and_engine_settings(monkeypatch) -> None:
    fake = install_fake_ultralytics(monkeypatch)
    assert export_main(["--model", "best.pt", "--format", "onnx", "--imgsz", "640", "--device", "cpu"]) == 0
    assert fake.export_calls == [{
        "format": "onnx", "imgsz": 640, "device": "cpu",
        "dynamic": False, "simplify": True, "opset": 12,
    }]

    monkeypatch.setattr("tools.export_target_detector.is_jetson", lambda: True)
    assert export_main(["--model", "best.pt", "--format", "engine", "--imgsz", "320", "--device", "0"]) == 0
    assert fake.export_calls[1] == {"format": "engine", "imgsz": 320, "device": "0", "half": True}


def test_engine_export_is_rejected_away_from_jetson(monkeypatch, capsys) -> None:
    install_fake_ultralytics(monkeypatch)
    monkeypatch.setattr("tools.export_target_detector.is_jetson", lambda: False)
    assert export_main(["--model", "best.pt", "--format", "engine"]) == 2
    assert "Jetson" in capsys.readouterr().err


def test_other_export_formats_forward_common_arguments(monkeypatch) -> None:
    fake = install_fake_ultralytics(monkeypatch)
    assert export_main(["--model", "best.pt", "--format", "openvino", "--imgsz", "512", "--device", "cpu"]) == 0
    assert fake.export_calls == [{"format": "openvino", "imgsz": 512, "device": "cpu"}]


def test_inspection_uses_save_false_and_writes_review_artifacts(monkeypatch, tmp_path: Path) -> None:
    fake = install_fake_ultralytics(monkeypatch)
    source = tmp_path / "images"
    source.mkdir()
    image = source / "sample.png"
    image.write_bytes(b"image")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    manifest = write_manifest(tmp_path, [("sample.png", "review-scene", "test", "difficult")])
    result = FakeResult([(0, 0.87, (30.0, 30.0, 70.0, 70.0))], path=image)
    fake.results = {"sample.png": result}
    output = tmp_path / "review"

    exit_code = inspect_main([
        "--model", "best.pt", "--source", str(source), "--output", str(output),
        "--manifest", str(manifest), "--labels", str(labels),
        "--confidence", "0.45", "--iou", "0.5", "--imgsz", "100", "--device", "cpu",
    ])

    assert exit_code == 0
    assert fake.predict_calls == [{
        "source": str(source), "conf": 0.45, "iou": 0.5, "imgsz": 100,
        "device": "cpu", "save": False, "verbose": False,
    }]
    assert result.save_calls == [str(output / "sample.png")]
    assert (output / "sample.png").read_bytes() == b"annotated"
    with (output / "review.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{
        "image": "sample.png", "difficulty": "difficult",
        "predicted_confidence": "0.870000", "matched_iou": "1.000000",
        "full_board_contained": "no", "notes": "",
    }]
