from __future__ import annotations

import argparse
import ast
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools import evaluate_classical_detector as subject


DEBUG_NAMES = (
    "normalized-gray",
    "white-mask",
    "edge-mask",
    "ring-arcs",
    "candidate-scores",
)


def write_frames(root: Path, count: int = 4) -> None:
    root.mkdir(parents=True)
    for index in range(count):
        image = np.full((120, 160, 3), 20 + index, dtype=np.uint8)
        assert cv2.imwrite(str(root / f"{index:04d}.png"), image)


class FakeDetector:
    SOURCES = ("FULL_BOARD", "CONCENTRIC_ARCS", "PREDICTED", "NONE")
    STATES = ("TRACKING", "TRACKING", "PREDICTING", "LOST")

    def __init__(self, config: object) -> None:
        self.config = config

    def detect(
        self,
        image: np.ndarray,
        *,
        captured_ns: int,
        source_sequence: int,
        include_debug: bool = False,
    ) -> SimpleNamespace:
        index = source_sequence % len(self.SOURCES)
        return SimpleNamespace(
            observation_source=self.SOURCES[index],
            tracking_state=self.STATES[index],
            target_valid=index != 3,
            center_px=(80.0, 60.0) if index != 3 else None,
            corners_px=(),
            debug_images={},
        )


class FakeDebugDetector(FakeDetector):
    def detect(
        self,
        image: np.ndarray,
        *,
        captured_ns: int,
        source_sequence: int,
        include_debug: bool = False,
    ) -> SimpleNamespace:
        assert include_debug is True
        result = super().detect(
            image,
            captured_ns=captured_ns,
            source_sequence=source_sequence,
            include_debug=include_debug,
        )
        result.corners_px = ((10.0, 10.0), (150.0, 10.0), (150.0, 110.0), (10.0, 110.0))
        result.debug_images = {
            name: np.full((24, 32), index + 1, dtype=np.uint8)
            for index, name in enumerate(DEBUG_NAMES)
        }
        return result


def test_iter_replay_frames_orders_image_directory(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 3)

    frames = list(subject.iter_replay_frames(replay, fps=20.0, max_frames=None))

    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert [frame.captured_ns for frame in frames] == [0, 50_000_000, 100_000_000]
    assert [frame.source_name for frame in frames] == ["0000.png", "0001.png", "0002.png"]


def test_iter_replay_frames_supports_single_image_and_limit(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 2)

    frames = list(subject.iter_replay_frames(replay / "0000.png", fps=25.0, max_frames=1))

    assert len(frames) == 1
    assert frames[0].image.shape == (120, 160, 3)


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_image_input_requires_positive_fps(tmp_path: Path, fps: float) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 1)

    with pytest.raises(ValueError, match="fps must be positive"):
        list(subject.iter_replay_frames(replay, fps=fps, max_frames=None))


def test_max_frames_must_be_positive(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 1)

    with pytest.raises(ValueError, match="max_frames must be positive"):
        list(subject.iter_replay_frames(replay, fps=20.0, max_frames=0))


def test_tar_archive_is_read_without_persisting_extracted_files(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 2)
    archive = tmp_path / "capture.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(replay / "0001.png", arcname="nested/0001.png")
        bundle.add(replay / "0000.png", arcname="nested/0000.png")

    frames = list(subject.iter_replay_frames(archive, fps=10.0, max_frames=None))

    assert [frame.sequence for frame in frames] == [0, 1]
    assert [frame.captured_ns for frame in frames] == [0, 100_000_000]
    assert [frame.source_name for frame in frames] == ["0000.png", "0001.png"]


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.png",
        r"..\escape.png",
        r"nested\..\escape.png",
        r"C:\escape.png",
        "C:/escape.png",
        r"\\server\share\escape.png",
        "/absolute.png",
    ],
)
def test_tar_archive_rejects_cross_platform_unsafe_paths(
    tmp_path: Path, member_name: str
) -> None:
    archive = tmp_path / "unsafe.tar"
    payload = b"not-an-image"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe archive member"):
        list(subject.iter_replay_frames(archive, fps=20.0, max_frames=None))


def test_tar_archive_is_extracted_member_by_member_without_extractall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 1)
    archive = tmp_path / "capture.tar"
    with tarfile.open(archive, "w") as bundle:
        bundle.add(replay / "0000.png", arcname="nested/0000.png")

    def forbidden_extractall(*args, **kwargs):
        raise AssertionError("extractall must not be used")

    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden_extractall)

    frames = list(subject.iter_replay_frames(archive, fps=20.0, max_frames=None))

    assert [item.source_name for item in frames] == ["0000.png"]


def test_evaluate_reports_measured_sources_latency_and_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 4)
    monkeypatch.setattr(subject, "ClassicalBoardDetector", FakeDetector)

    report = subject.evaluate(
        input_path=replay,
        output_path=tmp_path / "report.json",
        config_path=Path("config/default.yaml"),
        fps=20.0,
        save_debug=None,
        max_frames=None,
    )

    assert report["frames"] == 4
    assert report["source_counts"] == {
        "CONCENTRIC_ARCS": 1,
        "FULL_BOARD": 1,
        "NONE": 1,
        "PREDICTED": 1,
    }
    assert report["state_counts"] == {"LOST": 1, "PREDICTING": 1, "TRACKING": 2}
    assert report["valid_frames"] == 3
    assert report["full_board_frames"] == 1
    assert report["partial_frames"] == 1
    assert report["predicted_frames"] == 1
    assert report["lost_frames"] == 1
    assert report["max_prediction_streak"] == 1
    assert report["mean_processing_ms"] >= 0.0
    assert report["p95_processing_ms"] >= 0.0
    assert report["measured_detection_fps"] > 0.0
    assert json.loads((tmp_path / "report.json").read_text("utf-8")) == report


def test_debug_export_uses_sequence_named_subdirectories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay = tmp_path / "replay"
    write_frames(replay, 1)
    monkeypatch.setattr(subject, "ClassicalBoardDetector", FakeDebugDetector)

    subject.evaluate(
        input_path=replay,
        output_path=tmp_path / "report.json",
        config_path=Path("config/default.yaml"),
        fps=20.0,
        save_debug=tmp_path / "debug",
        max_frames=1,
    )

    frame_dir = tmp_path / "debug" / "000000"
    assert (frame_dir / "overlay.png").is_file()
    for name in DEBUG_NAMES:
        assert (frame_dir / f"{name}.png").is_file()


def test_cli_has_exact_public_options() -> None:
    parser = subject.build_parser()
    option_strings = {
        option
        for action in parser._actions
        if not isinstance(action, argparse._HelpAction)
        for option in action.option_strings
    }
    assert option_strings == {
        "--input",
        "--output",
        "--config",
        "--fps",
        "--save-debug",
        "--max-frames",
    }


def test_evaluator_does_not_import_mvs_or_ultralytics() -> None:
    tree = ast.parse(Path(subject.__file__).read_text("utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("MvCamera" in name or "ultralytics" in name for name in imported_names)
