from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import errno
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest
import yaml

from ev_vision.config import CameraConfig
from ev_vision.models import BoardObservation, Frame
from ev_vision.tuning.models import (
    CameraIdentity,
    CaptureSnapshot,
    DetectionSnapshot,
    EditableCameraParameters,
    ImageDiagnostics,
    OverlayOptions,
    ParameterBounds,
    RuntimeSnapshot,
)
from ev_vision.tuning.storage import PROFILE_SCHEMA_VERSION, TuningStorage


PARAMETERS = EditableCameraParameters(
    exposure_us=800.0,
    gain_db=6.0,
    acquisition_fps=120.0,
    auto_exposure=False,
    auto_gain=False,
    auto_white_balance=False,
)


def make_storage(tmp_path: Path, *, now: datetime | None = None) -> TuningStorage:
    fixed_now = now or datetime(2026, 7, 14, 1, 2, 3, 456789, tzinfo=timezone.utc)
    return TuningStorage(tmp_path, bounds=ParameterBounds(), now=lambda: fixed_now)


def make_capture_snapshot(*, include_diagnostics: bool = False) -> CaptureSnapshot:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :8] = (10, 20, 30)
    observation = BoardObservation(
        captured_ns=123456789,
        corners_px=((1.0, 1.0), (14.0, 1.0), (14.0, 10.0), (1.0, 10.0)),
        center_px=(7.5, 5.5),
        confidence=0.9,
        homography_valid=True,
    )
    return CaptureSnapshot(
        frame=Frame(sequence=42, captured_ns=123456789, image=image),
        parameters=PARAMETERS,
        runtime=RuntimeSnapshot(
            state="Connected",
            acquisition_fps=118.5,
            frame_count=100,
            timeout_count=2,
            sequence_gap_count=3,
            frame_age_ms=None,
            last_error=None,
        ),
        diagnostics=(
            ImageDiagnostics(
                source_sequence=42,
                computed_ns=123456999,
                gray_histogram=(1, 2, 3),
                blue_histogram=(4, 5, 6),
                green_histogram=(7, 8, 9),
                red_histogram=(10, 11, 12),
                dark_percent=1.5,
                bright_percent=2.5,
                focus_score=123.0,
                roi_px=(4, 3, 8, 6),
            )
            if include_diagnostics
            else None
        ),
        detection=DetectionSnapshot(
            enabled=True,
            detected=True,
            source_sequence=42,
            observation=observation,
            result_age_ms=4.5,
            error=None,
        ),
        overlay_options=OverlayOptions(),
        camera_config=CameraConfig(),
        camera_identity=CameraIdentity(model="MV-CA013-21UC", serial="00G02809155"),
    )


def write_profile(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def valid_profile_payload() -> dict[str, object]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "display_name": "室内普通光",
        "parameters": PARAMETERS.to_dict(),
    }


def test_profile_round_trip_and_sorted_listing(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    storage.save_profile("z-last", PARAMETERS, display_name="最后")
    storage.save_profile("indoor-normal", PARAMETERS, display_name="室内普通光")

    assert storage.list_profiles() == ["indoor-normal", "z-last"]
    profile = storage.load_profile("indoor-normal")
    assert profile.name == "indoor-normal"
    assert profile.display_name == "室内普通光"
    assert profile.parameters == PARAMETERS

    raw = yaml.safe_load((tmp_path / "profiles" / "indoor-normal.yaml").read_text(encoding="utf-8"))
    assert raw == valid_profile_payload()
    assert not [path for path in (tmp_path / "profiles").iterdir() if path.name.endswith(".tmp")]


@pytest.mark.parametrize(
    "name",
    ["../outside", "outside/child", "UPPER", "-leading", "a" * 65],
)
def test_profile_name_rejects_unsafe_or_invalid_slugs(tmp_path: Path, name: str) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(ValueError, match="profile name"):
        storage.save_profile(name, PARAMETERS)
    with pytest.raises(ValueError, match="profile name"):
        storage.load_profile(name)

    assert not (tmp_path.parent / "outside.yaml").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(schema_version=True), "schema_version"),
        (lambda payload: payload.update(unexpected=True), "profile keys"),
        (lambda payload: payload["parameters"].pop("gain_db"), "parameter keys"),
        (lambda payload: payload["parameters"].update(extra=1), "parameter keys"),
        (lambda payload: payload["parameters"].update(exposure_us=0), "exposure_us"),
        (lambda payload: payload["parameters"].update(auto_gain="false"), "auto_gain"),
    ],
)
def test_profile_load_validates_schema_exact_keys_types_and_bounds(
    tmp_path: Path,
    change: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    storage = make_storage(tmp_path)
    payload = valid_profile_payload()
    change(payload)
    write_profile(tmp_path / "profiles" / "invalid.yaml", payload)

    with pytest.raises(ValueError, match=message):
        storage.load_profile("invalid")


@pytest.mark.parametrize(
    "operation", ["save", "load"] + (["delete"] if os.name == "posix" else [])
)
def test_profile_operations_reject_target_swapped_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    storage = make_storage(tmp_path)
    storage.save_profile("race", PARAMETERS, display_name="before")
    target = tmp_path / "profiles" / "race.yaml"
    moved = tmp_path / "profiles" / "race-old.yaml"
    sentinel = target / "sentinel.txt"
    original_hook = storage._profile_operation_hook
    swapped = False

    def swap_target(stage: str, path: Path) -> None:
        nonlocal swapped
        original_hook(stage, path)
        if stage == {
            "save": "save:before-replace",
            "load": "load:before-open",
            "delete": "delete:before-unlink",
        }[operation] and not swapped:
            swapped = True
            target.rename(moved)
            target.mkdir()
            sentinel.write_text("safe", encoding="utf-8")

    monkeypatch.setattr(storage, "_profile_operation_hook", swap_target)

    error_type = (ValueError, OSError) if os.name != "posix" else ValueError
    with pytest.raises(
        error_type,
        match=(
            "profile path changed|symbolic link|regular file|"
            "does not overwrite existing profiles"
        ),
    ):
        if operation == "save":
            storage.save_profile("race", PARAMETERS, display_name="after")
        elif operation == "load":
            storage.load_profile("race")
        else:
            storage.delete_profile("race")

    assert sentinel.read_text(encoding="utf-8") == "safe"
    assert moved.exists()


@pytest.mark.skipif(os.name != "posix", reason="safe overwrite is POSIX-only")
def test_profile_save_atomically_overwrites_existing_regular_file(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    storage.save_profile("overwrite", PARAMETERS, display_name="before")
    updated = replace(PARAMETERS, exposure_us=900.0)

    storage.save_profile("overwrite", updated, display_name="after")

    profile = storage.load_profile("overwrite")
    assert profile.display_name == "after"
    assert profile.parameters == updated


@pytest.mark.skipif(os.name != "posix", reason="safe overwrite is POSIX-only")
def test_profile_save_is_atomic_and_preserves_previous_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = make_storage(tmp_path)
    storage.save_profile("stable", PARAMETERS, display_name="before")
    target = tmp_path / "profiles" / "stable.yaml"
    before = target.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        storage.save_profile("stable", PARAMETERS, display_name="after")

    assert target.read_bytes() == before
    assert not [path for path in target.parent.iterdir() if path.name.endswith(".tmp")]



@pytest.mark.skipif(os.name == "posix", reason="Windows fallback behavior")
def test_profile_save_fallback_rejects_overwrite_and_preserves_existing_file(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)
    storage.save_profile("stable", PARAMETERS, display_name="before")
    target = tmp_path / "profiles" / "stable.yaml"
    before = target.read_bytes()

    with pytest.raises(OSError, match="does not overwrite existing profiles"):
        storage.save_profile("stable", PARAMETERS, display_name="after")

    assert target.read_bytes() == before
    assert not [path for path in target.parent.iterdir() if path.name.endswith(".tmp")]

@pytest.mark.skipif(os.name == "posix", reason="Windows fallback behavior")
def test_profile_delete_fallback_is_disabled_and_preserves_file(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    storage.save_profile("stable", PARAMETERS)
    target = tmp_path / "profiles" / "stable.yaml"
    before = target.read_bytes()

    with pytest.raises(OSError, match="profile deletion is disabled"):
        storage.delete_profile("stable")

    assert target.read_bytes() == before


def test_profile_symlink_cannot_escape_profile_directory(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    outside = tmp_path / "outside.yaml"
    write_profile(outside, valid_profile_payload())
    link = tmp_path / "profiles" / "linked.yaml"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ValueError, match="profile path"):
        storage.load_profile("linked")



def test_capture_rejects_naive_created_datetime(tmp_path: Path) -> None:
    storage = make_storage(tmp_path, now=datetime(2026, 7, 14, 1, 2, 3, 456789))

    with pytest.raises(ValueError, match="timezone-aware"):
        storage.save_capture(
            make_capture_snapshot(), np.zeros((12, 16, 3), dtype=np.uint8)
        )


def test_capture_cleanup_does_not_remove_replacement_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot()
    original_write = storage._write_capture_png
    calls = 0

    def replace_capture_directory(temporary: Path, target: Path, image: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            capture_dir = target.parent
            moved = capture_dir.with_name(f"{capture_dir.name}-moved")
            capture_dir.rename(moved)
            capture_dir.mkdir()
            (capture_dir / "sentinel.txt").write_text("do not delete", encoding="utf-8")
            raise OSError("simulated capture failure")
        original_write(temporary, target, image)

    monkeypatch.setattr(storage, "_write_capture_png", replace_capture_directory)

    with pytest.raises(OSError, match="simulated capture failure"):
        storage.save_capture(snapshot, snapshot.frame.image)

    replacement = tmp_path / "captures" / "20260714_010203_456"
    assert (replacement / "sentinel.txt").read_text(encoding="utf-8") == "do not delete"
    assert (tmp_path / "captures" / "20260714_010203_456-moved").exists()


def test_capture_cleanup_survives_swap_after_identity_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot()
    capture_name = "20260714_010203_456"
    original_write = storage._write_capture_png
    original_lstat = Path.lstat
    cleanup_started = False
    swapped = False

    def fail_second_write(temporary: Path, target: Path, image: object) -> None:
        nonlocal cleanup_started
        if target.name == "overlay.png":
            cleanup_started = True
            raise OSError("simulated capture failure")
        original_write(temporary, target, image)

    def swap_after_lstat(path: Path):
        nonlocal swapped
        result = original_lstat(path)
        if cleanup_started and not swapped and path.name == capture_name:
            swapped = True
            moved = path.with_name(f"{capture_name}-moved")
            path.rename(moved)
            path.mkdir()
            (path / "sentinel.txt").write_text("do not delete", encoding="utf-8")
        return result

    monkeypatch.setattr(storage, "_write_capture_png", fail_second_write)
    monkeypatch.setattr(Path, "lstat", swap_after_lstat)

    with pytest.raises(OSError, match="simulated capture failure"):
        storage.save_capture(snapshot, snapshot.frame.image)

    replacement = tmp_path / "captures" / capture_name
    assert swapped
    assert (replacement / "sentinel.txt").read_text(encoding="utf-8") == "do not delete"
    assert (tmp_path / "captures" / f"{capture_name}-moved").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX cleanup uses a verified dir_fd")
def test_capture_cleanup_never_removes_public_directory_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot()
    capture_name = "20260714_010203_456"
    original_write = storage._write_capture_png
    original_lstat = Path.lstat
    cleanup_started = False
    validations = 0
    swapped = False

    def fail_second_write(temporary: Path, target: Path, image: object) -> None:
        nonlocal cleanup_started
        if target.name == "overlay.png":
            cleanup_started = True
            raise OSError("simulated capture failure")
        original_write(temporary, target, image)

    def swap_after_final_validation(path: Path):
        nonlocal validations, swapped
        result = original_lstat(path)
        if cleanup_started and path.name == capture_name:
            validations += 1
            if validations == 2 and not swapped:
                swapped = True
                moved = path.with_name(f"{capture_name}-moved-final")
                path.rename(moved)
                path.mkdir()
                (path / "sentinel.txt").write_text("do not delete", encoding="utf-8")
        return result

    monkeypatch.setattr(storage, "_write_capture_png", fail_second_write)
    monkeypatch.setattr(Path, "lstat", swap_after_final_validation)

    with pytest.raises(OSError, match="simulated capture failure"):
        storage.save_capture(snapshot, snapshot.frame.image)

    replacement = tmp_path / "captures" / capture_name
    assert swapped
    assert (replacement / "sentinel.txt").read_text(encoding="utf-8") == "do not delete"
    assert (tmp_path / "captures" / f"{capture_name}-moved-final").is_dir()


def test_capture_wraps_non_opencv_encoding_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = make_storage(tmp_path)

    def fail_encode(extension: str, image: object) -> object:
        raise TypeError("bad image")

    monkeypatch.setattr(cv2, "imencode", fail_encode)

    with pytest.raises(OSError, match="OpenCV failed to encode"):
        storage.save_capture(make_capture_snapshot(), object())


def test_yaml_value_recursively_converts_ndarrays(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot()
    snapshot = replace(
        snapshot,
        detection=replace(
            snapshot.detection,
            observation=None,
            error=np.array([["left", "right"]], dtype=object),
        ),
    )

    capture_dir = storage.save_capture(snapshot, snapshot.frame.image)
    metadata = yaml.safe_load((capture_dir / "metadata.yaml").read_text(encoding="utf-8"))

    assert metadata["detection"]["error"] == [["left", "right"]]


def test_capture_creates_png_pair_and_complete_yaml_metadata(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot()
    overlay = np.full_like(snapshot.frame.image, 200)

    capture_dir = storage.save_capture(snapshot, overlay)

    assert capture_dir.name == "20260714_010203_456"
    original = cv2.imdecode(
        np.fromfile(capture_dir / "original.png", dtype=np.uint8), cv2.IMREAD_COLOR
    )
    saved_overlay = cv2.imdecode(
        np.fromfile(capture_dir / "overlay.png", dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert np.array_equal(original, snapshot.frame.image)
    assert np.array_equal(saved_overlay, overlay)

    metadata = yaml.safe_load((capture_dir / "metadata.yaml").read_text(encoding="utf-8"))
    assert set(metadata) == {
        "schema_version",
        "created_utc",
        "frame",
        "fixed_format",
        "camera_parameters",
        "runtime",
        "diagnostics",
        "detection",
        "overlay_options",
    }
    assert metadata["schema_version"] == 1
    assert metadata["created_utc"] == "2026-07-14T01:02:03.456789Z"
    assert metadata["frame"] == {"sequence": 42, "captured_ns": 123456789}
    assert metadata["camera_parameters"] == PARAMETERS.to_dict()
    assert metadata["runtime"]["frame_age_ms"] is None
    assert metadata["runtime"]["last_error"] is None
    assert metadata["diagnostics"] is None
    assert metadata["detection"]["observation"]["center_px"] == [7.5, 5.5]
    assert metadata["fixed_format"]["camera_model"] == "MV-CA013-21UC"
    assert metadata["fixed_format"]["camera_serial"] == "00G02809155"
    assert not list(capture_dir.glob("*.tmp"))
    assert not list(capture_dir.glob("*.tmp.png"))


def test_capture_serializes_diagnostics_and_missing_detection_values(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot(include_diagnostics=True)
    snapshot = replace(
        snapshot,
        detection=DetectionSnapshot(enabled=True, detected=False),
    )

    capture_dir = storage.save_capture(snapshot, snapshot.frame.image)
    metadata = yaml.safe_load((capture_dir / "metadata.yaml").read_text(encoding="utf-8"))

    assert metadata["diagnostics"]["source_sequence"] == 42
    assert metadata["diagnostics"]["roi_px"] == [4, 3, 8, 6]
    assert metadata["detection"]["source_sequence"] is None
    assert metadata["detection"]["observation"] is None
    assert metadata["detection"]["result_age_ms"] is None
    assert metadata["detection"]["error"] is None


def test_capture_directories_are_unique_for_same_millisecond(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot()

    first = storage.save_capture(snapshot, snapshot.frame.image)
    second = storage.save_capture(snapshot, snapshot.frame.image)
    third = storage.save_capture(snapshot, snapshot.frame.image)

    assert [first.name, second.name, third.name] == [
        "20260714_010203_456",
        "20260714_010203_456-1",
        "20260714_010203_456-2",
    ]


@pytest.mark.skipif(os.name == "posix", reason="Windows fallback behavior")
def test_capture_failure_preserves_all_partial_files_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot()
    original_write = storage._write_capture_png

    def fail_second_write(temporary: Path, target: Path, image: object) -> None:
        if target.name == "overlay.png":
            temporary.write_bytes(b"partial overlay")
            raise OSError("simulated capture failure")
        original_write(temporary, target, image)

    monkeypatch.setattr(storage, "_write_capture_png", fail_second_write)

    with pytest.raises(OSError, match="simulated capture failure"):
        storage.save_capture(snapshot, snapshot.frame.image)

    partial = tmp_path / "captures" / "20260714_010203_456"
    assert (partial / "original.png").is_file()
    assert (partial / "overlay.tmp.png").read_bytes() == b"partial overlay"


@pytest.mark.skipif(os.name == "posix", reason="Windows fallback behavior")
def test_capture_cleanup_does_not_unlink_swapped_known_file_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot()
    original_write = storage._write_capture_png
    cleanup_started = False
    swapped = False
    original_unlink = Path.unlink

    def fail_second_write(temporary: Path, target: Path, image: object) -> None:
        nonlocal cleanup_started
        if target.name == "overlay.png":
            cleanup_started = True
            raise OSError("simulated capture failure")
        original_write(temporary, target, image)

    def swap_before_known_file_unlink(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal swapped
        if cleanup_started and not swapped and path.name == "original.png":
            swapped = True
            path.replace(path.with_name("original-created.png"))
            path.write_bytes(b"replacement")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(storage, "_write_capture_png", fail_second_write)
    monkeypatch.setattr(Path, "unlink", swap_before_known_file_unlink)

    with pytest.raises(OSError, match="simulated capture failure"):
        storage.save_capture(snapshot, snapshot.frame.image)

    partial = tmp_path / "captures" / "20260714_010203_456"
    assert not swapped
    assert (partial / "original.png").is_file()
    assert not (partial / "original-created.png").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX cleanup uses a verified dir_fd")
def test_capture_cleans_partial_directory_when_opencv_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = make_storage(tmp_path)
    snapshot = make_capture_snapshot()
    calls = 0
    original_imencode = cv2.imencode

    def controlled_encode(extension: str, image: np.ndarray) -> tuple[bool, np.ndarray]:
        nonlocal calls
        calls += 1
        if calls == 2:
            return False, np.empty(0, dtype=np.uint8)
        return original_imencode(extension, image)

    monkeypatch.setattr(cv2, "imencode", controlled_encode)
    with pytest.raises(OSError, match="OpenCV failed"):
        storage.save_capture(snapshot, snapshot.frame.image)

    captures = tmp_path / "captures"
    partial = captures / "20260714_010203_456"
    assert captures.is_dir()
    assert partial.is_dir()
    assert list(partial.iterdir()) == []
