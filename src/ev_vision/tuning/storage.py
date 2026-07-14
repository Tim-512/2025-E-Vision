from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

import cv2
import yaml

from .models import CaptureSnapshot, EditableCameraParameters, ParameterBounds


PROFILE_SCHEMA_VERSION = 1
CAPTURE_SCHEMA_VERSION = 1
PROFILE_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PARAMETER_KEYS = frozenset(
    {
        "exposure_us",
        "gain_db",
        "acquisition_fps",
        "auto_exposure",
        "auto_gain",
        "auto_white_balance",
    }
)
_PROFILE_REQUIRED_KEYS = frozenset({"schema_version", "parameters"})
_PROFILE_ALLOWED_KEYS = frozenset({"schema_version", "display_name", "parameters"})


@dataclass(frozen=True)
class NamedProfile:
    name: str
    display_name: str | None
    parameters: EditableCameraParameters


class TuningStorage:
    """Path-safe YAML profile and synchronized capture persistence."""

    def __init__(
        self,
        root: str | Path,
        *,
        bounds: ParameterBounds | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._root = _prepare_directory(Path(root), "storage root")
        self._profiles_dir = _prepare_directory(self._root / "profiles", "profile directory")
        self._captures_dir = _prepare_directory(self._root / "captures", "capture directory")
        self._bounds = bounds or ParameterBounds()
        self._now = now or (lambda: datetime.now(timezone.utc))

    @property
    def profiles_dir(self) -> Path:
        return self._profiles_dir

    @property
    def captures_dir(self) -> Path:
        return self._captures_dir

    def save_profile(
        self,
        name: str,
        parameters: EditableCameraParameters,
        *,
        display_name: str | None = None,
    ) -> NamedProfile:
        target = self._profile_path(name)
        if display_name is not None and not isinstance(display_name, str):
            raise ValueError("display_name must be a string or null")
        self._bounds.validate(parameters)
        self._reject_unsafe_existing_profile(target)

        payload: dict[str, object] = {
            "schema_version": PROFILE_SCHEMA_VERSION,
        }
        if display_name is not None:
            payload["display_name"] = display_name
        payload["parameters"] = parameters.to_dict()
        _atomic_write_yaml(target, payload)
        return NamedProfile(name=name, display_name=display_name, parameters=parameters)

    def load_profile(self, name: str) -> NamedProfile:
        target = self._profile_path(name)
        self._validate_existing_profile_path(target)
        try:
            raw = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid profile YAML: {exc}") from exc
        except OSError as exc:
            raise OSError(f"cannot read profile {name}: {exc}") from exc

        display_name, parameters = self._parse_profile(raw)
        return NamedProfile(name=name, display_name=display_name, parameters=parameters)

    def list_profiles(self) -> list[str]:
        names: list[str] = []
        for path in self._profiles_dir.glob("*.yaml"):
            name = path.stem
            if not PROFILE_SLUG_PATTERN.fullmatch(name):
                continue
            self._validate_existing_profile_path(path)
            names.append(name)
        return sorted(names)

    def delete_profile(self, name: str) -> None:
        target = self._profile_path(name)
        self._validate_existing_profile_path(target)
        target.unlink()

    def save_capture(self, snapshot: CaptureSnapshot, overlay_image: Any) -> Path:
        created = _as_utc(self._now())
        capture_dir = self._create_capture_directory(created)
        try:
            self._write_capture_png(
                capture_dir / "original.tmp.png",
                capture_dir / "original.png",
                snapshot.frame.image,
            )
            self._write_capture_png(
                capture_dir / "overlay.tmp.png",
                capture_dir / "overlay.png",
                overlay_image,
            )
            _atomic_write_yaml(
                capture_dir / "metadata.yaml",
                self._capture_metadata(snapshot, created),
            )
        except Exception:
            shutil.rmtree(capture_dir, ignore_errors=True)
            raise
        return capture_dir

    def _profile_path(self, name: str) -> Path:
        if not isinstance(name, str) or PROFILE_SLUG_PATTERN.fullmatch(name) is None:
            raise ValueError(
                "profile name must match ^[a-z0-9][a-z0-9-]{0,63}$"
            )
        target = self._profiles_dir / f"{name}.yaml"
        if target.parent.resolve(strict=True) != self._profiles_dir:
            raise ValueError("profile path escapes profile directory")
        return target

    def _reject_unsafe_existing_profile(self, target: Path) -> None:
        if target.is_symlink():
            raise ValueError("profile path must not be a symbolic link")
        if target.exists() and not target.is_file():
            raise ValueError("profile path must be a regular file")

    def _validate_existing_profile_path(self, target: Path) -> None:
        if not target.exists() and not target.is_symlink():
            raise FileNotFoundError(target)
        if target.is_symlink():
            raise ValueError("profile path must not be a symbolic link")
        if not target.is_file():
            raise ValueError("profile path must be a regular file")
        try:
            resolved = target.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"invalid profile path: {exc}") from exc
        if resolved.parent != self._profiles_dir:
            raise ValueError("profile path escapes profile directory")

    def _parse_profile(
        self, raw: object
    ) -> tuple[str | None, EditableCameraParameters]:
        if not isinstance(raw, Mapping):
            raise ValueError("profile must be a mapping")
        keys = frozenset(raw)
        if not _PROFILE_REQUIRED_KEYS <= keys or not keys <= _PROFILE_ALLOWED_KEYS:
            raise ValueError("profile keys must be exactly schema_version, optional display_name, and parameters")
        if raw["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PROFILE_SCHEMA_VERSION}")

        display_name = raw.get("display_name")
        if display_name is not None and not isinstance(display_name, str):
            raise ValueError("display_name must be a string or null")

        parameter_values = raw["parameters"]
        if not isinstance(parameter_values, Mapping):
            raise ValueError("parameters must be a mapping")
        if frozenset(parameter_values) != _PARAMETER_KEYS:
            raise ValueError("parameter keys must exactly match editable camera parameters")
        parameters = EditableCameraParameters(
            exposure_us=parameter_values["exposure_us"],
            gain_db=parameter_values["gain_db"],
            acquisition_fps=parameter_values["acquisition_fps"],
            auto_exposure=parameter_values["auto_exposure"],
            auto_gain=parameter_values["auto_gain"],
            auto_white_balance=parameter_values["auto_white_balance"],
        )
        return display_name, self._bounds.validate(parameters)

    def _create_capture_directory(self, created: datetime) -> Path:
        if self._captures_dir.is_symlink() or self._captures_dir.resolve(strict=True) != self._captures_dir:
            raise ValueError("capture directory path is unsafe")
        base_name = created.strftime("%Y%m%d_%H%M%S_") + f"{created.microsecond // 1000:03d}"
        suffix = 0
        while True:
            name = base_name if suffix == 0 else f"{base_name}-{suffix}"
            candidate = self._captures_dir / name
            try:
                candidate.mkdir()
            except FileExistsError:
                suffix += 1
                continue
            if candidate.is_symlink() or candidate.resolve(strict=True).parent != self._captures_dir:
                shutil.rmtree(candidate, ignore_errors=True)
                raise ValueError("capture path escapes capture directory")
            return candidate

    @staticmethod
    def _write_capture_png(temporary: Path, target: Path, image: Any) -> None:
        try:
            encoded, png = cv2.imencode(".png", image)
        except cv2.error as exc:
            raise OSError(f"OpenCV failed to encode {target.name}: {exc}") from exc
        if not encoded:
            raise OSError(f"OpenCV failed to encode {target.name}")
        try:
            with temporary.open("xb") as stream:
                stream.write(png.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _capture_metadata(snapshot: CaptureSnapshot, created: datetime) -> dict[str, object]:
        config = snapshot.camera_config
        identity = snapshot.camera_identity
        return {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "created_utc": created.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "frame": {
                "sequence": snapshot.frame.sequence,
                "captured_ns": snapshot.frame.captured_ns,
            },
            "fixed_format": {
                "camera_model": identity.model,
                "camera_serial": identity.serial,
                "width": config.width,
                "height": config.height,
                "pixel_format": config.pixel_format,
                "buffer_size": config.buffer_size,
            },
            "camera_parameters": snapshot.parameters.to_dict(),
            "runtime": _yaml_value(snapshot.runtime),
            "diagnostics": _yaml_value(snapshot.diagnostics),
            "detection": _yaml_value(snapshot.detection),
            "overlay_options": _yaml_value(snapshot.overlay_options),
        }


def _prepare_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _atomic_write_yaml(target: Path, payload: object) -> None:
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        text = yaml.safe_dump(
            _yaml_value(payload),
            allow_unicode=True,
            sort_keys=False,
        )
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _yaml_value(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _yaml_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _yaml_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_yaml_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value
