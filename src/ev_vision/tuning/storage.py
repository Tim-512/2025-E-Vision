from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import stat
from typing import Any, IO
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


@dataclass(frozen=True)
class _PathIdentity:
    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, result: os.stat_result) -> _PathIdentity:
        return cls(result.st_dev, result.st_ino, result.st_mode)

    def same_object(self, result: os.stat_result) -> bool:
        return (result.st_dev, result.st_ino) == (self.device, self.inode)


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
        self._profiles_identity = _directory_identity(
            self._profiles_dir, "profile directory"
        )
        self._captures_identity = _directory_identity(
            self._captures_dir, "capture directory"
        )
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
        _require_directory_identity(
            self._profiles_dir, self._profiles_identity, "profile directory"
        )
        expected = self._profile_identity_if_present(target)
        self._profile_operation_hook("save:validated", target)

        payload: dict[str, object] = {"schema_version": PROFILE_SCHEMA_VERSION}
        if display_name is not None:
            payload["display_name"] = display_name
        payload["parameters"] = parameters.to_dict()
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            _write_yaml_temporary(temporary, payload)
            temporary_identity = _regular_file_identity(temporary, "temporary profile")
            self._profile_operation_hook("save:before-replace", target)
            _replace_verified_profile(
                self._profiles_dir,
                self._profiles_identity,
                temporary,
                temporary_identity,
                target,
                expected,
            )
            _fsync_directory(self._profiles_dir)
        except Exception:
            _unlink_if_regular_file(temporary)
            raise
        return NamedProfile(name=name, display_name=display_name, parameters=parameters)

    def load_profile(self, name: str) -> NamedProfile:
        target = self._profile_path(name)
        _require_directory_identity(
            self._profiles_dir, self._profiles_identity, "profile directory"
        )
        expected = self._validate_profile_path(target)
        self._profile_operation_hook("load:validated", target)
        try:
            self._profile_operation_hook("load:before-open", target)
            with _open_verified_profile(
                target, expected, self._profiles_identity
            ) as stream:
                raw = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid profile YAML: {exc}") from exc
        except OSError as exc:
            raise OSError(f"cannot read profile {name}: {exc}") from exc
        _require_directory_identity(
            self._profiles_dir, self._profiles_identity, "profile directory"
        )
        self._require_profile_identity(target, expected)

        display_name, parameters = self._parse_profile(raw)
        return NamedProfile(name=name, display_name=display_name, parameters=parameters)

    def list_profiles(self) -> list[str]:
        _require_directory_identity(
            self._profiles_dir, self._profiles_identity, "profile directory"
        )
        names: list[str] = []
        for path in self._profiles_dir.glob("*.yaml"):
            name = path.stem
            if not PROFILE_SLUG_PATTERN.fullmatch(name):
                continue
            self._validate_profile_path(path)
            names.append(name)
        return sorted(names)

    def delete_profile(self, name: str) -> None:
        target = self._profile_path(name)
        _require_directory_identity(
            self._profiles_dir, self._profiles_identity, "profile directory"
        )
        expected = self._validate_profile_path(target)
        self._profile_operation_hook("delete:validated", target)
        self._profile_operation_hook("delete:before-unlink", target)
        _unlink_verified_profile(
            self._profiles_dir, self._profiles_identity, target, expected
        )
        _fsync_directory(self._profiles_dir)

    def save_capture(self, snapshot: CaptureSnapshot, overlay_image: Any) -> Path:
        _require_directory_identity(
            self._captures_dir, self._captures_identity, "capture directory"
        )
        created = _as_utc(self._now())
        capture_dir, identity = self._create_capture_directory(created)
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
            _remove_directory_if_identity_matches(capture_dir, identity)
            raise
        return capture_dir

    def _profile_path(self, name: str) -> Path:
        if not isinstance(name, str) or PROFILE_SLUG_PATTERN.fullmatch(name) is None:
            raise ValueError("profile name must match ^[a-z0-9][a-z0-9-]{0,63}$")
        target = self._profiles_dir / f"{name}.yaml"
        if target.parent.resolve(strict=True) != self._profiles_dir:
            raise ValueError("profile path escapes profile directory")
        return target

    def _profile_identity_if_present(self, target: Path) -> _PathIdentity | None:
        try:
            return self._validate_profile_path(target)
        except FileNotFoundError:
            return None

    def _validate_profile_path(self, target: Path) -> _PathIdentity:
        result = _safe_lstat(target)
        if _is_link_or_reparse(result):
            raise ValueError("profile path must not be a symbolic link or reparse point")
        if not stat.S_ISREG(result.st_mode):
            raise ValueError("profile path must be a regular file")
        try:
            resolved = target.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"invalid profile path: {exc}") from exc
        if resolved.parent != self._profiles_dir:
            raise ValueError("profile path escapes profile directory")
        return _PathIdentity.from_stat(result)

    def _require_profile_identity(
        self,
        target: Path,
        expected: _PathIdentity | None,
        *,
        allow_missing: bool = False,
    ) -> _PathIdentity | None:
        try:
            current = self._validate_profile_path(target)
        except FileNotFoundError:
            if expected is None and allow_missing:
                return None
            raise ValueError("profile path changed during operation") from None
        if expected is None:
            raise ValueError("profile path changed during operation")
        if current != expected:
            raise ValueError("profile path changed during operation")
        return current

    def _profile_operation_hook(self, stage: str, target: Path) -> None:
        """Test seam for simulating path replacement between validation and use."""

    def _parse_profile(
        self, raw: object
    ) -> tuple[str | None, EditableCameraParameters]:
        if not isinstance(raw, Mapping):
            raise ValueError("profile must be a mapping")
        keys = frozenset(raw)
        if not _PROFILE_REQUIRED_KEYS <= keys or not keys <= _PROFILE_ALLOWED_KEYS:
            raise ValueError(
                "profile keys must be exactly schema_version, optional display_name, and parameters"
            )
        schema_version = raw["schema_version"]
        if type(schema_version) is not int or schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be integer {PROFILE_SCHEMA_VERSION}")

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

    def _create_capture_directory(
        self, created: datetime
    ) -> tuple[Path, _PathIdentity]:
        _require_directory_identity(
            self._captures_dir, self._captures_identity, "capture directory"
        )
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
            result = _safe_lstat(candidate)
            _require_directory_identity(
                self._captures_dir, self._captures_identity, "capture directory"
            )
            if (
                _is_link_or_reparse(result)
                or not stat.S_ISDIR(result.st_mode)
                or candidate.resolve(strict=True).parent != self._captures_dir
            ):
                _remove_directory_if_identity_matches(
                    candidate, _PathIdentity.from_stat(result)
                )
                raise ValueError("capture path escapes capture directory")
            return candidate, _PathIdentity.from_stat(result)

    @staticmethod
    def _write_capture_png(temporary: Path, target: Path, image: Any) -> None:
        try:
            encoded, png = cv2.imencode(".png", image)
        except Exception as exc:
            raise OSError(f"OpenCV failed to encode {target.name}: {exc}") from exc
        if not encoded:
            raise OSError(f"OpenCV failed to encode {target.name}")
        try:
            with temporary.open("xb") as stream:
                stream.write(png.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        except Exception:
            _unlink_if_regular_file(temporary)
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


def _directory_identity(path: Path, label: str) -> _PathIdentity:
    result = _safe_lstat(path)
    if _is_link_or_reparse(result) or not stat.S_ISDIR(result.st_mode):
        raise ValueError(f"{label} must be a real directory")
    return _PathIdentity.from_stat(result)


def _require_directory_identity(
    path: Path, expected: _PathIdentity, label: str
) -> None:
    current = _directory_identity(path, label)
    if current != expected:
        raise ValueError(f"{label} path changed during operation")


def _regular_file_identity(path: Path, label: str) -> _PathIdentity:
    result = _safe_lstat(path)
    if _is_link_or_reparse(result) or not stat.S_ISREG(result.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return _PathIdentity.from_stat(result)


def _identity_if_regular_file(path: Path) -> _PathIdentity | None:
    try:
        return _regular_file_identity(path, "profile path")
    except FileNotFoundError:
        return None


def _replace_verified_profile(
    directory: Path,
    expected_directory: _PathIdentity,
    temporary: Path,
    temporary_identity: _PathIdentity,
    target: Path,
    expected_target: _PathIdentity | None,
) -> None:
    if os.name == "posix":
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(directory, directory_flags)
        try:
            if _PathIdentity.from_stat(os.fstat(directory_fd)) != expected_directory:
                raise ValueError("profile directory path changed during operation")
            if _identity_if_regular_file(temporary) != temporary_identity:
                raise ValueError("temporary profile path changed during operation")
            if _identity_if_regular_file(target) != expected_target:
                raise ValueError("profile path changed during operation")
            os.replace(
                temporary.name,
                target.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            opened = os.open(
                target.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                if _PathIdentity.from_stat(os.fstat(opened)) != temporary_identity:
                    raise ValueError("profile path changed during save")
            finally:
                os.close(opened)
        finally:
            os.close(directory_fd)
        return

    _require_directory_identity(directory, expected_directory, "profile directory")
    if _identity_if_regular_file(temporary) != temporary_identity:
        raise ValueError("temporary profile path changed during operation")
    if _identity_if_regular_file(target) != expected_target:
        raise ValueError("profile path changed during operation")
    os.replace(temporary, target)
    _require_directory_identity(directory, expected_directory, "profile directory")
    if _identity_if_regular_file(target) != temporary_identity:
        raise ValueError("profile path changed during save")


def _unlink_verified_profile(
    directory: Path,
    expected_directory: _PathIdentity,
    target: Path,
    expected_target: _PathIdentity,
) -> None:
    if os.name == "posix":
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(directory, directory_flags)
        try:
            if _PathIdentity.from_stat(os.fstat(directory_fd)) != expected_directory:
                raise ValueError("profile directory path changed during operation")
            descriptor = os.open(
                target.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                if _PathIdentity.from_stat(os.fstat(descriptor)) != expected_target:
                    raise ValueError("profile path changed during operation")
                os.unlink(target.name, dir_fd=directory_fd)
            finally:
                os.close(descriptor)
        finally:
            os.close(directory_fd)
    else:
        _require_directory_identity(directory, expected_directory, "profile directory")
        if _identity_if_regular_file(target) != expected_target:
            raise ValueError("profile path changed during operation")
        target.unlink()

    _require_directory_identity(directory, expected_directory, "profile directory")
    try:
        remaining = _safe_lstat(target)
    except FileNotFoundError:
        return
    if _is_link_or_reparse(remaining) or not stat.S_ISREG(remaining.st_mode):
        raise ValueError("profile path changed during delete")
    raise ValueError("profile path changed during delete")


def _open_verified_profile(
    target: Path, expected: _PathIdentity, expected_directory: _PathIdentity
) -> IO[str]:
    if os.name == "posix":
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(target.parent, directory_flags)
        descriptor: int | None = None
        try:
            if _PathIdentity.from_stat(os.fstat(directory_fd)) != expected_directory:
                raise ValueError("profile directory path changed during operation")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(target.name, flags, dir_fd=directory_fd)
            opened = _PathIdentity.from_stat(os.fstat(descriptor))
            if opened != expected:
                raise ValueError("profile path changed during load")
            stream = os.fdopen(descriptor, "r", encoding="utf-8")
            descriptor = None
            return stream
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory_fd)

    _require_directory_identity(target.parent, expected_directory, "profile directory")
    if _identity_if_regular_file(target) != expected:
        raise ValueError("profile path changed during load")
    stream = target.open("r", encoding="utf-8")
    opened = _PathIdentity.from_stat(os.fstat(stream.fileno()))
    if opened != expected:
        stream.close()
        raise ValueError("profile path changed during load")
    return stream


def _safe_lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"invalid storage path {path.name}: {exc}") from exc


def _is_link_or_reparse(result: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(result, "st_file_attributes", 0)
    return stat.S_ISLNK(result.st_mode) or bool(attributes & reparse_flag)


def _write_yaml_temporary(temporary: Path, payload: object) -> None:
    text = yaml.safe_dump(
        _yaml_value(payload),
        allow_unicode=True,
        sort_keys=False,
    )
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_yaml(target: Path, payload: object) -> None:
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        _write_yaml_temporary(temporary, payload)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        _unlink_if_regular_file(temporary)
        raise


def _unlink_if_regular_file(path: Path) -> None:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return
    if _is_link_or_reparse(result) or not stat.S_ISREG(result.st_mode):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _remove_directory_if_identity_matches(
    path: Path, expected: _PathIdentity
) -> bool:
    """Best-effort cleanup without recursively deleting a public path."""
    known_files = (
        "original.tmp.png",
        "original.png",
        "overlay.tmp.png",
        "overlay.png",
        "metadata.yaml",
    )
    try:
        current = path.lstat()
    except OSError:
        return False
    if (
        not expected.same_object(current)
        or _is_link_or_reparse(current)
        or not stat.S_ISDIR(current.st_mode)
    ):
        return False

    if os.name == "posix":
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            directory_fd = os.open(path, flags)
        except OSError:
            return False
        try:
            if _PathIdentity.from_stat(os.fstat(directory_fd)) != expected:
                return False
            for name in known_files:
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except OSError:
                    pass
        finally:
            os.close(directory_fd)
    else:
        for name in known_files:
            try:
                current = path.lstat()
            except OSError:
                return False
            if (
                not expected.same_object(current)
                or _is_link_or_reparse(current)
                or not stat.S_ISDIR(current.st_mode)
            ):
                return False
            _unlink_if_regular_file(path / name)

    try:
        current = path.lstat()
    except OSError:
        return False
    if (
        not expected.same_object(current)
        or _is_link_or_reparse(current)
        or not stat.S_ISDIR(current.st_mode)
    ):
        return False
    try:
        path.rmdir()
    except OSError:
        return False
    return True


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("capture datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _yaml_value(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _yaml_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _yaml_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_yaml_value(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _yaml_value(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _yaml_value(value.item())
        except (TypeError, ValueError):
            pass
    return value
