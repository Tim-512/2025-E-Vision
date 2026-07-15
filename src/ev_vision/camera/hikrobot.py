from __future__ import annotations

from typing import Any, Protocol

import cv2
import numpy as np

from ev_vision.camera.latest_frame import LatestFrameBuffer
from ev_vision.config import CameraConfig
from ev_vision.models import Frame


class CameraDisconnected(RuntimeError):
    pass


class MvsApi(Protocol):
    def list_devices(self) -> list[str]: ...
    def open_device(self, serial: str) -> Any: ...
    def set_enum(self, handle: Any, name: str, value: str) -> None: ...
    def set_int(self, handle: Any, name: str, value: int) -> None: ...
    def set_float(self, handle: Any, name: str, value: float) -> None: ...
    def start_grabbing(self, handle: Any) -> None: ...
    def get_frame(self, handle: Any, timeout_ms: int) -> Any | None: ...
    def release_frame(self, handle: Any, packet: Any) -> None: ...
    def stop_grabbing(self, handle: Any) -> None: ...
    def close_device(self, handle: Any) -> None: ...


class HikrobotCamera:
    def __init__(self, api: MvsApi, config: CameraConfig, *, serial_number: str | None = None) -> None:
        self.api = api
        self.config = config
        self.serial_number = serial_number
        self._handle: Any | None = None
        self._buffer = LatestFrameBuffer()

    def open(self) -> "HikrobotCamera":
        if self._handle is not None:
            return self
        devices = self.api.list_devices()
        if not devices:
            raise CameraDisconnected("no Hikrobot camera found")
        serial = self.serial_number or devices[0]
        if serial not in devices:
            raise CameraDisconnected(f"Hikrobot camera not found: {serial}")
        try:
            handle = self.api.open_device(serial)
            self.api.set_int(handle, "Width", self.config.width)
            self.api.set_int(handle, "Height", self.config.height)
            self.api.set_enum(handle, "PixelFormat", self.config.pixel_format)
            self.api.set_enum(handle, "ExposureAuto", "Continuous" if self.config.auto_exposure else "Off")
            self.api.set_float(handle, "ExposureTime", float(self.config.exposure_us))
            self.api.set_enum(handle, "GainAuto", "Continuous" if self.config.auto_gain else "Off")
            self.api.set_float(handle, "Gain", float(self.config.gain_db))
            self.api.set_enum(handle, "BalanceWhiteAuto", "Continuous" if self.config.auto_white_balance else "Off")
            self.api.set_float(handle, "AcquisitionFrameRate", float(self.config.acquisition_fps))
            self.api.set_int(handle, "StreamBufferCountManual", int(self.config.buffer_size))
            self.api.start_grabbing(handle)
        except Exception as exc:
            try:
                self.api.close_device(handle)
            except Exception:
                pass
            raise CameraDisconnected(f"failed to initialize Hikrobot camera {serial}") from exc
        self._handle = handle
        return self

    def reconfigure(self, config: CameraConfig) -> None:
        """Apply editable controls without closing or recreating the MVS handle."""
        handle = self._handle
        if handle is None:
            raise CameraDisconnected("camera is not open")
        fixed_fields = ("width", "height", "pixel_format", "buffer_size")
        changed_fixed = [
            name for name in fixed_fields if getattr(config, name) != getattr(self.config, name)
        ]
        if changed_fixed:
            raise ValueError(
                "cannot reconfigure fixed camera fields in place: "
                + ", ".join(changed_fixed)
            )

        self.api.stop_grabbing(handle)
        try:
            self.api.set_enum(
                handle,
                "ExposureAuto",
                "Continuous" if config.auto_exposure else "Off",
            )
            self.api.set_float(handle, "ExposureTime", float(config.exposure_us))
            self.api.set_enum(
                handle,
                "GainAuto",
                "Continuous" if config.auto_gain else "Off",
            )
            self.api.set_float(handle, "Gain", float(config.gain_db))
            self.api.set_enum(
                handle,
                "BalanceWhiteAuto",
                "Continuous" if config.auto_white_balance else "Off",
            )
            self.api.set_float(
                handle,
                "AcquisitionFrameRate",
                float(config.acquisition_fps),
            )
        except BaseException:
            self.api.start_grabbing(handle)
            raise
        self.api.start_grabbing(handle)
        self.config = config

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            self.api.stop_grabbing(handle)
        finally:
            self.api.close_device(handle)

    def __enter__(self) -> "HikrobotCamera":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def read(self, *, timeout_ms: int = 100) -> Frame:
        if self._handle is None:
            raise CameraDisconnected("camera is not open")
        packet = None
        try:
            packet = self.api.get_frame(self._handle, timeout_ms)
            if packet is None:
                raise TimeoutError("timed out waiting for Hikrobot frame")
            if packet.pixel_format != "BayerRG8":
                raise CameraDisconnected(f"unsupported pixel format: {packet.pixel_format}")
            raw = np.asarray(packet.data, dtype=np.uint8)
            if raw.shape != (self.config.height, self.config.width):
                raise CameraDisconnected(f"unexpected frame shape: {raw.shape}")
            image = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
            frame = Frame(sequence=int(packet.sequence), captured_ns=int(packet.timestamp_ns), image=image)
            self._buffer.publish(frame)
            return frame
        except TimeoutError:
            raise
        except CameraDisconnected:
            raise
        except Exception as exc:
            raise CameraDisconnected("Hikrobot camera disconnected or frame acquisition failed") from exc
        finally:
            if packet is not None:
                self.api.release_frame(self._handle, packet)

    def latest(self) -> Frame | None:
        return self._buffer.peek()


def create_native_api(module_name: str = "ev_vision_mvs_adapter") -> MvsApi:
    """Load the thin adapter supplied for the installed Jetson MVS SDK version."""
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"cannot import {module_name}; install Hikrobot MVS and provide its version-specific create_api() adapter"
        ) from exc
    factory = getattr(module, "create_api", None)
    if not callable(factory):
        raise RuntimeError(f"{module_name} must export a callable create_api()")
    return factory()
