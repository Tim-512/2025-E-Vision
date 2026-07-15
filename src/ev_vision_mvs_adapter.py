from __future__ import annotations

import time
import atexit
import logging
import threading
from ctypes import POINTER, cast
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


_LOGGER = logging.getLogger(__name__)


class MvsSdkError(RuntimeError):
    """Raised when the Hikrobot MVS SDK returns a non-success status."""

    def __init__(self, operation: str, code: int, detail: str | None = None) -> None:
        self.operation = operation
        self.code = int(code) & 0xFFFFFFFF
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{operation} failed with 0x{self.code:08x}{suffix}")


@dataclass(frozen=True)
class _SdkDevice:
    serial: str
    raw: Any
    owner: Any | None = None


@dataclass
class _Handle:
    serial: str
    camera: Any
    grabbing: bool = False
    closed: bool = False


@dataclass
class _FramePacket:
    """Zero-copy view of an MVS frame, valid only until release_frame()."""

    data: np.ndarray
    timestamp_ns: int
    sequence: int
    pixel_format: str
    sdk_frame: Any
    released: bool = False


class _MvsBindings:
    """Version-specific bindings for Hikrobot MVS 4.8.x on aarch64 Linux."""

    def __init__(self) -> None:
        try:
            import MvCameraControl_class as mvs
        except ImportError as exc:
            raise RuntimeError(
                "cannot import Hikrobot MVS Python SDK; add "
                "/opt/MVS/Samples/aarch64/Python/MvImport to PYTHONPATH"
            ) from exc
        self.mvs = mvs
        self.MV_ACCESS_Exclusive = int(mvs.MV_ACCESS_Exclusive)
        self.MV_E_NODATA = int(mvs.MV_E_NODATA) & 0xFFFFFFFF
        self.PIXEL_NAMES = {
            int(mvs.PixelType_Gvsp_BayerRG8): "BayerRG8",
        }
        self._initialized = False
        self._finalized = False

    def initialize(self) -> None:
        if self._initialized and not self._finalized:
            return
        ret = int(self.mvs.MvCamera.MV_CC_Initialize())
        if ret != 0:
            raise MvsSdkError("MV_CC_Initialize", ret)
        self._initialized = True
        self._finalized = False

    @staticmethod
    def _decode(value: Any) -> str:
        raw = memoryview(value).tobytes().split(b"\0", 1)[0]
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("latin-1", errors="replace")

    def enumerate_devices(self) -> list[_SdkDevice]:
        device_list = self.mvs.MV_CC_DEVICE_INFO_LIST()
        ret = int(self.mvs.MvCamera.MV_CC_EnumDevices(self.mvs.MV_USB_DEVICE, device_list))
        if ret != 0:
            raise MvsSdkError("MV_CC_EnumDevices", ret)
        devices: list[_SdkDevice] = []
        for index in range(int(device_list.nDeviceNum)):
            raw = cast(
                device_list.pDeviceInfo[index],
                POINTER(self.mvs.MV_CC_DEVICE_INFO),
            ).contents
            if int(raw.nTLayerType) != int(self.mvs.MV_USB_DEVICE):
                continue
            serial = self._decode(raw.SpecialInfo.stUsb3VInfo.chSerialNumber)
            devices.append(_SdkDevice(serial=serial, raw=raw, owner=device_list))
        return devices

    def create_camera(self) -> Any:
        return self.mvs.MvCamera()

    def create_frame(self) -> Any:
        return self.mvs.MV_FRAME_OUT()

    @staticmethod
    def array_from_frame(frame: Any) -> np.ndarray:
        frame_len = int(frame.stFrameInfo.nFrameLen)
        return np.ctypeslib.as_array(frame.pBufAddr, shape=(frame_len,))

    def finalize(self) -> None:
        if self._finalized or not self._initialized:
            return
        self._finalized = True
        self._initialized = False
        self.mvs.MvCamera.MV_CC_Finalize()


class NativeMvsApi:
    """Thin MVS API consumed by :class:`ev_vision.camera.hikrobot.HikrobotCamera`."""

    def __init__(
        self,
        bindings: Any | None = None,
        *,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self._bindings = bindings or _MvsBindings()
        self._clock_ns = clock_ns or time.monotonic_ns
        self._bindings.initialize()
        self._finalized = False
        atexit.register(self._bindings.finalize)

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._bindings.finalize()

    def __del__(self) -> None:
        try:
            self.finalize()
        except Exception:
            pass

    @staticmethod
    def _check(operation: str, code: int) -> None:
        if int(code) != 0:
            raise MvsSdkError(operation, code)

    def list_devices(self) -> list[str]:
        return [device.serial for device in self._bindings.enumerate_devices()]

    def open_device(self, serial: str) -> _Handle:
        device = next((item for item in self._bindings.enumerate_devices() if item.serial == serial), None)
        if device is None:
            raise LookupError(f"Hikrobot camera not found: {serial}")
        camera = self._bindings.create_camera()
        self._check("MV_CC_CreateHandle", camera.MV_CC_CreateHandle(device.raw))
        try:
            self._check(
                "MV_CC_OpenDevice",
                camera.MV_CC_OpenDevice(self._bindings.MV_ACCESS_Exclusive, 0),
            )
        except Exception:
            camera.MV_CC_DestroyHandle()
            raise
        return _Handle(serial=serial, camera=camera)

    def set_enum(self, handle: _Handle, name: str, value: str) -> None:
        self._require_open(handle)
        self._check(
            f"MV_CC_SetEnumValueByString({name})",
            handle.camera.MV_CC_SetEnumValueByString(name, value),
        )

    def set_int(self, handle: _Handle, name: str, value: int) -> None:
        self._require_open(handle)
        if name == "StreamBufferCountManual":
            setter = getattr(handle.camera, "MV_CC_SetImageNodeNum", None)
            if setter is not None:
                self._check("MV_CC_SetImageNodeNum", setter(int(value)))
                return
        self._check(f"MV_CC_SetIntValue({name})", handle.camera.MV_CC_SetIntValue(name, int(value)))

    def set_float(self, handle: _Handle, name: str, value: float) -> None:
        self._require_open(handle)
        self._check(f"MV_CC_SetFloatValue({name})", handle.camera.MV_CC_SetFloatValue(name, float(value)))

    def start_grabbing(self, handle: _Handle) -> None:
        self._require_open(handle)
        if handle.grabbing:
            return
        self._check("MV_CC_StartGrabbing", handle.camera.MV_CC_StartGrabbing())
        handle.grabbing = True

    def get_frame(self, handle: _Handle, timeout_ms: int) -> _FramePacket | None:
        self._require_open(handle)
        if not handle.grabbing:
            raise MvsSdkError("MV_CC_GetImageBuffer", 0x80000003, "camera is not grabbing")
        sdk_frame = self._bindings.create_frame()
        ret = int(handle.camera.MV_CC_GetImageBuffer(sdk_frame, int(timeout_ms))) & 0xFFFFFFFF
        received_ns = int(self._clock_ns())
        if ret == self._bindings.MV_E_NODATA:
            return None
        self._check("MV_CC_GetImageBuffer", ret)
        try:
            info = sdk_frame.stFrameInfo
            width = int(info.nWidth)
            height = int(info.nHeight)
            frame_len = int(info.nFrameLen)
            required = width * height
            if width <= 0 or height <= 0:
                raise MvsSdkError(
                    "MV_CC_GetImageBuffer",
                    0x80000027,
                    f"invalid frame dimensions: {width}x{height}",
                )
            if sdk_frame.pBufAddr is None:
                raise MvsSdkError("MV_CC_GetImageBuffer", 0x8000000D, "null frame buffer")
            if frame_len < required:
                raise MvsSdkError(
                    "MV_CC_GetImageBuffer",
                    0x8000000A,
                    f"short frame: {frame_len} bytes for {width}x{height}",
                )
            pixel_type = int(info.enPixelType)
            pixel_format = self._bindings.PIXEL_NAMES.get(
                pixel_type,
                f"0x{pixel_type & 0xFFFFFFFF:08x}",
            )
            data = self._bindings.array_from_frame(sdk_frame)[:required].reshape(height, width)
            # The installed MVS 4.8 Python report exposes nHostTimeStamp but
            # does not document its unit. Use receipt-time monotonic nanoseconds
            # until target-side measurements establish a safe conversion.
            return _FramePacket(
                data=data,
                timestamp_ns=received_ns,
                sequence=int(info.nFrameNum),
                pixel_format=pixel_format,
                sdk_frame=sdk_frame,
            )
        except Exception:
            handle.camera.MV_CC_FreeImageBuffer(sdk_frame)
            raise

    def release_frame(self, handle: _Handle, packet: _FramePacket) -> None:
        self._require_open(handle)
        if packet.released:
            return
        self._check("MV_CC_FreeImageBuffer", handle.camera.MV_CC_FreeImageBuffer(packet.sdk_frame))
        packet.released = True

    def _run_shutdown_stage(
        self,
        handle: _Handle,
        operation: str,
        callback: Callable[[], int],
    ) -> None:
        started_ns = time.monotonic_ns()
        _LOGGER.warning(
            "%s begin serial=%s thread=%s",
            operation,
            handle.serial,
            threading.current_thread().name,
        )
        try:
            self._check(operation, callback())
        except BaseException:
            elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
            _LOGGER.exception(
                "%s end status=error elapsed_ms=%.3f serial=%s",
                operation,
                elapsed_ms,
                handle.serial,
            )
            raise
        elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
        _LOGGER.warning(
            "%s end status=ok elapsed_ms=%.3f serial=%s",
            operation,
            elapsed_ms,
            handle.serial,
        )

    def stop_grabbing(self, handle: _Handle) -> None:
        if handle.closed or not handle.grabbing:
            return
        try:
            self._run_shutdown_stage(
                handle,
                "MV_CC_StopGrabbing",
                handle.camera.MV_CC_StopGrabbing,
            )
        finally:
            handle.grabbing = False

    def close_device(self, handle: _Handle) -> None:
        if handle.closed:
            return
        first_error: Exception | None = None
        try:
            self.stop_grabbing(handle)
        except Exception as exc:
            first_error = exc
        try:
            self._run_shutdown_stage(
                handle,
                "MV_CC_CloseDevice",
                handle.camera.MV_CC_CloseDevice,
            )
        except Exception as exc:
            first_error = first_error or exc
        try:
            self._run_shutdown_stage(
                handle,
                "MV_CC_DestroyHandle",
                handle.camera.MV_CC_DestroyHandle,
            )
        except Exception as exc:
            first_error = first_error or exc
        handle.closed = True
        if first_error is not None:
            raise first_error

    @staticmethod
    def _require_open(handle: _Handle) -> None:
        if handle.closed:
            raise MvsSdkError("camera operation", 0x80000000, "camera handle is closed")


def create_api() -> NativeMvsApi:
    return NativeMvsApi()
