from __future__ import annotations

import ctypes
import gc
import mmap
from pathlib import Path

JOINT_COUNT = 7
VISION_MAX_WIDTH = 1920
VISION_MAX_HEIGHT = 1200
VISION_MAX_PIXELS = VISION_MAX_WIDTH * VISION_MAX_HEIGHT


class ControlCommand(ctypes.Structure):
    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("action_deg", ctypes.c_float * JOINT_COUNT),
        ("valid", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 7),
    ]


class ControlState(ctypes.Structure):
    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("state_deg", ctypes.c_float * JOINT_COUNT),
        ("valid", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 7),
    ]


class ControlLeader(ctypes.Structure):
    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("leader_deg", ctypes.c_float * JOINT_COUNT),
        ("valid", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 7),
    ]


class ControlTelemetry(ctypes.Structure):
    _fields_ = [
        ("loop_count", ctypes.c_uint64),
        ("loop_dt_ms", ctypes.c_double),
        ("overruns", ctypes.c_uint64),
        ("leader_rx_timestamp_ns", ctypes.c_uint64),
        ("follower_tx_timestamp_ns", ctypes.c_uint64),
        ("follower_rx_timestamp_ns", ctypes.c_uint64),
        ("leader_rx_period_ms", ctypes.c_double),
        ("follower_tx_period_ms", ctypes.c_double),
    ]


class ControlSharedBlock(ctypes.Structure):
    _fields_ = [
        ("cmd_seq", ctypes.c_uint64),
        ("leader_seq", ctypes.c_uint64),
        ("state_seq", ctypes.c_uint64),
        ("stop_flag", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8 * 7),
        ("cmd", ControlCommand),
        ("leader", ControlLeader),
        ("state", ControlState),
        ("telemetry", ControlTelemetry),
    ]


class VisionTelemetry(ctypes.Structure):
    _fields_ = [
        ("frame_seq", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("zed_ms", ctypes.c_float),
        ("usb_ms", ctypes.c_float),
        ("yolo_ms", ctypes.c_float),
        ("pipeline_ms", ctypes.c_float),
        ("yolo_busy", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
    ]


class VisionSharedBlock(ctypes.Structure):
    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("stop_flag", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8 * 7),
        ("telemetry", VisionTelemetry),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("depth_width", ctypes.c_uint32),
        ("depth_height", ctypes.c_uint32),
        ("rgb", ctypes.c_uint8 * (VISION_MAX_PIXELS * 3)),
        ("yolo_rgb", ctypes.c_uint8 * (VISION_MAX_PIXELS * 3)),
        ("depth_mm", ctypes.c_uint16 * VISION_MAX_PIXELS),
    ]


class MappedStruct:
    def __init__(self, shm_name: str, struct_type: type[ctypes.Structure]):
        name = shm_name[1:] if shm_name.startswith("/") else shm_name
        self.path = Path("/dev/shm") / name
        self.struct_type = struct_type
        self.size = ctypes.sizeof(struct_type)
        self._fp = self.path.open("r+b")
        self._mmap = mmap.mmap(self._fp.fileno(), self.size, access=mmap.ACCESS_WRITE)
        self.view = struct_type.from_buffer(self._mmap)

    def close(self) -> None:
        # Release exported ctypes view before closing mmap.
        try:
            del self.view
        except Exception:
            pass
        # A caller may still hold transient ctypes references (for example, nested
        # fields from the latest loop iteration). Collect and retry close once.
        try:
            self._mmap.close()
        except BufferError:
            gc.collect()
            try:
                self._mmap.close()
            except BufferError:
                # Best-effort cleanup: process is usually shutting down already.
                pass
        self._fp.close()
