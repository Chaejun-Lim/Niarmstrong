import cv2
import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time


class CameraInterface(ABC):
    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """Return a real RGB uint8 frame in HWC format, or None on acquisition failure."""

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""


class ZedXCameraInterface(CameraInterface):
    """Synchronized left RGB and metric depth from a ZED X stereo camera."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth_mode: str = "performance",
        depth_sanitize: bool = False,
        depth_clip: bool = False,
        retrieve_scale: float = 1.0,
        capture_depth: bool = True,
    ):
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise RuntimeError(
                "pyzed.sl is unavailable. Install the ZED SDK Python API for this Jetson before recording."
            ) from exc

        self.sl = sl
        self.width = width
        self.height = height
        self.zed = sl.Camera()
        init = sl.InitParameters()
        # ZED X supports HD1200 natively (HD720 is not supported).
        # Frames are resized to the requested width/height below 
        # so all recorded camera streams share one schema.
        init.camera_resolution = sl.RESOLUTION.HD1200
        init.camera_fps = fps
        depth_mode_map = {
            "none": sl.DEPTH_MODE.NONE,
            "performance": sl.DEPTH_MODE.PERFORMANCE,
            "quality": sl.DEPTH_MODE.QUALITY,
            "ultra": sl.DEPTH_MODE.ULTRA,
            "neural": sl.DEPTH_MODE.NEURAL,
            "neural_plus": sl.DEPTH_MODE.NEURAL_PLUS,
            "neural_light": sl.DEPTH_MODE.NEURAL_LIGHT,
        }
        self.capture_depth = capture_depth
        mode_key = depth_mode.lower().strip()
        if mode_key not in depth_mode_map:
            raise ValueError(f"Unsupported depth_mode: {depth_mode}")
        init.depth_mode = depth_mode_map[mode_key] if self.capture_depth else sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.MILLIMETER
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError(f"Could not open ZED X: {status}")

        camera_info = self.zed.get_camera_information()
        selected_resolution = camera_info.camera_configuration.resolution
        self.native_width = int(selected_resolution.width)
        self.native_height = int(selected_resolution.height)

        self.runtime = sl.RuntimeParameters()
        self.left = sl.Mat()
        self.depth = sl.Mat() if self.capture_depth else None
        self.depth_sanitize = depth_sanitize
        self.depth_clip = depth_clip
        self.retrieve_scale = float(retrieve_scale)
        if not 0.0 < self.retrieve_scale <= 1.0:
            raise ValueError(f"retrieve_scale must be within (0, 1], got {retrieve_scale}")
        
        retrieve_w = max(1, int(self.width * self.retrieve_scale))
        retrieve_h = max(1, int(self.height * self.retrieve_scale))
        self._retrieve_resolution = sl.Resolution(retrieve_w, retrieve_h)

    @property
    def is_open(self) -> bool:
        return self.zed is not None and self.zed.is_opened()

    def get_frames(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Return left RGB and, when enabled, depth from the same ZED grab."""
        if not self.is_open or self.zed.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
            return None
        if self.zed.retrieve_image(self.left, self.sl.VIEW.LEFT, self.sl.MEM.CPU, self._retrieve_resolution) != self.sl.ERROR_CODE.SUCCESS:
            return None

        # ZED's U8_C4 image is BGRA for OpenCV interoperability.
        left_bgra = self.left.get_data()
        if left_bgra is None:
            return None
        rgb = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2RGB)
        depth_mm = None
        if self.capture_depth:
            if self.depth is None or self.zed.retrieve_measure(
                self.depth, self.sl.MEASURE.DEPTH, self.sl.MEM.CPU, self._retrieve_resolution
            ) != self.sl.ERROR_CODE.SUCCESS:
                return None
            depth_mm = self.depth.get_data()
            if depth_mm is None:
                return None

        # Fast path: skip expensive cleanup unless explicitly requested.
        if depth_mm is not None:
            depth_mm = depth_mm.astype(np.float32, copy=False)
            if self.depth_sanitize:
                depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
            if self.depth_clip:
                valid = depth_mm > 0.0
                depth_mm[valid] = np.clip(depth_mm[valid], 300.0, 5000.0)

        if (rgb.shape[1], rgb.shape[0]) != (self.width, self.height):
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
            if depth_mm is not None:
                depth_mm = cv2.resize(depth_mm, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        depth_out = None if depth_mm is None else depth_mm[..., np.newaxis] if depth_mm.ndim == 2 else depth_mm
        return np.ascontiguousarray(rgb), np.ascontiguousarray(depth_out)

    def get_frame(self) -> np.ndarray | None:
        frames = self.get_frames()
        return None if frames is None else frames[0]

    def release(self) -> None:
        if self.zed is not None:
            self.zed.close()
            self.zed = None


class OpenCVCameraInterface(CameraInterface):
    def __init__(self, camera_idx: int = 0, height: int = 480, width: int = 640, fps: int = 30):
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.height = height
        self.width = width
        if not self.is_open:
            raise RuntimeError(f"Could not open camera index {camera_idx}.")

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_frame(self) -> np.ndarray | None:
        if not self.is_open:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class NullCameraInterface(CameraInterface):
    """Camera stub used when USB capture is optional or unavailable."""

    def __init__(self):
        self._open = True

    @property
    def is_open(self) -> bool:
        return self._open

    def get_frame(self) -> np.ndarray | None:
        return None

    def release(self) -> None:
        self._open = False

class MultiCameraInterface:
    """Capture ZED left, optional depth, USB, and optional YOLO streams."""

    def __init__(
        self,
        zed: ZedXCameraInterface,
        usb: OpenCVCameraInterface,
        yolo_annotator=None,
        yolo_weights: Path | None = None,
        yolo_confidence: float = 0.1,
        yolo_device: str | int | None = None,
        yolo_imgsz: int = 320,
        yolo_max_det: int = 50,
        yolo_quantize: int | str | None = None,
        yolo_classes: list[str] | None = None,
        async_yolo: bool = True,
        require_usb: bool = True,
        render_yolo: bool = True,
        include_zed_depth: bool = True,
    ):
        self.zed = zed
        self.usb = usb
        if yolo_annotator is None:
            if yolo_weights is None:
                self.yolo_annotator = None
            else:
                from yolo_interface import YoloAnnotator

                self.yolo_annotator = YoloAnnotator(
                    yolo_weights,
                    yolo_confidence,
                    yolo_device,
                    class_filter=yolo_classes,
                    imgsz=yolo_imgsz,
                    max_det=yolo_max_det,
                    quantize=yolo_quantize,
                )
        else:
            self.yolo_annotator = yolo_annotator

        # 프레임 드롭 방지를 위한 버퍼
        self._last_frames: dict[str, np.ndarray] | None = None
        # 카메라 캡처를 동시 실행해 직렬 대기 시간을 줄인다.
        self._executor = ThreadPoolExecutor(max_workers=2)
        # YOLO 추론을 분리해 캡처 파이프라인을 막지 않도록 한다.
        self._yolo_executor = ThreadPoolExecutor(max_workers=1)
        self._yolo_future = None
        self._latest_annotated: np.ndarray | None = None
        self._latest_yolo_ms = 0.0
        self.async_yolo = async_yolo
        self.require_usb = require_usb
        self.render_yolo = render_yolo
        self.include_zed_depth = include_zed_depth
        self._usb_fallback_rgb = np.zeros((self.zed.height, self.zed.width, 3), dtype=np.uint8)
        self.last_profile: dict[str, float] = {
            "zed_ms": 0.0,
            "usb_ms": 0.0,
            "yolo_ms": 0.0,
            "pipeline_ms": 0.0,
            "used_cached": 0.0,
            "yolo_pending": 0.0,
            "frame_timestamp_ns": 0.0,
        }

    def _annotate_timed(self, zed_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        if self.yolo_annotator is None:
            return zed_rgb, 0.0
        annotated = self.yolo_annotator.annotate(zed_rgb, draw=self.render_yolo)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return annotated, elapsed_ms

    @property
    def is_open(self) -> bool:
        return self.zed.is_open and (self.usb.is_open or not self.require_usb)

    def get_frames(self) -> dict[str, np.ndarray] | None:
        """Get frames from all cameras. If a camera drops, use last valid frame."""
        pipeline_start = time.perf_counter()

        def _timed_call(fn):
            start = time.perf_counter()
            result = fn()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return result, elapsed_ms

        zed_future = self._executor.submit(_timed_call, self.zed.get_frames)
        usb_future = self._executor.submit(_timed_call, self.usb.get_frame)
        zed_frames, zed_ms = zed_future.result()
        usb_rgb, usb_ms = usb_future.result()

        if self._yolo_future is not None and self._yolo_future.done():
            try:
                self._latest_annotated, self._latest_yolo_ms = self._yolo_future.result()
            except Exception:
                self._latest_annotated = None
                self._latest_yolo_ms = 0.0
            self._yolo_future = None

        # 첫 번째 프레임 이후부터는 이전 버퍼 사용 (드롭 방지)
        if zed_frames is None or (self.require_usb and usb_rgb is None):
            if self._last_frames is not None:
                self.last_profile = {
                    "zed_ms": zed_ms,
                    "usb_ms": usb_ms,
                    "yolo_ms": self._latest_yolo_ms,
                    "pipeline_ms": (time.perf_counter() - pipeline_start) * 1000.0,
                    "used_cached": 1.0,
                    "yolo_pending": 1.0 if self._yolo_future is not None else 0.0,
                    "frame_timestamp_ns": self.last_profile.get("frame_timestamp_ns", 0.0),
                }
                return self._last_frames
            return None

        if usb_rgb is None:
            usb_rgb = self._usb_fallback_rgb
            usb_ms = 0.0

        zed_rgb, zed_depth = zed_frames
        if self.include_zed_depth and zed_depth is None:
            raise RuntimeError("ZED depth stream was requested but is unavailable.")

        if self.yolo_annotator is None:
            zed_yolo = None
            yolo_ms = 0.0
            yolo_pending = 0.0
        elif self.async_yolo:
            if self._yolo_future is None:
                self._yolo_future = self._yolo_executor.submit(self._annotate_timed, zed_rgb.copy())
            zed_yolo = self._latest_annotated if self._latest_annotated is not None else zed_rgb
            yolo_ms = self._latest_yolo_ms
            yolo_pending = 1.0 if self._yolo_future is not None else 0.0
        else:
            yolo_start = time.perf_counter()
            zed_yolo = self.yolo_annotator.annotate(zed_rgb, draw=self.render_yolo)
            yolo_ms = (time.perf_counter() - yolo_start) * 1000.0
            self._latest_annotated = zed_yolo
            self._latest_yolo_ms = yolo_ms
            yolo_pending = 0.0

        frame_timestamp_ns = float(time.perf_counter_ns())

        frames = {
            "observation.images.zed_left": zed_rgb,
            "observation.images.usb": usb_rgb,
        }
        if self.include_zed_depth:
            frames["observation.images.zed_depth"] = zed_depth
        if zed_yolo is not None:
            frames["observation.images.zed_left_yolo"] = zed_yolo

        self._last_frames = frames

        self.last_profile = {
            "zed_ms": zed_ms,
            "usb_ms": usb_ms,
            "yolo_ms": yolo_ms,
            "pipeline_ms": (time.perf_counter() - pipeline_start) * 1000.0,
            "used_cached": 0.0,
            "yolo_pending": yolo_pending,
            "frame_timestamp_ns": frame_timestamp_ns,
        }

        return frames

    def release(self) -> None:
        if self._yolo_future is not None:
            try:
                self._yolo_future.cancel()
            except Exception:
                pass
        self._yolo_executor.shutdown(wait=False)
        self._executor.shutdown(wait=False)
        self.zed.release()
        self.usb.release()
