"""Record teleop episodes and save as shard-based dataset layout.

Output layout:
- meta/info.json
- meta/stats.json
- meta/tasks.parquet
- meta/episodes/chunk-000/file-{episode_index:03d}.parquet
- data/chunk-000/file-{episode_index:03d}.parquet
- videos/{camera_key}/chunk-000/file-{episode_index:03d}.mp4

This script keeps the same capture pipeline as hybrid/main.py and only changes
how recordings are serialized on disk.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pyarrow is required for main2.py. Install it with: uv pip install pyarrow"
    ) from exc

from camera_interface import OpenCVCameraInterface
from main import (
    ControlSharedReader,
    LatestCameraStream,
    _keyboard_early_stop,
    record_episode,
    resolve_repo_id,
    upload_to_huggingface,
)
from shm_layout import MappedStruct, VisionSharedBlock


JOINT_NAMES = [
    "joint_1_deg",
    "joint_2_deg",
    "joint_3_deg",
    "joint_4_deg",
    "joint_5_deg",
    "joint_6_deg",
    "gripper_deg",
]

RECORDING_WIDTH = 640
RECORDING_HEIGHT = 480
DEPTH_MIN_MM = 200.0
DEPTH_MAX_MM = 1000.0
PLOT_JOINT_NAMES = JOINT_NAMES[:6]


class LiveJointPlot:
    """Display the six corresponding leader and follower joint signals."""

    def __init__(self, fps: int, duration_s: float, y_min: np.ndarray, y_max: np.ndarray, update_hz: float):
        import matplotlib.pyplot as plt

        self._plt = plt
        self._fps = float(fps)
        self._update_period_s = 1.0 / max(float(update_hz), 1e-6)
        self._start_time = time.perf_counter()
        self._next_update = self._start_time
        self._times: deque[float] = deque(maxlen=max(2, int(np.ceil(fps * duration_s)) + 1))
        self._leader: list[deque[float]] = [deque(maxlen=self._times.maxlen) for _ in PLOT_JOINT_NAMES]
        self._follower: list[deque[float]] = [deque(maxlen=self._times.maxlen) for _ in PLOT_JOINT_NAMES]
        self._closed = False
        self._figure, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        self._axes = axes.ravel()
        self._leader_lines = []
        self._follower_lines = []
        for idx, (axis, name) in enumerate(zip(self._axes, PLOT_JOINT_NAMES)):
            leader_line, = axis.plot([], [], color="#1976d2", linewidth=1.5, label="Leader")
            follower_line, = axis.plot([], [], color="#ef6c00", linewidth=1.5, label="Follower")
            axis.set_title(name, fontsize=10)
            axis.set_ylabel("deg")
            axis.set_ylim(float(y_min[idx]), float(y_max[idx]))
            axis.set_xlim(0.0, duration_s)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right", fontsize=8)
            self._leader_lines.append(leader_line)
            self._follower_lines.append(follower_line)
        self._axes[-1].set_xlabel("time (s)")
        self._axes[-2].set_xlabel("time (s)")
        self._figure.suptitle("Leader / Follower Joint Monitor")
        self._figure.tight_layout()
        self._figure.canvas.mpl_connect("close_event", self._on_close)
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

    def _on_close(self, _event) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed or not self._plt.fignum_exists(self._figure.number)

    def add_sample(self, leader: np.ndarray, follower: np.ndarray | None) -> None:
        if self.closed:
            return
        now = time.perf_counter()
        if now < self._next_update:
            return
        self._next_update = now + self._update_period_s
        self._times.append(now - self._start_time)
        for idx in range(len(PLOT_JOINT_NAMES)):
            self._leader[idx].append(float(leader[idx]))
            self._follower[idx].append(float(follower[idx]) if follower is not None else np.nan)
            self._leader_lines[idx].set_data(self._times, self._leader[idx])
            self._follower_lines[idx].set_data(self._times, self._follower[idx])
        self._figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def close(self) -> None:
        if not self.closed:
            self._plt.close(self._figure)


class GStreamerH264Writer:
    """Stream BGR frames to Jetson NVENC through a GStreamer subprocess."""

    def __init__(self, path: Path, fps: int, width: int, height: int, bitrate: int):
        if shutil.which("gst-launch-1.0") is None:
            raise RuntimeError("gst-launch-1.0 is required for GPU video encoding")

        self.path = path
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        pipeline = [
            "gst-launch-1.0",
            "-e",
            "-q",
            "fdsrc",
            "fd=0",
            f"blocksize={self.frame_bytes}",
            "!",
            "rawvideoparse",
            "format=bgr",
            f"width={width}",
            f"height={height}",
            f"framerate={fps}/1",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=I420",
            "!",
            "nvvidconv",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
            "!",
            "nvv4l2h264enc",
            f"bitrate={bitrate}",
            "insert-sps-pps=true",
            "!",
            "h264parse",
            "!",
            "mp4mux",
            "!",
            "filesink",
            f"location={path}",
        ]
        self._process = subprocess.Popen(
            pipeline,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None:
            self._process.kill()
            raise RuntimeError(f"Failed to open GPU encoder stdin for {path}")

    def write(self, frame: np.ndarray) -> None:
        if self._process.poll() is not None:
            error = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
            raise RuntimeError(f"GPU video encoder exited for {self.path}: {error[-1000:]}")
        image = np.asarray(frame)
        expected_shape = (self.height, self.width, 3)
        if image.shape != expected_shape or image.dtype != np.uint8:
            raise ValueError(
                f"GPU video frame must be uint8 {expected_shape}, got {image.shape} {image.dtype}"
            )
        try:
            self._process.stdin.write(np.ascontiguousarray(image).data)
        except (BrokenPipeError, OSError) as exc:
            error = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
            raise RuntimeError(f"GPU video encoder write failed for {self.path}: {error[-1000:]}") from exc

    def release(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
            self._process.stdin = None
        return_code = self._process.wait()
        if return_code != 0:
            error = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
            raise RuntimeError(f"GPU video encoder failed for {self.path}: {error[-1000:]}")


class CppVisionStream:
    """Read 640x480 C++ ZED RGB/depth/YOLO frames from POSIX shared memory."""

    def __init__(self, shm_name: str, poll_ms: float = 1.0):
        self._mapped = MappedStruct(shm_name, VisionSharedBlock)
        self._block = self._mapped.view
        self._poll_s = max(0.0001, poll_ms / 1000.0)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: dict[str, np.ndarray] | None = None
        self._last_seq = 0
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="cpp-vision-shm-stream", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                seq_before = int(self._block.seq)
                if seq_before == 0 or seq_before == self._last_seq or seq_before % 2:
                    time.sleep(self._poll_s)
                    continue

                width = int(self._block.width)
                height = int(self._block.height)
                depth_width = int(self._block.depth_width)
                depth_height = int(self._block.depth_height)
                if (width, height) != (RECORDING_WIDTH, RECORDING_HEIGHT):
                    time.sleep(self._poll_s)
                    continue
                if (depth_width, depth_height) != (RECORDING_WIDTH, RECORDING_HEIGHT):
                    time.sleep(self._poll_s)
                    continue

                rgb_array = np.ctypeslib.as_array(self._block.rgb)
                yolo_array = np.ctypeslib.as_array(self._block.yolo_rgb)
                depth_array = np.ctypeslib.as_array(self._block.depth_mm)
                frame = {
                    "observation.images.zed_left": rgb_array[: width * height * 3].reshape(height, width, 3).copy(),
                    "observation.images.zed_left_yolo": yolo_array[: width * height * 3].reshape(height, width, 3).copy(),
                    "observation.images.zed_depth": depth_array[: width * height].reshape(height, width, 1).astype(np.float32),
                }
                seq_after = int(self._block.seq)
                if seq_after != seq_before or seq_after % 2:
                    continue
                with self._lock:
                    self._latest = frame
                    self._last_seq = seq_after
        except Exception as exc:
            self._error = exc
            self._stop.set()

    def get_latest(self) -> dict[str, np.ndarray] | None:
        if self._error is not None:
            raise RuntimeError(f"C++ vision shared-memory reader failed: {self._error}") from self._error
        with self._lock:
            return None if self._latest is None else {key: value.copy() for key, value in self._latest.items()}

    def request_stop(self) -> None:
        self._block.stop_flag = 1

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._mapped.close()


class CombinedCppCamera:
    def __init__(self, cpp_stream: CppVisionStream, usb_camera: OpenCVCameraInterface):
        self.cpp_stream = cpp_stream
        self.usb_camera = usb_camera

    def get_frames(self) -> dict[str, np.ndarray] | None:
        zed_frames = self.cpp_stream.get_latest()
        usb_frame = self.usb_camera.get_frame()
        if zed_frames is None or usb_frame is None:
            return None
        return {**zed_frames, "observation.images.usb": usb_frame}


def _csv_floats(name: str) -> np.ndarray:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required; see hybrid/README.md for calibration setup.")
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"{name} must contain finite comma-separated values.")
    return values


def _plot_limits(raw: str, argument: str) -> np.ndarray:
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.size == 1:
        values = np.repeat(values, len(PLOT_JOINT_NAMES))
    if values.shape != (len(PLOT_JOINT_NAMES),) or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError(
            f"{argument} must be one finite value or six comma-separated values."
        )
    return values


def _normalize_range_m100_100(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    maximum = np.asarray(maximum, dtype=np.float32)
    if values.shape != minimum.shape or values.shape != maximum.shape:
        raise RuntimeError(
            f"Normalization shape mismatch: values={values.shape}, min={minimum.shape}, max={maximum.shape}"
        )
    if not np.all(maximum > minimum):
        raise RuntimeError("Every normalization minimum must be below its maximum.")
    bounded = np.clip(values, minimum, maximum)
    return (((bounded - minimum) / (maximum - minimum)) * 200.0 - 100.0).astype(np.float32, copy=False)


@dataclass
class EpisodeBlob:
    episode_index: int
    task: str
    skipped: int
    duration_s: float
    frames: list[dict[str, object]]


@dataclass(frozen=True)
class ExistingDatasetState:
    next_episode_index: int
    next_global_index: int
    total_episodes: int
    total_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Hugging Face dataset repo, e.g. account/arm-teleop")
    parser.add_argument("--dataset-root", type=Path, default=Path("recordings"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per episode")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--control-shm", default="/hybrid_control", help="POSIX shm name for control service output")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--yolo-onnx", type=Path, default=Path(__file__).resolve().parent.parent / "best.onnx")
    parser.add_argument("--yolo-weights", type=Path, default=None, help="Deprecated alias for --yolo-onnx")
    parser.add_argument("--yolo-confidence", type=float, default=0.45)
    parser.add_argument("--yolo-device", default="auto", help="Ignored by the C++ vision pipeline; kept for CLI compatibility")
    parser.add_argument("--yolo-quantize", default="", help="Ignored by the C++ vision pipeline; kept for CLI compatibility")
    parser.add_argument("--yolo-classes", type=str, default="", help="Ignored by the C++ vision pipeline; kept for CLI compatibility")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="Ignored by the C++ vision pipeline; kept for CLI compatibility")
    parser.add_argument("--sahi", action=argparse.BooleanOptionalAction, default=False, help="Ignored by the C++ vision pipeline; kept for CLI compatibility")
    parser.add_argument("--sahi-slice-size", type=int, default=320, help="Ignored by the C++ vision pipeline; kept for CLI compatibility")
    parser.add_argument("--sahi-overlap", type=float, default=0.2, help="Ignored by the C++ vision pipeline; kept for CLI compatibility")
    parser.add_argument("--video-bitrate", type=int, default=4000000, help="Jetson NVENC H.264 bitrate")
    parser.add_argument(
        "--zed-depth-mode",
        type=str,
        default="neural",
        choices=["none", "performance", "quality", "ultra", "neural", "neural_plus", "neural_light"],
        help="ZED depth mode used by the recorder camera pipeline",
    )
    parser.add_argument(
        "--zed-retrieve-scale",
        type=float,
        default=1.0,
        help="Deprecated; C++ vision output is fixed to 640x480.",
    )
    parser.add_argument("--vision-shm", default="/hybrid_vision", help="POSIX shm name for C++ vision service output")
    parser.add_argument(
        "--camera-service-bin",
        type=Path,
        default=Path(__file__).resolve().parent / "cpp_pipeline" / "build" / "camera_service_cpp",
        help="Path to the C++ ZED/YOLO camera service binary",
    )
    parser.add_argument("--vision-poll-ms", type=float, default=1.0, help="C++ vision shared-memory polling interval")
    parser.add_argument(
        "--startup-timeout-sec",
        type=float,
        default=600.0,
        help="Maximum wait for first C++ ZED/YOLO frame; first ONNX optimization can take several minutes",
    )
    parser.add_argument(
        "--sync-yolo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignored by the C++ vision pipeline; kept for CLI compatibility.",
    )
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--state-fallback",
        choices=["action", "strict"],
        default="action",
        help="When follower state is missing: 'action' uses action as observation.state, 'strict' requires real state",
    )
    parser.add_argument(
        "--auto-repo-id",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If --repo-id is a plain name, automatically prefix it with the current Hugging Face username.",
    )
    parser.add_argument("--codebase-version", default="v3.0-cpp-vision")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show six live leader/follower joint graphs during recording.",
    )
    parser.add_argument(
        "--plot-update-hz",
        type=float,
        default=5.0,
        help="Maximum live plot refresh rate; lower values reduce recording-loop latency.",
    )
    parser.add_argument(
        "--plot-y-min",
        default="-180,-180,-180,-180,-180,-180",
        help="Six comma-separated graph Y-axis minimums, or one value for all joints.",
    )
    parser.add_argument(
        "--plot-y-max",
        default="180,180,180,180,180,180",
        help="Six comma-separated graph Y-axis maximums, or one value for all joints.",
    )
    args = parser.parse_args()
    args.plot_y_min = _plot_limits(args.plot_y_min, "--plot-y-min")
    args.plot_y_max = _plot_limits(args.plot_y_max, "--plot-y-max")

    if (
        args.episodes < 1
        or args.duration <= 0
        or args.fps <= 0
        or not 0 <= args.yolo_confidence <= 1
        or not 0.0 < args.zed_retrieve_scale <= 1.0
        or args.sahi_slice_size < 32
        or not 0.0 <= args.sahi_overlap < 1.0
        or args.yolo_imgsz < 32
        or args.video_bitrate <= 0
        or args.plot_update_hz <= 0
        or args.vision_poll_ms <= 0
        or args.startup_timeout_sec <= 0
        or not np.all(args.plot_y_max > args.plot_y_min)
    ):
        parser.error("episodes, duration, and fps must be positive")
    if args.yolo_weights is not None:
        args.yolo_onnx = args.yolo_weights
    if not args.yolo_onnx.is_file():
        parser.error(f"YOLO ONNX model not found: {args.yolo_onnx}")
    return args


def _image_to_bgr(camera_name: str, image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if camera_name == "zed_depth":
        return _depth_to_colormap(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    raise RuntimeError(f"Unsupported image shape for {camera_name}: {img.shape}")


def _depth_to_colormap(depth_img: np.ndarray) -> np.ndarray:
    depth = depth_img[..., 0] if depth_img.ndim == 3 else depth_img
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip(depth, DEPTH_MIN_MM, DEPTH_MAX_MM)
    normalized[valid] = (
        255 * (DEPTH_MAX_MM - clipped[valid]) / (DEPTH_MAX_MM - DEPTH_MIN_MM)
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _to_rgb_norm(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _write_parquet_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _safe_rel(root: Path, target: Path) -> str:
    return target.relative_to(root).as_posix()


def _episode_file_index(episode_index: int) -> int:
    if episode_index < 0:
        raise RuntimeError(f"episode_index must be non-negative, got {episode_index}")
    return episode_index


def _shard_file_index(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("file-"):
        return None
    raw_index = stem.removeprefix("file-")
    if not raw_index.isdigit():
        return None
    return int(raw_index)


def _read_existing_dataset_state(root: Path) -> ExistingDatasetState:
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    data_dir = root / "data" / "chunk-000"
    videos_dir = root / "videos"
    max_episode_index = -1
    max_end_index = -1
    max_data_file_index = -1
    max_video_file_index = -1
    data_frame_count = 0
    total_episodes = 0

    if episodes_dir.exists():
        for path in sorted(episodes_dir.glob("file-*.parquet")):
            try:
                rows = pq.read_table(path).to_pylist()
            except Exception as exc:
                raise RuntimeError(f"Failed to read existing episode metadata: {path}") from exc
            for row in rows:
                if "episode_index" in row and row["episode_index"] is not None:
                    max_episode_index = max(max_episode_index, int(row["episode_index"]))
                if "end_index" in row and row["end_index"] is not None:
                    max_end_index = max(max_end_index, int(row["end_index"]))
                total_episodes += 1

    if data_dir.exists():
        for path in sorted(data_dir.glob("file-*.parquet")):
            file_index = _shard_file_index(path)
            if file_index is not None:
                max_data_file_index = max(max_data_file_index, file_index)
            try:
                rows = pq.read_table(path).to_pylist()
            except Exception as exc:
                raise RuntimeError(f"Failed to read existing data shard: {path}") from exc
            data_frame_count += len(rows)
            for row in rows:
                if "index" in row and row["index"] is not None:
                    max_end_index = max(max_end_index, int(row["index"]))

    if videos_dir.exists():
        for path in sorted(videos_dir.glob("*/chunk-000/file-*.mp4")):
            file_index = _shard_file_index(path)
            if file_index is not None:
                max_video_file_index = max(max_video_file_index, file_index)

    info_path = root / "meta" / "info.json"
    info_total_episodes = 0
    info_total_frames = 0
    if info_path.exists():
        try:
            with info_path.open("r", encoding="utf-8") as fh:
                info = json.load(fh)
            info_total_episodes = int(info.get("total_episodes") or 0)
            info_total_frames = int(info.get("total_frames") or 0)
        except Exception as exc:
            raise RuntimeError(f"Failed to read existing dataset info: {info_path}") from exc

    next_episode_index = max(
        max_episode_index + 1,
        max_data_file_index + 1,
        max_video_file_index + 1,
        info_total_episodes,
    )
    total_episodes = max(total_episodes, info_total_episodes, next_episode_index)
    next_global_index = max(max_end_index + 1, data_frame_count, info_total_frames)

    return ExistingDatasetState(
        next_episode_index=next_episode_index,
        next_global_index=next_global_index,
        total_episodes=total_episodes,
        total_frames=next_global_index,
    )


def _read_existing_stats(root: Path) -> dict[str, object]:
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        return {}
    try:
        with stats_path.open("r", encoding="utf-8") as fh:
            stats = json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to read existing dataset stats: {stats_path}") from exc
    if not isinstance(stats, dict):
        raise RuntimeError(f"Existing dataset stats must be a JSON object: {stats_path}")
    return stats


def _merge_stat_block(existing: object, incoming: object) -> object:
    if not isinstance(existing, dict) or not isinstance(incoming, dict):
        return incoming
    required = ("min", "max", "mean", "std", "count")
    if not all(key in existing and key in incoming for key in required):
        return incoming

    existing_count = int(np.asarray(existing["count"]).reshape(-1)[0])
    incoming_count = int(np.asarray(incoming["count"]).reshape(-1)[0])
    total_count = existing_count + incoming_count
    if existing_count <= 0:
        return incoming
    if incoming_count <= 0:
        return existing

    existing_mean = np.asarray(existing["mean"], dtype=np.float64)
    incoming_mean = np.asarray(incoming["mean"], dtype=np.float64)
    existing_std = np.asarray(existing["std"], dtype=np.float64)
    incoming_std = np.asarray(incoming["std"], dtype=np.float64)
    if existing_mean.shape != incoming_mean.shape or existing_std.shape != incoming_std.shape:
        return incoming

    merged_mean = (existing_mean * existing_count + incoming_mean * incoming_count) / total_count
    existing_var = np.square(existing_std)
    incoming_var = np.square(incoming_std)
    merged_var = (
        existing_count * (existing_var + np.square(existing_mean - merged_mean))
        + incoming_count * (incoming_var + np.square(incoming_mean - merged_mean))
    ) / total_count

    return {
        "min": np.minimum(np.asarray(existing["min"], dtype=np.float64), np.asarray(incoming["min"], dtype=np.float64)).tolist(),
        "max": np.maximum(np.asarray(existing["max"], dtype=np.float64), np.asarray(incoming["max"], dtype=np.float64)).tolist(),
        "mean": merged_mean.tolist(),
        "std": np.sqrt(np.maximum(merged_var, 0.0)).tolist(),
        "count": [int(total_count)],
    }


def _merge_stats(existing: dict[str, object], incoming: dict[str, object]) -> dict[str, object]:
    merged = dict(existing)
    for key, value in incoming.items():
        merged[key] = _merge_stat_block(merged[key], value) if key in merged else value
    return merged


def _serialize_v3(
    root: Path,
    episodes: list[EpisodeBlob],
    fps: int,
    codebase_version: str,
    state_action_min: np.ndarray,
    state_action_max: np.ndarray,
    video_bitrate: int = 4000000,
) -> None:
    if not episodes:
        raise RuntimeError("No episode data to serialize")

    existing_state = _read_existing_dataset_state(root)
    existing_stats = _read_existing_stats(root)

    meta_dir = root / "meta"
    episodes_meta_dir = meta_dir / "episodes" / "chunk-000"
    data_chunk_dir = root / "data" / "chunk-000"
    videos_root = root / "videos"
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_meta_dir.mkdir(parents=True, exist_ok=True)
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)

    first_frame = episodes[0].frames[0]
    camera_keys = sorted(k for k in first_frame.keys() if k.startswith("observation.images."))

    video_shapes: dict[str, tuple[int, int]] = {}
    for key in camera_keys:
        camera_name = key
        sample = np.asarray(first_frame[camera_name])
        _ = _image_to_bgr(camera_name.split(".")[-1], sample)
        video_shapes[camera_name] = (RECORDING_WIDTH, RECORDING_HEIGHT)

    data_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    data_paths: list[Path] = []
    video_paths: list[Path] = []

    action_all: list[np.ndarray] = []
    state_all: list[np.ndarray] = []
    timestamp_all: list[float] = []
    frame_index_all: list[int] = []
    episode_index_all: list[int] = []
    task_index_all: list[int] = []
    image_stats: dict[str, dict[str, np.ndarray | int]] = {}

    global_index = existing_state.next_global_index

    for ep in episodes:
        start_index = global_index
        n_ep = len(ep.frames)
        task_idx = 0
        file_index = _episode_file_index(ep.episode_index)
        episode_data_rows: list[dict[str, object]] = []
        episode_meta_row: dict[str, object] | None = None
        episode_video_paths: dict[str, Path] = {}
        writers: dict[str, GStreamerH264Writer] = {}

        for camera_key in camera_keys:
            path = videos_root / camera_key / "chunk-000" / f"file-{file_index:03d}.mp4"
            if path.exists():
                raise RuntimeError(f"Refusing to overwrite existing video shard: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[camera_key] = GStreamerH264Writer(
                path,
                fps,
                RECORDING_WIDTH,
                RECORDING_HEIGHT,
                video_bitrate,
            )
            episode_video_paths[camera_key] = path

        try:
            for frame_idx, frame in enumerate(ep.frames):
                action_v = _normalize_range_m100_100(
                    np.asarray(frame["action"], dtype=np.float32),
                    state_action_min,
                    state_action_max,
                )
                state_v = _normalize_range_m100_100(
                    np.asarray(frame["observation.state"], dtype=np.float32),
                    state_action_min,
                    state_action_max,
                )
                row: dict[str, object] = {
                    "index": global_index,
                    "episode_index": ep.episode_index,
                    "frame_index": frame_idx,
                    "timestamp": float(frame_idx / max(float(fps), 1e-6)),
                    "task_index": task_idx,
                    "action": action_v.tolist(),
                    "observation.state": state_v.tolist(),
                }

                for camera_key in camera_keys:
                    camera_name = camera_key.split(".")[-1]
                    bgr = _image_to_bgr(camera_name, np.asarray(frame[camera_key]))

                    target_shape = video_shapes[camera_key]
                    if (bgr.shape[1], bgr.shape[0]) != target_shape:
                        bgr = cv2.resize(bgr, target_shape, interpolation=cv2.INTER_AREA)

                    writers[camera_key].write(bgr)
                    row[camera_key] = _safe_rel(root, episode_video_paths[camera_key])

                    rgb_norm = _to_rgb_norm(bgr)
                    flat = rgb_norm.reshape(-1, 3)
                    stat = image_stats.setdefault(
                        camera_key,
                        {
                            "sum": np.zeros(3, dtype=np.float64),
                            "sum_sq": np.zeros(3, dtype=np.float64),
                            "min": np.full(3, np.inf, dtype=np.float64),
                            "max": np.full(3, -np.inf, dtype=np.float64),
                            "count": 0,
                        },
                    )
                    stat["sum"] = stat["sum"] + flat.sum(axis=0)
                    stat["sum_sq"] = stat["sum_sq"] + np.square(flat).sum(axis=0)
                    stat["min"] = np.minimum(stat["min"], flat.min(axis=0))
                    stat["max"] = np.maximum(stat["max"], flat.max(axis=0))
                    stat["count"] = int(stat["count"]) + int(flat.shape[0])

                action_all.append(action_v)
                state_all.append(state_v)
                timestamp_all.append(float(row["timestamp"]))
                frame_index_all.append(frame_idx)
                episode_index_all.append(ep.episode_index)
                task_index_all.append(task_idx)

                episode_data_rows.append(row)
                data_rows.append(row)
                global_index += 1

            end_index = global_index - 1
            episode_meta_row = {
                "episode_index": ep.episode_index,
                "task_index": 0,
                "task": ep.task,
                "length": n_ep,
                "start_index": start_index,
                "end_index": end_index,
                "skipped": ep.skipped,
                "duration_s": float(ep.duration_s),
            }
            episode_rows.append(episode_meta_row)
        finally:
            for writer in writers.values():
                writer.release()

        data_path = data_chunk_dir / f"file-{file_index:03d}.parquet"
        episodes_meta_path = episodes_meta_dir / f"file-{file_index:03d}.parquet"
        if data_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing data shard: {data_path}")
        if episodes_meta_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing episode metadata shard: {episodes_meta_path}")
        _write_parquet_rows(data_path, episode_data_rows)
        if episode_meta_row is not None:
            _write_parquet_rows(episodes_meta_path, [episode_meta_row])
        data_paths.append(data_path)
        video_paths.extend(episode_video_paths.values())

    tasks_path = meta_dir / "tasks.parquet"
    _write_parquet_rows(tasks_path, [{"task_index": 0, "task": episodes[0].task}])

    action_np = np.stack(action_all, axis=0)
    state_np = np.stack(state_all, axis=0)
    timestamp_np = np.asarray(timestamp_all, dtype=np.float64)
    frame_idx_np = np.asarray(frame_index_all, dtype=np.int64)
    episode_idx_np = np.asarray(episode_index_all, dtype=np.int64)
    index_np = np.asarray([int(row["index"]) for row in data_rows], dtype=np.int64)
    task_idx_np = np.asarray(task_index_all, dtype=np.int64)

    def _vec_stats(arr: np.ndarray) -> dict[str, object]:
        return {
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }

    def _scalar_stats(arr: np.ndarray) -> dict[str, object]:
        return {
            "min": [float(np.min(arr))],
            "max": [float(np.max(arr))],
            "mean": [float(np.mean(arr))],
            "std": [float(np.std(arr))],
            "count": [int(arr.shape[0])],
        }

    stats: dict[str, object] = {
        "timestamp": _scalar_stats(timestamp_np),
        "action": _vec_stats(action_np),
        "observation.state": _vec_stats(state_np),
        "episode_index": _scalar_stats(episode_idx_np.astype(np.float64)),
        "index": _scalar_stats(index_np.astype(np.float64)),
        "frame_index": _scalar_stats(frame_idx_np.astype(np.float64)),
        "task_index": _scalar_stats(task_idx_np.astype(np.float64)),
    }

    for camera_key, stat in image_stats.items():
        cnt = max(1, int(stat["count"]))
        mean = np.asarray(stat["sum"], dtype=np.float64) / cnt
        var = np.asarray(stat["sum_sq"], dtype=np.float64) / cnt - np.square(mean)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        mn = np.asarray(stat["min"], dtype=np.float64)
        mx = np.asarray(stat["max"], dtype=np.float64)

        stats[camera_key] = {
            "min": [[[float(mn[0])]], [[float(mn[1])]], [[float(mn[2])]]],
            "max": [[[float(mx[0])]], [[float(mx[1])]], [[float(mx[2])]]],
            "mean": [[[float(mean[0])]], [[float(mean[1])]], [[float(mean[2])]]],
            "std": [[[float(std[0])]], [[float(std[1])]], [[float(std[2])]]],
            "count": [int(len(data_rows))],
        }

    stats = _merge_stats(existing_stats, stats)

    with (meta_dir / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    features: dict[str, object] = {
        "action": {
            "dtype": "float32",
            "shape": [len(JOINT_NAMES)],
            "names": JOINT_NAMES,
            "fps": fps,
            "norm_mode": "range_m100_100",
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [len(JOINT_NAMES)],
            "names": JOINT_NAMES,
            "fps": fps,
            "norm_mode": "range_m100_100",
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": fps},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
    }

    for camera_key in camera_keys:
        w, h = video_shapes[camera_key]
        features[camera_key] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(fps),
                "video.height": h,
                "video.width": w,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    all_data_paths = list(data_chunk_dir.glob("file-*.parquet"))
    all_video_paths = list(videos_root.glob("*/chunk-000/file-*.mp4"))
    data_size_mb = sum(float(p.stat().st_size) for p in all_data_paths) / (1024.0 * 1024.0)
    videos_size_mb = sum(float(p.stat().st_size) for p in all_video_paths) / (1024.0 * 1024.0)
    total_episodes = existing_state.total_episodes + len(episodes)
    total_frames = existing_state.total_frames + len(data_rows)

    info = {
        "codebase_version": codebase_version,
        "fps": fps,
        "features": features,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": total_frames,
        "data_files_size_in_mb": round(data_size_mb, 3),
        "video_files_size_in_mb": round(videos_size_mb, 3),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": "hybrid_arm",
        "state_action_normalization": {
            "mode": "range_m100_100",
            "min": state_action_min.tolist(),
            "max": state_action_max.tolist(),
        },
        "splits": {"train": f"0:{total_episodes}"},
    }

    with (meta_dir / "info.json").open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)


def main() -> None:
    args = parse_args()
    if args.push_to_hub and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for upload; use --no-push-to-hub for local-only recording.")

    if not args.repo_id:
        raise RuntimeError("--repo-id is required unless you pass --auto-repo-id with a short name.")
    if args.auto_repo_id and "/" not in args.repo_id:
        args.repo_id = resolve_repo_id(args.repo_id, os.environ.get("HF_TOKEN"))

    state_action_min = _csv_floats("FOLLOWER_MIN_DEG")
    state_action_max = _csv_floats("FOLLOWER_MAX_DEG")
    if state_action_min.shape != state_action_max.shape:
        raise RuntimeError(
            f"FOLLOWER_MIN_DEG and FOLLOWER_MAX_DEG must have the same length: {state_action_min.shape} vs {state_action_max.shape}"
        )
    if state_action_min.shape[0] != len(JOINT_NAMES):
        raise RuntimeError(
            f"Expected {len(JOINT_NAMES)} follower limits, got {state_action_min.shape[0]}."
        )

    dataset_dir = args.dataset_root / args.repo_id
    existing_state = _read_existing_dataset_state(dataset_dir)
    if existing_state.total_episodes > 0:
        print(
            "[INFO] appending to existing dataset: "
            f"{existing_state.total_episodes} episodes, {existing_state.total_frames} frames; "
            f"next episode_index={existing_state.next_episode_index}"
        )

    service: subprocess.Popen | None = None
    control_reader: ControlSharedReader | None = None
    vision_stream: CppVisionStream | None = None
    camera_stream: LatestCameraStream | None = None
    usb: OpenCVCameraInterface | None = None
    plot: LiveJointPlot | None = None
    recording_completed = False
    interrupted = False
    episodes: list[EpisodeBlob] = []

    try:
        if not args.camera_service_bin.is_file():
            raise FileNotFoundError(f"C++ camera service binary not found: {args.camera_service_bin}")

        service = subprocess.Popen(
            [
                str(args.camera_service_bin),
                "--shm",
                args.vision_shm,
                "--fps",
                str(args.fps),
                "--depth-mode",
                args.zed_depth_mode,
                "--yolo-onnx",
                str(args.yolo_onnx),
                "--yolo-conf",
                str(args.yolo_confidence),
            ],
            stdout=None,
            stderr=None,
            text=True,
        )
        control_reader = ControlSharedReader(args.control_shm)
        print(
            "[INFO] recording performance mode: "
            f"cpp_vision=on, output={RECORDING_WIDTH}x{RECORDING_HEIGHT}, "
            f"plot={'on' if args.plot else 'off'}, plot_update_hz={args.plot_update_hz:g}"
        )

        vision_path = Path("/dev/shm") / args.vision_shm.lstrip("/")
        vision_deadline = time.perf_counter() + 30.0
        expected_vision_bytes = ctypes.sizeof(VisionSharedBlock)
        while not vision_path.exists() or vision_path.stat().st_size < expected_vision_bytes:
            if service.poll() is not None:
                raise RuntimeError(f"camera_service_cpp exited with code {service.returncode} before creating {vision_path}")
            if time.perf_counter() >= vision_deadline:
                actual_size = vision_path.stat().st_size if vision_path.exists() else 0
                raise RuntimeError(
                    f"Timed out waiting for C++ vision shared memory: {vision_path} "
                    f"(size={actual_size}, expected>={expected_vision_bytes})"
                )
            time.sleep(0.05)

        vision_stream = CppVisionStream(args.vision_shm, args.vision_poll_ms)
        vision_stream.start()
        usb = OpenCVCameraInterface(args.camera_index, height=480, width=640, fps=args.fps)
        camera_stream = LatestCameraStream(CombinedCppCamera(vision_stream, usb))
        camera_stream.start()

        print(f"[INFO] waiting up to {args.startup_timeout_sec:.0f}s for the first 640x480 C++ ZED/YOLO frame")
        first_deadline = time.perf_counter() + args.startup_timeout_sec
        while time.perf_counter() < first_deadline:
            images = camera_stream.get_latest()
            action, state = control_reader.get_latest()
            if images is not None and action is not None and (state is not None or args.state_fallback == "action"):
                break
            if service.poll() is not None:
                raise RuntimeError(f"camera_service_cpp exited with code {service.returncode}")
            time.sleep(0.01)
        else:
            cmd_valid = int(control_reader.block.cmd.valid)
            state_valid = int(control_reader.block.state.valid)
            loop_count = int(control_reader.block.telemetry.loop_count)
            raise RuntimeError(
                "Timed out waiting for control shared-memory samples. "
                f"cmd.valid={cmd_valid}, state.valid={state_valid}, loop_count={loop_count}. "
                "Check that control_service is still running and leader/follower serial input is active."
            )

        for local_episode_index in range(args.episodes):
            episode_index = existing_state.next_episode_index + local_episode_index
            input(f"\nEpisode {local_episode_index + 1}/{args.episodes}: press Enter to record. ('q' 키 = 조기 종료) ")
            if args.plot:
                plot = LiveJointPlot(args.fps, args.duration, args.plot_y_min, args.plot_y_max, args.plot_update_hz)
            episode_frames: list[dict[str, object]] = []
            early_stop_event = threading.Event()
            key_thread = threading.Thread(
                target=_keyboard_early_stop,
                args=(early_stop_event,),
                daemon=True,
                name="key-listener",
            )
            key_thread.start()
            try:
                recorded, skipped, interrupted, actual_duration_s = record_episode(
                    control_reader,
                    camera_stream,
                    args.task,
                    args.duration,
                    args.fps,
                    state_fallback=args.state_fallback,
                    frame_buffer=episode_frames,
                    early_stop_event=early_stop_event,
                    sample_callback=plot.add_sample if plot is not None else None,
                )
            finally:
                early_stop_event.set()
                key_thread.join(timeout=1.0)
                if plot is not None:
                    plot.close()
                    plot = None

            if recorded == 0:
                raise RuntimeError("No valid synchronized frames were captured; episode was not saved.")

            episodes.append(
                EpisodeBlob(
                    episode_index=episode_index,
                    task=args.task,
                    skipped=skipped,
                    duration_s=actual_duration_s,
                    frames=episode_frames,
                )
            )
            effective_fps = recorded / max(actual_duration_s, 1e-6)
            print(
                f"[INFO] buffered episode {episode_index + 1}: "
                f"{recorded} frames in {actual_duration_s:.2f}s ({effective_fps:.1f} fps)"
            )

            if interrupted:
                print("Recording stopped after the saved partial episode.")
                break

        _serialize_v3(
            dataset_dir,
            episodes,
            args.fps,
            args.codebase_version,
            state_action_min,
            state_action_max,
            args.video_bitrate,
        )
        print(f"[INFO] saved v3-style shard dataset to {dataset_dir}")
        recording_completed = True

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user.")

    finally:
        if args.push_to_hub and recording_completed:
            try:
                upload_to_huggingface(dataset_dir, args.repo_id, os.environ.get("HF_TOKEN"))
            except Exception as exc:
                print(f"[WARN] Hugging Face upload failed: {exc}")

        if control_reader is not None:
            if interrupted:
                try:
                    control_reader.request_stop()
                    time.sleep(0.15)
                except Exception:
                    pass
            control_reader.close()
        if camera_stream is not None:
            camera_stream.stop()
        if vision_stream is not None:
            try:
                vision_stream.request_stop()
            except Exception:
                pass
        if service is not None and service.poll() is None:
            try:
                service.wait(timeout=3.0)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if vision_stream is not None:
            vision_stream.stop()
        if plot is not None:
            plot.close()
        if usb is not None:
            usb.release()
        if service is not None and service.poll() is None:
            try:
                service.terminate()
                service.wait(timeout=3.0)
            except (subprocess.TimeoutExpired, OSError):
                service.kill()


if __name__ == "__main__":
    main()
