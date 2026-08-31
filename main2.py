"""Record teleop episodes and save as shard-based dataset layout.

Output layout:
- meta/info.json
- meta/stats.json
- meta/tasks.parquet
- meta/episodes/chunk-000/file-{file_index:03d}.parquet
- data/chunk-000/file-{file_index:03d}.parquet
- videos/{camera_key}/chunk-000/file-{file_index:03d}.mp4

This script keeps the same capture pipeline as hybrid/main.py and only changes
how recordings are serialized on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pyarrow is required for main2.py. Install it with: uv pip install pyarrow"
    ) from exc

from camera_interface import MultiCameraInterface, OpenCVCameraInterface, ZedXCameraInterface
from main import (
    ControlSharedReader,
    LatestCameraStream,
    _depth_to_colormap,
    record_episode,
    resolve_repo_id,
    upload_to_huggingface,
)
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


def _csv_floats(name: str) -> np.ndarray:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required; see hybrid/README.md for calibration setup.")
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"{name} must contain finite comma-separated values.")
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
class ExistingShardState:
    next_file_index: int
    next_episode_index: int
    next_global_index: int
    total_episodes: int
    total_frames: int


def _has_existing_dataset(root: Path) -> bool:
    return any(
        path.exists()
        for path in (
            root / "meta" / "info.json",
            root / "meta" / "stats.json",
            root / "meta" / "tasks.parquet",
            root / "data",
            root / "videos",
        )
    )


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
    parser.add_argument("--yolo-onnx", type=Path, default=Path(__file__).resolve().parent.parent / "best.engine")
    parser.add_argument("--yolo-weights", type=Path, default=None, help="Deprecated alias for --yolo-onnx")
    parser.add_argument("--record-yolo", action=argparse.BooleanOptionalAction, default=False, help="Record YOLO annotated ZED video stream")
    parser.add_argument("--yolo-confidence", type=float, default=0.45)
    parser.add_argument("--yolo-device", default="auto", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--yolo-quantize", default="", help="YOLO inference precision (16/fp16/32/fp32)")
    parser.add_argument("--yolo-classes", type=str, default="", help="Comma-separated class names or IDs to keep")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO square inference size; model export size must match")
    parser.add_argument("--sahi", action=argparse.BooleanOptionalAction, default=False, help="Use SAHI tiled YOLO inference")
    parser.add_argument("--sahi-slice-size", type=int, default=320, help="SAHI square slice size in pixels")
    parser.add_argument("--sahi-overlap", type=float, default=0.2, help="SAHI slice overlap ratio")
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
        help="Scale factor for ZED retrieve resolution (0<scale<=1)",
    )
    parser.add_argument(
        "--sync-yolo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize YOLO with each ZED frame (default: enabled). Use --no-sync-yolo for async.",
    )
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append to an existing LeRobot-style dataset instead of overwriting shards.",
    )
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
    parser.add_argument("--codebase-version", default="v3.0")
    args = parser.parse_args()

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
    ):
        parser.error("episodes, duration, and fps must be positive")
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


def _to_rgb_norm(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _write_parquet_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _read_task_mapping(root: Path) -> dict[str, int]:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    tasks_df = pd.read_parquet(tasks_path)
    mapping: dict[str, int] = {}
    if "task" in tasks_df.columns:
        for row in tasks_df.reset_index(drop=True).to_dict("records"):
            mapping[str(row["task"])] = int(row["task_index"])
    else:
        for task, row in tasks_df.iterrows():
            mapping[str(task)] = int(row["task_index"])
    return mapping


def _write_task_mapping(root: Path, task_to_index: dict[str, int]) -> None:
    tasks_path = root / "meta" / "tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_tasks = sorted(task_to_index, key=task_to_index.__getitem__)
    tasks_df = pd.DataFrame(
        {"task_index": [task_to_index[task] for task in ordered_tasks]},
        index=pd.Index(ordered_tasks, name="task"),
    )
    tasks_df.to_parquet(tasks_path)


def _safe_rel(root: Path, target: Path) -> str:
    return target.relative_to(root).as_posix()


def _keyboard_episode_control(stop_event: threading.Event) -> None:
    """Set stop_event during recording: q/right-arrow saves partial, r discards and retries."""
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (AttributeError, OSError, termios.error):
        return
    try:
        tty.setraw(fd)
        buf = b""
        while not stop_event.is_set():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = os.read(fd, 8)
                buf += ch
                if b"r" in buf or b"R" in buf:
                    setattr(stop_event, "reason", "rerecord")
                    stop_event.set()
                    break
                if b"q" in buf or b"Q" in buf or b"\x1b[C" in buf or b"[C" in buf:
                    setattr(stop_event, "reason", "early_stop")
                    stop_event.set()
                    break
                buf = buf[-8:]
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except (OSError, termios.error):
            pass


def _shard_file_index(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("file-"):
        return None
    raw_index = stem.removeprefix("file-")
    if not raw_index.isdigit():
        return None
    return int(raw_index)


def _read_existing_shard_state(root: Path) -> ExistingShardState:
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    data_dir = root / "data" / "chunk-000"
    videos_dir = root / "videos"
    max_file_index = -1
    max_episode_index = -1
    max_global_index = -1
    data_frame_count = 0
    total_episodes = 0

    if episodes_dir.exists():
        for path in sorted(episodes_dir.glob("file-*.parquet")):
            file_index = _shard_file_index(path)
            if file_index is not None:
                max_file_index = max(max_file_index, file_index)
            rows = pq.read_table(path).to_pylist()
            for row in rows:
                if "episode_index" in row and row["episode_index"] is not None:
                    max_episode_index = max(max_episode_index, int(row["episode_index"]))
                if "end_index" in row and row["end_index"] is not None:
                    max_global_index = max(max_global_index, int(row["end_index"]))
                total_episodes += 1

    if data_dir.exists():
        for path in sorted(data_dir.glob("file-*.parquet")):
            file_index = _shard_file_index(path)
            if file_index is not None:
                max_file_index = max(max_file_index, file_index)
            rows = pq.read_table(path).to_pylist()
            data_frame_count += len(rows)
            for row in rows:
                if "index" in row and row["index"] is not None:
                    max_global_index = max(max_global_index, int(row["index"]))

    if videos_dir.exists():
        for path in sorted(videos_dir.glob("*/chunk-000/file-*.mp4")):
            file_index = _shard_file_index(path)
            if file_index is not None:
                max_file_index = max(max_file_index, file_index)

    info_total_episodes = 0
    info_total_frames = 0
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as fh:
            info = json.load(fh)
        info_total_episodes = int(info.get("total_episodes") or 0)
        info_total_frames = int(info.get("total_frames") or 0)

    next_file_index = max_file_index + 1
    next_episode_index = max(max_episode_index + 1, info_total_episodes, total_episodes)
    next_global_index = max(max_global_index + 1, data_frame_count, info_total_frames)
    return ExistingShardState(
        next_file_index=next_file_index,
        next_episode_index=next_episode_index,
        next_global_index=next_global_index,
        total_episodes=max(total_episodes, info_total_episodes, next_episode_index),
        total_frames=next_global_index,
    )


def _read_existing_stats(root: Path) -> dict[str, object]:
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        return {}
    with stats_path.open("r", encoding="utf-8") as fh:
        stats = json.load(fh)
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
    resume: bool = True,
) -> None:
    if not episodes:
        raise RuntimeError("No episode data to serialize")

    if not resume and _has_existing_dataset(root):
        raise RuntimeError(f"Dataset already exists: {root}. Use --resume or choose a new --repo-id.")

    existing_state = _read_existing_shard_state(root) if resume else ExistingShardState(0, 0, 0, 0, 0)
    existing_stats = _read_existing_stats(root) if resume else {}

    meta_dir = root / "meta"
    episodes_meta_dir = meta_dir / "episodes" / "chunk-000"
    data_chunk_dir = root / "data" / "chunk-000"
    videos_root = root / "videos"
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_meta_dir.mkdir(parents=True, exist_ok=True)
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)
    file_index = existing_state.next_file_index

    task_to_index = _read_task_mapping(root) if resume else {}
    for ep in episodes:
        if ep.task not in task_to_index:
            task_to_index[ep.task] = len(task_to_index)

    first_frame = episodes[0].frames[0]
    camera_keys = sorted(k for k in first_frame.keys() if k.startswith("observation.images."))

    # Build one continuous video shard per camera key.
    writers: dict[str, GStreamerH264Writer] = {}
    video_shapes: dict[str, tuple[int, int]] = {}
    video_paths: dict[str, Path] = {}

    for key in camera_keys:
        camera_name = key
        sample = np.asarray(first_frame[camera_name])
        bgr = _image_to_bgr(camera_name.split(".")[-1], sample)
        video_shapes[camera_name] = (RECORDING_WIDTH, RECORDING_HEIGHT)

        path = videos_root / camera_name / "chunk-000" / f"file-{file_index:03d}.mp4"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing video shard: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = GStreamerH264Writer(
            path,
            fps,
            RECORDING_WIDTH,
            RECORDING_HEIGHT,
            video_bitrate,
        )
        writers[camera_name] = writer
        video_paths[camera_name] = path

    data_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []

    action_all: list[np.ndarray] = []
    state_all: list[np.ndarray] = []
    timestamp_all: list[float] = []
    frame_index_all: list[int] = []
    episode_index_all: list[int] = []
    task_index_all: list[int] = []
    image_stats: dict[str, dict[str, np.ndarray | int]] = {}

    global_index = existing_state.next_global_index

    video_timestamp_s = 0.0

    try:
        for ep in episodes:
            start_index = global_index
            n_ep = len(ep.frames)
            task_idx = task_to_index[ep.task]
            episode_index = existing_state.next_episode_index + ep.episode_index
            video_from_timestamp = video_timestamp_s
            video_timestamp_s += n_ep / max(float(fps), 1e-6)

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
                    "episode_index": episode_index,
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
                    row[camera_key] = _safe_rel(root, video_paths[camera_key])

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
                episode_index_all.append(episode_index)
                task_index_all.append(task_idx)

                data_rows.append(row)
                global_index += 1

            end_index = global_index - 1
            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "tasks": [ep.task],
                    "task_index": 0,
                    "task": ep.task,
                    "length": n_ep,
                    "dataset_from_index": start_index,
                    "dataset_to_index": end_index + 1,
                    "data/chunk_index": 0,
                    "data/file_index": file_index,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": file_index,
                    "start_index": start_index,
                    "end_index": end_index,
                    "skipped": ep.skipped,
                    "duration_s": float(ep.duration_s),
                    **{
                        f"videos/{camera_key}/chunk_index": 0
                        for camera_key in camera_keys
                    },
                    **{
                        f"videos/{camera_key}/file_index": file_index
                        for camera_key in camera_keys
                    },
                    **{
                        f"videos/{camera_key}/from_timestamp": video_from_timestamp
                        for camera_key in camera_keys
                    },
                    **{
                        f"videos/{camera_key}/to_timestamp": video_timestamp_s
                        for camera_key in camera_keys
                    },
                }
            )
    finally:
        for w in writers.values():
            w.release()

    data_path = data_chunk_dir / f"file-{file_index:03d}.parquet"
    episodes_meta_path = episodes_meta_dir / f"file-{file_index:03d}.parquet"

    if data_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing data shard: {data_path}")
    if episodes_meta_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing episode metadata shard: {episodes_meta_path}")

    _write_parquet_rows(data_path, data_rows)
    _write_parquet_rows(episodes_meta_path, episode_rows)
    _write_task_mapping(root, task_to_index)

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
        "total_tasks": len(task_to_index),
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
    if args.resume:
        existing_state = _read_existing_shard_state(dataset_dir)
        if existing_state.total_episodes > 0 or existing_state.next_file_index > 0:
            print(
                "[INFO] resuming existing dataset: "
                f"{existing_state.total_episodes} episodes, {existing_state.total_frames} frames; "
                f"next episode_index={existing_state.next_episode_index}, "
                f"next file_index={existing_state.next_file_index}"
            )
    elif _has_existing_dataset(dataset_dir):
        raise RuntimeError(f"Dataset already exists: {dataset_dir}. Use --resume or choose a new --repo-id.")

    control_reader: ControlSharedReader | None = None
    camera: MultiCameraInterface | None = None
    camera_stream: LatestCameraStream | None = None
    zed: ZedXCameraInterface | None = None
    usb: OpenCVCameraInterface | None = None
    recording_completed = False
    interrupted = False
    episodes: list[EpisodeBlob] = []

    try:
        control_reader = ControlSharedReader(args.control_shm)
        yolo = None
        if args.record_yolo:
            from yolo_interface import YoloAnnotator

            yolo_model_path = args.yolo_weights if args.yolo_weights is not None else args.yolo_onnx
            class_filter = [x.strip() for x in args.yolo_classes.split(",") if x.strip()]
            yolo_quantize: int | str | None = None
            if args.yolo_quantize:
                q = str(args.yolo_quantize).strip().lower()
                if q in ("16", "32"):
                    yolo_quantize = int(q)
                else:
                    yolo_quantize = q

            yolo = YoloAnnotator(
                yolo_model_path,
                args.yolo_confidence,
                args.yolo_device,
                class_filter=class_filter if class_filter else None,
                imgsz=args.yolo_imgsz,
                quantize=yolo_quantize,
                sahi=args.sahi,
                sahi_slice_size=args.sahi_slice_size,
                sahi_overlap_ratio=args.sahi_overlap,
            )

        zed = ZedXCameraInterface(
            width=640,
            height=480,
            fps=args.fps,
            depth_mode=args.zed_depth_mode,
            retrieve_scale=args.zed_retrieve_scale,
        )
        print(
            "[INFO] ZED capture resolution: "
            f"{zed.native_width}x{zed.native_height} "
            f"(retrieve_scale={args.zed_retrieve_scale}) -> recording {RECORDING_WIDTH}x{RECORDING_HEIGHT}; "
            f"yolo={'on' if args.record_yolo else 'off'}"
        )
        usb = OpenCVCameraInterface(args.camera_index, height=480, width=640, fps=args.fps)
        camera = MultiCameraInterface(zed, usb, yolo, async_yolo=not args.sync_yolo)

        if not camera.is_open:
            raise RuntimeError("A required device failed to open; no dataset was created.")

        camera_stream = LatestCameraStream(camera)
        camera_stream.start()

        first_deadline = time.perf_counter() + 5.0
        while time.perf_counter() < first_deadline:
            _ = camera_stream.get_latest()
            action, state = control_reader.get_latest()
            if action is not None and (state is not None or args.state_fallback == "action"):
                break
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

        episode_index = 0
        while episode_index < args.episodes:
            input(f"\nEpisode {episode_index + 1}/{args.episodes}: press Enter to record. ('q' 키 = 조기 종료) ")
            episode_frames: list[dict[str, object]] = []
            early_stop_event = threading.Event()
            key_thread = threading.Thread(
            target=_keyboard_episode_control,
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
                )
            finally:
                early_stop_event.set()
                key_thread.join(timeout=1.0)

            stop_reason = getattr(early_stop_event, "reason", None)
            if stop_reason == "rerecord":
                print(f"[INFO] discarded episode {episode_index + 1}; press Enter to rerecord.")
                interrupted = False
                continue

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
            print(f"[INFO] buffered episode {episode_index + 1}: {recorded} frames")
            episode_index += 1

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
            args.resume,
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
        if camera is not None:
            camera.release()
        else:
            if zed is not None:
                zed.release()
            if usb is not None:
                usb.release()


if __name__ == "__main__":
    main()
