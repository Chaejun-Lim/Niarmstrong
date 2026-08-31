"""Record calibrated teleoperation demonstrations in LeRobot Dataset v3 format.

Set HF_TOKEN (write access) before running with --push-to-hub.  The values sent
to the follower and stored as ``action`` use the follower's joint coordinate
system, in degrees.  ``observation.state`` must be reported in that same system.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import HfApi, login

from camera_interface import (
    MultiCameraInterface,
    OpenCVCameraInterface,
    ZedXCameraInterface,
)
from shm_layout import ControlSharedBlock, MappedStruct
from yolo_interface import YoloAnnotator


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


def _csv_floats(name: str) -> np.ndarray:
    """Read exactly seven comma-separated floats from an environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required; see README.md for calibration setup.")
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError(f"{name} must contain exactly 7 finite comma-separated values.")
    return values


@dataclass(frozen=True)
class LeaderToFollowerCalibration:
    """Map leader encoder angles to safe follower target angles, all in degrees."""

    leader_zero: np.ndarray
    direction: np.ndarray
    follower_zero: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def from_environment(cls) -> "LeaderToFollowerCalibration":
        calibration = cls(
            leader_zero=_csv_floats("LEADER_ZERO_DEG"),
            direction=_csv_floats("LEADER_DIRECTION"),
            follower_zero=_csv_floats("FOLLOWER_ZERO_DEG"),
            minimum=_csv_floats("FOLLOWER_MIN_DEG"),
            maximum=_csv_floats("FOLLOWER_MAX_DEG"),
        )
        if not np.isin(calibration.direction, (-1.0, 1.0)).all():
            raise RuntimeError("LEADER_DIRECTION values must each be -1 or 1.")
        if not np.all(calibration.minimum < calibration.maximum):
            raise RuntimeError("Every FOLLOWER_MIN_DEG value must be below FOLLOWER_MAX_DEG.")
        return calibration

    def to_target(self, leader_angles: np.ndarray) -> np.ndarray:
        if leader_angles.shape != (7,) or not np.isfinite(leader_angles).all():
            raise ValueError("leader angles must be seven finite values")
        target = (leader_angles - self.leader_zero) * self.direction + self.follower_zero
        return np.clip(target, self.minimum, self.maximum).astype(np.float32, copy=False)
def _write_fallback_episode(episode_dir: Path, frames: list[dict[str, object]], task: str, skipped: int) -> None:
    _write_lerobot_episode(episode_dir, frames, task, skipped)


DEPTH_MIN_MM = 200.0
DEPTH_MAX_MM = 1300.0


def _depth_to_colormap(depth_img: np.ndarray) -> np.ndarray:
    """Convert depth image in millimeters to a visually useful BGR colormap."""
    depth = depth_img[..., 0] if depth_img.ndim == 3 else depth_img
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip(depth, DEPTH_MIN_MM, DEPTH_MAX_MM)
    normalized[valid] = (
        255 * (DEPTH_MAX_MM - clipped[valid]) / (DEPTH_MAX_MM - DEPTH_MIN_MM)
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _write_lerobot_episode(
    episode_dir: Path,
    frames: list[dict[str, object]],
    task: str,
    skipped: int,
    fps: int = 30,
    duration_s: float | None = None,
) -> None:
    """Save one episode as MP4 videos + JSONL metadata compatible with LeRobot-style readers."""
    episode_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = episode_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(frames)
    if duration_s and duration_s > 0 and n_frames > 0:
        actual_fps = n_frames / duration_s
    else:
        actual_fps = float(fps)

    video_writers: dict[str, cv2.VideoWriter] = {}
    video_shapes: dict[str, tuple[int, int]] = {}

    for frame in frames:
        for name, image in frame.items():
            if not name.startswith("observation.images."):
                continue
            camera_name = name.split(".")[-1]
            if camera_name in video_writers:
                continue
            img = np.asarray(image)
            video_shapes[camera_name] = (RECORDING_WIDTH, RECORDING_HEIGHT)
            video_path = videos_dir / f"{camera_name}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                actual_fps,
                video_shapes[camera_name],
            )
            video_writers[camera_name] = writer
        if video_writers:
            break

    dataset_rows: list[dict[str, object]] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for frame_idx, frame in enumerate(frames):
        row = {
            "frame_index": frame_idx,
            "timestamp": frame_idx / max(actual_fps, 1e-6),
            "task": task,
        }

        for name, image in frame.items():
            if not name.startswith("observation.images."):
                continue
            camera_name = name.split(".")[-1]
            img = np.asarray(image)

            if camera_name == "zed_depth":
                img = _depth_to_colormap(img)
            else:
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            target_shape = video_shapes.get(camera_name)
            if target_shape is not None and (img.shape[1], img.shape[0]) != target_shape:
                img = cv2.resize(img, target_shape)

            writer = video_writers.get(camera_name)
            if writer is not None:
                writer.write(img)
                row[f"observation.images.{camera_name}"] = f"videos/{camera_name}.mp4"

        states.append(np.asarray(frame["observation.state"], dtype=np.float32))
        actions.append(np.asarray(frame["action"], dtype=np.float32))
        dataset_rows.append(row)

    for writer in video_writers.values():
        writer.release()

    with (episode_dir / "dataset.jsonl").open("w", encoding="utf-8") as fh:
        for row in dataset_rows:
            fh.write(json.dumps(row) + "\n")

    np.save(episode_dir / "state.npy", np.stack(states, axis=0).astype(np.float32))
    np.save(episode_dir / "action.npy", np.stack(actions, axis=0).astype(np.float32))

    features = {
        "observation.images": {
            name: {"shape": [shape[1], shape[0], 3]} for name, shape in video_shapes.items()
        },
        "observation.state": {"shape": [7]},
        "action": {"shape": [7]},
    }
    with (episode_dir / "features.json").open("w", encoding="utf-8") as fh:
        json.dump(features, fh, indent=2)

    with (episode_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "task": task,
                "frames": n_frames,
                "skipped": skipped,
                "fps": fps,
                "actual_fps": actual_fps,
            },
            fh,
            indent=2,
        )


def resolve_repo_id(repo_id: str | None, token: str | None = None) -> str:
    """Resolve a full Hugging Face repo id, using the current account name when needed."""
    if repo_id and "/" in repo_id:
        return repo_id
    if not repo_id:
        raise ValueError("repo_id is required")

    api = HfApi(token=token)
    try:
        identity = api.whoami()
        username = identity.get("name") or identity.get("id")
        if username:
            return f"{username}/{repo_id}"
    except Exception:
        pass

    if token:
        try:
            login(token=token, add_to_git_credential=False)
            identity = api.whoami()
            username = identity.get("name") or identity.get("id")
            if username:
                return f"{username}/{repo_id}"
        except Exception:
            pass

    return repo_id


def upload_to_huggingface(local_root: Path, repo_id: str, token: str | None = None) -> None:
    """Upload the local recording directory to Hugging Face Hub."""
    if token:
        login(token=token, add_to_git_credential=False)

    api = HfApi(token=token)
    repo_exists = False
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        repo_exists = True
    except Exception:
        repo_exists = False

    if not repo_exists:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(
                f"HF token cannot create a dataset under '{repo_id}'. "
                "Use a token with write access to that namespace or choose a repo_id you own."
            ) from exc

    try:
        api.upload_folder(
            folder_path=str(local_root),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload custom arm recording data",
        )
    except Exception as exc:
        raise RuntimeError(
            f"HF upload failed for '{repo_id}'. Check token permissions and repo namespace."
        ) from exc
    print("\n" + "=" * 60)
    print("[HF] Upload completed successfully")
    print(f"[HF] Dataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"[HF] Repo ID: {repo_id}")
    print("=" * 60)


class LatestCameraStream:
    """Continuously fetch camera frames and expose the latest snapshot."""

    def __init__(self, camera: MultiCameraInterface):
        self._camera = camera
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: dict[str, np.ndarray] | None = None
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="latest-camera-stream", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frames = self._camera.get_frames()
                if frames is not None:
                    with self._lock:
                        self._latest = dict(frames)
        except Exception as exc:
            self._error = exc
            self._stop.set()

    def get_latest(self) -> dict[str, np.ndarray] | None:
        if self._error is not None:
            raise RuntimeError(f"Camera stream worker failed: {self._error}") from self._error
        with self._lock:
            if self._latest is None:
                return None
            return dict(self._latest)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class ControlSharedReader:
    """Read latest action/state produced by C++ control_service."""

    def __init__(self, shm_name: str):
        self._mapped = MappedStruct(shm_name, ControlSharedBlock)
        self.block = self._mapped.view

    def get_latest(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        action = None
        state = None
        if self.block.cmd.valid:
            action = np.array(self.block.cmd.action_deg, dtype=np.float32)
        if self.block.state.valid:
            state = np.array(self.block.state.state_deg, dtype=np.float32)
        return action, state

    def get_latest_leader(self) -> np.ndarray | None:
        if self.block.leader.valid:
            return np.array(self.block.leader.leader_deg, dtype=np.float32)
        return None

    def request_stop(self) -> None:
        """Ask control_service to stop so it can release serial ports."""
        self.block.stop_flag = 1

    def close(self) -> None:
        self._mapped.close()


def _keyboard_early_stop(stop_event: threading.Event) -> None:
    """Background thread: set stop_event when 'q' key or right-arrow key is pressed."""
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
                if b"q" in buf or b"Q" in buf or b"\x1b[C" in buf or b"[C" in buf:
                    stop_event.set()
                    break
                buf = buf[-8:]  # keep tail for split escape sequences
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except (OSError, termios.error):
            pass


def record_episode(
    control_reader: ControlSharedReader,
    camera_stream: LatestCameraStream,
    task: str,
    duration_s: float,
    fps: int,
    state_fallback: str = "action",
    frame_buffer: list[dict[str, object]] | None = None,
    early_stop_event: threading.Event | None = None,
    sample_callback=None,
) -> tuple[int, int, bool, float]:
    """Record synchronized samples using camera stream + control shared memory.

    Returns (recorded, skipped, interrupted, actual_duration_s).
    """
    record_period = 1.0 / fps
    start = time.perf_counter()
    next_record = start
    end = start + duration_s
    recorded = skipped = 0
    fallback_warned = False

    try:
        while time.perf_counter() < end:
            if early_stop_event is not None and early_stop_event.is_set():
                reason = getattr(early_stop_event, "reason", "early_stop")
                if reason == "rerecord":
                    print("[INFO] 'r' 키 감지 — 에피소드 재녹화")
                else:
                    print("[INFO] 'q' 키 / 오른쪽 화살표 감지 — 에피소드 조기 종료")
                break
            now = time.perf_counter()
            if now < next_record:
                time.sleep(next_record - now)
                now = time.perf_counter()
            next_record += record_period
            if next_record < now:
                next_record = now + record_period

            images = camera_stream.get_latest()
            latest_action, latest_state = control_reader.get_latest()
            latest_leader = control_reader.get_latest_leader()
            follower_state_available = latest_state is not None
            if latest_action is not None and latest_state is None and state_fallback == "action":
                latest_state = latest_action.copy()
                if not fallback_warned:
                    print("[WARN] follower state unavailable; using action as observation.state fallback")
                    fallback_warned = True
            if images is None or latest_action is None or latest_state is None or latest_leader is None:
                skipped += 1
                continue

            sample = {
                **images,
                "observation.state": latest_state.copy(),
                "action": latest_action.copy(),
                "task": task,
            }
            if frame_buffer is not None:
                frame_buffer.append(sample)
            if sample_callback is not None:
                sample_callback(latest_leader, latest_state if follower_state_available else None)
            recorded += 1
    except KeyboardInterrupt:
        return recorded, skipped, True, time.perf_counter() - start

    return recorded, skipped, False, time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Hugging Face dataset repo, e.g. account/arm-teleop")
    parser.add_argument("--dataset-root", type=Path, default=Path("recordings"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per episode")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--control-shm", default="/hybrid_control", help="POSIX shm name for control service output")
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--yolo-onnx", type=Path, default=Path(__file__).resolve().parent.parent / "best.engine")
    parser.add_argument("--yolo-weights", type=Path, default=None, help="Deprecated alias for --yolo-onnx")
    parser.add_argument("--yolo-confidence", type=float, default=0.45)
    parser.add_argument("--yolo-device", default="auto", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--yolo-quantize", default="", help="YOLO inference precision (16/fp16/32/fp32)")
    parser.add_argument("--yolo-classes", type=str, default="", help="Comma-separated class names or IDs to keep")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO square inference size; model export size must match")
    parser.add_argument("--sahi", action=argparse.BooleanOptionalAction, default=False, help="Use SAHI tiled YOLO inference")
    parser.add_argument("--sahi-slice-size", type=int, default=320, help="SAHI square slice size in pixels")
    parser.add_argument("--sahi-overlap", type=float, default=0.2, help="SAHI slice overlap ratio")
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
    ):
        parser.error("episodes, duration, and fps must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.push_to_hub and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for upload; use --no-push-to-hub for local-only recording.")

    if not args.repo_id:
        raise RuntimeError("--repo-id is required unless you pass --auto-repo-id with a short name.")
    if args.auto_repo_id and "/" not in args.repo_id:
        args.repo_id = resolve_repo_id(args.repo_id, os.environ.get("HF_TOKEN"))

    control_reader: ControlSharedReader | None = None
    camera: MultiCameraInterface | None = None
    camera_stream: LatestCameraStream | None = None
    zed: ZedXCameraInterface | None = None
    usb: OpenCVCameraInterface | None = None
    active_episode_frames = 0
    recording_completed = False
    interrupted = False

    try:
        control_reader = ControlSharedReader(args.control_shm)
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
        usb = OpenCVCameraInterface(args.camera_index, height=480, width=640, fps=args.fps)
        camera = MultiCameraInterface(zed, usb, yolo, async_yolo=not args.sync_yolo)

        print(
            "[INFO] recorder config: "
            f"control_shm={args.control_shm} "
            f"sync_yolo={'on' if args.sync_yolo else 'off'} "
            f"yolo_conf={args.yolo_confidence} "
            f"zed_depth={args.zed_depth_mode} "
            f"zed_retrieve_scale={args.zed_retrieve_scale}"
        )

        if not camera.is_open:
            raise RuntimeError("A required device failed to open; no dataset was created.")

        camera_stream = LatestCameraStream(camera)
        camera_stream.start()

        # Wait for first control sample from the shared-memory publisher.
        first_deadline = time.perf_counter() + 5.0
        while time.perf_counter() < first_deadline:
            # Surface camera-worker failures immediately while waiting.
            if camera_stream is not None:
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

        for episode_index in range(args.episodes):
            input(f"\nEpisode {episode_index + 1}/{args.episodes}: press Enter to record. ('q' 키 = 조기 종료) ")
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
                active_episode_frames, skipped, interrupted, actual_duration_s = record_episode(
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
            if active_episode_frames == 0:
                raise RuntimeError("No valid synchronized frames were captured; episode was not saved.")
            episode_dir = args.dataset_root / args.repo_id / f"episode_{episode_index + 1:03d}"
            _write_lerobot_episode(
                episode_dir,
                episode_frames,
                args.task,
                skipped,
                fps=args.fps,
                duration_s=actual_duration_s,
            )
            print(f"[INFO] Saved episode to {episode_dir} ({active_episode_frames} frames) [LeRobot format]")
            active_episode_frames = 0
            if interrupted:
                print("Recording stopped after the saved partial episode.")
                break
        recording_completed = True
    except KeyboardInterrupt:
        interrupted = True
        if active_episode_frames:
            print(f"\nInterrupted: wrote the partial episode as fallback log ({active_episode_frames} frames).")
        else:
            print("\nInterrupted: no partial episode was saved.")
    finally:
        if args.push_to_hub and recording_completed:
            try:
                upload_to_huggingface(args.dataset_root / args.repo_id, args.repo_id, os.environ.get("HF_TOKEN"))
            except Exception as exc:
                print(f"[WARN] Hugging Face upload failed: {exc}")

        if control_reader is not None:
            if interrupted:
                try:
                    control_reader.request_stop()
                    print("[INFO] Ctrl+C detected: requested control_service stop (releases serial ports).")
                    # Give control_service a short window to observe stop_flag and close devices.
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
