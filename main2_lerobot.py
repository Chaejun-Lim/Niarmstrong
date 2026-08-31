"""Record teleop episodes and save them with the `lerobot` library's LeRobotDataset.

Same capture pipeline, CLI, and normalization behavior as hybrid/main2.py, but
serialization is delegated to `lerobot.datasets.lerobot_dataset.LeRobotDataset`
(add_frame/save_episode) instead of hand-rolled parquet/GStreamer shard writing.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from camera_interface import MultiCameraInterface, OpenCVCameraInterface, ZedXCameraInterface
from main import ControlSharedReader, LatestCameraStream, _depth_to_colormap, resolve_repo_id
from main2 import JOINT_NAMES, _csv_floats, _keyboard_episode_control, _normalize_range_m100_100

from lerobot.datasets.lerobot_dataset import LeRobotDataset

RECORDING_WIDTH = 640
RECORDING_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo, e.g. account/arm-teleop")
    parser.add_argument("--dataset-root", type=Path, default=Path("recordings"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per episode")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--control-shm", default="/hybrid_control", help="POSIX shm name for control service output")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--yolo-onnx", type=Path, default=Path(__file__).resolve().parent.parent / "best.engine")
    parser.add_argument("--record-yolo", action=argparse.BooleanOptionalAction, default=False, help="Record YOLO annotated ZED video stream")
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
    parser.add_argument("--zed-retrieve-scale", type=float, default=1.0)
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


def _to_dataset_rgb(camera_name: str, image: np.ndarray) -> np.ndarray:
    """Return a uint8 RGB HxWx3 frame at RECORDING_WIDTH x RECORDING_HEIGHT for dataset storage."""
    img = np.asarray(image)
    if camera_name == "zed_depth":
        bgr = _depth_to_colormap(img)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[:2] != (RECORDING_HEIGHT, RECORDING_WIDTH):
        img = cv2.resize(img, (RECORDING_WIDTH, RECORDING_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def _build_features(camera_keys: list[str]) -> dict[str, dict]:
    features: dict[str, dict] = {
        "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
        "observation.state": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
    }
    for camera_key in camera_keys:
        features[camera_key] = {
            "dtype": "video",
            "shape": (RECORDING_HEIGHT, RECORDING_WIDTH, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def main() -> None:
    args = parse_args()
    if args.push_to_hub and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for upload; use --no-push-to-hub for local-only recording.")

    if args.auto_repo_id and "/" not in args.repo_id:
        args.repo_id = resolve_repo_id(args.repo_id, os.environ.get("HF_TOKEN"))

    state_action_min = _csv_floats("FOLLOWER_MIN_DEG")
    state_action_max = _csv_floats("FOLLOWER_MAX_DEG")
    if state_action_min.shape != state_action_max.shape:
        raise RuntimeError(
            f"FOLLOWER_MIN_DEG and FOLLOWER_MAX_DEG must have the same length: {state_action_min.shape} vs {state_action_max.shape}"
        )
    if state_action_min.shape[0] != len(JOINT_NAMES):
        raise RuntimeError(f"Expected {len(JOINT_NAMES)} follower limits, got {state_action_min.shape[0]}.")

    dataset_dir = args.dataset_root / args.repo_id
    camera_keys = ["observation.images.zed_left", "observation.images.zed_depth", "observation.images.usb"]
    if args.record_yolo:
        camera_keys.append("observation.images.zed_left_yolo")

    control_reader: ControlSharedReader | None = None
    camera: MultiCameraInterface | None = None
    camera_stream: LatestCameraStream | None = None
    zed: ZedXCameraInterface | None = None
    usb: OpenCVCameraInterface | None = None
    dataset: LeRobotDataset | None = None
    recording_completed = False
    interrupted = False

    try:
        if (dataset_dir / "meta" / "info.json").exists():
            dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_dir)
            print(f"[INFO] resuming existing dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
        else:
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=args.fps,
                features=_build_features(camera_keys),
                root=dataset_dir,
                robot_type="hybrid_arm",
                use_videos=True,
            )

        control_reader = ControlSharedReader(args.control_shm)
        yolo = None
        if args.record_yolo:
            from yolo_interface import YoloAnnotator

            class_filter = [x.strip() for x in args.yolo_classes.split(",") if x.strip()]
            yolo_quantize: int | str | None = None
            if args.yolo_quantize:
                q = str(args.yolo_quantize).strip().lower()
                yolo_quantize = int(q) if q in ("16", "32") else q

            yolo = YoloAnnotator(
                args.yolo_onnx,
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

        record_period = 1.0 / args.fps
        episode_index = 0
        while episode_index < args.episodes:
            input(f"\nEpisode {episode_index + 1}/{args.episodes}: press Enter to record. ('q' 키 = 조기 종료, 'r' = 재녹화) ")
            early_stop_event = threading.Event()
            key_thread = threading.Thread(
                target=_keyboard_episode_control,
                args=(early_stop_event,),
                daemon=True,
                name="key-listener",
            )
            key_thread.start()

            recorded = skipped = 0
            fallback_warned = False
            start = time.perf_counter()
            next_record = start
            end = start + args.duration
            try:
                while time.perf_counter() < end:
                    if early_stop_event.is_set():
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
                    if latest_action is not None and latest_state is None and args.state_fallback == "action":
                        latest_state = latest_action.copy()
                        if not fallback_warned:
                            print("[WARN] follower state unavailable; using action as observation.state fallback")
                            fallback_warned = True
                    if images is None or latest_action is None or latest_state is None:
                        skipped += 1
                        continue

                    action_v = _normalize_range_m100_100(
                        np.asarray(latest_action, dtype=np.float32), state_action_min, state_action_max
                    )
                    state_v = _normalize_range_m100_100(
                        np.asarray(latest_state, dtype=np.float32), state_action_min, state_action_max
                    )
                    frame: dict[str, object] = {
                        "action": action_v,
                        "observation.state": state_v,
                        "task": args.task,
                    }
                    for camera_key in camera_keys:
                        camera_name = camera_key.split(".")[-1]
                        frame[camera_key] = _to_dataset_rgb(camera_name, images[camera_key])
                    dataset.add_frame(frame)
                    recorded += 1
            except KeyboardInterrupt:
                interrupted = True
            finally:
                early_stop_event.set()
                key_thread.join(timeout=1.0)

            stop_reason = getattr(early_stop_event, "reason", None)
            if stop_reason == "rerecord":
                dataset.clear_episode_buffer()
                print(f"[INFO] discarded episode {episode_index + 1}; press Enter to rerecord.")
                interrupted = False
                continue

            if recorded == 0:
                dataset.clear_episode_buffer()
                raise RuntimeError("No valid synchronized frames were captured; episode was not saved.")

            dataset.save_episode()
            print(f"[INFO] saved episode {episode_index + 1}: {recorded} frames ({skipped} skipped)")
            episode_index += 1

            if interrupted:
                print("Recording stopped after the saved partial episode.")
                break

        print(f"[INFO] saved LeRobotDataset to {dataset_dir}")
        recording_completed = True

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user.")

    finally:
        if dataset is not None:
            # Without this, meta/episodes/*.parquet footers never get written and the
            # dataset can't be reloaded (LeRobotDataset.finalize() docstring).
            dataset.finalize()

        if args.push_to_hub and recording_completed and dataset is not None:
            try:
                dataset.push_to_hub()
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
