from __future__ import annotations

import argparse
from pathlib import Path
import time

from camera_interface import MultiCameraInterface, NullCameraInterface, OpenCVCameraInterface, ZedXCameraInterface
from shm_layout import MappedStruct, VisionSharedBlock


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish real vision telemetry to shared memory")
    p.add_argument("--vision-shm", default="/hybrid_vision")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--camera-index", type=int, default=2)
    p.add_argument("--disable-usb", action="store_true", help="Run without opening USB camera")
    p.add_argument("--yolo-weights", type=Path, default=Path(__file__).resolve().parent.parent / "best.engine")
    p.add_argument("--yolo-confidence", type=float, default=0.45)
    p.add_argument("--yolo-device", default="auto")
    p.add_argument("--yolo-quantize", default="")
    p.add_argument("--yolo-max-det", type=int, default=10)
    p.add_argument("--zed-depth-mode", default="neural")
    p.add_argument("--zed-retrieve-scale", type=float, default=1.0)
    p.add_argument("--zed-depth-sanitize", action="store_true")
    p.add_argument("--zed-depth-clip", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    yolo_quantize: int | str | None = None
    if args.yolo_quantize:
        q = str(args.yolo_quantize).strip().lower()
        if q in ("16", "32"):
            yolo_quantize = int(q)
        else:
            yolo_quantize = q

    shm = MappedStruct(args.vision_shm, VisionSharedBlock)
    block = shm.view
    block.seq = 0
    block.stop_flag = 0

    zed = ZedXCameraInterface(
        width=640,
        height=480,
        fps=args.fps,
        depth_mode=args.zed_depth_mode,
        depth_sanitize=args.zed_depth_sanitize,
        depth_clip=args.zed_depth_clip,
        retrieve_scale=args.zed_retrieve_scale,
    )
    require_usb = not args.disable_usb
    if args.disable_usb:
        print("ℹ️  USB capture disabled for camera_service.py (--disable-usb).")
        usb = NullCameraInterface()
        require_usb = False
    else:
        try:
            usb = OpenCVCameraInterface(args.camera_index, width=640, height=480, fps=args.fps)
        except RuntimeError as exc:
            print(f"⚠️  USB camera open failed at index {args.camera_index}: {exc}")
            print("⚠️  Continuing with ZED+YOLO only. USB telemetry will be 0.")
            usb = NullCameraInterface()
            require_usb = False

    cam = MultiCameraInterface(
        zed,
        usb,
        yolo_weights=args.yolo_weights,
        yolo_confidence=args.yolo_confidence,
        yolo_device=args.yolo_device,
        yolo_quantize=yolo_quantize,
        yolo_imgsz=640,
        yolo_max_det=args.yolo_max_det,
        async_yolo=False,
        require_usb=require_usb,
        render_yolo=False,
    )

    try:
        while True:
            if block.stop_flag != 0:
                break
            frames = cam.get_frames()
            if frames is None:
                continue
            profile = cam.last_profile
            block.telemetry.frame_seq = block.seq + 1
            block.telemetry.timestamp_ns = time.perf_counter_ns()
            block.telemetry.zed_ms = float(profile.get("zed_ms", 0.0))
            block.telemetry.usb_ms = float(profile.get("usb_ms", 0.0))
            block.telemetry.yolo_ms = float(profile.get("yolo_ms", 0.0))
            block.telemetry.pipeline_ms = float(profile.get("pipeline_ms", 0.0))
            block.telemetry.yolo_busy = 1 if profile.get("yolo_pending", 0.0) else 0
            block.seq = block.telemetry.frame_seq
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        shm.close()


if __name__ == "__main__":
    main()
