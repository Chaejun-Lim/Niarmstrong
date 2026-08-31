"""Measure YOLO inference FPS and preview recording camera streams.

Press q or Esc to close.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from camera_interface import MultiCameraInterface, OpenCVCameraInterface, ZedXCameraInterface


def check_gpu_status() -> None:
    """Check and display PyTorch CUDA / GPU availability."""
    print("=" * 50)
    cuda_available = torch.cuda.is_available()
    print(f"🔍 CUDA Available : {cuda_available}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        current_device = torch.cuda.current_device()
        print(f"🚀 GPU Count      : {gpu_count}")
        print(f"🎮 Device Name    : {gpu_name}")
        print(f"📌 Active Device  : cuda:{current_device}")
    else:
        print("⚠️  GPU를 사용할 수 없습니다. CPU 모드로 동작합니다.")
    print("=" * 50)


def depth_to_colormap(depth_mm: np.ndarray, minimum_mm: float, maximum_mm: float) -> np.ndarray:
    """Convert metric depth to a display-only BGR color image."""
    depth = depth_mm[..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip(depth, minimum_mm, maximum_mm)
    normalized[valid] = (
        255 * (maximum_mm - clipped[valid]) / (maximum_mm - minimum_mm)
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def has_gui_backend() -> bool:
    """Return True when OpenCV can open a desktop window in this environment."""
    try:
        test = np.zeros((16, 16, 3), dtype=np.uint8)
        cv2.namedWindow("__copilot_gui_check__", cv2.WINDOW_NORMAL)
        cv2.imshow("__copilot_gui_check__", test)
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        return False


def labelled(image_bgr: np.ndarray, label: str) -> np.ndarray:
    result = image_bgr.copy()
    cv2.rectangle(result, (0, 0), (340, 34), (0, 0, 0), thickness=-1)
    cv2.putText(result, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return result


def fit_tile(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def mean_ms(total_ms: float, count: int) -> float:
    return (total_ms / count) if count > 0 else 0.0


def build_status_text(
    loop_fps: float,
    pipe_ms: float,
    pipe_avg_ms: float,
    loop_cap_ms: float,
    zed_ms: float,
    zed_avg_ms: float,
    usb_ms: float,
    usb_avg_ms: float,
    yolo_ms: float,
    yolo_fps_avg: float,
    yolo_pending: bool,
    cache_rate: float,
    loop_total_ms: float | None = None,
    loop_total_avg_ms: float | None = None,
) -> str:
    yolo_state = "busy" if yolo_pending else "idle"
    loop_total_part = ""
    if loop_total_ms is not None and loop_total_avg_ms is not None:
        loop_total_part = f" | LoopTotal: {loop_total_ms:.1f}ms ({loop_total_avg_ms:.1f} avg)"
    return (
        f"Loop FPS: {loop_fps:.1f} | Pipe: {pipe_ms:.1f}ms ({pipe_avg_ms:.1f} avg)"
        f"{loop_total_part} | LoopCap: {loop_cap_ms:.1f}ms | ZED: {zed_ms:.1f}ms ({zed_avg_ms:.1f} avg) | "
        f"USB: {usb_ms:.1f}ms ({usb_avg_ms:.1f} avg) | YOLO: {yolo_ms:.1f}ms ({yolo_fps_avg:.1f} FPS avg, {yolo_state}) | "
        f"Cache: {cache_rate:.1f}%"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usb-camera-index", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--zed-depth-mode",
        type=str,
        default="neural",
        choices=["none", "performance", "quality", "ultra", "neural", "neural_plus", "neural_light"],
        help="ZED depth mode (performance is fastest among depth-enabled modes)",
    )
    parser.add_argument(
        "--zed-retrieve-scale",
        type=float,
        default=1.0,
        help="Scale factor for ZED retrieve resolution (0<scale<=1, e.g. 0.5)",
    )
    parser.add_argument("--zed-depth-sanitize", action="store_true", help="Replace NaN/Inf depth values with 0")
    parser.add_argument("--zed-depth-clip", action="store_true", help="Clip valid depth values to 300-5000mm")
    parser.add_argument("--depth-min-mm", type=float, default=200.0)
    parser.add_argument("--depth-max-mm", type=float, default=2000.0)
    parser.add_argument("--yolo-weights", type=Path, default=Path(__file__).resolve().parent.parent / "best.engine")
    parser.add_argument("--yolo-confidence", type=float, default=0.45)
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--yolo-max-det", type=int, default=10, help="Maximum number of YOLO detections per frame")
    parser.add_argument(
        "--yolo-quantize",
        default="",
        help="YOLO inference precision (e.g. 16, fp16, 32, fp32). Replaces deprecated --yolo-half.",
    )
    parser.add_argument("--yolo-classes", type=str, default="", help="Comma-separated class names or IDs to annotate")
    parser.add_argument(
        "--sync-yolo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run YOLO synchronously for strict frame alignment (default: enabled). Use --no-sync-yolo for async.",
    )
    parser.add_argument("--fps-window-sec", type=float, default=2.0, help="FPS averaging window in seconds")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    return parser.parse_args()


def main() -> None:
    # 1. 프로그램 시작 시 GPU 인식 상태 체크
    check_gpu_status()

    args = parse_args()
    cameras: MultiCameraInterface | None = None
    gui_available = False
    frame_count = 0

    # FPS 측정용 변수
    fps_counter = 0
    yolo_fps = 0.0
    yolo_fps_avg = 0.0
    loop_time_ms = 0.0
    loop_total_ms = 0.0
    yolo_time_sum_ms = 0.0
    zed_time_sum_ms = 0.0
    usb_time_sum_ms = 0.0
    pipe_time_sum_ms = 0.0
    loop_total_sum_ms = 0.0
    cache_hits = 0
    zed_ms_avg = 0.0
    usb_ms_avg = 0.0
    pipe_ms_avg = 0.0
    loop_total_ms_avg = 0.0
    fps_start_time = time.perf_counter()

    try:
        zed = ZedXCameraInterface(
            width=640,
            height=480,
            fps=args.fps,
            depth_mode=args.zed_depth_mode,
            depth_sanitize=args.zed_depth_sanitize,
            depth_clip=args.zed_depth_clip,
            retrieve_scale=args.zed_retrieve_scale,
        )
        usb = OpenCVCameraInterface(args.usb_camera_index, width=640, height=480, fps=args.fps)
        class_filter = [x.strip() for x in args.yolo_classes.split(",") if x.strip()]
        yolo_quantize: int | str | None = None
        if args.yolo_quantize:
            q = args.yolo_quantize.strip().lower()
            if q in ("16", "32"):
                yolo_quantize = int(q)
            else:
                yolo_quantize = q
        
        cameras = MultiCameraInterface(
            zed,
            usb,
            yolo_weights=args.yolo_weights,
            yolo_confidence=args.yolo_confidence,
            yolo_device=args.yolo_device,
            yolo_imgsz=640,
            yolo_max_det=args.yolo_max_det,
            yolo_quantize=yolo_quantize,
            yolo_classes=class_filter if class_filter else None,
            async_yolo=not args.sync_yolo,
        )

        if not args.headless:
            try:
                if not has_gui_backend():
                    raise cv2.error("OpenCV GUI backend unavailable")
                cv2.namedWindow("YOLO Performance Benchmark", cv2.WINDOW_NORMAL)
                gui_available = True
            except cv2.error as e:
                print(f"⚠️  GUI를 사용할 수 없습니다: {e}")
                gui_available = False

        print("🚀 YOLO FPS 측정을 시작합니다...")

        while True:
            t0 = time.perf_counter()

            # 카메라 프레임 수신 (MultiCameraInterface 내부에서 YOLO 추론 실행)
            frames = cameras.get_frames()

            t1 = time.perf_counter()

            if frames is None:
                if gui_available:
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
                continue

            frame_count += 1
            fps_counter += 1

            # 전체 루프 시간 (카메라 + YOLO + 디스플레이)
            loop_time_ms = (t1 - t0) * 1000
            profile = getattr(cameras, "last_profile", {})
            yolo_time_ms = float(profile.get("yolo_ms", getattr(cameras.yolo_annotator, "last_inference_time_ms", 0.0)))
            zed_time_ms = float(profile.get("zed_ms", 0.0))
            usb_time_ms = float(profile.get("usb_ms", 0.0))
            pipeline_time_ms = float(profile.get("pipeline_ms", loop_time_ms))
            used_cached = bool(profile.get("used_cached", 0.0))
            yolo_pending = bool(profile.get("yolo_pending", 0.0))

            if yolo_time_ms > 0.0:
                yolo_time_sum_ms += yolo_time_ms
            zed_time_sum_ms += zed_time_ms
            usb_time_sum_ms += usb_time_ms
            pipe_time_sum_ms += pipeline_time_ms
            if used_cached:
                cache_hits += 1

            # 지정된 윈도우 단위로 평균 FPS 갱신
            now = time.perf_counter()
            elapsed = now - fps_start_time
            if elapsed >= args.fps_window_sec:
                yolo_fps = fps_counter / (now - fps_start_time)
                yolo_avg_ms = mean_ms(yolo_time_sum_ms, fps_counter)
                yolo_fps_avg = (1000.0 / yolo_avg_ms) if yolo_avg_ms > 1e-3 else 0.0
                zed_ms_avg = mean_ms(zed_time_sum_ms, fps_counter)
                usb_ms_avg = mean_ms(usb_time_sum_ms, fps_counter)
                pipe_ms_avg = mean_ms(pipe_time_sum_ms, fps_counter)
                loop_total_ms_avg = mean_ms(loop_total_sum_ms, fps_counter)
                fps_counter = 0
                yolo_time_sum_ms = 0.0
                zed_time_sum_ms = 0.0
                usb_time_sum_ms = 0.0
                pipe_time_sum_ms = 0.0
                loop_total_sum_ms = 0.0
                fps_start_time = now

            zed_rgb = frames["observation.images.zed_left"]
            zed_depth = frames["observation.images.zed_depth"]
            zed_yolo = frames["observation.images.zed_left_yolo"]
            usb_rgb = frames["observation.images.usb"]
            cache_rate = (100.0 * cache_hits / frame_count) if frame_count > 0 else 0.0

            fps_label = build_status_text(
                loop_fps=yolo_fps,
                pipe_ms=pipeline_time_ms,
                pipe_avg_ms=pipe_ms_avg,
                loop_cap_ms=loop_time_ms,
                zed_ms=zed_time_ms,
                zed_avg_ms=zed_ms_avg,
                usb_ms=usb_time_ms,
                usb_avg_ms=usb_ms_avg,
                yolo_ms=yolo_time_ms,
                yolo_fps_avg=yolo_fps_avg,
                yolo_pending=yolo_pending,
                cache_rate=cache_rate,
            )

            if gui_available:
                tiles = [
                    labelled(fit_tile(cv2.cvtColor(zed_rgb, cv2.COLOR_RGB2BGR)), "ZED X RGB"),
                    labelled(
                        fit_tile(depth_to_colormap(zed_depth, args.depth_min_mm, args.depth_max_mm)),
                        f"ZED Depth ({args.depth_min_mm:.0f}-{args.depth_max_mm:.0f}mm)",
                    ),
                    labelled(fit_tile(cv2.cvtColor(zed_yolo, cv2.COLOR_RGB2BGR)), fps_label),
                    labelled(fit_tile(cv2.cvtColor(usb_rgb, cv2.COLOR_RGB2BGR)), "USB RGB"),
                ]
                
                cv2.imshow("YOLO Performance Benchmark", np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:]))))

                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break

            loop_total_ms = (time.perf_counter() - t0) * 1000
            loop_total_sum_ms += loop_total_ms

            if not gui_available and frame_count % 30 == 0:
                fps_label_log = build_status_text(
                    loop_fps=yolo_fps,
                    pipe_ms=pipeline_time_ms,
                    pipe_avg_ms=pipe_ms_avg,
                    loop_cap_ms=loop_time_ms,
                    zed_ms=zed_time_ms,
                    zed_avg_ms=zed_ms_avg,
                    usb_ms=usb_time_ms,
                    usb_avg_ms=usb_ms_avg,
                    yolo_ms=yolo_time_ms,
                    yolo_fps_avg=yolo_fps_avg,
                    yolo_pending=yolo_pending,
                    cache_rate=cache_rate,
                    loop_total_ms=loop_total_ms,
                    loop_total_avg_ms=loop_total_ms_avg,
                )
                print(f"📊 Frame {frame_count:05d} | {fps_label_log}")

    finally:
        if cameras is not None:
            cameras.release()
        if gui_available:
            cv2.destroyAllWindows()
        if frame_count > 0:
            print(f"\n✅ 측정 종료 | 총 프레임: {frame_count}")


if __name__ == "__main__":
    main()