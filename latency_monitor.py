from __future__ import annotations

import argparse
from collections import deque
import gc
from pathlib import Path
import subprocess
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from camera_interface import MultiCameraInterface, OpenCVCameraInterface, ZedXCameraInterface
from shm_layout import ControlSharedBlock, MappedStruct, VisionSharedBlock


DEFAULT_YOLO_ONNX = Path(__file__).resolve().parent.parent / "best.onnx"


class ProcessOutputBuffer:
    """Drain child stdout continuously to avoid pipe backpressure."""

    def __init__(self, proc: subprocess.Popen[str], max_lines: int = 200):
        self._proc = proc
        self._lines = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._proc.stdout is None:
            return
        self._thread = threading.Thread(target=self._run, name="latency-monitor-cpp-log", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                text = line.rstrip("\n")
                if not text:
                    continue
                with self._lock:
                    self._lines.append(text)
        except Exception:
            pass

    def tail_text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=0.2)


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        if value >= 0.0:
            self.total += value
            self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else -1.0


def ns_to_ms(delta_ns: int) -> float:
    return delta_ns / 1_000_000.0 if delta_ns >= 0 else -1.0


def age_ms(now_ns: int, ts_ns: int) -> float:
    if ts_ns <= 0:
        return -1.0
    return ns_to_ms(now_ns - ts_ns)


def diff_ms(later_ns: int, earlier_ns: int) -> float:
    if later_ns <= 0 or earlier_ns <= 0 or later_ns < earlier_ns:
        return -1.0
    return ns_to_ms(later_ns - earlier_ns)


def fmt_ms(value: float) -> str:
    return "n/a" if value < 0.0 else f"{value:.1f}"


def fmt_ms_unit(value: float) -> str:
    return "n/a" if value < 0.0 else f"{value:.1f}ms"


def depth_to_colormap(depth_mm: np.ndarray, minimum_mm: float, maximum_mm: float) -> np.ndarray:
    depth = depth_mm[..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip(depth, minimum_mm, maximum_mm)
    normalized[valid] = (
        255 * (maximum_mm - clipped[valid]) / max(1.0, maximum_mm - minimum_mm)
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def labelled(image_bgr: np.ndarray, label: str) -> np.ndarray:
    out = image_bgr.copy()
    cv2.rectangle(out, (0, 0), (460, 34), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def blank_tile(width: int, height: int, label: str) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def fit_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a display frame without changing the capture resolution."""
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor leader/follower/camera latency from shared memory")
    parser.add_argument("--control-shm", default="/hybrid_control")
    parser.add_argument("--vision-shm", default="/hybrid_vision")
    parser.add_argument("--interval-ms", type=float, default=200.0)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--run-python-vision", action="store_true", help="Run Python ZED+USB+YOLO vision pipeline in-process")
    parser.add_argument("--run-cpp-vision", action="store_true", help="Run C++ camera_service_cpp with ZED native YOLO ONNX")
    parser.add_argument("--gui", action="store_true", help="Show 4-stream GUI: USB, ZED RGB, ZED depth, ZED YOLO")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--yolo-weights", type=Path, default=Path(__file__).resolve().parent.parent / "best.engine")
    parser.add_argument("--yolo-onnx", type=Path, default=DEFAULT_YOLO_ONNX)
    parser.add_argument("--yolo-confidence", type=float, default=0.45)
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--yolo-quantize", default="")
    parser.add_argument("--yolo-max-det", type=int, default=10)
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO square inference size; model export size must match")
    parser.add_argument(
        "--sync-yolo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize YOLO with each ZED frame (default: enabled). Use --no-sync-yolo to run async.",
    )
    parser.add_argument("--zed-depth-mode", default="neural")
    parser.add_argument("--zed-retrieve-scale", type=float, default=1.0)
    parser.add_argument("--depth-min-mm", type=float, default=200.0)
    parser.add_argument("--depth-max-mm", type=float, default=2000.0)
    parser.add_argument("--camera-service-bin", type=Path, default=Path("cpp_pipeline/build/camera_service_cpp"))
    parser.add_argument("--startup-timeout-sec", type=float, default=30.0, help="Wait timeout for first C++ vision telemetry frame")
    parser.add_argument("--reconnect-ms", type=float, default=1000.0, help="Retry interval for missing SHM mappings")
    parser.add_argument("--stale-vision-ms", type=float, default=1000.0, help="Mark vision telemetry older than this as unavailable")
    return parser.parse_args()


def launch_cpp_camera_service(args: argparse.Namespace) -> subprocess.Popen[str]:
    if not args.camera_service_bin.is_file():
        raise FileNotFoundError(f"C++ camera service binary not found: {args.camera_service_bin}")

    cmd = [
        str(args.camera_service_bin),
        "--shm",
        args.vision_shm,
        "--fps",
        str(args.fps),
        "--depth-mode",
        args.zed_depth_mode,
        "--retrieve-scale",
        str(args.zed_retrieve_scale),
        "--yolo-onnx",
        str(args.yolo_onnx),
        "--yolo-conf",
        str(args.yolo_confidence),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main() -> None:
    args = parse_args()
    if args.yolo_imgsz < 32:
        raise ValueError("--yolo-imgsz must be at least 32")
    if args.run_python_vision and args.run_cpp_vision:
        raise RuntimeError("Use only one of --run-python-vision or --run-cpp-vision.")
    if args.gui and not args.run_python_vision:
        raise RuntimeError("--gui requires --run-python-vision to show live 4-stream video.")

    cpp_service_proc: subprocess.Popen[str] | None = None
    cpp_service_logs: ProcessOutputBuffer | None = None
    if args.run_cpp_vision:
        cpp_service_proc = launch_cpp_camera_service(args)
        cpp_service_logs = ProcessOutputBuffer(cpp_service_proc)
        cpp_service_logs.start()
        print("ℹ️  started C++ camera_service_cpp with ZED native YOLO")

    control: MappedStruct | None = None
    try:
        control = MappedStruct(args.control_shm, ControlSharedBlock)
    except FileNotFoundError:
        print(f"⚠️  control shm not found: {args.control_shm}")
        print("⚠️  leader/follower latency will show as n/a until control_service starts.")

    vision: MappedStruct | None = None
    if not args.run_python_vision:
        try:
            vision = MappedStruct(args.vision_shm, VisionSharedBlock)
        except FileNotFoundError:
            if args.run_cpp_vision:
                print(f"ℹ️  waiting for vision shm from C++ service: {args.vision_shm}")
            else:
                print(f"⚠️  vision shm not found: {args.vision_shm}")
                print("⚠️  vision telemetry will show as n/a until camera service starts.")

    camera: MultiCameraInterface | None = None
    if args.run_python_vision:
        yolo_quantize: int | str | None = None
        if args.yolo_quantize:
            q = str(args.yolo_quantize).strip().lower()
            if q in ("16", "32"):
                yolo_quantize = int(q)
            else:
                yolo_quantize = q

        zed = ZedXCameraInterface(
            width=640,
            height=480,
            fps=args.fps,
            depth_mode=args.zed_depth_mode,
            retrieve_scale=args.zed_retrieve_scale,
        )
        usb = OpenCVCameraInterface(args.camera_index, width=640, height=480, fps=args.fps)
        camera = MultiCameraInterface(
            zed,
            usb,
            yolo_weights=args.yolo_weights,
            yolo_confidence=args.yolo_confidence,
            yolo_device=args.yolo_device,
            yolo_quantize=yolo_quantize,
            yolo_imgsz=args.yolo_imgsz,
            yolo_max_det=args.yolo_max_det,
            async_yolo=not args.sync_yolo,
            render_yolo=args.gui,
        )

    cmd_age_avg = RunningAverage()
    state_age_avg = RunningAverage()
    vision_age_avg = RunningAverage()
    leader_rx_period_avg = RunningAverage()
    follower_tx_period_avg = RunningAverage()
    rx_to_tx_delay_avg = RunningAverage()
    zed_avg = RunningAverage()
    yolo_avg = RunningAverage()
    usb_avg = RunningAverage()
    cam_pipe_avg = RunningAverage()

    sample_idx = 0
    last_reconnect_try = 0.0
    startup_deadline = time.perf_counter() + max(0.0, args.startup_timeout_sec)
    first_vision_frame_seen = False
    # Keep explicit handles so we can drop ctypes-backed references before close().
    cmd = state = ctrl_tm = vis_tm = None
    exit_code = 0

    print("🚦 통합 지연 모니터 시작")
    if args.run_python_vision:
        print(f"control_shm={args.control_shm} vision_mode=python gui={'on' if args.gui else 'off'}")
    elif args.run_cpp_vision:
        print(f"control_shm={args.control_shm} vision_mode=cpp gui=off yolo_onnx={args.yolo_onnx}")
    else:
        print(f"control_shm={args.control_shm} vision_shm={args.vision_shm} gui={'on' if args.gui else 'off'}")

    if args.gui:
        cv2.namedWindow("Hybrid Live Monitor", cv2.WINDOW_NORMAL)

    try:
        while True:
            now_ns = time.perf_counter_ns()
            now_s = time.perf_counter()

            if now_s - last_reconnect_try >= max(0.0, args.reconnect_ms) / 1000.0:
                if cpp_service_proc is not None and cpp_service_proc.poll() is not None:
                    output = cpp_service_logs.tail_text() if cpp_service_logs is not None else ""
                    hint = ""
                    if "CAMERA STREAM FAILED TO START" in output:
                        hint = "\nHint: ZED stream failed to start. Check if another process is using the camera, then retry."
                    raise RuntimeError(
                        "C++ camera_service_cpp terminated unexpectedly"
                        + (f"\n{output}" if output else "")
                        + hint
                    )
                if control is None:
                    try:
                        control = MappedStruct(args.control_shm, ControlSharedBlock)
                        print(f"✅ control shm connected: {args.control_shm}")
                    except FileNotFoundError:
                        pass
                if vision is None and not args.run_python_vision:
                    try:
                        vision = MappedStruct(args.vision_shm, VisionSharedBlock)
                        print(f"✅ vision shm connected: {args.vision_shm}")
                    except FileNotFoundError:
                        pass
                last_reconnect_try = now_s

            ctrl_loop_ms = -1.0
            cmd_age = -1.0
            state_age = -1.0
            leader_rx_period_ms = -1.0
            follower_tx_period_ms = -1.0
            rx_to_tx_delay_ms = -1.0

            if control is not None:
                cmd = control.view.cmd
                state = control.view.state
                ctrl_tm = control.view.telemetry
                ctrl_loop_ms = float(ctrl_tm.loop_dt_ms)
                cmd_age = age_ms(now_ns, int(cmd.timestamp_ns)) if cmd.valid else -1.0
                state_age = age_ms(now_ns, int(state.timestamp_ns)) if state.valid else -1.0
                leader_rx_period_ms = float(ctrl_tm.leader_rx_period_ms)
                follower_tx_period_ms = float(ctrl_tm.follower_tx_period_ms)
                rx_to_tx_delay_ms = diff_ms(
                    int(ctrl_tm.follower_tx_timestamp_ns),
                    int(ctrl_tm.leader_rx_timestamp_ns),
                )

            zed_ms = -1.0
            yolo_ms = -1.0
            yolo_busy = False
            usb_ms = -1.0
            cam_pipe_ms = -1.0
            vision_ts_ns = 0
            frames: dict[str, np.ndarray] | None = None

            if camera is not None:
                frames = camera.get_frames()
                profile = camera.last_profile
                if frames is not None:
                    vision_ts_ns = int(profile.get("frame_timestamp_ns", 0.0))
                    if vision_ts_ns <= 0:
                        vision_ts_ns = now_ns
                zed_ms = float(profile.get("zed_ms", -1.0))
                yolo_ms = float(profile.get("yolo_ms", -1.0))
                yolo_busy = bool(profile.get("yolo_pending", 0.0))
                usb_ms = float(profile.get("usb_ms", -1.0))
                cam_pipe_ms = float(profile.get("pipeline_ms", -1.0))
            elif vision is not None:
                vis_tm = vision.view.telemetry
                if vis_tm.frame_seq:
                    vision_ts_ns = int(vis_tm.timestamp_ns)
                    zed_ms = float(vis_tm.zed_ms)
                    yolo_ms = float(vis_tm.yolo_ms)
                    yolo_busy = bool(vis_tm.yolo_busy)
                    usb_ms = float(vis_tm.usb_ms)
                    cam_pipe_ms = float(vis_tm.pipeline_ms)

            vision_age = age_ms(now_ns, vision_ts_ns) if vision_ts_ns > 0 else -1.0
            if vision_ts_ns > 0:
                first_vision_frame_seen = True
            elif args.run_cpp_vision and not first_vision_frame_seen and time.perf_counter() > startup_deadline:
                output = cpp_service_logs.tail_text() if cpp_service_logs is not None else ""
                raise RuntimeError(
                    "Timed out waiting for first C++ vision frame."
                    + (f"\n[service tail]\n{output}" if output else "")
                )
            if vision_age >= 0.0 and vision_age > args.stale_vision_ms:
                vision_age = -1.0
                zed_ms = -1.0
                yolo_ms = -1.0
                usb_ms = -1.0
                cam_pipe_ms = -1.0
            cmd_age_avg.update(cmd_age)
            state_age_avg.update(state_age)
            vision_age_avg.update(vision_age)
            leader_rx_period_avg.update(leader_rx_period_ms)
            follower_tx_period_avg.update(follower_tx_period_ms)
            rx_to_tx_delay_avg.update(rx_to_tx_delay_ms)
            zed_avg.update(zed_ms)
            yolo_avg.update(yolo_ms)
            usb_avg.update(usb_ms)
            cam_pipe_avg.update(cam_pipe_ms)

            sample_idx += 1
            if sample_idx % args.print_every == 0:
                print(
                    " | ".join(
                        [
                            f"ctrl_loop={fmt_ms_unit(ctrl_loop_ms)}",
                            f"cmd_age={fmt_ms_unit(cmd_age)} ({fmt_ms_unit(cmd_age_avg.mean)} avg)",
                            f"state_age={fmt_ms_unit(state_age)} ({fmt_ms_unit(state_age_avg.mean)} avg)",
                            f"vision_age={fmt_ms_unit(vision_age)} ({fmt_ms_unit(vision_age_avg.mean)} avg)",
                            f"leader_rx_period={fmt_ms_unit(leader_rx_period_ms)} ({fmt_ms_unit(leader_rx_period_avg.mean)} avg)",
                            f"follower_tx_period={fmt_ms_unit(follower_tx_period_ms)} ({fmt_ms_unit(follower_tx_period_avg.mean)} avg)",
                            f"rx_to_tx_delay={fmt_ms_unit(rx_to_tx_delay_ms)} ({fmt_ms_unit(rx_to_tx_delay_avg.mean)} avg)",
                            f"zed={zed_ms:.1f}ms ({zed_avg.mean:.1f} avg)",
                            f"yolo={yolo_ms:.1f}ms ({yolo_avg.mean:.1f} avg, {'busy' if yolo_busy else 'idle'})",
                            f"usb={usb_ms:.1f}ms ({usb_avg.mean:.1f} avg)",
                            f"cam_pipe={cam_pipe_ms:.1f}ms ({cam_pipe_avg.mean:.1f} avg)",
                        ]
                    )
                )

            if args.gui:
                tile_w, tile_h = 640, 480
                if frames is None:
                    usb_tile = blank_tile(tile_w, tile_h, "USB image: waiting")
                    zed_tile = blank_tile(tile_w, tile_h, "ZED image: waiting")
                    depth_tile = blank_tile(tile_w, tile_h, "ZED depth: waiting")
                    yolo_tile = blank_tile(tile_w, tile_h, "ZED YOLO: waiting")
                else:
                    usb_rgb = frames.get("observation.images.usb")
                    zed_rgb = frames.get("observation.images.zed_left")
                    zed_depth = frames.get("observation.images.zed_depth")
                    zed_yolo = frames.get("observation.images.zed_left_yolo")

                    usb_tile = labelled(
                        fit_tile(cv2.cvtColor(usb_rgb, cv2.COLOR_RGB2BGR), tile_w, tile_h) if usb_rgb is not None else blank_tile(tile_w, tile_h, "USB image: none"),
                        "USB camera image",
                    )
                    zed_tile = labelled(
                        fit_tile(cv2.cvtColor(zed_rgb, cv2.COLOR_RGB2BGR), tile_w, tile_h) if zed_rgb is not None else blank_tile(tile_w, tile_h, "ZED image: none"),
                        "ZED normal image",
                    )
                    if zed_depth is not None:
                        depth_bgr = fit_tile(depth_to_colormap(zed_depth, args.depth_min_mm, args.depth_max_mm), tile_w, tile_h)
                    else:
                        depth_bgr = blank_tile(tile_w, tile_h, "ZED depth: none")
                    depth_tile = labelled(depth_bgr, "ZED depth image")
                    yolo_tile = labelled(
                        fit_tile(cv2.cvtColor(zed_yolo, cv2.COLOR_RGB2BGR), tile_w, tile_h) if zed_yolo is not None else blank_tile(tile_w, tile_h, "ZED YOLO: none"),
                        "ZED YOLO image",
                    )

                top = np.hstack((usb_tile, zed_tile))
                bottom = np.hstack((depth_tile, yolo_tile))
                mosaic = np.vstack((top, bottom))
                cv2.imshow("Hybrid Live Monitor", mosaic)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            time.sleep(max(0.0, args.interval_ms) / 1000.0)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        exit_code = 1
        print(f"❌ {exc}")
    finally:
        # Drop loop locals that may hold exported pointers to mmap-backed ctypes.
        cmd = state = ctrl_tm = vis_tm = None
        gc.collect()
        if control is not None:
            control.close()
        if vision is not None:
            vision.close()
        if camera is not None:
            camera.release()
        if cpp_service_proc is not None:
            if cpp_service_proc.poll() is None:
                cpp_service_proc.terminate()
                try:
                    cpp_service_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    cpp_service_proc.kill()
        if cpp_service_logs is not None:
            cpp_service_logs.stop()
        if args.gui:
            cv2.destroyAllWindows()

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
