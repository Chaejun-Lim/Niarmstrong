"""Measure camera performance using C++ ZED service path.

This benchmark launches `cpp_pipeline/build/camera_service_cpp` for ZED retrieval
(so ZED runs through the C++ API path) and concurrently samples USB camera timing
from Python for side-by-side comparison.

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from camera_interface import OpenCVCameraInterface
from shm_layout import MappedStruct, VisionSharedBlock


DEFAULT_YOLO_ONNX = Path(__file__).resolve().parent.parent / "best.onnx"


@dataclass
class RunningAverages:
    frame_count: int = 0
    zed_sum_ms: float = 0.0
    yolo_sum_ms: float = 0.0
    usb_read_sum_ms: float = 0.0
    usb_period_sum_ms: float = 0.0
    pipe_sum_ms: float = 0.0
    loop_total_sum_ms: float = 0.0

    def update(
        self,
        zed_ms: float,
        yolo_ms: float,
        usb_read_ms: float,
        usb_period_ms: float,
        pipe_ms: float,
        loop_total_ms: float,
    ) -> None:
        self.frame_count += 1
        self.zed_sum_ms += zed_ms
        self.yolo_sum_ms += yolo_ms
        self.usb_read_sum_ms += usb_read_ms
        self.usb_period_sum_ms += usb_period_ms
        self.pipe_sum_ms += pipe_ms
        self.loop_total_sum_ms += loop_total_ms

    def mean(self, value_sum: float) -> float:
        return (value_sum / self.frame_count) if self.frame_count > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--usb-camera-index", type=int, default=2)
    parser.add_argument("--no-usb", action="store_true", help="Run benchmark without USB camera")
    parser.add_argument("--fps-window-sec", type=float, default=2.0)

    parser.add_argument("--vision-shm", default="/hybrid_vision")
    parser.add_argument("--camera-service-bin", type=Path, default=Path("cpp_pipeline/build/camera_service_cpp"))
    parser.add_argument(
        "--with-yolo",
        action="store_true",
        help="Enable YOLO in C++ camera_service_cpp through ZED CUSTOM_YOLOLIKE_BOX_OBJECTS",
    )
    parser.add_argument("--zed-depth-mode", default="neural")
    parser.add_argument("--zed-retrieve-scale", type=float, default=1.0)
    parser.add_argument("--yolo-onnx", type=Path, default=DEFAULT_YOLO_ONNX)
    parser.add_argument("--yolo-confidence", type=float, default=0.45)

    parser.add_argument(
        "--poll-sleep-ms",
        type=float,
        default=0.2,
        help="Sleep duration while waiting for next SHM frame sequence",
    )
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument(
        "--startup-timeout-sec",
        type=float,
        default=0.0,
        help="Timeout waiting for first SHM frame (0: auto; C++ YOLO mode may need long warmup)",
    )
    return parser.parse_args()


def launch_camera_service(args: argparse.Namespace) -> subprocess.Popen[str]:
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
    ]
    if args.with_yolo:
        cmd += [
            "--yolo-onnx",
            str(args.yolo_onnx),
            "--yolo-conf",
            str(args.yolo_confidence),
        ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


class ProcessOutputBuffer:
    """Drain process stdout continuously to avoid pipe backpressure deadlocks."""

    def __init__(self, proc: subprocess.Popen[str], max_lines: int = 200):
        self._proc = proc
        self._lines = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._proc.stdout is None:
            return
        self._thread = threading.Thread(target=self._run, name="camera-service-log-drain", daemon=True)
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


class LatestUSBStream:
    def __init__(self, camera_index: int, fps: int):
        self._usb = OpenCVCameraInterface(camera_index, width=640, height=480, fps=fps)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="latest-usb-stream", daemon=True)
        self._latest_frame: np.ndarray | None = None
        self._last_read_ms = 0.0
        self._last_period_ms = 0.0
        self._last_ts = 0.0

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            frame = self._usb.get_frame()
            t1 = time.perf_counter()
            if frame is None:
                continue
            read_ms = (t1 - t0) * 1000.0
            period_ms = (t1 - self._last_ts) * 1000.0 if self._last_ts > 0.0 else 0.0
            self._last_ts = t1
            with self._lock:
                self._latest_frame = frame
                self._last_read_ms = read_ms
                self._last_period_ms = period_ms

    def get_latest(self) -> tuple[np.ndarray | None, float, float]:
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0, 0.0
            return self._latest_frame, self._last_read_ms, self._last_period_ms

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._usb.release()


def main() -> None:
    args = parse_args()

    # Start selected camera service first so SHM is initialized.
    service_proc = launch_camera_service(args)
    service_logs = ProcessOutputBuffer(service_proc)
    service_logs.start()
    time.sleep(0.5)

    if service_proc.poll() is not None:
        output = service_logs.tail_text()
        raise RuntimeError(
            "camera service failed during startup"
            + (f"\n{output}" if output else "")
        )

    shm = MappedStruct(args.vision_shm, VisionSharedBlock)
    block = shm.view

    usb_stream: LatestUSBStream | None = None
    usb_available = False
    if not args.no_usb:
        try:
            usb_stream = LatestUSBStream(args.usb_camera_index, args.fps)
            usb_stream.start()
            usb_available = True
        except RuntimeError as exc:
            print(f"⚠️  USB 카메라를 열 수 없습니다: {exc}")
            print("⚠️  USB 없이 ZED(C++) 경로만 측정을 계속합니다. --no-usb 옵션과 동일")
    else:
        print("ℹ️  --no-usb 옵션이 설정되어 USB 측정은 비활성화됩니다.")

    frame_counter = 0
    fps_counter = 0
    last_seq = block.seq
    loop_fps = 0.0
    fps_start = time.perf_counter()
    stats = RunningAverages()

    service_label = "C++ camera_service_cpp + ZED YOLO" if args.with_yolo else "C++ camera_service_cpp"
    print(f"🚀 {service_label} 기반 FPS 측정을 시작합니다...")
    if not usb_available:
        print("ℹ️  USB 상태: off")

    startup_timeout_sec = args.startup_timeout_sec
    if startup_timeout_sec <= 0.0:
        startup_timeout_sec = 180.0 if args.with_yolo else 5.0
    if args.with_yolo:
        print(
            "ℹ️  C++ YOLO 초기 실행에서는 ONNX->TensorRT 최적화 캐시 생성으로 "
            f"첫 프레임이 지연될 수 있습니다. (대기 타임아웃: {startup_timeout_sec:.0f}s)"
        )
    startup_deadline = time.perf_counter() + startup_timeout_sec

    try:
        while True:
            t0 = time.perf_counter()

            # Wait for a fresh frame published by C++ service.
            seq = block.seq
            while seq == last_seq:
                time.sleep(max(0.0, args.poll_sleep_ms) / 1000.0)
                if service_proc.poll() is not None:
                    output = service_logs.tail_text()
                    raise RuntimeError(
                        "camera service terminated unexpectedly."
                        + (f"\n{output}" if output else "")
                    )
                if frame_counter == 0 and time.perf_counter() > startup_deadline:
                    output = service_logs.tail_text()
                    raise RuntimeError(
                        "Timed out waiting for first frame from camera service."
                        + (f"\n[service tail]\n{output}" if output else "")
                    )
                seq = block.seq
            last_seq = seq

            zed_ms = float(block.telemetry.zed_ms)
            yolo_ms = float(block.telemetry.yolo_ms)
            yolo_busy = bool(block.telemetry.yolo_busy)
            cxx_pipe_ms = float(block.telemetry.pipeline_ms)

            # USB is captured asynchronously to avoid coupling with ZED wait.
            usb_read_ms = 0.0
            usb_period_ms = 0.0
            if usb_available and usb_stream is not None:
                usb_frame, usb_read_ms, usb_period_ms = usb_stream.get_latest()
                if usb_frame is None:
                    continue

            t1 = time.perf_counter()
            loop_total_ms = (t1 - t0) * 1000.0

            frame_counter += 1
            fps_counter += 1
            stats.update(
                zed_ms=zed_ms,
                yolo_ms=yolo_ms,
                usb_read_ms=usb_read_ms,
                usb_period_ms=usb_period_ms,
                pipe_ms=cxx_pipe_ms,
                loop_total_ms=loop_total_ms,
            )

            now = time.perf_counter()
            elapsed = now - fps_start
            if elapsed >= args.fps_window_sec:
                loop_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = now

            if frame_counter % args.print_every == 0:
                yolo_ms_avg = stats.mean(stats.yolo_sum_ms)
                yolo_fps_avg = (1000.0 / yolo_ms_avg) if yolo_ms_avg > 1e-3 else 0.0
                print(
                    f"📊 Frame {frame_counter:05d} | "
                    f"Loop FPS: {loop_fps:.1f} | "
                    f"C++Pipe: {cxx_pipe_ms:.1f}ms ({stats.mean(stats.pipe_sum_ms):.1f} avg) | "
                    f"LoopTotal: {loop_total_ms:.1f}ms ({stats.mean(stats.loop_total_sum_ms):.1f} avg) | "
                    f"ZED: {zed_ms:.1f}ms ({stats.mean(stats.zed_sum_ms):.1f} avg) | "
                    f"YOLO: {yolo_ms:.1f}ms ({yolo_ms_avg:.1f} avg, {yolo_fps_avg:.1f} FPS avg, {'busy' if yolo_busy else 'idle'}) | "
                    f"USB read: {usb_read_ms:.1f}ms ({stats.mean(stats.usb_read_sum_ms):.1f} avg) | "
                    f"USB period: {usb_period_ms:.1f}ms ({stats.mean(stats.usb_period_sum_ms):.1f} avg) | "
                    f"USB: {'on' if usb_available else 'off'}"
                )
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"❌ {exc}")
    finally:
        try:
            block.stop_flag = 1
        except Exception:
            pass
        # Release local ctypes reference before closing mmap.
        del block
        shm.close()
        if usb_stream is not None:
            usb_stream.stop()

        if service_proc.poll() is None:
            service_proc.terminate()
            try:
                service_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                service_proc.kill()
        service_logs.stop()

        if frame_counter > 0:
            print(
                f"\n✅ 측정 종료 | 총 프레임: {frame_counter} | "
                f"ZED: {stats.mean(stats.zed_sum_ms):.1f}ms avg | "
                f"YOLO: {stats.mean(stats.yolo_sum_ms):.1f}ms avg | "
                f"USB read: {stats.mean(stats.usb_read_sum_ms):.1f}ms avg | "
                f"USB period: {stats.mean(stats.usb_period_sum_ms):.1f}ms avg | "
                f"C++Pipe: {stats.mean(stats.pipe_sum_ms):.1f}ms avg | "
                f"LoopTotal: {stats.mean(stats.loop_total_sum_ms):.1f}ms avg"
            )


if __name__ == "__main__":
    main()
