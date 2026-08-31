"""Camera preview script for adjusting ZED X, USB camera, and YOLO detection angles.

Usage:
    uv run python hybrid/camera_preview.py --camera-index 2
    uv run python hybrid/camera_preview.py --camera-index 0 --no-yolo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from camera_interface import MultiCameraInterface, OpenCVCameraInterface, ZedXCameraInterface
from yolo_interface import YoloAnnotator

DEPTH_MIN_MM = 300.0
DEPTH_MAX_MM = 2000.0


def depth_to_colormap(depth_img: np.ndarray) -> np.ndarray:
    """Convert depth matrix (mm) to BGR colormap for display."""
    depth = depth_img[..., 0] if depth_img.ndim == 3 else depth_img
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip(depth, DEPTH_MIN_MM, DEPTH_MAX_MM)
    normalized[valid] = (
        255 * (DEPTH_MAX_MM - clipped[valid]) / (DEPTH_MAX_MM - DEPTH_MIN_MM)
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live camera alignment and preview tool.")
    parser.add_argument("--camera-index", type=int, default=0, help="USB camera index (e.g. 0, 1, 2)")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS")
    parser.add_argument(
        "--zed-depth-mode",
        type=str,
        default="performance",
        choices=["none", "performance", "quality", "ultra", "neural", "neural_plus", "neural_light"],
        help="ZED depth sensing mode",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path("best.engine"),
        help="Path to YOLO weights (.engine, .onnx, .pt)",
    )
    parser.add_argument("--yolo-confidence", type=float, default=0.45)
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO annotation")
    parser.add_argument("--grid-scale", type=float, default=1.0, help="Scale factor for preview window size")
    return parser.parse_args()


def draw_label(img: np.ndarray, text: str, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw a semi-transparent header bar with text."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()

    yolo = None
    if not args.no_yolo:
        weights = args.yolo_weights
        if not weights.is_file():
            # Try searching in parent or workspace root if path relative
            alt_weights = Path(__file__).resolve().parent.parent / weights.name
            if alt_weights.is_file():
                weights = alt_weights

        if weights.is_file():
            print(f"[INFO] Loading YOLO model: {weights}")
            try:
                yolo = YoloAnnotator(
                    weights=weights,
                    confidence=args.yolo_confidence,
                    device="auto",
                )
            except Exception as exc:
                print(f"[WARN] Failed to load YOLO ({exc}); continuing without YOLO.")
        else:
            print(f"[WARN] YOLO weights not found ({args.yolo_weights}); previewing without YOLO.")

    print(f"[INFO] Opening ZED X camera (depth_mode={args.zed_depth_mode})...")
    zed = ZedXCameraInterface(
        width=640,
        height=480,
        fps=args.fps,
        depth_mode=args.zed_depth_mode,
    )

    print(f"[INFO] Opening USB camera index {args.camera_index}...")
    usb = OpenCVCameraInterface(args.camera_index, height=480, width=640, fps=args.fps)

    multi_cam = MultiCameraInterface(zed, usb, yolo_annotator=yolo, render_yolo=True)

    if not multi_cam.is_open:
        raise RuntimeError("Failed to initialize cameras. Check camera connections and index.")

    window_name = "Camera Preview - [q]: Quit | [y]: Toggle YOLO | [s]: Save Snapshot"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 60)
    print(" Camera Preview Running")
    print(" Controls:")
    print("   [q] / [ESC] : Quit")
    print("   [y]         : Toggle YOLO annotation")
    print("   [s]         : Save current frame snapshot")
    print("=" * 60 + "\n")

    render_yolo = True
    fps_history = []
    last_time = time.perf_counter()

    snapshot_dir = Path("recordings/snapshots")

    try:
        while True:
            frame_start = time.perf_counter()
            multi_cam.render_yolo = render_yolo
            frames = multi_cam.get_frames()

            if frames is None:
                time.sleep(0.01)
                continue

            zed_rgb = frames["observation.images.zed_left"]
            zed_yolo = frames["observation.images.zed_left_yolo"]
            zed_depth = frames["observation.images.zed_depth"]
            usb_rgb = frames["observation.images.usb"]

            # Convert RGB to BGR for OpenCV display
            zed_bgr = cv2.cvtColor(zed_rgb, cv2.COLOR_RGB2BGR)
            zed_yolo_bgr = cv2.cvtColor(zed_yolo, cv2.COLOR_RGB2BGR)
            usb_bgr = cv2.cvtColor(usb_rgb, cv2.COLOR_RGB2BGR)
            depth_bgr = depth_to_colormap(zed_depth)

            # Labels
            main_zed = zed_yolo_bgr if render_yolo else zed_bgr
            v1 = draw_label(main_zed, f"ZED Left {'(YOLO)' if render_yolo else ''}")
            v2 = draw_label(usb_bgr, f"USB Camera (Index {args.camera_index})")
            v3 = draw_label(depth_bgr, "ZED Depth (Colormap)", (0, 255, 255))

            # FPS calculation
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps_history.append(1.0 / dt)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            avg_fps = np.mean(fps_history) if fps_history else 0.0

            # Blank 4th tile for stats overlay
            stats_tile = np.zeros_like(v1)
            cv2.rectangle(stats_tile, (0, 0), (stats_tile.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(stats_tile, "Status & Info", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

            info_lines = [
                f"FPS: {avg_fps:.1f}",
                f"USB Index: {args.camera_index}",
                f"ZED Res: {zed_rgb.shape[1]}x{zed_rgb.shape[0]}",
                f"USB Res: {usb_rgb.shape[1]}x{usb_rgb.shape[0]}",
                f"YOLO Mode: {'ON' if render_yolo and yolo else 'OFF'}",
                "Press 's' to save snapshot",
                "Press 'q' to quit",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(stats_tile, line, (20, 60 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

            # Combine 2x2 grid
            top_row = np.hstack([v1, v2])
            bottom_row = np.hstack([v3, stats_tile])
            grid = np.vstack([top_row, bottom_row])

            if args.grid_scale != 1.0:
                h, w = grid.shape[:2]
                grid = cv2.resize(grid, (int(w * args.grid_scale), int(h * args.grid_scale)))

            cv2.imshow(window_name, grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("y"):
                render_yolo = not render_yolo
                print(f"[INFO] YOLO render toggled: {render_yolo}")
            elif key == ord("s"):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(snapshot_dir / f"zed_left_{timestamp}.png"), zed_bgr)
                cv2.imwrite(str(snapshot_dir / f"zed_yolo_{timestamp}.png"), zed_yolo_bgr)
                cv2.imwrite(str(snapshot_dir / f"usb_{timestamp}.png"), usb_bgr)
                cv2.imwrite(str(snapshot_dir / f"depth_{timestamp}.png"), depth_bgr)
                cv2.imwrite(str(snapshot_dir / f"grid_{timestamp}.png"), grid)
                print(f"[INFO] Saved snapshots to {snapshot_dir}/ (*_{timestamp}.png)")

    finally:
        multi_cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera preview closed.")


if __name__ == "__main__":
    main()
