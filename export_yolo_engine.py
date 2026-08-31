"""Export a YOLO .pt checkpoint to a Jetson-native ONNX model.

The ZED native custom-object API expects ONNX for CUSTOM_YOLOLIKE_BOX_OBJECTS,
so the default export path is ONNX. TensorRT engine export is still supported
for experimental direct TRT inference, but the native ZED pipeline should use ONNX.

Examples:
    cd hybrid
    uv run python export_yolo_engine.py --weights ../best.pt
    uv run python export_yolo_engine.py --weights ../best.pt --format engine
    uv run python export_yolo_engine.py --weights ../best.pt --imgsz 640 --half
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("../best.pt"), help="Source YOLO .pt checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the exported model (default: alongside --weights)",
    )
    parser.add_argument("--format", choices=["engine", "onnx"], default="onnx", help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size baked into the model")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True, help="FP16 engine (recommended on Orin)")
    parser.add_argument("--int8", action="store_true", help="INT8 engine (requires --data for calibration)")
    parser.add_argument("--workspace-gib", type=float, default=4.0, help="TensorRT builder workspace in GiB")
    parser.add_argument("--batch", type=int, default=1, help="Static batch size")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes (slower on Jetson)")
    parser.add_argument("--device", default="0", help="CUDA device index used for export/build")
    parser.add_argument("--data", type=Path, default=None, help="Dataset yaml for INT8 calibration")
    args = parser.parse_args()
    if args.imgsz < 32:
        parser.error("--imgsz must be >= 32")
    if args.batch < 1:
        parser.error("--batch must be >= 1")
    if args.int8 and args.data is None:
        parser.error("--int8 requires --data for calibration")
    return args


def import_tensorrt():
    """Import tensorrt, falling back to the JetPack system site-packages.

    uv venvs do not see the apt-installed tensorrt bindings, so when the import
    fails inside a venv we append the JetPack dist-packages path. This must run
    before ultralytics builds the engine because it imports tensorrt internally.
    """
    try:
        import tensorrt

        return tensorrt
    except ImportError:
        for candidate in ("/usr/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages"):
            if candidate not in sys.path and Path(candidate, "tensorrt").is_dir():
                sys.path.append(candidate)
        try:
            import tensorrt
        except ImportError as exc:
            raise RuntimeError(
                "tensorrt Python bindings are unavailable. On JetPack they are installed "
                "system-wide via apt (python3-libnvinfer); run this script with the system "
                "python or keep the dist-packages fallback path working."
            ) from exc
        return tensorrt


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if weights.suffix != ".pt":
        raise ValueError(f"--weights must be a .pt checkpoint, got: {weights.name}")

    tensorrt = import_tensorrt()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required to build a TensorRT engine")

    print(f"[export] torch={torch.__version__} cuda={torch.version.cuda} trt={tensorrt.__version__}")
    print(f"[export] weights={weights} format={args.format} imgsz={args.imgsz}")

    model = YOLO(str(weights), task="detect")
    export_device = args.device if args.format == "engine" else "cpu"
    export_half = args.half and not args.int8 and args.format == "engine"
    if args.format == "onnx" and args.device != "cpu":
        print("[export] ONNX export uses CPU to avoid the PyTorch 2.11 CUDA arange exporter issue")
    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": export_half,
        "int8": args.int8,
        "data": str(args.data) if args.data else None,
        "batch": args.batch,
        "dynamic": args.dynamic,
        "device": export_device,
        "simplify": True,
    }
    if args.format == "engine":
        export_kwargs["workspace"] = args.workspace_gib

    exported = model.export(**export_kwargs)

    exported_path = Path(str(exported)).resolve()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target = args.output_dir / exported_path.name
        shutil.move(str(exported_path), target)
        exported_path = target

    size_mb = exported_path.stat().st_size / (1024 * 1024)
    print(f"[export] wrote {exported_path} ({size_mb:.1f} MiB)")
    print("[export] note: this engine is only valid for TensorRT", tensorrt.__version__)


if __name__ == "__main__":
    main()
