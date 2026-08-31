"""Ultralytics YOLO annotation for the ZED left RGB stream."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


def enable_system_tensorrt() -> None:
    """Make JetPack's system TensorRT bindings visible inside the uv venv."""
    try:
        import tensorrt  # noqa: F401

        return
    except ImportError:
        pass

    for site_packages in ("/usr/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages"):
        if Path(site_packages, "tensorrt").is_dir() and site_packages not in sys.path:
            sys.path.append(site_packages)

    try:
        import tensorrt  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT Python bindings are required for a .engine model. "
            "Install the JetPack python3-libnvinfer package or use an ONNX model."
        ) from exc


class YoloAnnotator:
    """Load one custom YOLO checkpoint and draw detections on RGB frames."""

    def __init__(
        self,
        weights: Path,
        confidence: float = 0.1,
        device: str | int | None = None,
        class_filter: list[str] | None = None,
        imgsz: int = 416,
        max_det: int = 100,
        quantize: int | str | None = None,
        sahi: bool = False,
        sahi_slice_size: int = 320,
        sahi_overlap_ratio: float = 0.2,
    ):
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights were not found: {weights}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1")
        if sahi_slice_size < 32:
            raise ValueError("SAHI slice size must be at least 32 pixels")
        if not 0.0 <= sahi_overlap_ratio < 1.0:
            raise ValueError("SAHI overlap ratio must be in [0, 1)")
        if weights.suffix.lower() == ".engine":
            enable_system_tensorrt()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the 'ultralytics' dependency before enabling YOLO annotation.") from exc

        import torch
        self._inject_torchvision_nms_fallback()

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, str):
            device_str = device.lower().strip()
            if device_str in ("auto", "gpu"):
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            elif device_str.isdigit():
                device = f"cuda:{device_str}" if torch.cuda.is_available() else "cpu"
            elif "," in device_str:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            elif device_str in ("cpu", "mps") or device_str.startswith("cuda"):
                device = device_str
            else:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, int):
            device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(str(weights), task="detect")
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        self.max_det = max_det
        self.quantize = quantize
        self.sahi = sahi
        self.sahi_slice_size = sahi_slice_size
        self.sahi_overlap_ratio = sahi_overlap_ratio
        self._sahi_model = None
        self.last_inference_time_ms = 0.0

        # Warm-up / Auto-detect model input size for compiled backends (TensorRT, ONNX, etc.)
        try:
            dummy = np.zeros((16, 16, 3), dtype=np.uint8)
            _ = self.model.predict(source=dummy, conf=self.confidence, device=self.device, verbose=False)
            if hasattr(self.model, "predictor") and hasattr(self.model.predictor, "model"):
                backend = getattr(self.model.predictor.model, "backend", None)
                if backend is not None and hasattr(backend, "imgsz"):
                    backend_imgsz = backend.imgsz
                    if isinstance(backend_imgsz, (list, tuple)) and len(backend_imgsz) >= 2:
                        detected_sz = int(backend_imgsz[0])
                        if detected_sz > 0 and detected_sz != self.imgsz:
                            print(
                                f"[INFO] YoloAnnotator: Overriding imgsz {self.imgsz} -> {detected_sz} "
                                f"to match TensorRT/backend model size."
                            )
                            self.imgsz = detected_sz
        except Exception:
            pass

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if class_filter is None:
            class_filter = [
                str(name)
                for class_id, name in self.model.names.items()
                if "fruit" in str(name).lower()
            ]
        self.class_filter = class_filter
        self.class_filter_ids: list[int] | None = self._resolve_class_ids(self.class_filter) if self.class_filter else None

        if self.sahi:
            try:
                from sahi import AutoDetectionModel
            except ImportError as exc:
                raise RuntimeError(
                    "SAHI inference was requested but the 'sahi' package is not installed."
                ) from exc
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model=self.model,
                confidence_threshold=self.confidence,
                device=self.device,
                image_size=self.imgsz,
            )

    def _resolve_class_ids(self, class_filter: list[str]) -> list[int]:
        if not self.model.names:
            return []

        names_to_ids = {str(name): int(class_id) for class_id, name in self.model.names.items()}
        resolved: list[int] = []
        for entry in class_filter:
            key = entry.strip()
            if not key:
                continue
            if key.isdigit():
                resolved.append(int(key))
            elif key in names_to_ids:
                resolved.append(names_to_ids[key])
            else:
                raise ValueError(f"YOLO class filter value not found in model names: {key}")
        return sorted(set(resolved))

    def _inject_torchvision_nms_fallback(self) -> None:
        try:
            import torch
            import torchvision
            from ultralytics.utils.nms import TorchNMS
        except ImportError:
            return

        # If the current torchvision build lacks CUDA nms support, force CPU fallback.
        try:
            test_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda:0")
            test_scores = torch.tensor([0.5], device="cuda:0")
            torchvision.ops.nms(test_boxes, test_scores, 0.5)
        except Exception:
            def _nms_cpu(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
                if boxes.device.type != "cpu":
                    boxes = boxes.to("cpu")
                    scores = scores.to("cpu")
                keep = TorchNMS.nms(boxes, scores, iou_threshold)
                return keep.to(boxes.device)

            try:
                torchvision.ops.nms = _nms_cpu  # type: ignore[assignment]
                torchvision.ops.boxes.nms = _nms_cpu  # type: ignore[assignment]
            except Exception:
                pass

            try:
                torch.ops.torchvision.nms = _nms_cpu  # type: ignore[assignment]
            except Exception:
                pass

    @staticmethod
    def _color(class_id: int) -> tuple[int, int, int]:
        # RGB colour chosen deterministically by class, so identical classes are
        # visually stable across frames and recorded episodes.
        # Brighter colors for better visibility on dark backgrounds
        return ((100 + 37 * class_id) % 256, (200 + 17 * class_id) % 256, (80 + 29 * class_id) % 256)

    def annotate(self, rgb: np.ndarray, draw: bool = True) -> np.ndarray:
        """Return an RGB copy with detection boxes, classes, and confidence.
        
        Enhanced with preprocessing for better small object detection like strawberries.
        """
        # Ensure the image is BGR when passed to Ultralytics if the model expects OpenCV format.
        # The ZED interface provides RGB frames, and Ultralytics may accept either, but explicit
        # conversion avoids class confusion caused by color channel mismatch.
        src = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        original_height, original_width = rgb.shape[:2]
        resized_src = src
        scale_x = scale_y = 1.0

        if (original_width, original_height) != (self.imgsz, self.imgsz):
            resized_src = cv2.resize(src, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
            scale_x = original_width / self.imgsz
            scale_y = original_height / self.imgsz

        if self.sahi:
            return self._annotate_sahi(rgb, draw)

        predict_kwargs = {
            "source": resized_src,
            "conf": self.confidence,
            "device": self.device,
            "task": "detect",
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
            "augment": False,
            "iou": 0.45,
        }
        if self.class_filter_ids is not None:
            predict_kwargs["classes"] = self.class_filter_ids

        if self.quantize is not None:
            predict_kwargs["quantize"] = self.quantize

        inference_start = cv2.getTickCount()
        results = self.model(**predict_kwargs)
        inference_end = cv2.getTickCount()
        self.last_inference_time_ms = (
            (inference_end - inference_start) / cv2.getTickFrequency()
        ) * 1000.0

        annotated = rgb.copy()
        result = results[0]
        if result.boxes is None or not draw:
            return annotated

        height, width = rgb.shape[:2]
        names = result.names
        for box in result.boxes:
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = np.rint(box.xyxy[0].detach().cpu().numpy()).astype(int)
            if (original_width, original_height) != (self.imgsz, self.imgsz):
                x1 = int(np.clip(x1 * scale_x, 0, width - 1))
                y1 = int(np.clip(y1 * scale_y, 0, height - 1))
                x2 = int(np.clip(x2 * scale_x, 0, width - 1))
                y2 = int(np.clip(y2 * scale_y, 0, height - 1))
            else:
                x1, x2 = np.clip((x1, x2), 0, width - 1)
                y1, y2 = np.clip((y1, y2), 0, height - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = str(names[class_id])
            label = f"{class_name} {confidence:.2f}"
            color = self._color(class_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
        return annotated

    def _annotate_sahi(self, rgb: np.ndarray, draw: bool) -> np.ndarray:
        from sahi.predict import get_sliced_prediction

        inference_start = cv2.getTickCount()
        result = get_sliced_prediction(
            rgb,
            self._sahi_model,
            slice_height=self.sahi_slice_size,
            slice_width=self.sahi_slice_size,
            overlap_height_ratio=self.sahi_overlap_ratio,
            overlap_width_ratio=self.sahi_overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.45,
            verbose=0,
        )
        inference_end = cv2.getTickCount()
        self.last_inference_time_ms = (
            (inference_end - inference_start) / cv2.getTickFrequency()
        ) * 1000.0

        annotated = rgb.copy()
        if not draw:
            return annotated

        height, width = rgb.shape[:2]
        names = self.model.names
        for prediction in result.object_prediction_list:
            class_id = int(prediction.category.id)
            if self.class_filter_ids is not None and class_id not in self.class_filter_ids:
                continue
            confidence = float(prediction.score.value)
            x1, y1, x2, y2 = np.rint(prediction.bbox.to_xyxy()).astype(int)
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = str(names[class_id])
            color = self._color(class_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(
                annotated,
                f"{class_name} {confidence:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated
