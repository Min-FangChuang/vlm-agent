from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def _normalize_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape={image.shape}")
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


@dataclass
class GroundingDetection:
    bbox: np.ndarray
    label: str
    score: float


def draw_bbox(
    rgb: np.ndarray,
    bbox: np.ndarray | list[float] | tuple[float, float, float, float],
    target_object: str,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    image_rgb = _normalize_rgb_image(rgb).copy()
    box = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if box.shape[0] != 4:
        raise ValueError(f"Expected bbox with 4 values, got shape={box.shape}")

    x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)

    label = str(target_object)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = max(1, thickness)
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    text_x = max(0, x1)
    text_y = y1 - 8
    if text_y - text_height - baseline < 0:
        text_y = min(image_bgr.shape[0] - baseline - 1, y1 + text_height + 8)

    bg_x2 = min(image_bgr.shape[1] - 1, text_x + text_width + 8)
    bg_y1 = max(0, text_y - text_height - baseline - 4)
    bg_y2 = min(image_bgr.shape[0] - 1, text_y + 4)
    cv2.rectangle(image_bgr, (text_x, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.putText(
        image_bgr,
        label,
        (text_x + 4, max(text_height, text_y - 2)),
        font,
        font_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class YOLOWorldDetector:
    def __init__(
        self,
        model: str = "yolov8s-worldv2.pt",
        score_threshold: float = 0.25,
    ) -> None:
        self.model = model
        self.score_threshold = score_threshold
        self._yolo_world_model: Any = None

    @staticmethod
    def _resolve_label(names: Any, class_id: int, default: str) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, default))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return default

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    @property
    def yolo_world_model(self) -> Any:
        if self._yolo_world_model is None:
            try:
                from ultralytics import YOLOWorld
            except ImportError as exc:
                raise ImportError("ultralytics is required to use YOLOWorldDetector.") from exc
            self._yolo_world_model = YOLOWorld(self.model)
        return self._yolo_world_model

    def detect_detections(self, rgb: np.ndarray, query_text: str) -> list[GroundingDetection]:
        prompt = query_text.strip()
        if not prompt:
            return []

        image = _normalize_rgb_image(np.asarray(rgb))
        model = self.yolo_world_model
        model.set_classes([prompt])
        results = model.predict(source=image, conf=self.score_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = self._to_numpy(boxes.xyxy)
        confidences = self._to_numpy(boxes.conf).reshape(-1)
        class_ids = self._to_numpy(boxes.cls).reshape(-1).astype(np.int32)
        names = getattr(result, "names", {})

        detections: list[GroundingDetection] = []
        for bbox, score, class_id in zip(xyxy, confidences, class_ids):
            detections.append(
                GroundingDetection(
                    bbox=np.asarray(bbox, dtype=np.float32).reshape(4),
                    label=self._resolve_label(names, int(class_id), prompt),
                    score=float(score),
                )
            )
        return detections
