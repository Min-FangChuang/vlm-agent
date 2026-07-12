from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    from .detector import GroundingDetection, YOLOWorldDetector, _normalize_rgb_image
except ImportError:
    from detector import GroundingDetection, YOLOWorldDetector, _normalize_rgb_image  # type: ignore


class YOLOEDetector:
    def __init__(
        self,
        model: str = "yoloe-11s-seg.pt",
        score_threshold: float = 0.25,
        brightness_enable: bool = True,
        brightness_mean_threshold: float = 40.0,
        brightness_alpha: float = 1.2,
        brightness_beta: float = 20.0,
        min_bbox_area_px: float = 400.0,
        min_bbox_area_ratio: float = 0.02,
        max_bbox_aspect_ratio: float = 6.0,
        edge_touch_margin_px: float = 2.0,
        containment_threshold: float = 0.95,
    ) -> None:
        self.model = model
        self.score_threshold = score_threshold
        self.brightness_enable = brightness_enable
        self.brightness_mean_threshold = brightness_mean_threshold
        self.brightness_alpha = brightness_alpha
        self.brightness_beta = brightness_beta
        self.min_bbox_area_px = min_bbox_area_px
        self.min_bbox_area_ratio = min_bbox_area_ratio
        self.max_bbox_aspect_ratio = max_bbox_aspect_ratio
        self.edge_touch_margin_px = edge_touch_margin_px
        self.containment_threshold = containment_threshold
        self._yoloe_model: Any = None

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).reshape(4).tolist()
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    @staticmethod
    def _bbox_intersection(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = np.asarray(bbox_a, dtype=np.float32).reshape(4).tolist()
        bx1, by1, bx2, by2 = np.asarray(bbox_b, dtype=np.float32).reshape(4).tolist()
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        return inter_w * inter_h

    def _filter_detections_like_agent(
        self,
        detections: list[GroundingDetection],
        image_shape: tuple[int, int, int],
    ) -> list[GroundingDetection]:
        boxes = [
            np.asarray(detection.bbox, dtype=np.float32).reshape(4)
            for detection in detections
        ]
        kept: list[GroundingDetection] = []
        image_h, image_w = image_shape[:2]
        image_area = max(float(image_h * image_w), 1.0)

        for index, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox.tolist()
            width = max(0.0, float(x2 - x1))
            height = max(0.0, float(y2 - y1))
            bbox_area = width * height
            bbox_area_ratio = bbox_area / image_area
            short_side = max(min(width, height), 1.0)
            aspect_ratio = max(width, height) / short_side
            edge_touch_count = (
                int(x1 <= self.edge_touch_margin_px)
                + int(y1 <= self.edge_touch_margin_px)
                + int(x2 >= image_w - self.edge_touch_margin_px)
                + int(y2 >= image_h - self.edge_touch_margin_px)
            )

            keep = (
                bbox_area >= self.min_bbox_area_px
                and bbox_area_ratio >= self.min_bbox_area_ratio
                and aspect_ratio <= self.max_bbox_aspect_ratio
                and edge_touch_count <= 1
            )
            if keep:
                kept.append(detections[index])
        return kept

    def _suppress_containing_large_boxes(
        self,
        detections: list[GroundingDetection],
    ) -> list[GroundingDetection]:
        keep = [True] * len(detections)
        for i, det_i in enumerate(detections):
            if not keep[i]:
                continue
            bbox_i = np.asarray(det_i.bbox, dtype=np.float32).reshape(4)
            area_i = self._bbox_area(bbox_i)
            for j, det_j in enumerate(detections):
                if i == j or not keep[j]:
                    continue
                bbox_j = np.asarray(det_j.bbox, dtype=np.float32).reshape(4)
                area_j = self._bbox_area(bbox_j)
                if area_j <= area_i:
                    continue
                intersection = self._bbox_intersection(bbox_i, bbox_j)
                containment = intersection / max(area_i, 1e-6)
                if containment >= self.containment_threshold:
                    keep[j] = False
        return [det for det, flag in zip(detections, keep) if flag]

    @property
    def yoloe_model(self) -> Any:
        if self._yoloe_model is None:
            try:
                from ultralytics import YOLOE
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required to use YOLOEDetector."
                ) from exc
            self._yoloe_model = YOLOE(self.model)
        return self._yoloe_model

    def detect_detections(
        self, rgb: np.ndarray, query_text: str
    ) -> list[GroundingDetection]:
        prompt = query_text.strip()
        if not prompt:
            return []

        image = _normalize_rgb_image(np.asarray(rgb))
        if self.brightness_enable:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean = float(gray.mean())
            if mean < self.brightness_mean_threshold:
                image = cv2.convertScaleAbs(
                    image,
                    alpha=float(self.brightness_alpha),
                    beta=float(self.brightness_beta),
                )
        model = self.yoloe_model
        model.set_classes([prompt])
        results = model.predict(source=image, conf=self.score_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = YOLOWorldDetector._to_numpy(boxes.xyxy)
        confidences = YOLOWorldDetector._to_numpy(boxes.conf).reshape(-1)
        class_ids = YOLOWorldDetector._to_numpy(boxes.cls).reshape(-1).astype(np.int32)
        names = getattr(result, "names", {})

        detections: list[GroundingDetection] = []
        for bbox, score, class_id in zip(xyxy, confidences, class_ids):
            bbox_array = np.asarray(bbox, dtype=np.float32).reshape(4)
            detections.append(
                GroundingDetection(
                    bbox=bbox_array,
                    label=YOLOWorldDetector._resolve_label(
                        names, int(class_id), prompt
                    ),
                    score=float(score),
                )
            )
        detections = self._filter_detections_like_agent(detections, image.shape)
        return detections
