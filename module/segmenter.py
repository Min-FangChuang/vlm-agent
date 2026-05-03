from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as e:
    raise ImportError(
        "segment-anything is not installed. Install with: "
        "pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


@dataclass
class SAMConfig:
    checkpoint_path: str
    model_type: str = "vit_h"
    device: str = "cpu"


class SAMSegmenter:
    """
    Lightweight SAM wrapper.

    Input:
        - RGB image: np.ndarray of shape (H, W, 3)
        - prompt: point or box

    Output:
        - binary mask: np.ndarray of shape (H, W), dtype uint8
    """

    def __init__(self, checkpoint_path: str, model_type: str = "vit_h", device: str = "cpu") -> None:
        self.config = SAMConfig(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device=device,
        )

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._validate_image(image_rgb)
        self.predictor.set_image(image_rgb)

    def segment_from_point(
        self,
        image_rgb: np.ndarray,
        point_xy: Tuple[int, int],
        point_label: int = 1,
        multimask_output: bool = True,
    ) -> np.ndarray:
        self.set_image(image_rgb)

        point_coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
        point_labels = np.array([point_label], dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )

        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(np.uint8)

    def segment_from_box(
        self,
        image_rgb: np.ndarray,
        box_xyxy: Sequence[float],
        multimask_output: bool = False,
    ) -> np.ndarray:
        self.set_image(image_rgb)

        box = np.array(box_xyxy, dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=multimask_output,
        )

        if masks.ndim == 3:
            best_idx = int(np.argmax(scores))
            return masks[best_idx].astype(np.uint8)
        return masks.astype(np.uint8)

    @staticmethod
    def _validate_image(image_rgb: np.ndarray) -> None:
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError("image_rgb must be a numpy array.")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (H, W, 3).")
