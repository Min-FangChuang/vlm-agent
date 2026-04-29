from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import cv2
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

try:
    from agent_schema import CandidateObject, ObjectView
except ImportError:
    from ..agent_schema import CandidateObject, ObjectView

MODULE_ROOT = Path(__file__).resolve().parent.parent
PATS_ROOT = MODULE_ROOT / "pats"
if str(PATS_ROOT) not in sys.path:
    sys.path.append(str(PATS_ROOT))

from models.pats import PATS  # type: ignore
from utils.utils import Resize_img  # type: ignore

DEFAULT_PATS_CONFIG = PATS_ROOT / "configs" / "test_scannet.yaml"
MIN_TOTAL_MATCHES = 1000
MIN_FINAL_MATCHES = 100


@dataclass
class ViewMatchResult:
    image0_points: np.ndarray
    image1_points: np.ndarray

    @property
    def num_matches(self) -> int:
        return int(len(self.image0_points))


@dataclass
class ObjectViewMatchResult:
    total_matches: int
    num_bbox_matches: int
    num_mask_matches: int
    num_filtered_matches: int
    is_match: bool


def _normalize_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape={image.shape}")
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def _extract_object_view_rgb(object_view: ObjectView) -> np.ndarray:
    return _normalize_rgb_image(np.asarray(object_view.rgb))


def _extract_object_view_bbox(object_view: ObjectView) -> np.ndarray:
    bbox_array = np.asarray(object_view.bbox_2d, dtype=np.float32).reshape(-1)
    if bbox_array.shape[0] != 4:
        raise ValueError(f"Expected bbox with 4 values, got shape={bbox_array.shape}")
    return bbox_array


def _extract_object_view_mask(object_view: ObjectView) -> np.ndarray:
    mask = object_view.mask_2d
    if mask is None:
        raise ValueError("object_view.mask_2d must not be None.")
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_array.shape}")
    return mask_array


def _points_inside_bbox(points_xy: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=bool)
    x1, y1, x2, y2 = bbox.tolist()
    return (
        (points_xy[:, 0] >= x1)
        & (points_xy[:, 0] <= x2)
        & (points_xy[:, 1] >= y1)
        & (points_xy[:, 1] <= y2)
    )


def _points_inside_mask(points_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=bool)
    rounded = np.round(points_xy).astype(np.int32)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < mask.shape[0])
    )
    hits = np.zeros((len(points_xy),), dtype=bool)
    if np.any(valid):
        hits[valid] = mask[rounded[valid, 1], rounded[valid, 0]] > 0
    return hits


class PATSMatcher:
    def __init__(
        self,
        config_path: str | Path = DEFAULT_PATS_CONFIG,
        device: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._args = self._load_config(self.config_path)
        self._model = None

    def _load_config(self, config_path: Path) -> SimpleNamespace:
        if not config_path.is_file():
            raise FileNotFoundError(f"PATS config does not exist: {config_path}.")
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        args = SimpleNamespace(**config)
        seed = getattr(args, "seed", 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return args

    def _build_model(self):
        model = PATS(self._args)
        model.config.checkpoint = str(PATS_ROOT / model.config.checkpoint)
        model.config.checkpoint2 = str(PATS_ROOT / model.config.checkpoint2)
        model.config.checkpoint3 = str(PATS_ROOT / model.config.checkpoint3)
        model.load_state_dict()
        return model.to(self.device).eval()

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _preprocess_image(self, rgb: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
        image_rgb = _normalize_rgb_image(rgb)
        ori_h, ori_w = image_rgb.shape[:2]
        max_shape = max(ori_h, ori_w)
        scale_factor = size / max_shape
        resized = Resize_img(image_rgb, np.array([int(ori_w * scale_factor), int(ori_h * scale_factor)]))
        resized_h, resized_w = resized.shape[:2]
        if resized_h > 480 or resized_w > 640:
            raise ValueError(f"Image becomes larger than 480x640 after resizing: {image_rgb.shape}")
        padded = cv2.copyMakeBorder(
            resized,
            0,
            480 - resized_h,
            0,
            640 - resized_w,
            cv2.BORDER_CONSTANT,
            None,
            0,
        )
        return padded, float(scale_factor), (ori_h, ori_w)

    def _filter_matches(
        self,
        kp0: torch.Tensor,
        kp1: torch.Tensor,
        scale_factor: float,
        image0_shape: tuple[int, int],
        image1_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        kp0 = torch.round(kp0 / scale_factor).to(torch.int16)
        kp1 = torch.round(kp1 / scale_factor).to(torch.int16)
        image0_h, image0_w = image0_shape
        image1_h, image1_w = image1_shape
        mask0 = torch.logical_and(
            torch.logical_and(kp0[:, 1] >= 0, kp0[:, 1] < image0_w),
            torch.logical_and(kp0[:, 0] >= 0, kp0[:, 0] < image0_h),
        )
        mask1 = torch.logical_and(
            torch.logical_and(kp1[:, 1] >= 0, kp1[:, 1] < image1_w),
            torch.logical_and(kp1[:, 0] >= 0, kp1[:, 0] < image1_h),
        )
        mask = torch.logical_and(mask0, mask1)
        return kp0[mask].cpu().numpy(), kp1[mask].cpu().numpy()

    def match_views(self, rgb0: np.ndarray, rgb1: np.ndarray) -> ViewMatchResult:
        image0_input, scale_factor0, image0_shape = self._preprocess_image(rgb0)
        image1_input, scale_factor1, image1_shape = self._preprocess_image(rgb1)
        if abs(scale_factor0 - scale_factor1) > 1e-6:
            raise ValueError("The two images must produce the same scale factor, matching PATS preprocessing behavior.")
        data = {
            "image0_name": ["view0"],
            "image0": torch.from_numpy(image0_input).unsqueeze(0).float().to(self.device),
            "image1_name": ["view1"],
            "image1": torch.from_numpy(image1_input).unsqueeze(0).float().to(self.device),
        }
        with torch.no_grad():
            model = self.model
            with autocast(enabled=self.device.type == "cuda"):
                result = model(data)
        kp0 = result["matches_l"]
        kp1 = result["matches_r"]
        if len(kp0) == 0 or len(kp1) == 0:
            empty_points = np.empty((0, 2), dtype=np.int16)
            return ViewMatchResult(image0_points=empty_points, image1_points=empty_points.copy())
        kp0, kp1 = self._filter_matches(kp0, kp1, scale_factor0, image0_shape, image1_shape)
        image0_points = np.asarray([[int(point0[1]), int(point0[0])] for point0 in kp0.tolist()], dtype=np.int16)
        image1_points = np.asarray([[int(point1[1]), int(point1[0])] for point1 in kp1.tolist()], dtype=np.int16)
        return ViewMatchResult(image0_points=image0_points, image1_points=image1_points)

    def match_object_views(
        self,
        object_view: ObjectView,
        candidate_object_view: ObjectView,
        min_final_matches: int = MIN_FINAL_MATCHES,
    ) -> ObjectViewMatchResult:
        object_rgb = _extract_object_view_rgb(object_view)
        candidate_rgb = _extract_object_view_rgb(candidate_object_view)
        object_bbox = _extract_object_view_bbox(object_view)
        candidate_bbox = _extract_object_view_bbox(candidate_object_view)

        view_match = self.match_views(object_rgb, candidate_rgb)
        if view_match.num_matches <= MIN_TOTAL_MATCHES:
            return ObjectViewMatchResult(
                total_matches=view_match.num_matches,
                num_bbox_matches=0,
                num_mask_matches=0,
                num_filtered_matches=0,
                is_match=False,
            )

        object_points = view_match.image0_points
        candidate_points = view_match.image1_points

        bbox_keep = _points_inside_bbox(candidate_points, candidate_bbox)
        object_points = object_points[bbox_keep]
        candidate_points = candidate_points[bbox_keep]

        object_bbox_keep = _points_inside_bbox(object_points, object_bbox)
        object_points = object_points[object_bbox_keep]
        candidate_points = candidate_points[object_bbox_keep]
        num_bbox_matches = int(len(candidate_points))

        if num_bbox_matches <= int(min_final_matches):
            return ObjectViewMatchResult(
                total_matches=view_match.num_matches,
                num_bbox_matches=num_bbox_matches,
                num_mask_matches=0,
                num_filtered_matches=num_bbox_matches,
                is_match=False,
            )

        if candidate_object_view.mask_2d is None:
            num_mask_matches = num_bbox_matches
            num_filtered_matches = num_bbox_matches
        else:
            candidate_mask = _extract_object_view_mask(candidate_object_view)
            mask_keep = _points_inside_mask(candidate_points, candidate_mask)
            object_points = object_points[mask_keep]
            candidate_points = candidate_points[mask_keep]
            num_mask_matches = int(len(candidate_points))
            num_filtered_matches = num_mask_matches

        return ObjectViewMatchResult(
            total_matches=view_match.num_matches,
            num_bbox_matches=num_bbox_matches,
            num_mask_matches=num_mask_matches,
            num_filtered_matches=num_filtered_matches,
            is_match=num_filtered_matches > int(min_final_matches),
        )

    def match_object_view_to_candidate(
        self,
        object_view: ObjectView,
        candidate: CandidateObject,
        min_final_matches: int = MIN_FINAL_MATCHES,
    ) -> ObjectViewMatchResult:
        candidate_object_views = candidate.object_view
        if len(candidate_object_views) == 0:
            raise ValueError("candidate.object_view must be non-empty.")
        best_id = int(candidate.best_id)
        if best_id < 0 or best_id >= len(candidate_object_views):
            raise IndexError(f"candidate.best_id out of range: {best_id}")
        candidate_object_view = candidate_object_views[best_id]
        return self.match_object_views(object_view, candidate_object_view, min_final_matches=min_final_matches)
