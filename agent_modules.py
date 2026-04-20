from __future__ import annotations

import base64
import mimetypes
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import requests
import torch
import yaml
from requests.exceptions import ProxyError, ReadTimeout
from torch.cuda.amp import autocast

MODULE_ROOT = Path(__file__).resolve().parent
PATS_ROOT = MODULE_ROOT / "pats"
if str(PATS_ROOT) not in sys.path:
    sys.path.append(str(PATS_ROOT))

from models.pats import PATS  # type: ignore
from utils.utils import Resize_img  # type: ignore

try:
    from utils.my_gdino import GroundingDINOAPI  # type: ignore
except ModuleNotFoundError:
    GroundingDINOAPI = None  # type: ignore[assignment]

try:
    from .agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View
except ImportError:
    from agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View  # type: ignore

API_ENDPOINT = "api.deepdataspace.com"
API_PATH = "/v2/task/grounding_dino/detection"
DEFAULT_PATS_CONFIG = PATS_ROOT / "configs" / "test_scannet.yaml"


@dataclass
class GroundingDetection:
    bbox: np.ndarray
    label: str
    score: float


@dataclass
class MatchPair:
    image0_xy: np.ndarray
    image1_xy: np.ndarray


@dataclass
class ViewMatchResult:
    matches: list[MatchPair]
    image0_shape: tuple[int, int]
    image1_shape: tuple[int, int]

    @property
    def num_matches(self) -> int:
        return len(self.matches)


@dataclass
class ObjectViewMatchResult:
    view_match: ViewMatchResult
    matched_object_points: np.ndarray
    matched_candidate_points: np.ndarray
    candidate_bbox: np.ndarray
    num_bbox_matches: int
    num_mask_matches: int
    is_match: bool


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 40,
    max_delay: int = 30,
    errors: tuple[type[BaseException], ...] = (ProxyError, ReadTimeout, RuntimeError),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors as exc:
                num_retries += 1
                if num_retries > max_retries:
                    raise RuntimeError(
                        f"Maximum retries exceeded while calling {func.__name__}: {exc}"
                    ) from exc
                time.sleep(min(delay, max_delay))
                delay *= exponential_base * (1 + jitter * random.random())

    return wrapper


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _extract_object_view_rgb(object_view: Any) -> np.ndarray:
    rgb = _get_attr_or_key(object_view, "rgb")
    if rgb is None:
        view = _get_attr_or_key(object_view, "view")
        if view is not None:
            rgb = _get_attr_or_key(view, "rgb")
    if rgb is None:
        raise ValueError("object_view must provide `rgb` or `view.rgb`.")
    return _normalize_rgb_image(np.asarray(rgb))


def _extract_object_view_bbox(object_view: Any) -> np.ndarray:
    bbox = _get_attr_or_key(object_view, "bbox_2d")
    if bbox is None:
        raise ValueError("object_view must provide `bbox_2d`.")
    bbox_array = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox_array.shape[0] != 4:
        raise ValueError(f"Expected bbox with 4 values, got shape={bbox_array.shape}")
    return bbox_array


def _extract_object_view_mask(object_view: Any) -> np.ndarray:
    mask = _get_attr_or_key(object_view, "mask_2d")
    if mask is None:
        mask = _get_attr_or_key(object_view, "mask")
    if mask is None:
        bbox = _extract_object_view_bbox(object_view)
        rgb = _extract_object_view_rgb(object_view)
        height, width = rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox.tolist()]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        mask_array = np.zeros((height, width), dtype=np.uint8)
        if x2 >= x1 and y2 >= y1:
            mask_array[y1 : y2 + 1, x1 : x2 + 1] = 1
        return mask_array
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_array.shape}")
    return mask_array


def _extract_candidate_best_object_view(candidate: Any) -> Any:
    object_views = _get_attr_or_key(candidate, "object_view")
    if object_views is None:
        object_views = _get_attr_or_key(candidate, "object_views")
    if object_views is None or len(object_views) == 0:
        raise ValueError("candidate must provide non-empty `object_view` or `object_views`.")
    best_id = int(_get_attr_or_key(candidate, "best_id", 0))
    if best_id < 0 or best_id >= len(object_views):
        raise IndexError(f"candidate.best_id out of range: {best_id}")
    return object_views[best_id]


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


def _build_submit_url(endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint.rstrip("/") + API_PATH
    return f"https://{endpoint.rstrip('/')}" + API_PATH


def _build_status_url(submit_url: str, task_uuid: str) -> str:
    prefix, separator, _ = submit_url.partition("/v2/task/")
    if not separator:
        raise ValueError(f"Unexpected submit url format: {submit_url}")
    return f"{prefix}/v2/task_status/{task_uuid}"


def _normalize_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape={image.shape}")
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def _encode_rgb_to_data_uri(image: np.ndarray, image_format: str = ".png") -> str:
    image = _normalize_rgb_image(image)
    success, encoded = cv2.imencode(image_format, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Failed to encode RGB image.")
    mime_type, _ = mimetypes.guess_type(f"image{image_format}")
    mime_type = mime_type or "image/png"
    image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def _get_default_groundingdino_token() -> str | None:
    token = os.getenv("GDINO_API_TOKEN") or os.getenv("PA_TOKEN")
    if token:
        return token
    if GroundingDINOAPI is not None:
        return getattr(GroundingDINOAPI, "api_key", None)
    return None


def parse_groundingdino_result(result_response: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objects = result_response.get("data", {}).get("result", {}).get("objects", [])
    boxes: list[Any] = []
    categorys: list[Any] = []
    scores: list[float] = []

    for obj in objects:
        bbox = obj.get("bbox")
        if bbox is None:
            continue
        boxes.append(bbox)
        categorys.append(obj.get("category", "object"))
        scores.append(float(obj.get("score", 1.0)))

    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(categorys),
        np.asarray(scores, dtype=np.float32),
    )


def draw_groundingdino_bbox(
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


class GroundingDinoDetector:
    def __init__(
        self,
        token: str | None = None,
        endpoint: str = API_ENDPOINT,
        model: str = "GroundingDino-1.5-Pro",
        bbox_threshold: float = 0.25,
        iou_threshold: float = 0.8,
        score_threshold: float = 0.25,
        poll_interval: float = 1.0,
        timeout: float = 120.0,
        request_timeout: float = 30.0,
        token_header: str = "Token",
        targets: tuple[str, ...] = ("bbox",),
    ) -> None:
        self.token = token or _get_default_groundingdino_token()
        if not self.token:
            raise ValueError("GroundingDINO token is required via token arg or env.")
        self.submit_url = _build_submit_url(endpoint)
        self.model = model
        self.bbox_threshold = bbox_threshold
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.token_header = token_header
        self.targets = list(targets)

    def _submit_task(self, rgb: np.ndarray, prompt: str) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model,
            "image": _encode_rgb_to_data_uri(rgb),
            "prompt": {"type": "text", "text": prompt},
            "targets": self.targets,
            "bbox_threshold": self.bbox_threshold,
            "iou_threshold": self.iou_threshold,
        }
        headers = {self.token_header: self.token, "Content-Type": "application/json"}
        response = requests.post(
            self.submit_url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        response_json = response.json()
        if response_json.get("code") != 0:
            raise RuntimeError(f"Failed to submit GroundingDINO task: {response_json}")
        return response_json["data"]["task_uuid"], response_json

    def _wait_for_result(self, task_uuid: str) -> dict[str, Any]:
        status_url = _build_status_url(self.submit_url, task_uuid)
        headers = {self.token_header: self.token}
        start_time = time.time()
        while True:
            response = requests.get(status_url, headers=headers, timeout=self.request_timeout)
            response.raise_for_status()
            response_json = response.json()
            if response_json.get("code") != 0:
                raise RuntimeError(f"Failed to query GroundingDINO status: {response_json}")
            data = response_json["data"]
            status = data.get("status")
            if status == "success":
                return response_json
            if status == "failed":
                raise RuntimeError(f"GroundingDINO task failed: {data.get('error')}")
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Timed out waiting for GroundingDINO task {task_uuid}: {status}")
            time.sleep(self.poll_interval)

    def _parse_detections(self, result_response: dict[str, Any]) -> list[GroundingDetection]:
        objects = result_response.get("data", {}).get("result", {}).get("objects", [])
        detections: list[GroundingDetection] = []
        for obj in objects:
            bbox = obj.get("bbox")
            score = float(obj.get("score", 0.0))
            if bbox is None or score < self.score_threshold:
                continue
            detections.append(
                GroundingDetection(
                    bbox=np.asarray(bbox, dtype=np.float32),
                    label=str(obj.get("category", "object")),
                    score=score,
                )
            )
        return detections

    def _package_result(
        self,
        submit_response: dict[str, Any],
        result_response: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "submit_response": submit_response,
            "result_response": result_response,
        }

    @retry_with_exponential_backoff
    def detect(self, rgb: np.ndarray, query_text: str) -> dict[str, Any]:
        task_uuid, submit_response = self._submit_task(rgb, query_text)
        result = self._wait_for_result(task_uuid)
        return self._package_result(submit_response, result)

    def detect_detections(self, rgb: np.ndarray, query_text: str) -> list[GroundingDetection]:
        response = self.detect(rgb, query_text)
        return self._parse_detections(response["result_response"])


class PATSMatcher:
    def __init__(
        self,
        config_path: str | Path = DEFAULT_PATS_CONFIG,
        device: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._args = self._load_config(self.config_path)
        self._model: Any = None

    def _load_config(self, config_path: Path) -> SimpleNamespace:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"PATS config does not exist: {config_path}."
            )
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        args = SimpleNamespace(**config)
        seed = getattr(args, "seed", 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return args

    def _build_model(self) -> Any:
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
            raise ValueError(
                "The two images must produce the same scale factor, matching PATS preprocessing behavior."
            )
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
            return ViewMatchResult(matches=[], image0_shape=image0_shape, image1_shape=image1_shape)
        kp0, kp1 = self._filter_matches(kp0, kp1, scale_factor0, image0_shape, image1_shape)
        matches = [
            MatchPair(
                image0_xy=np.asarray([int(point0[1]), int(point0[0])], dtype=np.int16),
                image1_xy=np.asarray([int(point1[1]), int(point1[0])], dtype=np.int16),
            )
            for point0, point1 in zip(kp0.tolist(), kp1.tolist())
        ]
        return ViewMatchResult(matches=matches, image0_shape=image0_shape, image1_shape=image1_shape)

    def match_candidate_views(self, candidate_views: list[np.ndarray], query_view: np.ndarray) -> list[ViewMatchResult]:
        return [self.match_views(candidate_view, query_view) for candidate_view in candidate_views]

    def match_best_view(self, candidate_views: list[np.ndarray], query_view: np.ndarray) -> tuple[int, ViewMatchResult] | None:
        best_index = -1
        best_result: ViewMatchResult | None = None
        best_score = -1
        for index, candidate_view in enumerate(candidate_views):
            result = self.match_views(candidate_view, query_view)
            if result.num_matches > best_score:
                best_index = index
                best_result = result
                best_score = result.num_matches
        if best_result is None:
            return None
        return best_index, best_result

    def match_object_views(
        self,
        object_view: Any,
        candidate_object_view: Any,
        min_mask_pixels: int = 10,
    ) -> ObjectViewMatchResult:
        object_rgb = _extract_object_view_rgb(object_view)
        candidate_rgb = _extract_object_view_rgb(candidate_object_view)
        candidate_bbox = _extract_object_view_bbox(candidate_object_view)
        candidate_mask = _extract_object_view_mask(candidate_object_view)

        view_match = self.match_views(object_rgb, candidate_rgb)
        if view_match.num_matches == 0:
            empty_points = np.empty((0, 2), dtype=np.int16)
            return ObjectViewMatchResult(
                view_match=view_match,
                matched_object_points=empty_points,
                matched_candidate_points=empty_points.copy(),
                candidate_bbox=candidate_bbox,
                num_bbox_matches=0,
                num_mask_matches=0,
                is_match=False,
            )

        object_points = np.asarray([match.image0_xy for match in view_match.matches], dtype=np.int16)
        candidate_points = np.asarray([match.image1_xy for match in view_match.matches], dtype=np.int16)

        bbox_keep = _points_inside_bbox(candidate_points, candidate_bbox)
        object_points = object_points[bbox_keep]
        candidate_points = candidate_points[bbox_keep]
        num_bbox_matches = int(len(candidate_points))

        if num_bbox_matches == 0:
            empty_points = np.empty((0, 2), dtype=np.int16)
            return ObjectViewMatchResult(
                view_match=view_match,
                matched_object_points=empty_points,
                matched_candidate_points=empty_points.copy(),
                candidate_bbox=candidate_bbox,
                num_bbox_matches=0,
                num_mask_matches=0,
                is_match=False,
            )

        mask_keep = _points_inside_mask(candidate_points, candidate_mask)
        object_points = object_points[mask_keep]
        candidate_points = candidate_points[mask_keep]
        num_mask_matches = int(len(candidate_points))

        return ObjectViewMatchResult(
            view_match=view_match,
            matched_object_points=object_points,
            matched_candidate_points=candidate_points,
            candidate_bbox=candidate_bbox,
            num_bbox_matches=num_bbox_matches,
            num_mask_matches=num_mask_matches,
            is_match=num_mask_matches >= int(min_mask_pixels),
        )

    def match_object_view_to_candidate(
        self,
        object_view: Any,
        candidate: Any,
        min_mask_pixels: int = 10,
    ) -> ObjectViewMatchResult:
        candidate_object_view = _extract_candidate_best_object_view(candidate)
        return self.match_object_views(object_view, candidate_object_view, min_mask_pixels=min_mask_pixels)

    def is_same_candidate(
        self,
        object_view: Any,
        candidate: Any,
        min_mask_pixels: int = 10,
    ) -> bool:
        return self.match_object_view_to_candidate(
            object_view,
            candidate,
            min_mask_pixels=min_mask_pixels,
        ).is_match
