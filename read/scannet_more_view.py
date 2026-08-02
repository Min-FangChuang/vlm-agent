from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import cv2

try:
    from ..agent_schema import View
    from ..module.detector_yoloe import draw_bbox
except ImportError:
    from agent_schema import View  # type: ignore
    from module.detector_yoloe import draw_bbox  # type: ignore


@dataclass
class ProjectedView:
    view_id: str
    image_file: str
    projected_bbox_2d: np.ndarray
    bbox_area: float
    camera_xy: np.ndarray
    image_size: tuple[int, int] = (0, 0)
    projected_points_total: int = 0
    projected_points_in_frame: int = 0
    visible_projected_points: int = 0
    occluded_projected_points: int = 0
    depth_missing_points: int = 0
    visible_ratio: float = 0.0
    occluded_ratio: float = 0.0
    uv: np.ndarray | None = None
    visible_mask: np.ndarray | None = None
    occluded_mask: np.ndarray | None = None
    depth_missing_mask: np.ndarray | None = None
    projected_depths: np.ndarray | None = None
    sampled_depths: np.ndarray | None = None
    selection_reason: str = ""


def bbox_to_array(bbox: Any) -> np.ndarray:
    return np.asarray(bbox, dtype=np.float32).reshape(4)


def bbox3d_to_corners(bbox_3d: Any) -> np.ndarray:
    bbox = np.asarray(bbox_3d, dtype=np.float64).reshape(-1)
    if bbox.shape[0] != 6:
        raise ValueError("bbox_3d must contain 6 values")
    center = bbox[:3]
    dims = bbox[3:]
    half = dims / 2.0
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return center + signs * half


def bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    mask_array = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask_array)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(np.min(xs))
    y1 = int(np.min(ys))
    x2 = int(np.max(xs))
    y2 = int(np.max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def bbox_intersection(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = bbox_to_array(bbox_a).tolist()
    bx1, by1, bx2, by2 = bbox_to_array(bbox_b).tolist()
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return float(inter_w * inter_h)


def bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = bbox_to_array(bbox_a).tolist()
    bx1, by1, bx2, by2 = bbox_to_array(bbox_b).tolist()
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def bbox_center_distance(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = bbox_to_array(bbox_a).tolist()
    bx1, by1, bx2, by2 = bbox_to_array(bbox_b).tolist()
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return float(np.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2))


def choose_detection_for_projected_bbox(
    detections: list[Any],
    projected_bbox: np.ndarray,
    min_projected_coverage: float = 0.75,
) -> tuple[Any | None, dict[str, Any]]:
    if not detections:
        return None, {"found": False, "reason": "no_detections"}

    best_detection = None
    best_metrics = None
    best_score = None
    projected_area = max(bbox_area(projected_bbox), 1e-6)
    for detection in detections:
        detection_bbox = bbox_to_array(detection.bbox)
        coverage = bbox_intersection(projected_bbox, detection_bbox) / projected_area
        if coverage < min_projected_coverage:
            continue
        iou = bbox_iou(projected_bbox, detection_bbox)
        center_distance = bbox_center_distance(projected_bbox, detection_bbox)
        metrics = {
            "found": True,
            "projected_coverage": round(float(coverage), 4),
            "iou_with_projected": round(float(iou), 4),
            "center_distance": round(float(center_distance), 2),
            "detection_bbox_2d": detection_bbox.tolist(),
            "detection_score": round(float(detection.score), 4),
        }
        score = iou * 5.0 + float(detection.score) - center_distance * 0.002
        if best_score is None or score > best_score:
            best_score = score
            best_detection = detection
            best_metrics = metrics

    if best_detection is None or best_metrics is None:
        return None, {
            "found": False,
            "reason": "no_detection_covers_projected_bbox",
            "min_projected_coverage": float(min_projected_coverage),
        }
    return best_detection, best_metrics


def bbox_from_projected_points(
    uv: np.ndarray,
    image_shape: tuple[int, int, int],
    min_inside_points: int = 20,
) -> np.ndarray | None:
    if uv.shape[0] == 0:
        return None
    height, width = image_shape[:2]
    x = uv[:, 0]
    y = uv[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if np.count_nonzero(inside) < int(min_inside_points):
        return None
    x = x[inside]
    y = y[inside]
    x1 = max(0, int(np.floor(np.min(x))))
    y1 = max(0, int(np.floor(np.min(y))))
    x2 = min(width - 1, int(np.ceil(np.max(x))))
    y2 = min(height - 1, int(np.ceil(np.max(y))))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).reshape(4).tolist()
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _bbox_edge_side(projected_view: ProjectedView) -> str:
    bbox = np.asarray(projected_view.projected_bbox_2d, dtype=np.float32).reshape(4)
    frame_width = max(float(projected_view.image_size[0]), 1.0)
    frame_height = max(float(projected_view.image_size[1]), 1.0)
    center_x = float((bbox[0] + bbox[2]) / 2.0)
    center_y = float((bbox[1] + bbox[3]) / 2.0)
    center = np.asarray([center_x, center_y], dtype=np.float32)
    edge_centers = {
        "left": np.asarray([0.0, frame_height / 2.0], dtype=np.float32),
        "right": np.asarray([frame_width, frame_height / 2.0], dtype=np.float32),
        "top": np.asarray([frame_width / 2.0, 0.0], dtype=np.float32),
        "bottom": np.asarray([frame_width / 2.0, frame_height], dtype=np.float32),
    }
    distances = {
        side: float(np.linalg.norm(center - edge_center))
        for side, edge_center in edge_centers.items()
    }
    return sorted(distances.items(), key=lambda pair: pair[1])[0][0]


def _bbox_region(projected_view: ProjectedView) -> str:
    bbox = np.asarray(projected_view.projected_bbox_2d, dtype=np.float32).reshape(4)
    frame_width = max(float(projected_view.image_size[0]), 1.0)
    frame_height = max(float(projected_view.image_size[1]), 1.0)
    center_x = float((bbox[0] + bbox[2]) / 2.0)
    center_y = float((bbox[1] + bbox[3]) / 2.0)
    center = np.asarray([center_x, center_y], dtype=np.float32)
    anchors = {
        "left": np.asarray([0.0, frame_height / 2.0], dtype=np.float32),
        "right": np.asarray([frame_width, frame_height / 2.0], dtype=np.float32),
        "top": np.asarray([frame_width / 2.0, 0.0], dtype=np.float32),
        "bottom": np.asarray([frame_width / 2.0, frame_height], dtype=np.float32),
        "center": np.asarray([frame_width / 2.0, frame_height / 2.0], dtype=np.float32),
    }
    distances = {
        name: float(np.linalg.norm(center - anchor)) for name, anchor in anchors.items()
    }
    return sorted(distances.items(), key=lambda pair: pair[1])[0][0]


def _bbox_region_score(projected_view: ProjectedView) -> float:
    bbox = np.asarray(projected_view.projected_bbox_2d, dtype=np.float32).reshape(4)
    frame_width = max(float(projected_view.image_size[0]), 1.0)
    frame_height = max(float(projected_view.image_size[1]), 1.0)
    center_x = float((bbox[0] + bbox[2]) / 2.0)
    center_y = float((bbox[1] + bbox[3]) / 2.0)
    anchors = {
        "left": np.asarray([0.0, frame_height / 2.0], dtype=np.float32),
        "right": np.asarray([frame_width, frame_height / 2.0], dtype=np.float32),
        "top": np.asarray([frame_width / 2.0, 0.0], dtype=np.float32),
        "bottom": np.asarray([frame_width / 2.0, frame_height], dtype=np.float32),
        "center": np.asarray([frame_width / 2.0, frame_height / 2.0], dtype=np.float32),
    }
    center = np.asarray([center_x, center_y], dtype=np.float32)
    region = _bbox_region(projected_view)
    return float(np.linalg.norm(center - anchors[region]))


def _bbox_missing_edges(
    projected_view: ProjectedView, margin: float = 16.0
) -> set[str]:
    bbox = np.asarray(projected_view.projected_bbox_2d, dtype=np.float32).reshape(4)
    frame_width = max(float(projected_view.image_size[0]), 1.0)
    frame_height = max(float(projected_view.image_size[1]), 1.0)
    x1, y1, x2, y2 = bbox.tolist()
    edges: set[str] = set()
    if x1 <= margin:
        edges.add("left")
    if x2 >= frame_width - margin:
        edges.add("right")
    if y1 <= margin:
        edges.add("top")
    if y2 >= frame_height - margin:
        edges.add("bottom")
    return edges


def _bbox_covered_edges(
    projected_view: ProjectedView, margin: float = 16.0
) -> set[str]:
    return {"left", "right", "top", "bottom"} - _bbox_missing_edges(
        projected_view, margin=margin
    )


def _is_full_in_frame(projected_view: ProjectedView, margin: float = 16.0) -> bool:
    return len(_bbox_missing_edges(projected_view, margin=margin)) == 0


def _bbox_dimensions(bbox: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_to_array(bbox).tolist()
    return max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))


def _is_edge_touching_small_bbox(
    projected_view: ProjectedView,
    *,
    min_side_threshold: float = 100.0,
    margin: float = 16.0,
) -> bool:
    if not _bbox_missing_edges(projected_view, margin=margin):
        return False
    width, height = _bbox_dimensions(projected_view.projected_bbox_2d)
    return width < float(min_side_threshold) or height < float(min_side_threshold)


def _expand_bbox_toward_edges(
    bbox: np.ndarray,
    *,
    target_edges: set[str],
    image_size: tuple[int, int],
    scale: float = 0.5,
) -> np.ndarray:
    bbox_array = bbox_to_array(bbox)
    x1, y1, x2, y2 = bbox_array.tolist()
    width, height = _bbox_dimensions(bbox_array)
    frame_width = max(int(image_size[0]), 1)
    frame_height = max(int(image_size[1]), 1)

    if "left" in target_edges:
        x1 -= width * float(scale)
    if "right" in target_edges:
        x2 += width * float(scale)
    if "top" in target_edges:
        y1 -= height * float(scale)
    if "bottom" in target_edges:
        y2 += height * float(scale)

    x1 = float(np.clip(x1, 0.0, frame_width - 1.0))
    y1 = float(np.clip(y1, 0.0, frame_height - 1.0))
    x2 = float(np.clip(x2, 0.0, frame_width - 1.0))
    y2 = float(np.clip(y2, 0.0, frame_height - 1.0))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def project_bbox3d_to_view(
    bbox_3d: Any,
    *,
    view: Any,
    intrinsic_matrix: np.ndarray,
    world_to_axis_align_matrix: np.ndarray | None,
) -> np.ndarray | None:
    corners = bbox3d_to_corners(bbox_3d)
    uv = project_points_to_view(
        corners,
        intrinsic_matrix,
        np.asarray(view.camera_to_world, dtype=np.float64),
        None
        if world_to_axis_align_matrix is None
        else np.asarray(world_to_axis_align_matrix, dtype=np.float64),
    )
    return bbox_from_projected_points(uv, tuple(view.rgb.shape), min_inside_points=1)


def camera_xyz(
    view: Any,
    world_to_axis_align_matrix: np.ndarray | None = None,
) -> np.ndarray:
    camera_to_world = np.asarray(view.camera_to_world, dtype=np.float64)
    camera_position = np.asarray(camera_to_world[:3, 3], dtype=np.float64)
    if world_to_axis_align_matrix is None:
        return camera_position
    camera_position_h = np.concatenate(
        [camera_position, np.ones((1,), dtype=np.float64)]
    )
    aligned_camera_position = (
        np.asarray(world_to_axis_align_matrix, dtype=np.float64) @ camera_position_h
    )[:3]
    return np.asarray(aligned_camera_position, dtype=np.float64)


def camera_xy(
    view: Any,
    world_to_axis_align_matrix: np.ndarray | None = None,
) -> np.ndarray:
    return camera_xyz(view, world_to_axis_align_matrix)[:2]


def view_angle_relative_to_object(
    camera_xy_position: np.ndarray,
    object_center_xy: np.ndarray,
) -> float:
    delta = np.asarray(camera_xy_position, dtype=np.float64) - np.asarray(
        object_center_xy, dtype=np.float64
    )
    return float(np.arctan2(delta[1], delta[0]))


def project_points_to_view(
    points_xyz_aligned: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    world_to_axis_align_matrix: np.ndarray | None,
) -> np.ndarray:
    points_xyz = np.asarray(points_xyz_aligned, dtype=np.float64)
    if world_to_axis_align_matrix is not None:
        axis_align = np.asarray(world_to_axis_align_matrix, dtype=np.float64)
        aligned_h = np.concatenate(
            [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        raw_world_h = (np.linalg.inv(axis_align) @ aligned_h.T).T
        points_xyz = raw_world_h[:, :3]

    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    cam = (world_to_camera @ points_h.T).T
    valid = cam[:, 2] > 1e-6
    cam = cam[valid]
    if cam.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    proj = (np.asarray(intrinsic, dtype=np.float64) @ cam.T).T
    return proj[:, :2] / proj[:, 2:3]


def _depth_visibility_stats(
    points_xyz_aligned: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    world_to_axis_align_matrix: np.ndarray | None,
    rgb_shape: tuple[int, int, int],
    depth_image: np.ndarray,
    depth_scale: float,
    visibility_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    points_xyz = np.asarray(points_xyz_aligned, dtype=np.float64)
    if world_to_axis_align_matrix is not None:
        axis_align = np.asarray(world_to_axis_align_matrix, dtype=np.float64)
        aligned_h = np.concatenate(
            [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        raw_world_h = (np.linalg.inv(axis_align) @ aligned_h.T).T
        points_xyz = raw_world_h[:, :3]

    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    cam = (world_to_camera @ points_h.T).T
    valid = cam[:, 2] > 1e-6
    cam = cam[valid]
    if cam.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=bool),
            {
                "projected_points_total": 0,
                "projected_points_in_frame": 0,
                "visible_projected_points": 0,
                "occluded_projected_points": 0,
                "depth_missing_points": 0,
                "visible_ratio": 0.0,
                "occluded_ratio": 0.0,
                "projected_depths": np.empty((0,), dtype=np.float64),
                "sampled_depths": np.empty((0,), dtype=np.float64),
            },
        )

    proj = (np.asarray(intrinsic, dtype=np.float64) @ cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    z_proj = cam[:, 2]

    rgb_height, rgb_width = rgb_shape[:2]
    depth_height, depth_width = depth_image.shape[:2]
    scale_x = depth_width / max(rgb_width, 1)
    scale_y = depth_height / max(rgb_height, 1)
    x = uv[:, 0]
    y = uv[:, 1]
    inside = (x >= 0) & (x < rgb_width) & (y >= 0) & (y < rgb_height)
    in_frame_uv = uv[inside]
    in_frame_z = z_proj[inside]

    if in_frame_uv.shape[0] == 0:
        return (
            uv,
            np.zeros((uv.shape[0],), dtype=bool),
            np.zeros((uv.shape[0],), dtype=bool),
            np.zeros((uv.shape[0],), dtype=bool),
            {
                "projected_points_total": int(uv.shape[0]),
                "projected_points_in_frame": 0,
                "visible_projected_points": 0,
                "occluded_projected_points": 0,
                "depth_missing_points": 0,
                "visible_ratio": 0.0,
                "occluded_ratio": 0.0,
                "projected_depths": np.full((uv.shape[0],), np.nan, dtype=np.float64),
                "sampled_depths": np.full((uv.shape[0],), np.nan, dtype=np.float64),
            },
        )

    depth_u = in_frame_uv[:, 0] * scale_x
    depth_v = in_frame_uv[:, 1] * scale_y
    ix = np.clip(np.round(depth_u).astype(np.int32), 0, depth_width - 1)
    iy = np.clip(np.round(depth_v).astype(np.int32), 0, depth_height - 1)

    nearest_index_by_pixel: dict[tuple[int, int], int] = {}
    for local_index, (px, py, pz) in enumerate(zip(ix, iy, in_frame_z)):
        key = (int(px), int(py))
        existing = nearest_index_by_pixel.get(key)
        if existing is None or pz < in_frame_z[existing]:
            nearest_index_by_pixel[key] = local_index

    nearest_local_indices = np.asarray(
        sorted(nearest_index_by_pixel.values()), dtype=np.int32
    )
    nearest_ix = ix[nearest_local_indices]
    nearest_iy = iy[nearest_local_indices]
    nearest_z = in_frame_z[nearest_local_indices]

    raw_depth_values = np.asarray(depth_image[nearest_iy, nearest_ix], dtype=np.float64)
    valid_depth_values = np.isfinite(raw_depth_values) & (raw_depth_values > 0)
    depth_values = np.full(raw_depth_values.shape, np.nan, dtype=np.float64)
    depth_values[valid_depth_values] = raw_depth_values[valid_depth_values] * float(
        depth_scale
    )
    depth_missing = (~np.isfinite(depth_values)) | (depth_values <= 0)
    occluded = (~depth_missing) & (depth_values < (nearest_z - visibility_margin))
    visible = (~depth_missing) & (~occluded)

    in_frame_count = int(nearest_local_indices.shape[0])
    visible_count = int(np.count_nonzero(visible))
    occluded_count = int(np.count_nonzero(occluded))
    missing_count = int(np.count_nonzero(depth_missing))
    visible_mask = np.zeros((uv.shape[0],), dtype=bool)
    occluded_mask = np.zeros((uv.shape[0],), dtype=bool)
    depth_missing_mask = np.zeros((uv.shape[0],), dtype=bool)
    inside_indices = np.where(inside)[0]
    mapped_indices = inside_indices[nearest_local_indices]
    visible_mask[mapped_indices] = visible
    occluded_mask[mapped_indices] = occluded
    depth_missing_mask[mapped_indices] = depth_missing
    sampled_depths = np.full((uv.shape[0],), np.nan, dtype=np.float64)
    sampled_depths[mapped_indices] = depth_values
    projected_depths = np.full((uv.shape[0],), np.nan, dtype=np.float64)
    projected_depths[mapped_indices] = nearest_z
    return (
        uv,
        visible_mask,
        occluded_mask,
        depth_missing_mask,
        {
            "projected_points_total": int(uv.shape[0]),
            "projected_points_in_frame": in_frame_count,
            "visible_projected_points": visible_count,
            "occluded_projected_points": occluded_count,
            "depth_missing_points": missing_count,
            "visible_ratio": float(visible_count / max(in_frame_count, 1)),
            "occluded_ratio": float(occluded_count / max(in_frame_count, 1)),
            "projected_depths": projected_depths,
            "sampled_depths": sampled_depths,
        },
    )


def reproject_candidate_to_scene_views(
    *,
    frame_ids: list[str],
    build_view_fn: Callable[[str], Any],
    intrinsic_matrix: np.ndarray,
    world_to_axis_align_matrix: np.ndarray | None,
    points_3d: np.ndarray,
    min_inside_points: int = 20,
    min_visible_points: int = 50,
    min_visible_ratio: float = 0.3,
    depth_scale: float = 0.001,
    visibility_margin: float = 0.05,
) -> list[ProjectedView]:
    projected_views: list[ProjectedView] = []
    for frame_id in frame_ids:
        view = build_view_fn(frame_id)
        uv, visible_mask, occluded_mask, depth_missing_mask, visibility_stats = (
            _depth_visibility_stats(
                points_3d[:, :3],
                intrinsic_matrix,
                np.asarray(view.camera_to_world, dtype=np.float64),
                None
                if world_to_axis_align_matrix is None
                else np.asarray(world_to_axis_align_matrix, dtype=np.float64),
                tuple(view.rgb.shape),
                np.asarray(view.depth, dtype=np.float64),
                float(depth_scale),
                float(visibility_margin),
            )
        )
        visible_uv = uv[visible_mask]
        if int(visibility_stats["projected_points_in_frame"]) == 0:
            continue
        if int(visibility_stats["visible_projected_points"]) == 0:
            continue
        if int(visibility_stats["visible_projected_points"]) < int(min_visible_points):
            continue
        if float(visibility_stats["visible_ratio"]) < float(min_visible_ratio):
            continue
        bbox = bbox_from_projected_points(
            visible_uv, view.rgb.shape, min_inside_points=min_inside_points
        )
        if bbox is None:
            continue
        projected_views.append(
            ProjectedView(
                view_id=str(view.view_id),
                image_file=f"{view.view_id}.jpg",
                projected_bbox_2d=np.asarray(bbox, dtype=np.float32),
                bbox_area=bbox_area(bbox),
                camera_xy=camera_xy(view, world_to_axis_align_matrix),
                image_size=(int(view.rgb.shape[1]), int(view.rgb.shape[0])),
                projected_points_total=int(visibility_stats["projected_points_total"]),
                projected_points_in_frame=int(
                    visibility_stats["projected_points_in_frame"]
                ),
                visible_projected_points=int(
                    visibility_stats["visible_projected_points"]
                ),
                occluded_projected_points=int(
                    visibility_stats["occluded_projected_points"]
                ),
                depth_missing_points=int(visibility_stats["depth_missing_points"]),
                visible_ratio=float(visibility_stats["visible_ratio"]),
                occluded_ratio=float(visibility_stats["occluded_ratio"]),
                uv=np.asarray(uv, dtype=np.float64),
                visible_mask=np.asarray(visible_mask, dtype=bool),
                occluded_mask=np.asarray(occluded_mask, dtype=bool),
                depth_missing_mask=np.asarray(depth_missing_mask, dtype=bool),
            )
        )
    return projected_views


def select_projected_views_forward(
    projected_views: list[ProjectedView],
    num_views: int,
    object_center_xy: np.ndarray,
    existing_view_ids: set[str],
    existing_angles: list[float],
    covered_edges: set[str],
    max_fallback_distance: float,
) -> list[ProjectedView]:
    sorted_views = sorted(
        projected_views, key=lambda item: item.bbox_area, reverse=True
    )
    used_view_ids = set(existing_view_ids)
    selected: list[ProjectedView] = []
    selected_angles: list[float] = [float(angle) for angle in existing_angles]

    def angle_novelty(item: ProjectedView) -> float:
        item_angle = view_angle_relative_to_object(item.camera_xy, object_center_xy)
        if not selected_angles:
            return float(np.pi)
        return min(
            float(
                abs(
                    np.arctan2(
                        np.sin(item_angle - existing_angle),
                        np.cos(item_angle - existing_angle),
                    )
                )
            )
            for existing_angle in selected_angles
        )

    remaining = [item for item in sorted_views if item.view_id not in used_view_ids]

    while len(selected) < int(num_views) and remaining:
        missing_edges = {"left", "right", "top", "bottom"} - covered_edges

        item = None
        if missing_edges:
            edge_candidates = [
                item for item in remaining if _bbox_covered_edges(item) & missing_edges
            ]
            if edge_candidates:
                item = sorted(
                    edge_candidates,
                    key=lambda view: (
                        -len(_bbox_covered_edges(view) & missing_edges),
                        not _is_full_in_frame(view),
                        -float(view.bbox_area),
                    ),
                )[0]

        if item is None:
            candidates = remaining
            angle_candidates = [
                item for item in candidates if angle_novelty(item) >= np.pi / 6.0
            ]
            if angle_candidates:
                item = sorted(
                    angle_candidates,
                    key=lambda view: (
                        not _is_full_in_frame(view),
                        -angle_novelty(view),
                        -float(view.bbox_area),
                    ),
                )[0]
            else:
                full_views = [item for item in candidates if _is_full_in_frame(item)]
                if full_views:
                    item = sorted(full_views, key=lambda view: -float(view.bbox_area))[
                        0
                    ]
                else:
                    item = sorted(candidates, key=lambda view: -float(view.bbox_area))[
                        0
                    ]

        if item is None:
            break

        selected.append(
            ProjectedView(
                view_id=item.view_id,
                image_file=item.image_file,
                projected_bbox_2d=np.asarray(item.projected_bbox_2d, dtype=np.float32),
                bbox_area=float(item.bbox_area),
                camera_xy=np.asarray(item.camera_xy, dtype=np.float64),
                image_size=(int(item.image_size[0]), int(item.image_size[1])),
                selection_reason="forward_priority",
            )
        )
        used_view_ids.add(item.view_id)
        covered_edges.update(_bbox_covered_edges(item))
        selected_angles.append(
            view_angle_relative_to_object(item.camera_xy, object_center_xy)
        )
        remaining = [
            candidate for candidate in remaining if candidate.view_id != item.view_id
        ]

    return selected


def select_projected_views_yaw(
    projected_views: list[ProjectedView],
    num_views: int,
    object_center_xy: np.ndarray,
    existing_view_ids: set[str],
    max_fallback_distance: float,
) -> list[ProjectedView]:
    filtered_views = [
        item
        for item in projected_views
        if float(
            np.linalg.norm(
                np.asarray(item.camera_xy, dtype=np.float64)
                - np.asarray(object_center_xy, dtype=np.float64)
            )
        )
        <= float(max_fallback_distance)
    ]
    sorted_views = sorted(filtered_views or projected_views, key=_bbox_region_score)
    selected: list[ProjectedView] = []
    used_view_ids = set(existing_view_ids)
    covered_regions: set[str] = set()
    for item in sorted_views:
        if len(selected) >= int(num_views):
            break
        if item.view_id in used_view_ids:
            continue
        region = _bbox_region(item)
        if region in covered_regions:
            continue
        selected.append(
            ProjectedView(
                view_id=item.view_id,
                image_file=item.image_file,
                projected_bbox_2d=np.asarray(item.projected_bbox_2d, dtype=np.float32),
                bbox_area=float(item.bbox_area),
                camera_xy=np.asarray(item.camera_xy, dtype=np.float64),
                image_size=(int(item.image_size[0]), int(item.image_size[1])),
                selection_reason="yaw_region_priority",
            )
        )
        used_view_ids.add(item.view_id)
        covered_regions.add(region)

    if len(selected) < int(num_views):
        for item in sorted_views:
            if len(selected) >= int(num_views):
                break
            if item.view_id in used_view_ids:
                continue
            selected.append(
                ProjectedView(
                    view_id=item.view_id,
                    image_file=item.image_file,
                    projected_bbox_2d=np.asarray(
                        item.projected_bbox_2d, dtype=np.float32
                    ),
                    bbox_area=float(item.bbox_area),
                    camera_xy=np.asarray(item.camera_xy, dtype=np.float64),
                    image_size=(int(item.image_size[0]), int(item.image_size[1])),
                    selection_reason="yaw_fallback",
                )
            )
            used_view_ids.add(item.view_id)
    return selected


def select_projected_views_backward(
    projected_views: list[ProjectedView],
    num_views: int,
    object_center_xy: np.ndarray,
    existing_view_ids: set[str],
    existing_angles: list[float],
    covered_edges: set[str],
    max_fallback_distance: float,
) -> list[ProjectedView]:
    del existing_angles, covered_edges, max_fallback_distance
    remaining = [
        item for item in projected_views if item.view_id not in existing_view_ids
    ]
    sorted_views = sorted(
        remaining,
        key=lambda item: float(
            np.linalg.norm(
                np.asarray(item.camera_xy, dtype=np.float64)
                - np.asarray(object_center_xy, dtype=np.float64)
            )
        ),
        reverse=True,
    )
    selected: list[ProjectedView] = []
    for item in sorted_views[: int(num_views)]:
        selected.append(
            ProjectedView(
                view_id=item.view_id,
                image_file=item.image_file,
                projected_bbox_2d=np.asarray(item.projected_bbox_2d, dtype=np.float32),
                bbox_area=float(item.bbox_area),
                camera_xy=np.asarray(item.camera_xy, dtype=np.float64),
                image_size=(int(item.image_size[0]), int(item.image_size[1])),
                selection_reason="backward_distance_priority",
            )
        )
    return selected


def select_projected_views_yaw(
    projected_views: list[ProjectedView],
    num_views: int,
    object_center_xy: np.ndarray,
    existing_view_ids: set[str],
    max_fallback_distance: float,
) -> list[ProjectedView]:
    filtered_views = [
        item
        for item in projected_views
        if float(
            np.linalg.norm(
                np.asarray(item.camera_xy, dtype=np.float64)
                - np.asarray(object_center_xy, dtype=np.float64)
            )
        )
        <= float(max_fallback_distance)
    ]
    sorted_views = sorted(filtered_views or projected_views, key=_bbox_region_score)
    selected: list[ProjectedView] = []
    used_view_ids = set(existing_view_ids)
    covered_regions: set[str] = set()
    for item in sorted_views:
        if len(selected) >= int(num_views):
            break
        if item.view_id in used_view_ids:
            continue
        region = _bbox_region(item)
        if region in covered_regions:
            continue
        selected.append(
            ProjectedView(
                view_id=item.view_id,
                image_file=item.image_file,
                projected_bbox_2d=np.asarray(item.projected_bbox_2d, dtype=np.float32),
                bbox_area=float(item.bbox_area),
                camera_xy=np.asarray(item.camera_xy, dtype=np.float64),
                image_size=(int(item.image_size[0]), int(item.image_size[1])),
                selection_reason="yaw_region_priority",
            )
        )
        used_view_ids.add(item.view_id)
        covered_regions.add(region)

    if len(selected) < int(num_views):
        for item in sorted_views:
            if len(selected) >= int(num_views):
                break
            if item.view_id in used_view_ids:
                continue
            selected.append(
                ProjectedView(
                    view_id=item.view_id,
                    image_file=item.image_file,
                    projected_bbox_2d=np.asarray(
                        item.projected_bbox_2d, dtype=np.float32
                    ),
                    bbox_area=float(item.bbox_area),
                    camera_xy=np.asarray(item.camera_xy, dtype=np.float64),
                    image_size=(int(item.image_size[0]), int(item.image_size[1])),
                    selection_reason="yaw_fallback",
                )
            )
            used_view_ids.add(item.view_id)
    return selected


def complete_candidate_with_more_views(
    *,
    agent: Any,
    candidate: Any,
    num_bootstrap_views: int = 3,
    max_fallback_distance: float = 1.5,
    action_mode: str = "forward",
) -> Any:
    if (
        agent.segmenter is None
        or agent.mapper_2d3d is None
        or agent.intrinsic_matrix is None
    ):
        return candidate
    frame_ids = getattr(agent.motion, "frame_ids", [])
    build_view_fn = getattr(agent.motion, "_build_view", None)
    if not frame_ids or build_view_fn is None:
        return candidate

    agent.complete_candidate_masks(candidate)
    points_3d, bbox_3d = agent.map_candidate_to_3d(candidate)
    candidate.points_3d = points_3d
    candidate.bbox_3d = bbox_3d
    object_center_xy = np.asarray(bbox_3d, dtype=np.float64).reshape(-1)[:2]
    axis_align_matrix = (
        None
        if agent.world_to_axis_align_matrix is None
        else np.asarray(agent.world_to_axis_align_matrix, dtype=np.float64)
    )

    object_view_type = type(candidate.object_view[0])
    existing_view_ids: set[str] = set()
    existing_angles: list[float] = []
    covered_edges: set[str] = set()
    for object_view in getattr(candidate, "object_view", []) or []:
        source = str(getattr(object_view, "source", ""))
        if "turn_around" in source:
            continue
        view = getattr(object_view, "view", None)
        if view is None:
            continue
        view_id = getattr(view, "view_id", None)
        if view_id is not None:
            existing_view_ids.add(str(view_id))
        camera_position_xy = camera_xy(view, axis_align_matrix)
        existing_angles.append(
            view_angle_relative_to_object(camera_position_xy, object_center_xy)
        )
        if getattr(object_view, "bbox_2d", None) is not None:
            proxy_view = ProjectedView(
                view_id=str(view.view_id),
                image_file=f"{view.view_id}.jpg",
                projected_bbox_2d=np.asarray(object_view.bbox_2d, dtype=np.float32),
                bbox_area=float(bbox_area(object_view.bbox_2d)),
                camera_xy=np.asarray(camera_position_xy, dtype=np.float64),
                image_size=(int(view.rgb.shape[1]), int(view.rgb.shape[0])),
            )
            covered_edges.update(_bbox_covered_edges(proxy_view))

    original_missing_edges = {"left", "right", "top", "bottom"} - covered_edges
    should_segment_bootstrap_views = bool(original_missing_edges)

    projected_views = reproject_candidate_to_scene_views(
        frame_ids=list(frame_ids),
        build_view_fn=build_view_fn,
        intrinsic_matrix=np.asarray(agent.intrinsic_matrix, dtype=np.float64),
        world_to_axis_align_matrix=axis_align_matrix,
        points_3d=np.asarray(points_3d, dtype=np.float64),
    )

    if action_mode == "yaw":
        bootstrap_views = select_projected_views_yaw(
            projected_views,
            num_views=int(num_bootstrap_views),
            object_center_xy=object_center_xy,
            existing_view_ids=existing_view_ids,
            max_fallback_distance=float(max_fallback_distance),
        )
    elif action_mode == "backward":
        bootstrap_views = select_projected_views_backward(
            projected_views,
            num_views=int(num_bootstrap_views),
            object_center_xy=object_center_xy,
            existing_view_ids=existing_view_ids,
            existing_angles=existing_angles,
            covered_edges=covered_edges,
            max_fallback_distance=float(max_fallback_distance),
        )
    else:
        bootstrap_views = select_projected_views_forward(
            projected_views,
            num_views=int(num_bootstrap_views),
            object_center_xy=object_center_xy,
            existing_view_ids=existing_view_ids,
            existing_angles=existing_angles,
            covered_edges=covered_edges,
            max_fallback_distance=float(max_fallback_distance),
        )

    bootstrap_object_views: list[Any] = []
    for projected_view in bootstrap_views:
        view = build_view_fn(projected_view.view_id)
        projected_bbox = np.asarray(
            projected_view.projected_bbox_2d, dtype=np.float32
        ).reshape(4)
        is_yaw = action_mode == "yaw"
        is_backward = action_mode == "backward"
        is_support_only = _is_edge_touching_small_bbox(projected_view)

        if is_support_only:
            final_bbox = projected_bbox
            mask = None
            status = "support_only"
            source = (
                "projected_yaw_support_only"
                if is_yaw
                else "projected_backward_support_only"
                if is_backward
                else "projected_bootstrap_support_only"
            )
        else:
            detections = agent.detect_target_objects(view)
            chosen_detection, _ = choose_detection_for_projected_bbox(
                detections, projected_bbox
            )
            chosen_index = -1
            if chosen_detection is not None:
                for detection_index, detection in enumerate(detections):
                    if detection is chosen_detection:
                        chosen_index = detection_index
                        break
            if chosen_index >= 0:
                view.reference = agent._other_detection_bboxes(detections, chosen_index)
            else:
                view.reference = [
                    np.asarray(detection.bbox, dtype=np.float32).reshape(4)
                    for detection in detections
                ]
            if chosen_detection is not None:
                refined_bbox = bbox_to_array(chosen_detection.bbox)
            else:
                refined_bbox = projected_bbox

            if should_segment_bootstrap_views and chosen_detection is None:
                segment_bbox = _expand_bbox_toward_edges(
                    refined_bbox,
                    target_edges=original_missing_edges,
                    image_size=(int(view.rgb.shape[1]), int(view.rgb.shape[0])),
                    scale=0.5,
                )
                mask = agent.segmenter.segment_from_box(view.rgb, segment_bbox.tolist())
                final_bbox = bbox_from_mask(mask)
                if final_bbox is None:
                    final_bbox = refined_bbox
            else:
                mask = None
                final_bbox = refined_bbox

            status = "active"
            if is_yaw:
                source = (
                    "projected_yaw_segmented"
                    if mask is not None
                    else "projected_yaw_refined"
                )
            elif is_backward:
                source = (
                    "projected_backward_segmented"
                    if mask is not None
                    else "projected_backward_refined"
                )
            else:
                source = (
                    "projected_bootstrap_segmented"
                    if mask is not None
                    else "projected_bootstrap_refined"
                )
        bootstrap_object_views.append(
            object_view_type(
                object_id=f"bootstrap_{projected_view.view_id}",
                label=str(candidate.label),
                score=1.0,
                view=view,
                bbox_2d=final_bbox,
                mask_2d=None if mask is None else np.asarray(mask, dtype=np.uint8),
                points_3d=None,
                status=status,
                source=source,
            )
        )

    for object_view in bootstrap_object_views:
        candidate.add_object_view(object_view)
    agent.ensure_candidate_best_view_mask(candidate)
    candidate.status = "expanded"
    return candidate


def complete_candidate_with_turn_around_views(
    *,
    agent: Any,
    candidate: Any,
    max_distance: float = 1,
    angle_step: float = 45.0,
) -> Any:
    object_views = list(getattr(candidate, "object_view", []) or [])
    if not object_views:
        return candidate

    best_id = int(getattr(candidate, "best_id", 0))
    if best_id < 0 or best_id >= len(object_views):
        best_id = 0
    best_object_view = object_views[best_id]
    best_view = getattr(best_object_view, "view", None)
    if best_view is None or getattr(best_view, "camera_to_world", None) is None:
        return candidate

    frame_ids = getattr(agent.motion, "frame_ids", [])
    build_view_fn = getattr(agent.motion, "_build_view", None)
    if not frame_ids or build_view_fn is None:
        return candidate

    reference_position = np.asarray(best_view.camera_to_world[:3, 3], dtype=np.float64)
    reference_forward = np.asarray(
        best_view.camera_to_world[:3, :3], dtype=np.float64
    ) @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    forward_norm = float(np.linalg.norm(reference_forward))
    if forward_norm <= 1e-8:
        return candidate
    reference_forward = reference_forward / forward_norm
    reference_yaw = float(
        np.degrees(np.arctan2(reference_forward[1], reference_forward[0]))
    )

    centers: list[float] = []
    current = 180.0 - float(angle_step)
    while current > 0.0:
        centers.append(current)
        current -= float(angle_step)
    centers.append(0.0)
    current = -float(angle_step)
    while current >= -180.0 + 1e-6:
        centers.append(current)
        current -= float(angle_step)

    half_step = float(angle_step) * 0.5
    chosen: list[tuple[float, Any, float, float]] = []
    used_view_ids: set[str] = {str(getattr(best_view, "view_id", ""))}

    for center in centers:
        if abs(center) < 1e-6:
            chosen.append((0.0, best_view, 0.0, 0.0))
            continue

        matches: list[tuple[float, float, Any, float]] = []
        for frame_id in frame_ids:
            view = build_view_fn(str(frame_id))
            if getattr(view, "camera_to_world", None) is None:
                continue
            view_id = str(getattr(view, "view_id", frame_id))
            if view_id in used_view_ids:
                continue

            position = np.asarray(view.camera_to_world[:3, 3], dtype=np.float64)
            distance_m = float(np.linalg.norm(position - reference_position))
            if distance_m > float(max_distance):
                continue

            forward = np.asarray(
                view.camera_to_world[:3, :3], dtype=np.float64
            ) @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            norm = float(np.linalg.norm(forward))
            if norm <= 1e-8:
                continue
            forward = forward / norm
            yaw = float(np.degrees(np.arctan2(forward[1], forward[0])))
            yaw_delta = float(((yaw - reference_yaw + 180.0) % 360.0) - 180.0)
            diff = abs(float(((yaw_delta - center + 180.0) % 360.0) - 180.0))
            if diff > half_step:
                continue
            matches.append((diff, distance_m, view, yaw_delta))

        if not matches:
            continue

        matches.sort(
            key=lambda item: (item[0], item[1], str(getattr(item[2], "view_id", "")))
        )
        _, distance_m, selected_view, yaw_delta = matches[0]
        used_view_ids.add(str(getattr(selected_view, "view_id", "")))
        chosen.append(
            (float(center), selected_view, float(yaw_delta), float(distance_m))
        )

    existing_turn_around_indices = [
        index
        for index, item in enumerate(list(getattr(candidate, "object_view", []) or []))
        if str(getattr(item, "source", "")) == "turn_around"
    ]
    for index in reversed(existing_turn_around_indices):
        del candidate.object_view[index]

    if len(chosen) <= 1:
        return candidate

    object_view_type = type(best_object_view)
    best_bbox = np.asarray(best_object_view.bbox_2d, dtype=np.float32).reshape(4)
    stitched_tiles: list[np.ndarray] = []

    for center, view, yaw_delta, _distance_m in chosen:
        image_rgb = np.asarray(view.rgb, dtype=np.uint8).copy()
        if abs(center) < 1e-6:
            image_rgb = draw_bbox(
                image_rgb,
                best_bbox,
                "",
                color=(0, 255, 0),
            )
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        label = f"{yaw_delta:+.0f} deg"
        cv2.putText(
            image_bgr,
            label,
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            label,
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        stitched_tiles.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    if stitched_tiles:
        count = len(stitched_tiles)
        if count <= 3:
            columns, rows = 3, 1
        elif count <= 4:
            columns, rows = 2, 2
        elif count <= 6:
            columns, rows = 3, 2
        elif count <= 8:
            columns, rows = 4, 2
        else:
            columns = 4
            rows = int(np.ceil(count / columns))
        tile_height = max(tile.shape[0] for tile in stitched_tiles)
        tile_width = max(tile.shape[1] for tile in stitched_tiles)
        canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
        for index, tile in enumerate(stitched_tiles):
            row = index // columns
            col = index % columns
            y1 = row * tile_height
            x1 = col * tile_width
            resized = tile
            if tile.shape[0] != tile_height or tile.shape[1] != tile_width:
                resized = cv2.resize(
                    tile, (tile_width, tile_height), interpolation=cv2.INTER_AREA
                )
            canvas[y1 : y1 + tile_height, x1 : x1 + tile_width] = resized
        turn_around_tile_descriptions = [
            f"tile {index}: view_id={getattr(view, 'view_id', '')}, delta_deg={yaw_delta:.1f}"
            for index, (_center, view, yaw_delta, _distance_m) in enumerate(chosen)
        ]
        stitched_view = View(
            rgb=canvas,
            depth=None,
            camera_to_world=None,
            view_id="turn_around_stitched",
            reference={
                "tile_descriptions": turn_around_tile_descriptions,
                "reference_view_id": str(getattr(best_view, "view_id", "")),
            },
        )
        candidate.object_view.append(
            object_view_type(
                object_id="turn_around_stitched",
                label=str(candidate.label),
                score=1.0,
                view=stitched_view,
                bbox_2d=best_bbox,
                mask_2d=None,
                points_3d=None,
                status="support_only",
                source="turn_around",
            )
        )

    candidate.status = "expanded"
    return candidate
