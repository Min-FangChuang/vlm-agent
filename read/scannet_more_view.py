from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


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

    depth_values = np.asarray(
        depth_image[nearest_iy, nearest_ix], dtype=np.float64
    ) * float(depth_scale)
    depth_missing = depth_values <= 0
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


def select_distinct_position_views(
    projected_views: list[ProjectedView],
    num_views: int,
    object_center_xy: np.ndarray | None = None,
    angle_threshold_rad: float = np.pi / 6.0,
    existing_view_ids: set[str] | None = None,
    existing_angles: list[float] | None = None,
    max_fallback_distance: float | None = None,
) -> list[ProjectedView]:
    def _bbox_side_score(projected_view: ProjectedView) -> float:
        bbox = np.asarray(projected_view.projected_bbox_2d, dtype=np.float32).reshape(4)
        frame_width = max(float(projected_view.image_size[0]), 1.0)
        center_x = float((bbox[0] + bbox[2]) / 2.0)
        horizontal_side = _bbox_horizontal_side(projected_view)
        if horizontal_side == "left":
            return center_x
        if horizontal_side == "right":
            return max(frame_width - center_x, 0.0)
        return abs(center_x - (frame_width / 2.0))

    def _target_distance(projected_view: ProjectedView) -> float:
        if object_center_xy is None:
            return float("inf")
        return float(
            np.linalg.norm(
                np.asarray(projected_view.camera_xy, dtype=np.float64)
                - np.asarray(object_center_xy, dtype=np.float64)
            )
        )

    def _bbox_horizontal_side(projected_view: ProjectedView) -> str:
        bbox = np.asarray(projected_view.projected_bbox_2d, dtype=np.float32).reshape(4)
        frame_width = max(float(projected_view.image_size[0]), 1.0)
        center_x = float((bbox[0] + bbox[2]) / 2.0)
        normalized_x = center_x / frame_width
        if normalized_x <= 0.25:
            return "left"
        if normalized_x >= 0.75:
            return "right"
        return "center"

    sorted_views = sorted(
        projected_views, key=lambda item: item.bbox_area, reverse=True
    )
    selected: list[ProjectedView] = []
    selected_angles: list[float] = [float(angle) for angle in (existing_angles or [])]
    used_view_ids: set[str] = set(existing_view_ids or set())
    selected_horizontal_sides: set[str] = set()
    for item in sorted_views:
        if len(selected) >= int(num_views):
            break
        if item.view_id in used_view_ids:
            continue
        if object_center_xy is not None:
            item_angle = view_angle_relative_to_object(item.camera_xy, object_center_xy)
            min_angle = min(
                (
                    float(
                        abs(
                            np.arctan2(
                                np.sin(item_angle - existing_angle),
                                np.cos(item_angle - existing_angle),
                            )
                        )
                    )
                    for existing_angle in selected_angles
                ),
                default=float("inf"),
            )
            angle_conflict = min_angle < angle_threshold_rad
        else:
            item_angle = None
            min_angle = float("nan")
            angle_conflict = False
        if angle_conflict:
            continue
        selected.append(
            ProjectedView(
                view_id=item.view_id,
                image_file=item.image_file,
                projected_bbox_2d=np.asarray(item.projected_bbox_2d, dtype=np.float32),
                bbox_area=float(item.bbox_area),
                camera_xy=np.asarray(item.camera_xy, dtype=np.float64),
                image_size=(int(item.image_size[0]), int(item.image_size[1])),
                selection_reason="distinct_angle",
            )
        )
        if item_angle is not None:
            selected_angles.append(item_angle)
        used_view_ids.add(item.view_id)
        selected_horizontal_sides.add(_bbox_horizontal_side(item))

    if len(selected) < int(num_views):
        remaining_items = [
            item for item in sorted_views if item.view_id not in used_view_ids
        ]
        while len(selected) < int(num_views) and remaining_items:
            gated_remaining_items = remaining_items
            if max_fallback_distance is not None and object_center_xy is not None:
                gated_remaining_items = [
                    item
                    for item in remaining_items
                    if _target_distance(item) <= float(max_fallback_distance)
                ]
            ranking_items = gated_remaining_items or remaining_items
            fallback_sorted_views = sorted(
                ranking_items,
                key=lambda item: (
                    _bbox_horizontal_side(item) in selected_horizontal_sides,
                    _bbox_side_score(item),
                    -float(item.bbox_area),
                ),
            )
            item = fallback_sorted_views[0]
            remaining_items = [
                candidate_item
                for candidate_item in remaining_items
                if candidate_item.view_id != item.view_id
            ]
            if len(selected) >= int(num_views):
                break
            horizontal_side = _bbox_horizontal_side(item)
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
                    selection_reason="side_fallback",
                )
            )
            used_view_ids.add(item.view_id)
            selected_horizontal_sides.add(horizontal_side)

    return selected


def complete_candidate_with_more_views(
    *,
    agent: Any,
    candidate: Any,
    num_bootstrap_views: int = 3,
    max_fallback_distance: float = 1.5,
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
    for object_view in getattr(candidate, "object_view", []) or []:
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

    projected_views = reproject_candidate_to_scene_views(
        frame_ids=list(frame_ids),
        build_view_fn=build_view_fn,
        intrinsic_matrix=np.asarray(agent.intrinsic_matrix, dtype=np.float64),
        world_to_axis_align_matrix=axis_align_matrix,
        points_3d=np.asarray(points_3d, dtype=np.float64),
    )

    bootstrap_views = select_distinct_position_views(
        projected_views,
        num_views=int(num_bootstrap_views),
        object_center_xy=object_center_xy,
        existing_view_ids=existing_view_ids,
        existing_angles=existing_angles,
        max_fallback_distance=float(max_fallback_distance),
    )

    bootstrap_object_views: list[Any] = []
    for projected_view in bootstrap_views:
        view = build_view_fn(projected_view.view_id)
        projected_bbox = np.asarray(
            projected_view.projected_bbox_2d, dtype=np.float32
        ).reshape(4)
        detections = agent.detect_target_objects(view)
        chosen_detection, _ = choose_detection_for_projected_bbox(
            detections, projected_bbox
        )
        refined_bbox = (
            bbox_to_array(chosen_detection.bbox)
            if chosen_detection is not None
            else projected_bbox
        )
        mask = agent.segmenter.segment_from_box(view.rgb, refined_bbox.tolist())
        final_bbox = bbox_from_mask(mask)
        if final_bbox is None:
            final_bbox = refined_bbox
        bootstrap_object_views.append(
            object_view_type(
                object_id=f"bootstrap_{projected_view.view_id}",
                label=str(candidate.label),
                score=1.0,
                view=view,
                bbox_2d=final_bbox,
                mask_2d=np.asarray(mask, dtype=np.uint8),
                points_3d=None,
                status="active",
                source="projected_bootstrap",
            )
        )

    for object_view in bootstrap_object_views:
        candidate.add_object_view(object_view)
    candidate.status = "expanded"
    return candidate
