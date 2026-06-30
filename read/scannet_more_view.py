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
) -> tuple[Any | None, dict[str, Any]]:
    if not detections:
        return None, {"found": False, "reason": "no_detections"}

    best_detection = None
    best_metrics = None
    best_score = None
    for detection in detections:
        detection_bbox = bbox_to_array(detection.bbox)
        iou = bbox_iou(projected_bbox, detection_bbox)
        center_distance = bbox_center_distance(projected_bbox, detection_bbox)
        metrics = {
            "found": True,
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
        return None, {"found": False, "reason": "no_valid_detection"}
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


def camera_xy(view: Any) -> np.ndarray:
    camera_to_world = np.asarray(view.camera_to_world, dtype=np.float64)
    return np.asarray(camera_to_world[:2, 3], dtype=np.float64)


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
                camera_xy=camera_xy(view),
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
    position_threshold: float,
    object_center_xy: np.ndarray | None = None,
    angle_threshold_rad: float = np.pi / 6.0,
) -> list[ProjectedView]:
    sorted_views = sorted(
        projected_views, key=lambda item: item.bbox_area, reverse=True
    )
    max_bbox_area = max((float(item.bbox_area) for item in sorted_views), default=1.0)
    selected: list[ProjectedView] = []
    selected_positions: list[np.ndarray] = []
    selected_angles: list[float] = []
    used_view_ids: set[str] = set()

    while len(selected) < int(num_views):
        best_item = None
        best_angle = None
        best_score = None

        for item in sorted_views:
            if item.view_id in used_view_ids:
                continue

            too_close = any(
                np.linalg.norm(item.camera_xy - existing_xy) < position_threshold
                for existing_xy in selected_positions
            )
            if object_center_xy is not None:
                item_angle = view_angle_relative_to_object(
                    item.camera_xy, object_center_xy
                )
                angle_gaps = [
                    abs(
                        np.arctan2(
                            np.sin(item_angle - existing_angle),
                            np.cos(item_angle - existing_angle),
                        )
                    )
                    for existing_angle in selected_angles
                ]
                min_angle_gap = min(angle_gaps) if angle_gaps else np.pi
                angle_conflict = (
                    bool(selected_angles) and min_angle_gap < angle_threshold_rad
                )
            else:
                item_angle = None
                min_angle_gap = 0.0
                angle_conflict = False

            if too_close or angle_conflict:
                continue

            area_score = np.sqrt(float(item.bbox_area) / max(max_bbox_area, 1e-6))
            if object_center_xy is not None:
                angle_score = float(min_angle_gap / np.pi)
            else:
                angle_score = 0.0

            candidate_score = 0.3 * area_score + 0.7 * angle_score

            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_item = item
                best_angle = item_angle

        if best_item is None:
            break

        selected.append(
            ProjectedView(
                view_id=best_item.view_id,
                image_file=best_item.image_file,
                projected_bbox_2d=np.asarray(
                    best_item.projected_bbox_2d, dtype=np.float32
                ),
                bbox_area=float(best_item.bbox_area),
                camera_xy=np.asarray(best_item.camera_xy, dtype=np.float64),
                selection_reason="distinct_camera_xy_angle_scored",
            )
        )
        selected_positions.append(best_item.camera_xy)
        if best_angle is not None:
            selected_angles.append(best_angle)
        used_view_ids.add(best_item.view_id)

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
                    selection_reason="bbox_area_fallback",
                )
            )
            used_view_ids.add(item.view_id)

    return selected


def complete_candidate_with_more_views(
    *,
    agent: Any,
    candidate: Any,
    num_bootstrap_views: int = 3,
    num_final_views: int = 5,
    position_threshold: float = 0.3,
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

    projected_views = reproject_candidate_to_scene_views(
        frame_ids=list(frame_ids),
        build_view_fn=build_view_fn,
        intrinsic_matrix=np.asarray(agent.intrinsic_matrix, dtype=np.float64),
        world_to_axis_align_matrix=None
        if agent.world_to_axis_align_matrix is None
        else np.asarray(agent.world_to_axis_align_matrix, dtype=np.float64),
        points_3d=np.asarray(points_3d, dtype=np.float64),
    )
    bootstrap_views = select_distinct_position_views(
        projected_views,
        num_views=int(num_bootstrap_views),
        position_threshold=float(position_threshold),
        object_center_xy=object_center_xy,
    )

    bootstrap_object_views = []
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
            type(candidate.object_view[0])(
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

    agent.complete_candidate_masks(candidate)
    points_3d, bbox_3d = agent.map_candidate_to_3d(candidate)
    candidate.points_3d = points_3d
    candidate.bbox_3d = bbox_3d
    object_center_xy = np.asarray(bbox_3d, dtype=np.float64).reshape(-1)[:2]

    projected_views = reproject_candidate_to_scene_views(
        frame_ids=list(frame_ids),
        build_view_fn=build_view_fn,
        intrinsic_matrix=np.asarray(agent.intrinsic_matrix, dtype=np.float64),
        world_to_axis_align_matrix=None
        if agent.world_to_axis_align_matrix is None
        else np.asarray(agent.world_to_axis_align_matrix, dtype=np.float64),
        points_3d=np.asarray(points_3d, dtype=np.float64),
    )
    final_views = select_distinct_position_views(
        projected_views,
        num_views=int(num_final_views),
        position_threshold=float(position_threshold),
        object_center_xy=object_center_xy,
    )
    existing_view_ids = {
        str(object_view.view.view_id) for object_view in candidate.object_view
    }
    for projected_view in final_views:
        if projected_view.view_id in existing_view_ids:
            continue
        view = build_view_fn(projected_view.view_id)
        bbox = np.asarray(projected_view.projected_bbox_2d, dtype=np.float32).reshape(4)
        mask = agent.segmenter.segment_from_box(view.rgb, bbox.tolist())
        final_bbox = bbox_from_mask(mask)
        if final_bbox is None:
            final_bbox = bbox
        candidate.add_object_view(
            type(candidate.object_view[0])(
                object_id=f"projected_{projected_view.view_id}",
                label=str(candidate.label),
                score=1.0,
                view=view,
                bbox_2d=final_bbox,
                mask_2d=np.asarray(mask, dtype=np.uint8),
                points_3d=None,
                status="active",
                source="projected_final",
            )
        )

    agent.complete_candidate_masks(candidate)
    points_3d, bbox_3d = agent.map_candidate_to_3d(candidate)
    candidate.points_3d = points_3d
    candidate.bbox_3d = bbox_3d
    candidate.status = "expanded"
    return candidate
