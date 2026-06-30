from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from agent import Agent
    from agent_schema import CandidateObject, ObjectView
    from benchmark.utils import calc_iou, load_pc
    from module.detector import draw_bbox
    from module.projection import TwoDToThreeDTool
    from module.segmenter import SAMSegmenter
    from read import Read
except ImportError:
    from .agent import Agent
    from .agent_schema import CandidateObject, ObjectView
    from .benchmark.utils import calc_iou, load_pc
    from .module.detector import draw_bbox
    from .module.projection import TwoDToThreeDTool
    from .module.segmenter import SAMSegmenter
    from .read import Read


def _bbox_to_array(bbox: Any) -> np.ndarray:
    return np.asarray(bbox, dtype=np.float32).reshape(4)


def _bbox_to_list(bbox: Any) -> list[float]:
    return [round(float(value), 2) for value in _bbox_to_array(bbox).tolist()]


def _bbox_from_projected_points(
    uv: np.ndarray, image_shape: tuple[int, int, int]
) -> np.ndarray | None:
    if uv.shape[0] == 0:
        return None
    height, width = image_shape[:2]
    x = uv[:, 0]
    y = uv[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if np.count_nonzero(inside) < 20:
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


def _project_points_to_view(
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


def _build_candidate_from_json(
    candidate_data: dict[str, Any], reader: Read
) -> CandidateObject:
    candidate = CandidateObject(
        object_id=int(candidate_data["candidate_id"]),
        label=str(candidate_data.get("label", "object")),
        best_id=int(candidate_data.get("best_id", 0)),
    )
    view_lookup = {
        frame_id: reader._build_view(frame_id) for frame_id in reader.frame_ids
    }
    for object_view_data in candidate_data.get("object_views", []):
        view_id = str(object_view_data["view_id"])
        view = view_lookup[view_id]
        object_view = ObjectView(
            object_id=str(object_view_data["object_view_id"]),
            label=str(object_view_data["label"]),
            score=float(object_view_data["score"]),
            view=view,
            bbox_2d=_bbox_to_array(object_view_data["bbox_2d"]),
            mask_2d=None,
            points_3d=None,
            status=str(object_view_data.get("status", "active")),
        )
        candidate.add_object_view(object_view)
    candidate.best_id = int(candidate_data.get("best_id", candidate.best_id))
    return candidate


def _score_projected_view(view: Any, bbox: np.ndarray) -> dict[str, float]:
    height, width = view.rgb.shape[:2]
    x1, y1, x2, y2 = bbox.tolist()
    bbox_width = max(0.0, float(x2 - x1))
    bbox_height = max(0.0, float(y2 - y1))
    area = bbox_width * bbox_height
    area_ratio = area / max(float(width * height), 1.0)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    normalized_center_distance = np.sqrt(
        ((center_x - width / 2.0) / max(width / 2.0, 1.0)) ** 2
        + ((center_y - height / 2.0) / max(height / 2.0, 1.0)) ** 2
    )
    short_side = max(min(bbox_width, bbox_height), 1.0)
    aspect_ratio = max(bbox_width, bbox_height) / short_side
    aspect_score = 1.0 / max(aspect_ratio, 1.0)
    return {
        "area_ratio": float(area_ratio),
        "center_distance": float(normalized_center_distance),
        "aspect_ratio": float(aspect_ratio),
        "aspect_score": float(aspect_score),
        "selection_score": float(
            area_ratio * 3.0 + aspect_score - normalized_center_distance
        ),
    }


def _bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = _bbox_to_array(bbox_a).tolist()
    bx1, by1, bx2, by2 = _bbox_to_array(bbox_b).tolist()
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def _expand_bbox(
    bbox: np.ndarray, image_shape: tuple[int, int, int], expand_ratio: float
) -> np.ndarray:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox.tolist()
    bbox_width = max(1.0, float(x2 - x1))
    bbox_height = max(1.0, float(y2 - y1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = bbox_width * (1.0 + expand_ratio) / 2.0
    half_h = bbox_height * (1.0 + expand_ratio) / 2.0
    expanded = np.asarray(
        [
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(width - 1.0, cx + half_w),
            min(height - 1.0, cy + half_h),
        ],
        dtype=np.float32,
    )
    return expanded


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
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


def _bbox_center_distance(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = _bbox_to_array(bbox_a).tolist()
    bx1, by1, bx2, by2 = _bbox_to_array(bbox_b).tolist()
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return float(np.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2))


def _choose_detection_for_projected_bbox(
    detections: list[Any],
    projected_bbox: np.ndarray,
) -> tuple[Any | None, dict[str, Any]]:
    if not detections:
        return None, {"found": False, "reason": "no_detections"}

    best_detection = None
    best_metrics = None
    best_score = None
    for detection in detections:
        detection_bbox = _bbox_to_array(detection.bbox)
        iou = _bbox_iou(projected_bbox, detection_bbox)
        center_distance = _bbox_center_distance(projected_bbox, detection_bbox)
        metrics = {
            "found": True,
            "iou_with_projected": round(float(iou), 4),
            "center_distance": round(float(center_distance), 2),
            "detection_bbox_2d": _bbox_to_list(detection_bbox),
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


def _build_scored_projected_views(
    projected_views_data: list[dict[str, Any]], reader: Read
) -> list[dict[str, Any]]:
    candidates = []
    for item in projected_views_data:
        view = reader._build_view(str(item["view_id"]))
        bbox = _bbox_to_array(item["projected_bbox_2d"])
        scores = _score_projected_view(view, bbox)
        candidates.append(
            {
                "view_id": str(item["view_id"]),
                "image_file": item["image_file"],
                "projected_bbox_2d": _bbox_to_list(bbox),
                "bbox_area": round(float(_bbox_area(bbox)), 2),
                "scores": scores,
            }
        )

    candidates.sort(key=lambda item: item["scores"]["selection_score"], reverse=True)
    return candidates


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = _bbox_to_array(bbox).tolist()
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _camera_xy(view: Any) -> np.ndarray:
    camera_to_world = np.asarray(view.camera_to_world, dtype=np.float64)
    return np.asarray(camera_to_world[:2, 3], dtype=np.float64)


def _select_distinct_position_views(
    reprojected_views: list[dict[str, Any]],
    reader: Read,
    num_views: int,
    position_threshold: float,
) -> list[dict[str, Any]]:
    sorted_views = sorted(
        reprojected_views,
        key=lambda item: item["bbox_area"],
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_positions: list[np.ndarray] = []
    used_view_ids: set[str] = set()

    for item in sorted_views:
        if len(selected) >= num_views:
            break
        view = reader._build_view(str(item["view_id"]))
        xy = _camera_xy(view)
        if any(
            np.linalg.norm(xy - existing_xy) < position_threshold
            for existing_xy in selected_positions
        ):
            continue
        enriched = dict(item)
        enriched["camera_xy"] = [round(float(xy[0]), 4), round(float(xy[1]), 4)]
        enriched["selection_reason"] = "distinct_camera_xy"
        selected.append(enriched)
        selected_positions.append(xy)
        used_view_ids.add(str(item["view_id"]))

    if len(selected) < num_views:
        for item in sorted_views:
            if len(selected) >= num_views:
                break
            if str(item["view_id"]) in used_view_ids:
                continue
            view = reader._build_view(str(item["view_id"]))
            xy = _camera_xy(view)
            enriched = dict(item)
            enriched["camera_xy"] = [round(float(xy[0]), 4), round(float(xy[1]), 4)]
            enriched["selection_reason"] = "bbox_area_fallback"
            selected.append(enriched)
            used_view_ids.add(str(item["view_id"]))

    return selected


def _make_object_view(
    label: str,
    view: Any,
    bbox: np.ndarray,
    mask: np.ndarray,
) -> ObjectView:
    return ObjectView(
        object_id=f"projected_{view.view_id}",
        label=str(label),
        score=1.0,
        view=view,
        bbox_2d=np.asarray(bbox, dtype=np.float32),
        mask_2d=np.asarray(mask, dtype=np.uint8),
        points_3d=None,
        status="active",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study projected-view selection, bbox refinement, and reprojection completion."
    )
    parser.add_argument(
        "--candidate-json",
        required=True,
        type=Path,
        help="Path to original candidate JSON",
    )
    parser.add_argument(
        "--projected-json",
        required=True,
        type=Path,
        help="Path to candidate projected views JSON, e.g. candidate_000_projected_views.json",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="checkpoints/SAM/sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file",
    )
    parser.add_argument("--sam-model-type", default="vit_h", help="SAM model type")
    parser.add_argument(
        "--sam-device", default="cpu", help="Device for SAM inference, e.g. cpu or cuda"
    )
    parser.add_argument(
        "--bbox-expand-ratio",
        type=float,
        default=0.15,
        help="Expand projected bbox by this ratio before mask refinement",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=5,
        help="Number of refined projected views to keep for multi-view projection",
    )
    parser.add_argument(
        "--position-threshold",
        type=float,
        default=0.3,
        help="Minimum camera XY distance to treat reprojected views as distinct positions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "candidate_projection_completion",
        help="Directory to save study outputs",
    )
    parser.add_argument(
        "--visualize-3d",
        action="store_true",
        help="Visualize original and refined 3D points with AABB using Open3D.",
    )
    args = parser.parse_args()

    candidate_data = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    projected_data = json.loads(args.projected_json.read_text(encoding="utf-8"))
    scene = str(candidate_data["scene"])

    reader = Read(scene, max_frames_per_find=999999, frame_skip=1)
    candidate = _build_candidate_from_json(candidate_data, reader)

    segmenter = SAMSegmenter(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.sam_device,
    )
    agent = Agent(
        motion=reader,
        segmenter=segmenter,
        mapper_2d3d=TwoDToThreeDTool(),
        intrinsic_matrix=reader.intrinsic_matrix,
        world_to_axis_align_matrix=reader.world_to_axis_align_matrix,
        debug=True,
    )
    agent.reset(str(candidate_data.get("query", candidate.label)))

    agent.complete_candidate_masks(candidate)
    original_points_3d, original_bbox_3d = agent.map_candidate_to_3d(candidate)

    bootstrap_views = _select_distinct_position_views(
        _build_scored_projected_views(
            projected_data.get("projected_views", []), reader
        ),
        reader,
        3,
        float(args.position_threshold),
    )
    output_dir = args.output_dir / scene / f"candidate_{int(candidate.object_id):03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_image_dir = output_dir / "bootstrap_selected_views"
    bootstrap_image_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_object_views = []
    bootstrap_view_records = []
    for index, item in enumerate(bootstrap_views):
        view = reader._build_view(str(item["view_id"]))
        projected_bbox = _bbox_to_array(item["projected_bbox_2d"])
        detections = agent.detect_target_objects(view)
        chosen_detection, detection_result = _choose_detection_for_projected_bbox(
            detections,
            projected_bbox,
        )
        if chosen_detection is not None:
            refined_bbox = _bbox_to_array(chosen_detection.bbox)
            refinement_source = "detector"
        else:
            refined_bbox = _expand_bbox(
                projected_bbox, view.rgb.shape, args.bbox_expand_ratio
            )
            refinement_source = "projected_expand_fallback"
        mask = segmenter.segment_from_box(view.rgb, refined_bbox.tolist())
        mask_bbox = _bbox_from_mask(mask)
        if mask_bbox is not None:
            final_bbox = mask_bbox
            final_bbox_source = "mask"
        else:
            final_bbox = refined_bbox
            final_bbox_source = "refined_bbox"
        projected_vis = draw_bbox(
            view.rgb,
            projected_bbox,
            f"projected rank {index + 1}",
            color=(255, 200, 0),
        )
        refined_vis = draw_bbox(
            projected_vis,
            final_bbox,
            f"refined rank {index + 1}",
            color=(0, 255, 0),
        )
        selected_vis_file = f"{view.view_id}.png"
        cv2.imwrite(
            str(bootstrap_image_dir / selected_vis_file),
            cv2.cvtColor(refined_vis, cv2.COLOR_RGB2BGR),
        )
        bootstrap_object_views.append(
            _make_object_view(
                candidate.label, view, final_bbox, np.asarray(mask, dtype=np.uint8)
            )
        )
        bootstrap_view_records.append(
            {
                "rank": index + 1,
                "view_id": str(view.view_id),
                "image_file": f"{view.view_id}.jpg",
                "projected_bbox_2d": _bbox_to_list(projected_bbox),
                "refined_bbox_2d": _bbox_to_list(refined_bbox),
                "final_bbox_2d": _bbox_to_list(final_bbox),
                "final_bbox_area": round(float(_bbox_area(final_bbox)), 2),
                "refinement_source": refinement_source,
                "final_bbox_source": final_bbox_source,
                "detection_result": detection_result,
                "num_detections": len(detections),
                "scores": item["scores"],
                "visualization_file": selected_vis_file,
            }
        )

    bootstrap_candidate = CandidateObject(
        object_id=int(candidate.object_id),
        label=str(candidate.label),
        best_id=0,
        object_view=bootstrap_object_views,
    )
    bootstrap_points_3d, bootstrap_bbox_3d = agent.map_candidate_to_3d(
        bootstrap_candidate
    )

    bootstrap_reprojected_views = []
    bootstrap_reprojected_image_dir = output_dir / "bootstrap_reprojected_views"
    bootstrap_reprojected_image_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in reader.frame_ids:
        view = reader._build_view(frame_id)
        uv = _project_points_to_view(
            bootstrap_points_3d[:, :3],
            reader.intrinsic_matrix,
            np.asarray(view.camera_to_world, dtype=np.float64),
            None
            if reader.world_to_axis_align_matrix is None
            else np.asarray(reader.world_to_axis_align_matrix, dtype=np.float64),
        )
        bbox = _bbox_from_projected_points(uv, view.rgb.shape)
        if bbox is None:
            continue
        vis = draw_bbox(
            view.rgb,
            bbox,
            f"bootstrap candidate {int(candidate.object_id)}",
            color=(0, 255, 0),
        )
        vis_file = f"{view.view_id}.png"
        cv2.imwrite(
            str(bootstrap_reprojected_image_dir / vis_file),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )
        bootstrap_reprojected_views.append(
            {
                "view_id": str(view.view_id),
                "image_file": f"{view.view_id}.jpg",
                "projected_bbox_2d": _bbox_to_list(bbox),
                "bbox_area": round(float(_bbox_area(bbox)), 2),
                "visualization_file": vis_file,
            }
        )

    final_mask_input_views = _select_distinct_position_views(
        bootstrap_reprojected_views,
        reader,
        max(1, int(args.num_views)),
        float(args.position_threshold),
    )
    final_mask_image_dir = output_dir / "final_mask_views"
    final_mask_image_dir.mkdir(parents=True, exist_ok=True)
    final_object_views = []
    final_view_records = []
    for index, item in enumerate(final_mask_input_views, start=1):
        view = reader._build_view(str(item["view_id"]))
        projected_bbox = _bbox_to_array(item["projected_bbox_2d"])
        mask = segmenter.segment_from_box(view.rgb, projected_bbox.tolist())
        final_bbox = _bbox_from_mask(mask)
        if final_bbox is None:
            final_bbox = projected_bbox
            final_bbox_source = "projected_bbox"
        else:
            final_bbox_source = "mask"
        vis = draw_bbox(
            view.rgb,
            projected_bbox,
            f"top5 projected rank {index}",
            color=(255, 200, 0),
        )
        vis = draw_bbox(
            vis,
            final_bbox,
            f"final mask rank {index}",
            color=(0, 255, 0),
        )
        vis_file = f"{view.view_id}.png"
        cv2.imwrite(
            str(final_mask_image_dir / vis_file),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )
        final_object_views.append(
            _make_object_view(
                candidate.label, view, final_bbox, np.asarray(mask, dtype=np.uint8)
            )
        )
        final_view_records.append(
            {
                "rank": index,
                "view_id": str(view.view_id),
                "image_file": f"{view.view_id}.jpg",
                "projected_bbox_2d": _bbox_to_list(projected_bbox),
                "final_bbox_2d": _bbox_to_list(final_bbox),
                "bbox_area": round(float(_bbox_area(projected_bbox)), 2),
                "camera_xy": item["camera_xy"],
                "selection_reason": item["selection_reason"],
                "final_bbox_source": final_bbox_source,
                "visualization_file": vis_file,
            }
        )

    final_candidate = CandidateObject(
        object_id=int(candidate.object_id),
        label=str(candidate.label),
        best_id=0,
        object_view=final_object_views,
    )
    final_points_3d, final_bbox_3d = agent.map_candidate_to_3d(final_candidate)

    gt_bbox_3d = None
    original_iou = None
    bootstrap_iou = None
    final_iou = None
    try:
        obj_ids, obj_labels, obj_locs = load_pc(scene)
        target_id = int(candidate_data.get("target_id", -1))
        if target_id in obj_ids:
            target_index = obj_ids.index(target_id)
            gt_bbox_3d = np.asarray(obj_locs[target_index], dtype=np.float64)
            original_iou = float(
                calc_iou(
                    np.asarray(original_bbox_3d, dtype=np.float64),
                    gt_bbox_3d,
                )
            )
            bootstrap_iou = float(
                calc_iou(
                    np.asarray(bootstrap_bbox_3d, dtype=np.float64),
                    gt_bbox_3d,
                )
            )
            final_iou = float(
                calc_iou(
                    np.asarray(final_bbox_3d, dtype=np.float64),
                    gt_bbox_3d,
                )
            )
    except Exception:
        gt_bbox_3d = None
        original_iou = None
        bootstrap_iou = None
        final_iou = None

    payload = {
        "scene": scene,
        "query": candidate_data.get("query", ""),
        "candidate_id": int(candidate.object_id),
        "source_candidate_json": str(args.candidate_json),
        "source_projected_json": str(args.projected_json),
        "bootstrap_used_num_views": len(bootstrap_view_records),
        "final_used_num_views": len(final_view_records),
        "selection_goal": ["bootstrap_complete_3d", "largest_reprojected_bboxes"],
        "selection_heuristics": [
            "select 3 projected views by distinct camera xy with bbox fallback",
            "refine them with detection and mask to bootstrap a more complete 3D",
            "reproject bootstrap 3D to all views and keep top-5 largest projected bboxes",
            "mask top-5 projected bboxes and run final multi-view projection",
        ],
        "bootstrap_selected_views": bootstrap_view_records,
        "bootstrap_reprojected_views": bootstrap_reprojected_views,
        "final_mask_views": final_view_records,
        "original_bbox_3d": [
            round(float(value), 4)
            for value in np.asarray(original_bbox_3d).reshape(-1).tolist()
        ],
        "bootstrap_bbox_3d": [
            round(float(value), 4)
            for value in np.asarray(bootstrap_bbox_3d).reshape(-1).tolist()
        ],
        "final_bbox_3d": [
            round(float(value), 4)
            for value in np.asarray(final_bbox_3d).reshape(-1).tolist()
        ],
        "gt_bbox_3d": None
        if gt_bbox_3d is None
        else [
            round(float(value), 4)
            for value in np.asarray(gt_bbox_3d).reshape(-1).tolist()
        ],
        "original_iou": None if original_iou is None else round(float(original_iou), 4),
        "bootstrap_iou": None
        if bootstrap_iou is None
        else round(float(bootstrap_iou), 4),
        "final_iou": None if final_iou is None else round(float(final_iou), 4),
    }
    output_path = output_dir / "study_result.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.visualize_3d:
        print("[3D] visualize original candidate points and bbox")
        TwoDToThreeDTool.visualize_points_and_aabb(original_points_3d, original_bbox_3d)
        print("[3D] visualize bootstrap candidate points and bbox")
        TwoDToThreeDTool.visualize_points_and_aabb(
            bootstrap_points_3d, bootstrap_bbox_3d
        )
        print("[3D] visualize final candidate points and bbox")
        TwoDToThreeDTool.visualize_points_and_aabb(final_points_3d, final_bbox_3d)

    print(f"scene={scene}")
    print(f"candidate_id={int(candidate.object_id)}")
    print(f"bootstrap_selected_views={len(bootstrap_view_records)}")
    print(f"bootstrap_reprojected_views={len(bootstrap_reprojected_views)}")
    print(f"final_mask_views={len(final_view_records)}")
    print(f"output_dir={output_dir}")
    print(f"bootstrap_image_dir={bootstrap_image_dir}")
    print(f"bootstrap_reprojected_image_dir={bootstrap_reprojected_image_dir}")
    print(f"final_mask_image_dir={final_mask_image_dir}")
    print(f"output_path={output_path}")
    if original_iou is not None and bootstrap_iou is not None and final_iou is not None:
        print(f"original_iou={original_iou:.4f}")
        print(f"bootstrap_iou={bootstrap_iou:.4f}")
        print(f"final_iou={final_iou:.4f}")


if __name__ == "__main__":
    main()
