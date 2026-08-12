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
    from module.detector_yoloe import YOLOEDetector
    from module.detector_yoloe import draw_bbox
    from module.projection import PointFilterConfig, TwoDToThreeDTool
    from module.segmenter import SAMSegmenter
    from read import Read
    from read.scannet_more_view import (
        _depth_visibility_stats,
        bbox_from_projected_points,
    )
except ImportError:
    from .agent import Agent
    from .agent_schema import CandidateObject, ObjectView
    from .module.detector_yoloe import YOLOEDetector
    from .module.detector_yoloe import draw_bbox
    from .module.projection import PointFilterConfig, TwoDToThreeDTool
    from .module.segmenter import SAMSegmenter
    from .read import Read
    from .read.scannet_more_view import (
        _depth_visibility_stats,
        bbox_from_projected_points,
    )


def _bbox_to_list(bbox: Any) -> list[float]:
    array = np.asarray(bbox, dtype=np.float32).reshape(-1)
    return [round(float(value), 2) for value in array.tolist()]


def _stats(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "N/A"
    return f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"


def _print_depth_samples(
    *,
    view_id: str,
    label: str,
    uv: np.ndarray,
    projected_depths: np.ndarray,
    sampled_depths: np.ndarray,
    mask: np.ndarray,
    limit: int,
) -> None:
    indices = np.where(mask)[0][: max(0, int(limit))]
    if indices.size == 0:
        print(f"[depth_samples] view={view_id} {label}: none")
        return
    print(f"[depth_samples] view={view_id} {label}")
    for index in indices:
        point = uv[index]
        px = int(round(point[0]))
        py = int(round(point[1]))
        z_proj = (
            float(projected_depths[index])
            if index < len(projected_depths)
            else float("nan")
        )
        z_depth = (
            float(sampled_depths[index])
            if index < len(sampled_depths)
            else float("nan")
        )
        diff = (
            z_proj - z_depth
            if np.isfinite(z_proj) and np.isfinite(z_depth)
            else float("nan")
        )
        print(
            f"  pixel=({px}, {py}) uv=({point[0]:.1f}, {point[1]:.1f}) "
            f"z_proj={z_proj:.4f} z_depth={z_depth:.4f} diff={diff:.4f}"
        )


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
        object_id=int(candidate_data.get("candidate_id", 0)),
        label=str(candidate_data.get("label", "object")),
        best_id=0,
    )
    view_lookup = {
        frame_id: reader._build_view(frame_id) for frame_id in reader.frame_ids
    }
    for object_view_data in candidate_data["object_views"]:
        view_id = str(object_view_data["view_id"])
        view = view_lookup[view_id]
        object_view = ObjectView(
            object_id=str(
                object_view_data.get(
                    "object_view_id",
                    f"{view_id}_{int(object_view_data.get('index', 0))}",
                )
            ),
            label=str(object_view_data.get("label", candidate.label)),
            score=float(object_view_data.get("score", 1.0)),
            view=view,
            bbox_2d=np.asarray(object_view_data["bbox_2d"], dtype=np.float32),
            mask_2d=None,
            points_3d=None,
            status=str(object_view_data.get("status", "active")),
        )
        candidate.add_object_view(object_view)
    if candidate.object_view:
        candidate.best_id = max(
            0,
            min(
                int(candidate_data.get("best_id", candidate.best_id)),
                len(candidate.object_view) - 1,
            ),
        )
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project one exported candidate into other unseen views in the same scene."
    )
    parser.add_argument(
        "--candidate-json",
        required=True,
        type=Path,
        help="Path to one candidate JSON file",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Optional scene override. If omitted, read from candidate JSON.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="checkpoints/SAM/sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file",
    )
    parser.add_argument(
        "--detector-model",
        default="yoloe-11s-seg.pt",
        help="YOLOE checkpoint name or path",
    )
    parser.add_argument("--sam-model-type", default="vit_h", help="SAM model type")
    parser.add_argument(
        "--sam-device", default="cpu", help="Device for SAM inference, e.g. cpu or cuda"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output json path. Default: alongside candidate json.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save visualization images with projected bbox overlays.",
    )
    parser.add_argument(
        "--debug-depth",
        action="store_true",
        help="Print projected/depth visibility statistics for each projected view.",
    )
    parser.add_argument(
        "--debug-depth-samples",
        type=int,
        default=0,
        help="Print up to N sample point-to-depth comparisons per category.",
    )
    parser.add_argument(
        "--debug-rejects",
        action="store_true",
        help="Print why a projected view was rejected before bbox output.",
    )
    parser.add_argument(
        "--visualize-3d",
        action="store_true",
        help="Visualize candidate 3D points and bbox using Open3D.",
    )
    parser.add_argument(
        "--visualize-raw-3d",
        action="store_true",
        help="Visualize raw projected 3D points before filtering.",
    )
    args = parser.parse_args()

    candidate_data = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    scene_value = (
        args.scene or candidate_data.get("scene") or candidate_data.get("scan_id")
    )
    if not scene_value:
        raise ValueError(
            "Scene must be provided via --scene or candidate JSON scene/scan_id"
        )
    scene = str(scene_value)
    reader = Read(scene, max_frames_per_find=999999, frame_skip=1)
    candidate = _build_candidate_from_json(candidate_data, reader)
    detector = YOLOEDetector(model=args.detector_model)

    segmenter = SAMSegmenter(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.sam_device,
    )
    agent = Agent(
        motion=reader,
        detector=detector,
        segmenter=segmenter,
        mapper_2d3d=TwoDToThreeDTool(point_filter=PointFilterConfig()),
        intrinsic_matrix=reader.intrinsic_matrix,
        world_to_axis_align_matrix=reader.world_to_axis_align_matrix,
        debug=True,
    )
    if candidate_data.get("query"):
        agent.reset(str(candidate_data.get("query", "")))

    agent.complete_candidate_masks(candidate)
    if args.visualize_raw_3d:
        projection_inputs = agent.mapper_2d3d.build_projection_inputs_from_candidate(
            candidate,
            intrinsic_matrix=reader.intrinsic_matrix,
            world_to_axis_align_matrix=reader.world_to_axis_align_matrix,
        )
        raw_points = agent.mapper_2d3d.project_views_to_3d(projection_inputs)
        raw_bbox = agent.mapper_2d3d.calculate_aabb(raw_points)
        try:
            TwoDToThreeDTool.visualize_points_and_aabb(raw_points, raw_bbox)
        except ImportError as exc:
            print(f"raw_visualization_skipped={exc}")
    points_3d, bbox_3d = agent.map_candidate_to_3d(candidate)
    if args.visualize_3d:
        try:
            TwoDToThreeDTool.visualize_points_and_aabb(points_3d, bbox_3d)
        except ImportError as exc:
            print(f"visualization_skipped={exc}")

    existing_view_ids = {
        str(object_view.view.view_id) for object_view in candidate.object_view
    }
    projected_views = []
    for frame_id in reader.frame_ids:
        if frame_id in existing_view_ids:
            continue
        view = reader._build_view(frame_id)
        (
            uv,
            visible_mask,
            occluded_mask,
            depth_missing_mask,
            background_mismatch_mask,
            visibility_stats,
        ) = _depth_visibility_stats(
            points_3d[:, :3],
            reader.intrinsic_matrix,
            np.asarray(view.camera_to_world, dtype=np.float64),
            None
            if reader.world_to_axis_align_matrix is None
            else np.asarray(reader.world_to_axis_align_matrix, dtype=np.float64),
            tuple(view.rgb.shape),
            np.asarray(view.depth, dtype=np.float64),
            0.001,
            0.05,
        )
        visible_uv = uv[visible_mask]
        in_frame_points = int(visibility_stats["projected_points_in_frame"])
        visible_points = int(visibility_stats["visible_projected_points"])
        visible_ratio = float(visibility_stats["visible_ratio"])
        if in_frame_points == 0:
            if args.debug_rejects:
                print(
                    f"[depth_reject] view={view.view_id} reason=no_projection_in_frame"
                )
            continue
        if visible_points == 0:
            if args.debug_rejects:
                print(
                    f"[depth_reject] view={view.view_id} reason=no_visible_projection "
                    f"in_frame={in_frame_points}"
                )
            continue
        if visible_points < 50:
            if args.debug_rejects:
                print(
                    f"[depth_reject] view={view.view_id} reason=visible_points_too_low "
                    f"in_frame={in_frame_points} visible={visible_points} threshold=50 visible_ratio={visible_ratio:.4f}"
                )
            continue
        if visible_ratio < 0.3:
            if args.debug_rejects:
                print(
                    f"[depth_reject] view={view.view_id} reason=visible_ratio_too_low "
                    f"visible={visible_points} visible_ratio={visible_ratio:.4f} threshold=0.3000"
                )
            continue
        bbox = bbox_from_projected_points(visible_uv, view.rgb.shape)
        if bbox is None:
            if args.debug_rejects:
                print(
                    f"[depth_reject] view={view.view_id} reason=invalid_visible_bbox "
                    f"visible={visible_points} visible_ratio={visible_ratio:.4f}"
                )
            continue
        if args.debug_depth:
            projected_depths = np.asarray(
                visibility_stats["projected_depths"], dtype=np.float64
            )
            sampled_depths = np.asarray(
                visibility_stats["sampled_depths"], dtype=np.float64
            )
            diff = projected_depths - sampled_depths
            print(f"[depth_debug] view={view.view_id}")
            print(
                f"  projected_total={int(visibility_stats['projected_points_total'])} "
                f"in_frame={int(visibility_stats['projected_points_in_frame'])} "
                f"visible={int(visibility_stats['visible_projected_points'])} "
                f"occluded={int(visibility_stats['occluded_projected_points'])} "
                f"missing={int(visibility_stats['depth_missing_points'])} "
                f"background_mismatch={int(visibility_stats['background_mismatch_points'])}"
            )
            print(
                f"  visible_ratio={float(visibility_stats['visible_ratio']):.4f} "
                f"occluded_ratio={float(visibility_stats['occluded_ratio']):.4f} "
                f"background_mismatch_ratio={float(visibility_stats['background_mismatch_ratio']):.4f}"
            )
            print(f"  z_proj[{_stats(projected_depths)}]")
            print(f"  z_depth[{_stats(sampled_depths)}]")
            print(f"  diff=z_proj-z_depth[{_stats(diff)}]")
            print(f"  visible_z_proj[{_stats(projected_depths[visible_mask])}]")
            print(f"  occluded_z_proj[{_stats(projected_depths[occluded_mask])}]")
            print(f"  visible_diff[{_stats(diff[visible_mask])}]")
            print(f"  occluded_diff[{_stats(diff[occluded_mask])}]")
            if int(args.debug_depth_samples) > 0:
                _print_depth_samples(
                    view_id=str(view.view_id),
                    label="visible",
                    uv=uv,
                    projected_depths=projected_depths,
                    sampled_depths=sampled_depths,
                    mask=visible_mask,
                    limit=int(args.debug_depth_samples),
                )
                _print_depth_samples(
                    view_id=str(view.view_id),
                    label="occluded",
                    uv=uv,
                    projected_depths=projected_depths,
                    sampled_depths=sampled_depths,
                    mask=occluded_mask,
                    limit=int(args.debug_depth_samples),
                )
                _print_depth_samples(
                    view_id=str(view.view_id),
                    label="depth_missing",
                    uv=uv,
                    projected_depths=projected_depths,
                    sampled_depths=sampled_depths,
                    mask=depth_missing_mask,
                    limit=int(args.debug_depth_samples),
                )
                _print_depth_samples(
                    view_id=str(view.view_id),
                    label="background_mismatch",
                    uv=uv,
                    projected_depths=projected_depths,
                    sampled_depths=sampled_depths,
                    mask=background_mismatch_mask,
                    limit=int(args.debug_depth_samples),
                )
        reference_detections = []
        if getattr(getattr(agent, "query", None), "reference_object", ""):
            detected_references = agent.detect_reference_objects(view)
            for detection in detected_references:
                reference_detections.append(
                    {
                        "label": str(detection.label),
                        "score": round(float(detection.score), 4),
                        "bbox_2d": _bbox_to_list(detection.bbox),
                    }
                )

        visualization_file = None
        if args.save_images:
            image_output_dir = (
                args.output_path
                if args.output_path is not None
                else args.candidate_json.with_name(
                    args.candidate_json.stem + "_projected_views.json"
                )
            ).with_suffix("")
            image_output_dir.mkdir(parents=True, exist_ok=True)
            vis = draw_bbox(
                view.rgb,
                bbox,
                f"candidate {int(candidate.object_id)} projection",
                color=(0, 255, 0),
            )
            for point in uv[occluded_mask]:
                vis = cv2.circle(
                    vis,
                    (int(round(point[0])), int(round(point[1]))),
                    2,
                    (255, 0, 0),
                    -1,
                )
            for point in uv[depth_missing_mask]:
                vis = cv2.circle(
                    vis,
                    (int(round(point[0])), int(round(point[1]))),
                    2,
                    (255, 255, 0),
                    -1,
                )
            for point in uv[background_mismatch_mask]:
                vis = cv2.circle(
                    vis,
                    (int(round(point[0])), int(round(point[1]))),
                    2,
                    (255, 0, 255),
                    -1,
                )
            for point in visible_uv:
                vis = cv2.circle(
                    vis,
                    (int(round(point[0])), int(round(point[1]))),
                    2,
                    (0, 255, 0),
                    -1,
                )
            visualization_file = f"{view.view_id}.png"
            cv2.imwrite(
                str(image_output_dir / visualization_file),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            )
        projected_views.append(
            {
                "view_id": str(view.view_id),
                "image_file": f"{view.view_id}.jpg",
                "projected_bbox_2d": _bbox_to_list(bbox),
                "num_projected_points": int(visibility_stats["projected_points_total"]),
                "projected_points_in_frame": int(
                    visibility_stats["projected_points_in_frame"]
                ),
                "visible_projected_points": int(
                    visibility_stats["visible_projected_points"]
                ),
                "occluded_projected_points": int(
                    visibility_stats["occluded_projected_points"]
                ),
                "depth_missing_points": int(visibility_stats["depth_missing_points"]),
                "visible_ratio": round(float(visibility_stats["visible_ratio"]), 4),
                "occluded_ratio": round(float(visibility_stats["occluded_ratio"]), 4),
                "reference_detections": reference_detections,
                "visualization_file": visualization_file,
            }
        )

    output_path = args.output_path
    if output_path is None:
        output_path = args.candidate_json.with_name(
            args.candidate_json.stem + "_projected_views.json"
        )

    image_output_dir = output_path.with_suffix("")

    payload = {
        "scene": scene,
        "query": candidate_data.get("query", ""),
        "candidate_id": int(candidate.object_id),
        "source_candidate_json": str(args.candidate_json),
        "num_source_views": len(candidate.object_view),
        "bbox_3d": [
            round(float(value), 4) for value in np.asarray(bbox_3d).reshape(-1).tolist()
        ],
        "projected_views": projected_views,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"candidate_json={args.candidate_json}")
    print(f"scene={scene}")
    print(f"num_source_views={len(candidate.object_view)}")
    print(f"num_projected_views={len(projected_views)}")
    print(f"output_path={output_path}")
    if args.save_images:
        print(f"image_output_dir={image_output_dir}")


if __name__ == "__main__":
    main()
