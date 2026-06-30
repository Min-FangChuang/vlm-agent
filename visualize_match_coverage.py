from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from agent import Agent
    from agent_schema import CandidateObject, ObjectView
    from module.detector import draw_bbox
    from module.matcher import (
        PATSMatcher,
        _bbox_from_points,
        _extract_object_view_bbox,
        _extract_object_view_mask,
        _points_inside_bbox,
        _points_inside_mask,
    )
    from module.projection import TwoDToThreeDTool
    from module.segmenter import SAMSegmenter
    from read import Read
except ImportError:
    from .agent import Agent
    from .agent_schema import CandidateObject, ObjectView
    from .module.detector import draw_bbox
    from .module.matcher import (
        PATSMatcher,
        _bbox_from_points,
        _extract_object_view_bbox,
        _extract_object_view_mask,
        _points_inside_bbox,
        _points_inside_mask,
    )
    from .module.projection import TwoDToThreeDTool
    from .module.segmenter import SAMSegmenter
    from .read import Read


def _draw_points(
    image: np.ndarray, points_xy: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    output = image.copy()
    for point in points_xy:
        output = cv2.circle(output, (int(point[0]), int(point[1])), 2, color, -1)
    return output


def _build_candidate_from_views(
    label: str, object_views: list[ObjectView], best_id: int
) -> CandidateObject:
    return CandidateObject(
        object_id=0,
        label=label,
        best_id=best_id,
        object_view=object_views,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize matcher mask-supported coverage for one best-view / incoming-view pair."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--best-view-id", required=True)
    parser.add_argument("--best-bbox", required=True, help="x1,y1,x2,y2")
    parser.add_argument("--incoming-view-id", required=True)
    parser.add_argument("--incoming-bbox", required=True, help="x1,y1,x2,y2")
    parser.add_argument(
        "--sam-checkpoint", default="checkpoints/SAM/sam_vit_h_4b8939.pth"
    )
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--sam-device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output") / "match_coverage"
    )
    args = parser.parse_args()

    reader = Read(args.scene, max_frames_per_find=999999, frame_skip=1)
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
    agent.reset(args.query)
    matcher = PATSMatcher()

    best_view = reader._build_view(args.best_view_id)
    incoming_view = reader._build_view(args.incoming_view_id)
    best_bbox = np.asarray(
        [float(x) for x in args.best_bbox.split(",")], dtype=np.float32
    )
    incoming_bbox = np.asarray(
        [float(x) for x in args.incoming_bbox.split(",")], dtype=np.float32
    )

    best_mask = segmenter.segment_from_box(best_view.rgb, best_bbox.tolist())
    best_object_view = ObjectView(
        object_id=f"{args.best_view_id}_0",
        label=args.query,
        score=1.0,
        view=best_view,
        bbox_2d=best_bbox,
        mask_2d=np.asarray(best_mask, dtype=np.uint8),
        points_3d=None,
        status="active",
        source="detected",
    )
    incoming_object_view = ObjectView(
        object_id=f"{args.incoming_view_id}_0",
        label=args.query,
        score=1.0,
        view=incoming_view,
        bbox_2d=incoming_bbox,
        mask_2d=None,
        points_3d=None,
        status="active",
        source="detected",
    )

    view_match = matcher.match_views(
        np.asarray(incoming_object_view.rgb), np.asarray(best_object_view.rgb)
    )
    object_points = view_match.image0_points
    candidate_points = view_match.image1_points

    candidate_bbox = _extract_object_view_bbox(best_object_view)
    object_bbox = _extract_object_view_bbox(incoming_object_view)
    candidate_mask = _extract_object_view_mask(best_object_view)

    all_mask_keep = _points_inside_mask(candidate_points, candidate_mask)
    all_object_mask_points = object_points[all_mask_keep]
    all_candidate_mask_points = candidate_points[all_mask_keep]
    all_object_mask_bbox = _bbox_from_points(all_object_mask_points)

    bbox_keep = _points_inside_bbox(candidate_points, candidate_bbox)
    object_points = object_points[bbox_keep]
    candidate_points = candidate_points[bbox_keep]

    object_bbox_keep = _points_inside_bbox(object_points, object_bbox)
    object_points = object_points[object_bbox_keep]
    candidate_points = candidate_points[object_bbox_keep]

    mask_keep = _points_inside_mask(candidate_points, candidate_mask)
    object_mask_points = object_points[mask_keep]
    candidate_mask_points = candidate_points[mask_keep]

    best_vis = draw_bbox(best_view.rgb, best_bbox, "best_view_bbox", color=(0, 255, 0))
    best_vis = _draw_points(best_vis, candidate_points, (255, 0, 0))
    best_vis = _draw_points(best_vis, candidate_mask_points, (0, 255, 0))
    best_vis = _draw_points(best_vis, all_candidate_mask_points, (0, 255, 255))

    incoming_vis = draw_bbox(
        incoming_view.rgb, incoming_bbox, "incoming_bbox", color=(0, 255, 0)
    )
    incoming_vis = _draw_points(incoming_vis, object_points, (255, 0, 0))
    incoming_vis = _draw_points(incoming_vis, object_mask_points, (0, 255, 0))
    incoming_vis = _draw_points(incoming_vis, all_object_mask_points, (0, 255, 255))
    if all_object_mask_bbox is not None:
        incoming_vis = draw_bbox(
            incoming_vis,
            all_object_mask_bbox,
            "projected_match_bbox",
            color=(255, 255, 0),
        )

    output_dir = (
        args.output_dir / args.scene / f"{args.best_view_id}_to_{args.incoming_view_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_dir / "best_view.png"), cv2.cvtColor(best_vis, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        str(output_dir / "incoming_view.png"),
        cv2.cvtColor(incoming_vis, cv2.COLOR_RGB2BGR),
    )
    print(f"output_dir={output_dir}")
