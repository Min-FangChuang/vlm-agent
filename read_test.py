from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from agent import Agent
    from eval_read_test import (
        _candidate_meets_vlm_threshold,
        _candidate_snapshot,
        _select_unsure_candidate,
    )
    from module.detector import YOLOWorldDetector
    from module.projection import TwoDToThreeDTool
    from module.segmenter import SAMSegmenter
    from prompt import build_candidate_summary
    from read import Read
    from read.scannet_more_view import complete_candidate_with_more_views
except ImportError:
    from .agent import Agent
    from .eval_read_test import (
        _candidate_meets_vlm_threshold,
        _candidate_snapshot,
        _select_unsure_candidate,
    )
    from .module.detector import YOLOWorldDetector
    from .module.projection import TwoDToThreeDTool
    from .module.segmenter import SAMSegmenter
    from .prompt import build_candidate_summary
    from .read import Read
    from .read.scannet_more_view import complete_candidate_with_more_views


if __name__ == "__main__":
    min_selected_object_views = 5
    parser = argparse.ArgumentParser(
        description="Read one ScanNet scene with the updated single-candidate debug flow."
    )
    parser.add_argument(
        "--scene",
        default="scene0207_00",
        help="Scene name under vlm-agent/scannet/posed_images",
    )
    parser.add_argument(
        "--query", default="chair", help="Search query passed to Agent.reset()"
    )
    parser.add_argument(
        "--max-frames", type=int, default=10, help="Maximum frames per read unit"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Sample every Nth frame when building views",
    )
    parser.add_argument(
        "--max-units", type=int, default=8, help="Maximum read units to process"
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="checkpoints/SAM/sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_h",
        help="SAM model type passed to sam_model_registry",
    )
    parser.add_argument(
        "--sam-device",
        default="cpu",
        help="Device for SAM inference, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--detector-model",
        default="yolov8x-worldv2.pt",
        help="YOLO-World model checkpoint name or path",
    )
    args = parser.parse_args()

    reader = Read(
        args.scene, max_frames_per_find=args.max_frames, frame_skip=args.frame_skip
    )
    detector = YOLOWorldDetector(model=args.detector_model)
    segmenter = SAMSegmenter(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.sam_device,
    )
    agent = Agent(
        motion=reader,
        detector=detector,
        segmenter=segmenter,
        mapper_2d3d=TwoDToThreeDTool(),
        intrinsic_matrix=reader.intrinsic_matrix,
        world_to_axis_align_matrix=reader.world_to_axis_align_matrix,
        debug=True,
    )
    agent.reset(args.query)
    debug_output_dir = (
        Path("output") / "candidate_debug" / args.scene / "_".join(args.query.split())
    )

    def save_candidate_snapshot(prefix: str, candidate) -> None:
        if str(getattr(candidate, "status", "")) == "false":
            return
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        status = str(getattr(candidate, "status", "unknown"))
        verification_round = int(getattr(candidate, "verification_round", 0))
        candidate_id = int(getattr(candidate, "object_id", -1))
        file_path = (
            debug_output_dir
            / f"{prefix}_candidate_{candidate_id:03d}_round_{verification_round}_{status}.json"
        )
        file_path.write_text(
            json.dumps(_candidate_snapshot(candidate), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    total_views = 0
    total_object_views = 0
    selected_candidate = None
    final_decision = "false"

    unit_index = 0
    while args.max_units < 0 or unit_index < args.max_units:
        views = reader.find()
        if not views:
            if agent.candidates.exist():
                print("fallback_no_more_frames_with_unsure_candidates")
                selected_candidate = _select_unsure_candidate(agent)
                if selected_candidate is not None:
                    final_decision = "true"
            break

        total_views += len(views)
        object_views_before = sum(
            len(candidate.object_view) for candidate in agent.candidates.values()
        )
        agent.consume_views(views)
        object_views_after = sum(
            len(candidate.object_view) for candidate in agent.candidates.values()
        )
        object_views = object_views_after - object_views_before
        total_object_views += max(0, object_views)

        print(f"unit={unit_index}")
        print(f"views={len(views)}")
        print(f"object_views={object_views}")
        print(f"candidates={len(agent.candidates.values())}")

        processed_any_candidate = False
        while True:
            pending_candidate = agent.pick_candidate_for_verification()
            if pending_candidate is None:
                if not processed_any_candidate:
                    print("decision=no_pending_candidate_read_next_unit")
                break

            processed_any_candidate = True
            decision = agent.verify_candidate_once(pending_candidate)
            if decision == "unsure" and agent.can_retry_candidate(pending_candidate):
                save_candidate_snapshot(
                    f"unit_{unit_index:03d}_before_more_view",
                    pending_candidate,
                )
                pending_candidate = complete_candidate_with_more_views(
                    agent=agent,
                    candidate=pending_candidate,
                )
                save_candidate_snapshot(
                    f"unit_{unit_index:03d}_after_more_view",
                    pending_candidate,
                )
                decision = agent.verify_candidate_once(pending_candidate)
                if decision not in {"true", "false"}:
                    pending_candidate.status = "unsure"
                save_candidate_snapshot(
                    f"unit_{unit_index:03d}_after_retry",
                    pending_candidate,
                )
            final_decision = decision
            print(f"decision={decision}")
            if decision == "true":
                selected_candidate = pending_candidate
                break

        if selected_candidate is not None:
            break

        unit_index += 1

    print(f"scene={args.scene}")
    print(f"query={args.query}")
    print(f"total_views={total_views}")
    print(f"total_object_views={total_object_views}")
    print(f"total_candidates={len(agent.candidates.values())}")
    print(f"final_decision={final_decision}")
    print(f"vlm_image_counts={agent.vlm_image_counts}")
    print(f"detector_call_count={agent.detector_call_count}")

    if selected_candidate is not None:
        if not _candidate_meets_vlm_threshold(selected_candidate):
            print("selected_candidate_below_vlm_threshold_skip_projection")
            selected_candidate = None
        else:
            try:
                print("before_complete_candidate_masks")
                agent.complete_candidate_masks(selected_candidate)
                print("after_complete_candidate_masks")
                print("before_map_candidate_to_3d")
                points_3d, bbox_3d = agent.map_candidate_to_3d(selected_candidate)
                selected_candidate.points_3d = points_3d
                selected_candidate.bbox_3d = bbox_3d
                print("after_map_candidate_to_3d")
                print(f"bbox_3d={np.asarray(bbox_3d).tolist()}")
                try:
                    TwoDToThreeDTool.visualize_points_and_aabb(points_3d, bbox_3d)
                except ImportError as exc:
                    print(f"visualization_skipped={exc}")
            except ValueError as exc:
                print(f"projection_skipped={exc}")
            print(
                f"final_selected_candidate={build_candidate_summary(selected_candidate)}"
            )
    elif final_decision == "unsure":
        print("saving_all_candidates_for_unsure_case")
        for index, candidate in enumerate(agent.candidates.values(), start=1):
            if not _candidate_meets_vlm_threshold(candidate):
                print(f"unsure_candidate[{index}] skipped_below_vlm_threshold")
                continue
            print(f"unsure_candidate[{index}] {build_candidate_summary(candidate)}")

    for index, candidate in enumerate(agent.candidates.values(), start=1):
        print(f"candidate[{index}] {build_candidate_summary(candidate)}")
