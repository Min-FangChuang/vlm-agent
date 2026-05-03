from __future__ import annotations

import argparse

try:
    from agent import Agent
    from module.projection import TwoDToThreeDTool
    from module.segmenter import SAMSegmenter
    from prompt import build_candidate_summary
    from read import Read
except ImportError:
    from .agent import Agent
    from .module.projection import TwoDToThreeDTool
    from .module.segmenter import SAMSegmenter
    from .prompt import build_candidate_summary
    from .read import Read


if __name__ == "__main__":
    min_selected_object_views = 5
    parser = argparse.ArgumentParser(description="Read posed images in chunks and run agent detection.")
    parser.add_argument("--scene", default="scene0207_00", help="Scene name under vlm-agent/scannet/posed_images")
    parser.add_argument("--query", default="chair", help="Search query passed to Agent.reset()")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum frames per read unit")
    parser.add_argument("--max-units", type=int, default=1, help="Maximum read units to process")
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
    args = parser.parse_args()

    reader = Read(args.scene, max_frames_per_find=args.max_frames)
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
    )
    agent.reset(args.query)

    total_views = 0
    total_object_views = 0
    selected_candidate = None
    final_decision = "false"
    for unit_index in range(args.max_units):
        views = reader.find()
        if not views:
            break

        total_views += len(views)
        object_views = agent.collect_object_views(views)
        total_object_views += len(object_views)
        agent.update_candidates(object_views)

        print(f"unit={unit_index}")
        print(f"views={len(views)}")
        print(f"object_views={len(object_views)}")
        print(f"candidates={len(agent.candidates.values())}")

        if selected_candidate is not None:
            selected_object_views = len(selected_candidate.object_view)
            print(f"selected_object_views={selected_object_views}")
            if selected_object_views < min_selected_object_views:
                print("decision=skip_vlm_collect_more_views")
                continue
            final_decision = "true"
            print("decision=selected_candidate_ready")
            break

        decision_candidate, decision = agent.evaluate_candidates()
        final_decision = decision
        print(f"decision={decision}")
        if decision_candidate is not None:
            selected_candidate = decision_candidate
            if len(selected_candidate.object_view) < min_selected_object_views:
                print(
                    f"selected_object_views={len(selected_candidate.object_view)}"
                )
                print("decision=collect_more_views_before_stop")
                continue
            break

    print(f"scene={args.scene}")
    print(f"query={args.query}")
    print(f"total_views={total_views}")
    print(f"total_object_views={total_object_views}")
    print(f"total_candidates={len(agent.candidates.values())}")
    print(f"final_decision={final_decision}")
    if selected_candidate is not None:
        try:
            agent.complete_candidate_masks(selected_candidate)
            points_3d, bbox_3d = agent.map_candidate_to_3d(selected_candidate)
            print(f"bbox_3d={bbox_3d.tolist()}")
            try:
                TwoDToThreeDTool.visualize_points_and_aabb(points_3d, bbox_3d)
            except ImportError as exc:
                print(f"visualization_skipped={exc}")
        except ValueError as exc:
            print(f"projection_skipped={exc}")
        print(f"final_selected_candidate={build_candidate_summary(selected_candidate)}")
    for index, candidate in enumerate(agent.candidates.values(), start=1):
        print(f"candidate[{index}] {build_candidate_summary(candidate)}")
