from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

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


def _bbox_to_list(bbox: Any) -> list[float]:
    array = np.asarray(bbox, dtype=np.float32).reshape(-1)
    return [round(float(value), 2) for value in array.tolist()]


def _status_reason(result: Any) -> str:
    if int(result.total_matches) <= 1000:
        return "low_total_matches"
    if int(result.num_bbox_matches) <= 100:
        return "low_bbox_matches"
    if float(getattr(result, "mask_back_project_coverage", 0.0)) < 0.8:
        return "low_mask_back_project_coverage"
    if int(result.num_mask_matches) < int(result.num_bbox_matches):
        return "mask_filtered"
    return "accepted"


def _collect_candidate_match_debug(
    agent: Agent,
    object_view: Any,
) -> dict[str, Any]:
    candidates = agent.candidates.values()
    candidate_matches: list[dict[str, Any]] = []
    for candidate in candidates:
        same_view_exists = any(
            existing_object_view.view.view_id == object_view.view.view_id
            for existing_object_view in candidate.object_view
        )
        best_view = candidate.object_view[candidate.best_id]
        if same_view_exists:
            candidate_matches.append(
                {
                    "candidate_object_id": candidate.object_id,
                    "candidate_label": candidate.label,
                    "candidate_best_id": candidate.best_id,
                    "candidate_best_view_id": best_view.view.view_id,
                    "candidate_best_bbox": _bbox_to_list(best_view.bbox_2d),
                    "candidate_num_views": len(candidate.object_view),
                    "same_view_blocked": True,
                    "match_status": "same_view_blocked",
                }
            )
            continue

        result = agent.matcher.match_object_view_to_candidate(
            object_view,
            candidate,
        )
        candidate_matches.append(
            {
                "candidate_object_id": candidate.object_id,
                "candidate_label": candidate.label,
                "candidate_best_id": candidate.best_id,
                "candidate_best_view_id": best_view.view.view_id,
                "candidate_best_bbox": _bbox_to_list(best_view.bbox_2d),
                "candidate_num_views": len(candidate.object_view),
                "same_view_blocked": False,
                "total_matches": int(result.total_matches),
                "bbox_matches": int(result.num_bbox_matches),
                "mask_matches": int(result.num_mask_matches),
                "filtered_matches": int(result.num_filtered_matches),
                "mask_back_project_coverage": round(
                    float(result.mask_back_project_coverage), 4
                ),
                "is_match": bool(result.is_match),
                "match_status": _status_reason(result),
            }
        )

    candidate_matches.sort(
        key=lambda item: (
            not item.get("same_view_blocked", False),
            item.get("is_match", False),
            item.get("filtered_matches", -1),
            item.get("bbox_matches", -1),
            item.get("total_matches", -1),
        ),
        reverse=True,
    )

    return {
        "object_view_id": object_view.object_id,
        "object_label": object_view.label,
        "object_score": round(float(object_view.score), 4),
        "view_id": object_view.view.view_id,
        "status": getattr(object_view, "status", "active"),
        "bbox": _bbox_to_list(object_view.bbox_2d),
        "existing_candidates": len(candidates),
        "candidate_matches": candidate_matches,
    }


def _summarize_filtered_detection(
    view: Any, detection_index: int, detection: Any
) -> dict[str, Any]:
    return {
        "object_view_id": f"{view.view_id}_{detection_index}",
        "object_label": detection.label,
        "object_score": round(float(detection.score), 4),
        "view_id": view.view_id,
        "status": "filtered_out_before_match",
        "bbox": _bbox_to_list(detection.bbox),
        "existing_candidates": 0,
        "candidate_matches": [],
        "filtered_out_before_match": True,
    }


def _print_object_view_debug(item: dict[str, Any]) -> None:
    print(
        "object_view="
        f"{item['object_view_id']} view={item['view_id']} label={item['object_label']} "
        f"score={item['object_score']:.3f} status={item['status']} bbox={item['bbox']}"
    )
    if item.get("filtered_out_before_match"):
        print("  classification=filtered_out_before_match")
        return

    candidate_matches = item["candidate_matches"]
    if not candidate_matches:
        print("  classification=new_candidate_no_existing_candidates")
    else:
        top_match = candidate_matches[0]
        if top_match.get("same_view_blocked"):
            print(
                "  top_match="
                f"candidate[{top_match['candidate_object_id']}] "
                f"best_view={top_match['candidate_best_view_id']} status=same_view_blocked"
            )
        else:
            print(
                "  top_match="
                f"candidate[{top_match['candidate_object_id']}] "
                f"best_view={top_match['candidate_best_view_id']} "
                f"is_match={top_match['is_match']} status={top_match['match_status']} "
                f"total={top_match['total_matches']} filtered={top_match['filtered_matches']} "
                f"bbox={top_match['bbox_matches']} mask={top_match['mask_matches']} "
                f"mask_coverage={top_match['mask_back_project_coverage']}"
            )

        for match in candidate_matches[:5]:
            if match.get("same_view_blocked"):
                print(
                    "  candidate_check="
                    f"candidate[{match['candidate_object_id']}] best_view={match['candidate_best_view_id']} "
                    "status=same_view_blocked"
                )
                continue
            print(
                "  candidate_check="
                f"candidate[{match['candidate_object_id']}] best_view={match['candidate_best_view_id']} "
                f"best_bbox={match['candidate_best_bbox']} is_match={match['is_match']} "
                f"status={match['match_status']} total={match['total_matches']} "
                f"filtered={match['filtered_matches']} bbox={match['bbox_matches']} "
                f"mask={match['mask_matches']} mask_coverage={match['mask_back_project_coverage']}"
            )

    assigned_candidate_id = item.get("assigned_candidate_id")
    if assigned_candidate_id is not None:
        print(
            "  assignment="
            f"candidate[{assigned_candidate_id}] matched_existing={item['matched_existing']} "
            f"candidate_views_after={item['candidate_num_views_after']}"
        )


def _save_unit_debug(
    output_dir: Path, unit_index: int, items: list[dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"unit_{unit_index:03d}.json"
    file_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch matcher inspection with eval_read_test.py-style chunking."
    )
    parser.add_argument(
        "--scene", default="scene0207_00", help="Scene name under scannet/posed_images"
    )
    parser.add_argument(
        "--query", default="chair", help="Search query passed to Agent.reset()"
    )
    parser.add_argument(
        "--max-frames", type=int, default=40, help="Maximum frames per read unit"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Sample every Nth frame when building views",
    )
    parser.add_argument(
        "--max-units", type=int, default=3, help="Maximum read units to process"
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
        "--debug-output-dir",
        type=Path,
        default=Path("output") / "matcher_batches",
        help="Directory to save per-unit matcher debug JSON",
    )
    args = parser.parse_args()

    reader = Read(
        args.scene, max_frames_per_find=args.max_frames, frame_skip=args.frame_skip
    )
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

    total_views = 0
    total_object_views = 0
    for unit_index in range(args.max_units):
        views = reader.find()
        if not views:
            break

        unit_debug_items: list[dict[str, Any]] = []
        unit_views = 0
        unit_object_views = 0

        for view in views:
            detections = agent.detect_target_objects(view)
            agent.attach_reference(view)
            unit_views += 1

            if not detections:
                continue

            filtered_detections = agent._filter_detections_for_object_views(
                view, detections
            )
            kept_detection_indices = {index for index, _ in filtered_detections}

            for detection_index, detection in enumerate(detections):
                if detection_index in kept_detection_indices:
                    continue
                unit_debug_items.append(
                    _summarize_filtered_detection(view, detection_index, detection)
                )

            object_views = agent.collect_view_object_views(view, detections)
            unit_object_views += len(object_views)

            for object_view in object_views:
                debug_item = _collect_candidate_match_debug(agent, object_view)
                candidate, matched_existing = agent.candidates.add_ObjectView(
                    object_view,
                    lambda incoming_object_view, candidate_obj: (
                        agent.matcher.match_object_view_to_candidate(
                            incoming_object_view,
                            candidate_obj,
                        )
                    ),
                )
                agent.ensure_candidate_best_view_mask(candidate)
                debug_item["assigned_candidate_id"] = candidate.object_id
                debug_item["matched_existing"] = matched_existing
                debug_item["candidate_num_views_after"] = len(candidate.object_view)
                unit_debug_items.append(debug_item)

        total_views += unit_views
        total_object_views += unit_object_views
        _save_unit_debug(args.debug_output_dir, unit_index, unit_debug_items)

        print(f"unit={unit_index}")
        print(f"views={unit_views}")
        print(f"object_views={unit_object_views}")
        print(f"candidates={len(agent.candidates.values())}")
        for debug_item in unit_debug_items:
            _print_object_view_debug(debug_item)

    print(f"scene={args.scene}")
    print(f"query={args.query}")
    print(f"total_views={total_views}")
    print(f"total_object_views={total_object_views}")
    print(f"total_candidates={len(agent.candidates.values())}")
    for index, candidate in enumerate(agent.candidates.values(), start=1):
        print(f"candidate[{index}] {build_candidate_summary(candidate)}")
