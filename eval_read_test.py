from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from agent import Agent
    from benchmark.utils import calc_iou, load_pc
    from module.projection import TwoDToThreeDTool
    from module.segmenter import SAMSegmenter
    from prompt import build_candidate_summary, build_multi_candidate_selection_prompt
    from read import Read
except ImportError:
    from .agent import Agent
    from .benchmark.utils import calc_iou, load_pc
    from .module.projection import TwoDToThreeDTool
    from .module.segmenter import SAMSegmenter
    from .prompt import build_candidate_summary, build_multi_candidate_selection_prompt
    from .read import Read


def _normalize_multi_candidate_selection(result: Any) -> int | None:
    if not isinstance(result, dict):
        return None
    selected_index = result.get("selected_index")
    if isinstance(selected_index, bool):
        return None
    if isinstance(selected_index, int):
        return selected_index
    if isinstance(selected_index, str) and selected_index.strip().isdigit():
        return int(selected_index.strip())
    return None


def _select_unsure_candidate(agent: Agent):
    active_candidates = [
        candidate
        for candidate in agent.candidates.values()
        if getattr(candidate, "status", "active") == "active"
    ]
    if not active_candidates:
        return None
    if len(active_candidates) == 1:
        print("fallback_selected_single_unsure_candidate")
        return active_candidates[0]

    print("saving_active_candidates_for_multi_candidate_selection")
    for index, candidate in enumerate(active_candidates, start=1):
        print(f"active_candidate[{index}] {build_candidate_summary(candidate)}")

    prompt = build_multi_candidate_selection_prompt(agent.query, active_candidates)
    image_count = agent._count_prompt_images(prompt)
    agent.vlm_image_counts.append(image_count)
    print(f"[Agent] vlm_stitched_image_count={image_count}")
    result = agent._normalize_vlm_result(agent.vlm(prompt, candidates=active_candidates))
    print("[Agent] vlm_multi_candidate_result")
    if isinstance(result, (dict, list)):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)
    selected_index = _normalize_multi_candidate_selection(result)
    if selected_index is None or selected_index < 0 or selected_index >= len(active_candidates):
        print("fallback_multi_candidate_selection_unsure")
        return None
    print(f"fallback_selected_candidate_index={selected_index}")
    return active_candidates[selected_index]


def run_one_case(
    *,
    scene: str,
    query: str,
    sam_checkpoint: str,
    sam_model_type: str,
    sam_device: str,
    max_frames: int,
    frame_skip: int,
    max_units: int,
    min_selected_object_views: int,
) -> dict[str, Any]:
    reader = Read(scene, max_frames_per_find=max_frames, frame_skip=frame_skip)
    segmenter = SAMSegmenter(
        checkpoint_path=sam_checkpoint,
        model_type=sam_model_type,
        device=sam_device,
    )
    agent = Agent(
        motion=reader,
        segmenter=segmenter,
        mapper_2d3d=TwoDToThreeDTool(),
        intrinsic_matrix=reader.intrinsic_matrix,
        world_to_axis_align_matrix=reader.world_to_axis_align_matrix,
        debug=True,
    )
    agent.reset(query)

    total_views = 0
    total_object_views = 0
    selected_candidate = None
    final_decision = "false"
    vlm_used = False

    unit_index = 0
    while max_units < 0 or unit_index < max_units:
        views = reader.find()
        if not views:
            if final_decision == "unsure" and agent.candidates.exist():
                print("fallback_no_more_frames_with_unsure_candidates")
                selected_candidate = _select_unsure_candidate(agent)
                if selected_candidate is not None:
                    final_decision = "true"
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

        vlm_used = True
        decision_candidate, decision = agent.evaluate_candidates()
        final_decision = decision
        print(f"decision={decision}")
        if decision_candidate is not None:
            selected_candidate = decision_candidate
            if len(selected_candidate.object_view) < min_selected_object_views:
                print(f"selected_object_views={len(selected_candidate.object_view)}")
                print("decision=collect_more_views_before_stop")
                unit_index += 1
                continue
            break

        unit_index += 1

    bbox_3d = None
    if selected_candidate is not None:
        try:
            print("before_complete_candidate_masks")
            agent.complete_candidate_masks(selected_candidate)
            print("after_complete_candidate_masks")
            print("before_map_candidate_to_3d")
            points_3d, bbox_3d = agent.map_candidate_to_3d(selected_candidate)
            print("after_map_candidate_to_3d")
            print(f"bbox_3d={bbox_3d.tolist()}")
            try:
                TwoDToThreeDTool.visualize_points_and_aabb(points_3d, bbox_3d)
            except ImportError as exc:
                print(f"visualization_skipped={exc}")
        except ValueError as exc:
            print(f"projection_skipped={exc}")
    elif final_decision == "unsure":
        print("saving_all_candidates_for_unsure_case")
        for index, candidate in enumerate(agent.candidates.values(), start=1):
            print(f"unsure_candidate[{index}] {build_candidate_summary(candidate)}")

    return {
        "final_decision": final_decision,
        "selected_candidate": selected_candidate,
        "bbox_3d": None if bbox_3d is None else np.asarray(bbox_3d, dtype=np.float64),
        "total_views": total_views,
        "total_object_views": total_object_views,
        "total_candidates": len(agent.candidates.values()),
        "vlm_used": vlm_used,
        "vlm_image_counts": list(agent.vlm_image_counts),
        "detector_call_count": int(agent.detector_call_count),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate read_test.py style pipeline on ScanRefer benchmark tasks.")
    parser.add_argument("--data-path", default="benchmark/scanrefer_250.json", help="Path to the benchmark json file")
    parser.add_argument("--case-index", type=int, default=-1, help="Run only one benchmark case by 0-based index")
    parser.add_argument("--max-cases", type=int, default=-1, help="Maximum number of benchmark cases to run")
    parser.add_argument("--query-field", choices=["caption", "obj_name"], default="caption", help="Which field to use as query")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum frames per read unit")
    parser.add_argument("--frame-skip", type=int, default=4, help="Sample every Nth frame when building views")
    parser.add_argument("--max-units", type=int, default=-1, help="Maximum read units to process per case; use -1 to read until exhausted")
    parser.add_argument("--min-selected-object-views", type=int, default=5, help="Minimum selected candidate object views before stopping")
    parser.add_argument("--sam-checkpoint", default="checkpoints/SAM/sam_vit_h_4b8939.pth", help="Path to the SAM checkpoint file")
    parser.add_argument("--sam-model-type", default="vit_h", help="SAM model type passed to sam_model_registry")
    parser.add_argument("--sam-device", default="cpu", help="Device for SAM inference, e.g. cpu or cuda")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    with data_path.open("r", encoding="utf-8") as file:
        eval_data = json.load(file)

    if args.case_index >= 0:
        eval_data = [eval_data[args.case_index]]
    if args.max_cases > 0:
        eval_data = eval_data[: args.max_cases]

    correct_25 = 0
    correct_50 = 0
    unique_25 = 0
    unique_50 = 0
    total = 0
    unique_total = 0
    except_total = 0
    vlm_total = 0
    eps = 1e-6

    for case_index, task in enumerate(eval_data):
        scene_id = str(task["scan_id"])
        target_id = int(task["target_id"])
        query = str(task[args.query_field])

        print(f"Case: {case_index}")
        print(f"scene_id: {scene_id}")
        print(f"query: {query}")
        print(f"target_id: {target_id}")

        total += 1

        try:
            obj_ids, obj_labels, obj_locs = load_pc(scene_id)
            target_index = obj_ids.index(target_id)
            target_box = np.asarray(obj_locs[target_index], dtype=np.float64)
            target_label = obj_labels[target_index]
            print(f"gt_label: {target_label}")
            print(f"gt_bbox_3d: {target_box.tolist()}")
            unique = sum(label == target_label for label in obj_labels) == 1
            if unique:
                unique_total += 1

            result = run_one_case(
                scene=scene_id,
                query=query,
                sam_checkpoint=args.sam_checkpoint,
                sam_model_type=args.sam_model_type,
                sam_device=args.sam_device,
                max_frames=args.max_frames,
                frame_skip=args.frame_skip,
                max_units=args.max_units,
                min_selected_object_views=args.min_selected_object_views,
            )

            if result["vlm_used"]:
                vlm_total += 1

            print(f"vlm_image_counts={result['vlm_image_counts']}")
            print(f"detector_call_count={result['detector_call_count']}")

            pred_box = result["bbox_3d"]
            selected_candidate = result["selected_candidate"]
            if selected_candidate is not None:
                print(f"final_selected_candidate={build_candidate_summary(selected_candidate)}")
            if pred_box is None:
                except_total += 1
            else:
                iou = float(calc_iou(pred_box, target_box))
                print(f"IoU: {iou:.3f}")
                if iou >= 0.25:
                    correct_25 += 1
                    if unique:
                        unique_25 += 1
                if iou >= 0.5:
                    correct_50 += 1
                    if unique:
                        unique_50 += 1
        except Exception as exc:
            except_total += 1
            print(f"case_error={exc}")

        accuracy_msgs = [
            "Overall@25: {:.3f}".format(correct_25 / total),
            "Overall@50: {:.3f}".format(correct_50 / total),
            "Unique@25: {:.3f}".format(unique_25 / (unique_total + eps)),
            "Unique@50: {:.3f}".format(unique_50 / (unique_total + eps)),
            "Multiple@25: {:.3f}".format((correct_25 - unique_25) / (total - unique_total + eps)),
            "Multiple@50: {:.3f}".format((correct_50 - unique_50) / (total - unique_total + eps)),
            "Unique Ratio: {} / {}".format(unique_25, unique_total),
            "Multiple Ratio: {} / {}".format(correct_25 - unique_25, total - unique_total),
            "Except Ratio: {} / {}".format(except_total, total),
            "VLM Usage Ratio: {} / {}".format(vlm_total, total),
            "",
        ]
        print("\n".join(accuracy_msgs))
