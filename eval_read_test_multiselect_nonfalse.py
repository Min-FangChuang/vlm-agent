from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    from agent import Agent
    from benchmark.utils import calc_iou, load_pc
    from module.detector_yoloe import YOLOEDetector
    from module.projection import TwoDToThreeDTool
    from module.segmenter import SAMSegmenter
    from prompt import build_candidate_summary, build_multi_candidate_selection_prompt
    from read import Read
except ImportError:
    from .agent import Agent
    from .benchmark.utils import calc_iou, load_pc
    from .module.detector_yoloe import YOLOEDetector
    from .module.projection import TwoDToThreeDTool
    from .module.segmenter import SAMSegmenter
    from .prompt import build_candidate_summary, build_multi_candidate_selection_prompt
    from .read import Read


MIN_VLM_CANDIDATE_VIEWS = 1
SKIP_CASE_INDICES = set()


def _candidate_meets_vlm_threshold(candidate: Any) -> bool:
    object_views = getattr(candidate, "object_view", []) or []
    return len(object_views) >= MIN_VLM_CANDIDATE_VIEWS


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


def _select_nonfalse_candidate(agent: Agent):
    active_candidates = [
        candidate
        for candidate in agent.candidates.values()
        if str(getattr(candidate, "status", "new")) != "false"
        and _candidate_meets_vlm_threshold(candidate)
    ]
    if not active_candidates:
        print("fallback_no_candidates_meet_vlm_threshold")
        return None
    if len(active_candidates) == 1:
        print("fallback_selected_single_nonfalse_candidate")
        return active_candidates[0]

    print("saving_active_candidates_for_multi_candidate_selection")
    for index, candidate in enumerate(active_candidates, start=1):
        print(f"active_candidate[{index}] {build_candidate_summary(candidate)}")

    prompt = build_multi_candidate_selection_prompt(agent.query, active_candidates)
    image_count = agent._count_prompt_images(prompt)
    agent.vlm_image_counts.append(image_count)
    print(f"[Agent] vlm_stitched_image_count={image_count}")
    result = agent._normalize_vlm_result(
        agent.vlm(prompt, candidates=active_candidates)
    )
    print("[Agent] vlm_multi_candidate_result")
    if isinstance(result, (dict, list)):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)
    selected_index = _normalize_multi_candidate_selection(result)
    if (
        selected_index is None
        or selected_index < 0
        or selected_index >= len(active_candidates)
    ):
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
    detector_model: str | None,
    shared_detector: Any,
    shared_segmenter: Any,
    shared_mapper_2d3d: Any,
) -> dict[str, Any]:
    del min_selected_object_views, detector_model
    reader = Read(scene, max_frames_per_find=max_frames, frame_skip=frame_skip)
    agent = Agent(
        motion=reader,
        detector=shared_detector,
        segmenter=shared_segmenter,
        mapper_2d3d=shared_mapper_2d3d,
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
            if agent.candidates.exist():
                print("fallback_no_more_frames_with_nonfalse_candidates")
                selected_candidate = _select_nonfalse_candidate(agent)
                if selected_candidate is not None:
                    final_decision = str(
                        getattr(selected_candidate, "status", "unsure")
                    )
                    if final_decision == "expanded":
                        final_decision = "unsure"
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
            vlm_used = True
            pending_candidate, decision = agent.verify_candidate_once(pending_candidate)
            final_decision = decision
            print(f"decision={decision}")

        unit_index += 1

    bbox_3d = None
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
                print("after_map_candidate_to_3d")
                print(f"bbox_3d={bbox_3d.tolist()}")
                #try:
                #    TwoDToThreeDTool.visualize_points_and_aabb(points_3d, bbox_3d)
                #except ImportError as exc:
                #    print(f"visualization_skipped={exc}")
            except ValueError as exc:
                print(f"projection_skipped={exc}")
    elif final_decision in {"unsure", "true"}:
        print("saving_all_candidates_for_nonfalse_case")
        for index, candidate in enumerate(agent.candidates.values(), start=1):
            if str(getattr(candidate, "status", "")) == "false":
                continue
            if not _candidate_meets_vlm_threshold(candidate):
                print(f"nonfalse_candidate[{index}] skipped_below_vlm_threshold")
                continue
            print(f"nonfalse_candidate[{index}] {build_candidate_summary(candidate)}")

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
    parser = argparse.ArgumentParser(
        description="Evaluate ScanRefer-style cases but defer all non-false candidates to final multiselect."
    )
    parser.add_argument(
        "--data-path",
        default="benchmark/scanrefer_250.json",
        help="Path to the benchmark json file",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=-1,
        help="Run only one benchmark case by 0-based index",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=-1,
        help="Maximum number of benchmark cases to run",
    )
    parser.add_argument(
        "--query-field",
        choices=["caption", "obj_name"],
        default="caption",
        help="Which field to use as query",
    )
    parser.add_argument(
        "--max-frames", type=int, default=10, help="Maximum frames per read unit"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Sample every Nth frame when building views",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=-1,
        help="Maximum read units to process per case; use -1 to read until exhausted",
    )
    parser.add_argument(
        "--min-selected-object-views",
        type=int,
        default=5,
        help="Minimum selected candidate object views before stopping",
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
        "--sam-device", default="cpu", help="Device for SAM inference, e.g. cpu or cuda"
    )
    parser.add_argument(
        "--detector-model",
        default=None,
        help="Optional YOLOE checkpoint name or path.",
    )
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
    case_iou_rows: list[dict[str, Any]] = []

    shared_detector = YOLOEDetector(model=args.detector_model or "yoloe-11s-seg.pt")
    shared_segmenter = SAMSegmenter(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.sam_device,
    )
    shared_mapper_2d3d = TwoDToThreeDTool()

    for case_index, task in enumerate(eval_data):
        original_case_index = args.case_index if args.case_index >= 0 else case_index
        scene_id = str(task["scan_id"])
        target_id = int(task["target_id"])
        query = str(task[args.query_field])

        print(f"Case: {case_index}")
        print(f"scene_id: {scene_id}")
        print(f"query: {query}")
        print(f"target_id: {target_id}")

        if original_case_index in SKIP_CASE_INDICES:
            print(f"skip_case={original_case_index}")
            total += 1
            case_iou_rows.append({"case_id": int(original_case_index), "iou": ""})
            continue

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
                detector_model=args.detector_model,
                shared_detector=shared_detector,
                shared_segmenter=shared_segmenter,
                shared_mapper_2d3d=shared_mapper_2d3d,
            )

            if result["vlm_used"]:
                vlm_total += 1

            print(f"vlm_image_counts={result['vlm_image_counts']}")
            print(f"detector_call_count={result['detector_call_count']}")

            pred_box = result["bbox_3d"]
            selected_candidate = result["selected_candidate"]
            if selected_candidate is not None:
                print(
                    f"final_selected_candidate={build_candidate_summary(selected_candidate)}"
                )
            if pred_box is None:
                except_total += 1
                case_iou_rows.append({"case_id": int(original_case_index), "iou": ""})
            else:
                iou = float(calc_iou(pred_box, target_box))
                print(f"IoU: {iou:.3f}")
                case_iou_rows.append(
                    {"case_id": int(original_case_index), "iou": round(iou, 6)}
                )
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
            traceback.print_exc()
            case_iou_rows.append({"case_id": int(original_case_index), "iou": ""})

        accuracy_msgs = [
            "Overall@25: {:.3f}".format(correct_25 / total),
            "Overall@50: {:.3f}".format(correct_50 / total),
            "Unique@25: {:.3f}".format(unique_25 / (unique_total + eps)),
            "Unique@50: {:.3f}".format(unique_50 / (unique_total + eps)),
            "Multiple@25: {:.3f}".format(
                (correct_25 - unique_25) / (total - unique_total + eps)
            ),
            "Multiple@50: {:.3f}".format(
                (correct_50 - unique_50) / (total - unique_total + eps)
            ),
            "Unique Ratio: {} / {}".format(unique_25, unique_total),
            "Multiple Ratio: {} / {}".format(
                correct_25 - unique_25, total - unique_total
            ),
            "Except Ratio: {} / {}".format(except_total, total),
            "VLM Usage Ratio: {} / {}".format(vlm_total, total),
            "",
        ]
        print("\n".join(accuracy_msgs))

    if args.max_cases > 0:
        csv_path = Path(
            f"eval_multiselect_nonfalse_max_cases_{args.max_cases}_ious.csv"
        )
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=["case_id", "iou"])
            writer.writeheader()
            writer.writerows(case_iou_rows)
