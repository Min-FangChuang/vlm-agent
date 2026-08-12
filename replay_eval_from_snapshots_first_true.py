from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmark.utils import calc_iou, load_pc
    from agent import Agent
    from agent_schema import CandidateObject, ObjectView
    from module.detector_yoloe import YOLOEDetector
    from module.projection import TwoDToThreeDTool
    from module.segmenter import SAMSegmenter
    from read import Read
except ImportError:
    from .benchmark.utils import calc_iou, load_pc  # type: ignore
    from .agent import Agent  # type: ignore
    from .agent_schema import CandidateObject, ObjectView  # type: ignore
    from .module.detector_yoloe import YOLOEDetector  # type: ignore
    from .module.projection import TwoDToThreeDTool  # type: ignore
    from .module.segmenter import SAMSegmenter  # type: ignore
    from .read import Read  # type: ignore


SKIP_CASE_INDICES = {22, 66}


def _load_snapshots(snapshot_dir: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for path in sorted(snapshot_dir.glob("case_*.json")):
        snapshots.append(json.loads(path.read_text(encoding="utf-8")))
    return snapshots


def _build_unit_index_map(
    scene_id: str, *, max_frames: int = 10, frame_skip: int = 2
) -> dict[str, tuple[int, int]]:
    try:
        from read import Read
    except ImportError:
        from .read import Read  # type: ignore

    reader = Read(scene_id, max_frames_per_find=max_frames, frame_skip=frame_skip)
    mapping: dict[str, tuple[int, int]] = {}
    unit_index = 0
    while True:
        views = reader.find()
        if not views:
            break
        for order_in_unit, view in enumerate(views):
            mapping[str(getattr(view, "view_id", ""))] = (unit_index, order_in_unit)
        unit_index += 1
    return mapping


def _detected_prefix_object_views(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    for object_view in candidate.get("object_views", []) or []:
        if str(object_view.get("source", "")) != "detected":
            break
        prefix.append(object_view)
    return prefix


def _true_candidate_order_key(
    candidate: dict[str, Any], unit_index_map: dict[str, tuple[int, int]]
) -> tuple[int, int, int, int]:
    prefix = _detected_prefix_object_views(candidate)
    if not prefix:
        return (10**9, 10**9, 10**9, int(candidate.get("candidate_id", 10**9)))

    first_view_id = str(prefix[0].get("view_id", ""))
    first_unit_index, first_view_order = unit_index_map.get(
        first_view_id, (10**9, 10**9)
    )
    prefix_count_in_first_unit = 0
    for object_view in prefix:
        view_id = str(object_view.get("view_id", ""))
        unit_index, _ = unit_index_map.get(view_id, (10**9, 10**9))
        if unit_index == first_unit_index:
            prefix_count_in_first_unit += 1

    return (
        int(first_unit_index),
        -int(prefix_count_in_first_unit),
        int(first_view_order),
        int(candidate.get("candidate_id", 10**9)),
    )


def _select_first_true_or_original(
    snapshot: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    candidates = snapshot.get("candidates", []) or []
    true_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("status", "")) in {"true", "expanded"}
    ]
    if true_candidates:
        scene_id = str(snapshot.get("scene_id", ""))
        unit_index_map = _build_unit_index_map(scene_id)
        print(
            f"[Replay] case_id={snapshot.get('case_id')} true_candidates_before_sort="
        )
        for candidate in true_candidates:
            key = _true_candidate_order_key(candidate, unit_index_map)
            prefix = _detected_prefix_object_views(candidate)
            prefix_view_ids = [str(item.get("view_id", "")) for item in prefix]
            print(
                f"  candidate_id={candidate.get('candidate_id')} "
                f"status={candidate.get('status')} "
                f"prefix_view_ids={prefix_view_ids} "
                f"order_key={key}"
            )
        true_candidates.sort(
            key=lambda candidate: _true_candidate_order_key(candidate, unit_index_map)
        )
        print(
            f"[Replay] case_id={snapshot.get('case_id')} selected_true_candidate_id={true_candidates[0].get('candidate_id')}"
        )
        return true_candidates[0], "true"

    selected_candidate_id = snapshot.get("result", {}).get("selected_candidate_id")
    if selected_candidate_id is None:
        return None, str(snapshot.get("result", {}).get("final_decision", ""))

    for candidate in candidates:
        if int(candidate.get("candidate_id", -1)) == int(selected_candidate_id):
            return candidate, str(snapshot.get("result", {}).get("final_decision", ""))
    return None, str(snapshot.get("result", {}).get("final_decision", ""))


def _score_scanrefer_bbox(
    pred_box: np.ndarray | None, gt_bbox_3d: np.ndarray
) -> float | None:
    if pred_box is None:
        return None
    return float(calc_iou(pred_box, gt_bbox_3d))


def _score_nr3d_bbox(
    pred_box: np.ndarray | None, scene_id: str, target_id: int
) -> dict[str, Any] | None:
    if pred_box is None:
        return None
    obj_ids, _, obj_locs = load_pc(scene_id)
    scene_centers = np.asarray(obj_locs, dtype=np.float64)[:, :3]
    pred_center = pred_box[:3]
    center_distances = np.linalg.norm(scene_centers - pred_center, axis=1)
    nearest_index = int(np.argmin(center_distances))
    nearest_target_id = int(obj_ids[nearest_index])
    min_center_distance = float(center_distances[nearest_index])
    acc = int(nearest_target_id == target_id)
    return {
        "iou": "",
        "nearest_target_id": nearest_target_id,
        "min_center_distance": min_center_distance,
        "acc": acc,
        "acc_tf": "T" if acc else "F",
    }


def _trim_object_views_for_replay(
    candidate_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    seen_non_detect = False
    for object_view in candidate_snapshot.get("object_views", []) or []:
        source = str(object_view.get("source", ""))
        if source == "turn_around":
            continue
        if not seen_non_detect:
            if source == "detected":
                trimmed.append(object_view)
                continue
            seen_non_detect = True
            trimmed.append(object_view)
            continue
        if source == "detected":
            break
        trimmed.append(object_view)
    return trimmed


def _rebuild_candidate_bbox_3d(
    snapshot: dict[str, Any],
    candidate_snapshot: dict[str, Any],
) -> np.ndarray | None:
    scene_id = str(snapshot.get("scene_id", ""))
    query = str(snapshot.get("query", ""))
    query_analysis = snapshot.get("query_analysis")
    if not isinstance(query_analysis, dict):
        query_analysis = None

    reader = Read(scene_id, max_frames_per_find=10, frame_skip=2)
    detector = YOLOEDetector(model="yoloe-11l-seg.pt")
    segmenter = SAMSegmenter(
        checkpoint_path="checkpoints/SAM/sam_vit_h_4b8939.pth",
        model_type="vit_h",
        device="cpu",
    )
    agent = Agent(
        motion=reader,
        detector=detector,
        segmenter=segmenter,
        mapper_2d3d=TwoDToThreeDTool(),
        intrinsic_matrix=reader.intrinsic_matrix,
        world_to_axis_align_matrix=reader.world_to_axis_align_matrix,
        debug=False,
    )
    agent.reset(query, parsed_query=query_analysis)

    trimmed_views = _trim_object_views_for_replay(candidate_snapshot)
    if not trimmed_views:
        return None

    candidate = CandidateObject(
        object_id=int(candidate_snapshot.get("candidate_id", 0)),
        label=str(candidate_snapshot.get("label", "object")),
        status=str(candidate_snapshot.get("status", "")),
        best_id=int(candidate_snapshot.get("best_id", 0)),
        object_view=[],
    )

    for object_view_data in trimmed_views:
        view = reader._build_view(str(object_view_data["view_id"]))
        object_view = ObjectView(
            object_id=str(object_view_data.get("object_view_id", view.view_id)),
            label=str(candidate.label),
            score=float(object_view_data.get("score", 1.0)),
            view=view,
            bbox_2d=np.asarray(object_view_data["bbox_2d"], dtype=np.float32),
            mask_2d=None,
            points_3d=None,
            status=str(object_view_data.get("status", "active")),
            source=str(object_view_data.get("source", "detected")),
        )
        candidate.object_view.append(object_view)

    if not candidate.object_view:
        return None
    if candidate.best_id >= len(candidate.object_view):
        candidate.best_id = 0

    agent.complete_candidate_masks(candidate)
    _, bbox_3d = agent.map_candidate_to_3d(candidate)
    return np.asarray(bbox_3d, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay evaluation from saved candidate snapshots using first-true selection."
    )
    parser.add_argument(
        "--snapshot-dir",
        default="output/candidate_snapshots_multiselect_nonfalse",
        help="Directory of case_XXXX.json snapshots",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["scanrefer", "nr3d"],
        default="scanrefer",
        help="Scoring mode",
    )
    parser.add_argument(
        "--output-csv",
        default="output/replay_first_true_ious.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    snapshots = _load_snapshots(snapshot_dir)

    correct_25 = 0
    correct_50 = 0
    unique_25 = 0
    unique_50 = 0
    correct_easy_25 = 0
    correct_hard_25 = 0
    correct_dep_25 = 0
    correct_indep_25 = 0
    total = 0
    unique_total = 0
    easy_total = 0
    hard_total = 0
    dep_total = 0
    indep_total = 0
    except_total = 0
    eps = 1e-6
    rows: list[dict[str, Any]] = []

    for snapshot in snapshots:
        case_id = int(snapshot["case_id"])
        total += 1
        if case_id in SKIP_CASE_INDICES:
            rows.append({"case_id": case_id, "iou": ""})
            continue

        gt = snapshot.get("gt", {})
        gt_bbox_3d = np.asarray(gt.get("gt_bbox_3d", []), dtype=np.float64)
        unique = bool(gt.get("unique", False))
        if unique:
            unique_total += 1
        is_easy = bool(gt.get("easy", False))
        is_dep = bool(gt.get("view_dep", False))
        if args.eval_mode == "nr3d":
            if is_easy:
                easy_total += 1
            else:
                hard_total += 1
            if is_dep:
                dep_total += 1
            else:
                indep_total += 1

        selected_candidate, final_decision = _select_first_true_or_original(snapshot)
        original_selected_candidate_id = snapshot.get("result", {}).get(
            "selected_candidate_id"
        )

        if selected_candidate is None:
            except_total += 1
            rows.append({"case_id": case_id, "iou": ""})
            continue

        selected_candidate_id = int(selected_candidate.get("candidate_id", -1))
        same_as_original = (
            original_selected_candidate_id is not None
            and selected_candidate_id == int(original_selected_candidate_id)
        )

        reused_bbox_3d = None
        if same_as_original:
            bbox_3d = selected_candidate.get("bbox_3d")
            if bbox_3d is not None:
                reused_bbox_3d = np.asarray(bbox_3d, dtype=np.float64)

        rebuilt_bbox_3d = reused_bbox_3d
        if rebuilt_bbox_3d is None:
            rebuilt_bbox_3d = _rebuild_candidate_bbox_3d(snapshot, selected_candidate)

        if args.eval_mode == "nr3d":
            scene_id = str(snapshot.get("scene_id", ""))
            target_id = int(gt.get("target_id", -1))
            scored = _score_nr3d_bbox(rebuilt_bbox_3d, scene_id, target_id)
            if scored is None:
                except_total += 1
                rows.append({"case_id": case_id, "iou": ""})
                continue
            row = {"case_id": case_id}
            row.update(scored)
            rows.append(row)
            if int(scored["acc"]):
                correct_25 += 1
                if unique:
                    unique_25 += 1
                if is_easy:
                    correct_easy_25 += 1
                else:
                    correct_hard_25 += 1
                if is_dep:
                    correct_dep_25 += 1
                else:
                    correct_indep_25 += 1
        else:
            iou = _score_scanrefer_bbox(rebuilt_bbox_3d, gt_bbox_3d)
            if iou is None:
                except_total += 1
                rows.append({"case_id": case_id, "iou": ""})
                continue
            rows.append({"case_id": case_id, "iou": round(iou, 6)})
            if iou >= 0.25:
                correct_25 += 1
                if unique:
                    unique_25 += 1
            if iou >= 0.5:
                correct_50 += 1
                if unique:
                    unique_50 += 1

    out_path = Path(args.output_csv)
    fieldnames = ["case_id", "iou"]
    if args.eval_mode == "nr3d":
        fieldnames = [
            "case_id",
            "iou",
            "nearest_target_id",
            "min_center_distance",
            "acc",
            "acc_tf",
        ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    accuracy_msgs = [
        (
            "Acc: {:.3f}".format(correct_25 / total)
            if args.eval_mode == "nr3d"
            else "Overall@25: {:.3f}".format(correct_25 / total)
        ),
        (
            "Overall@50: {:.3f}".format(correct_50 / total)
            if args.eval_mode != "nr3d"
            else "Overall@50(obs): {:.3f}".format(correct_50 / total)
        ),
    ]
    if args.eval_mode == "nr3d":
        accuracy_msgs.extend(
            [
                "EasyAcc: {:.3f}".format(correct_easy_25 / (easy_total + eps)),
                "HardAcc: {:.3f}".format(correct_hard_25 / (hard_total + eps)),
                "ViewDepAcc: {:.3f}".format(correct_dep_25 / (dep_total + eps)),
                "ViewIndepAcc: {:.3f}".format(correct_indep_25 / (indep_total + eps)),
                "EasyAcc Ratio: {} / {}".format(correct_easy_25, easy_total),
                "HardAcc Ratio: {} / {}".format(correct_hard_25, hard_total),
                "ViewDepAcc Ratio: {} / {}".format(correct_dep_25, dep_total),
                "ViewIndepAcc Ratio: {} / {}".format(correct_indep_25, indep_total),
            ]
        )
    else:
        accuracy_msgs.extend(
            [
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
            ]
        )
    accuracy_msgs.extend(
        [
            "Except Ratio: {} / {}".format(except_total, total),
            f"saved_replay_csv={out_path}",
            "",
        ]
    )
    print("\n".join(accuracy_msgs))


if __name__ == "__main__":
    main()
