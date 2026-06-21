from __future__ import annotations

import json
from typing import Any

import numpy as np

MIN_BBOX_AREA_PX = 400.0
MIN_BBOX_AREA_RATIO = 0.02
MAX_BBOX_ASPECT_RATIO = 6.0
MIN_EXCLUSIVE_AREA_RATIO = 0.5
ACTIVE_EXCLUSIVE_AREA_RATIO = 0.8
EDGE_TOUCH_MARGIN_PX = 2.0

try:
    from .module.detector import GroundingDetection, YOLOWorldDetector
    from .agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View
    from .module.matcher import PATSMatcher
    from .motion import Motion
    from .prompt import build_candidate_judgement_prompt
    from .vlm_api_bridge import call_vlm_api_messages
    from .vlm_bridge import call_vlm_messages
except ImportError:
    from module.detector import GroundingDetection, YOLOWorldDetector  # type: ignore
    from agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View  # type: ignore
    from module.matcher import PATSMatcher  # type: ignore
    from motion import Motion  # type: ignore
    from prompt import build_candidate_judgement_prompt  # type: ignore
    from vlm_bridge import call_vlm_messages  # type: ignore
    from vlm_api_bridge import call_vlm_api_messages  # type: ignore


class Agent:
    def __init__(
        self,
        motion: Motion | Any,
        detector: YOLOWorldDetector | None = None,
        segmenter: Any = None,
        matcher: PATSMatcher | None = None,
        mapper_2d3d: Any = None,
        intrinsic_matrix: Any = None,
        world_to_axis_align_matrix: Any = None,
        view_selector: Any = None,
        debug: bool = True,
    ) -> None:
        self.detector = detector or YOLOWorldDetector()
        self.segmenter = segmenter
        self.matcher = matcher or PATSMatcher()
        self.mapper_2d3d = mapper_2d3d
        self.intrinsic_matrix = intrinsic_matrix
        self.world_to_axis_align_matrix = world_to_axis_align_matrix
        self.view_selector = view_selector
        self.motion = motion
        self.debug = debug
        self.current_view: View | None = None
        self.query: Query | None = None
        self.candidates = CandidateMemory()
        self.detector_call_count = 0
        self.vlm_image_counts: list[int] = []

    def vlm(self, prompt, **_: Any) -> Any:
        # OpenAI API route manual switch:
        # return call_vlm_api_messages(prompt)
        return call_vlm_messages(prompt)

    def reset(self, query_text: str) -> None:
        self.query = Query(query_text)
        self.current_view = None
        self.candidates = CandidateMemory()
        self.detector_call_count = 0
        self.vlm_image_counts = []

    def observe(self) -> View:
        return self.motion._current_view()

    def _require_query(self) -> Query:
        if self.query is None:
            raise ValueError(
                "Agent query is not initialized. Call `reset(query_text)` first."
            )
        return self.query

    def detect_target_objects(self, view: View) -> list[GroundingDetection]:
        query = self._require_query()
        self.detector_call_count += 1
        return self.detector.detect_detections(view.rgb, query.target_object)

    def detect_reference_objects(self, view: View) -> list[GroundingDetection]:
        query = self._require_query()
        if not query.reference_object:
            return []
        self.detector_call_count += 1
        return self.detector.detect_detections(view.rgb, query.reference_object)

    @staticmethod
    def _count_prompt_images(prompt: Any) -> int:
        if not isinstance(prompt, list):
            return 0
        total = 0
        for message in prompt:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    total += 1
        return total

    def attach_reference(self, view: View) -> None:
        view.reference = self.detect_reference_objects(view)

    def build_object_view(
        self, view: View, detection: GroundingDetection, object_id: str | int
    ) -> ObjectView:
        bbox = np.asarray(detection.bbox, dtype=np.float32).reshape(4)
        return ObjectView(
            object_id=object_id,
            label=detection.label,
            score=float(detection.score),
            view=view,
            bbox_2d=bbox,
            mask_2d=None,
            points_3d=None,
        )

    @staticmethod
    def _build_object_view_status(
        detections: list[GroundingDetection], keep_index: int
    ) -> str:
        bbox = np.asarray(detections[keep_index].bbox, dtype=np.float32).reshape(4)
        x1, y1, x2, y2 = bbox.tolist()
        bbox_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        overlap_area = 0.0
        for index, detection in enumerate(detections):
            if index == keep_index:
                continue
            other_bbox = np.asarray(detection.bbox, dtype=np.float32).reshape(4)
            ox1, oy1, ox2, oy2 = other_bbox.tolist()
            inter_w = max(0.0, float(min(x2, ox2) - max(x1, ox1)))
            inter_h = max(0.0, float(min(y2, oy2) - max(y1, oy1)))
            overlap_area += inter_w * inter_h
        exclusive_ratio = max(0.0, bbox_area - overlap_area) / max(bbox_area, 1.0)
        return (
            "support_only"
            if exclusive_ratio < ACTIVE_EXCLUSIVE_AREA_RATIO
            else "active"
        )

    @staticmethod
    def _other_detection_bboxes(
        detections: list[GroundingDetection],
        keep_index: int,
    ) -> list[np.ndarray]:
        others: list[np.ndarray] = []
        for index, detection in enumerate(detections):
            if index == keep_index:
                continue
            others.append(np.asarray(detection.bbox, dtype=np.float32).reshape(4))
        return others

    def _filter_detections_for_object_views(
        self,
        view: View,
        detections: list[GroundingDetection],
    ) -> list[tuple[int, GroundingDetection]]:
        image_shape = tuple(np.asarray(view.rgb).shape[:2])
        boxes = [
            np.asarray(detection.bbox, dtype=np.float32).reshape(4)
            for detection in detections
        ]
        kept: list[tuple[int, GroundingDetection]] = []
        for index, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox.tolist()
            width = max(0.0, float(x2 - x1))
            height = max(0.0, float(y2 - y1))
            bbox_area = width * height
            overlap_area = 0.0
            max_iou = 0.0
            for other_index, other_bbox in enumerate(boxes):
                if index == other_index:
                    continue
                ox1, oy1, ox2, oy2 = other_bbox.tolist()
                inter_w = max(0.0, float(min(x2, ox2) - max(x1, ox1)))
                inter_h = max(0.0, float(min(y2, oy2) - max(y1, oy1)))
                inter_area = inter_w * inter_h
                overlap_area += inter_area
                other_area = max(0.0, float(ox2 - ox1)) * max(0.0, float(oy2 - oy1))
                union_area = max(bbox_area + other_area - inter_area, 1.0)
                max_iou = max(max_iou, inter_area / union_area)

            exclusive_area = max(0.0, bbox_area - overlap_area)
            exclusive_ratio = exclusive_area / max(bbox_area, 1.0)

            bbox_area_ratio = bbox_area / max(
                float(image_shape[0] * image_shape[1]), 1.0
            )
            short_side = max(min(width, height), 1.0)
            aspect_ratio = max(width, height) / short_side
            edge_touch_count = (
                int(x1 <= EDGE_TOUCH_MARGIN_PX)
                + int(y1 <= EDGE_TOUCH_MARGIN_PX)
                + int(x2 >= image_shape[1] - EDGE_TOUCH_MARGIN_PX)
                + int(y2 >= image_shape[0] - EDGE_TOUCH_MARGIN_PX)
            )

            keep = (
                bbox_area >= MIN_BBOX_AREA_PX
                and bbox_area_ratio >= MIN_BBOX_AREA_RATIO
                and aspect_ratio <= MAX_BBOX_ASPECT_RATIO
                and exclusive_ratio >= MIN_EXCLUSIVE_AREA_RATIO
                and edge_touch_count <= 1
            )
            if keep:
                kept.append((index, detections[index]))
        return kept

    def collect_view_object_views(
        self, view: View, detections: list[GroundingDetection]
    ) -> list[ObjectView]:
        object_views: list[ObjectView] = []
        # filtered_detections = self._filter_detections_for_object_views(view, detections)
        filtered_detections = list(enumerate(detections))
        for detection_index, detection in filtered_detections:
            object_view = self.build_object_view(
                view, detection, f"{view.view_id}_{detection_index}"
            )
            object_view.status = self._build_object_view_status(
                detections, detection_index
            )
            object_views.append(object_view)
        return object_views

    def update_candidates_for_view(
        self,
        object_views: list[ObjectView],
        detections: list[GroundingDetection],
    ) -> None:
        for object_view in object_views:
            current_object_id = str(object_view.object_id)
            current_detection_index = (
                int(current_object_id.rsplit("_", 1)[1])
                if "_" in current_object_id
                else -1
            )
            candidate, _ = self.candidates.add_ObjectView(
                object_view,
                lambda incoming_object_view, candidate_obj: (
                    self.matcher.match_object_view_to_candidate(
                        incoming_object_view,
                        candidate_obj,
                    )
                ),
            )
            self.ensure_candidate_best_view_mask(candidate)

    def consume_view(self, view: View) -> None:
        detections = self.detect_target_objects(view)
        self.attach_reference(view)
        if not detections:
            return
        object_views = self.collect_view_object_views(view, detections)
        self.update_candidates_for_view(object_views, detections)

    def _normalize_vlm_decision(self, result: Any) -> str:
        if isinstance(result, bool):
            return "true" if result else "false"
        if isinstance(result, dict):
            decision = (
                result.get("decision") or result.get("answer") or result.get("result")
            )
            if isinstance(decision, str):
                lowered = decision.strip().lower()
                if lowered in {"true", "false", "unsure"}:
                    return lowered
        return "unsure"

    def _normalize_vlm_result(self, result: Any) -> Any:
        if isinstance(result, dict):
            return result

        text = "" if result is None else str(result).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "decision": "unsure",
                "confidence": "low",
                "reasoning": text or "Model returned non-JSON output.",
                "matched_conditions": [],
                "missing_conditions": [],
                "suggested_action": "yaw",
            }

    def _debug_print(self, title: str, payload: Any) -> None:
        if not self.debug or title not in {"vlm_raw_result", "vlm_normalized_decision"}:
            return
        print(f"[Agent] {title}")
        if isinstance(payload, (dict, list)):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload)

    def evaluate_candidate(self, candidate: CandidateObject) -> str:
        object_views = getattr(candidate, "object_view", []) or []
        if len(object_views) <= 0:
            print(f"[Agent] skip_vlm_candidate_views={len(object_views)}")
            return "unsure"

        prompt = build_candidate_judgement_prompt(self._require_query(), candidate)
        image_count = self._count_prompt_images(prompt)
        self.vlm_image_counts.append(image_count)
        print(f"[Agent] vlm_stitched_image_count={image_count}")
        result = self._normalize_vlm_result(self.vlm(prompt, candidate=candidate))
        self._debug_print("vlm_raw_result", result)
        decision = self._normalize_vlm_decision(result)
        self._debug_print("vlm_normalized_decision", decision)
        return decision

    def complete_candidate_masks(self, candidate: CandidateObject) -> None:
        if self.segmenter is None:
            raise ValueError("segmenter is not configured.")

        active_object_views = [
            object_view
            for object_view in candidate.object_view
            if getattr(object_view, "status", "active") == "active"
        ]
        if len(active_object_views) < 2:
            ranked_object_views = sorted(
                candidate.object_view,
                key=lambda object_view: (
                    max(
                        0.0,
                        float(
                            np.asarray(object_view.bbox_2d, dtype=np.float32).reshape(
                                4
                            )[2]
                            - np.asarray(object_view.bbox_2d, dtype=np.float32).reshape(
                                4
                            )[0]
                        ),
                    )
                    * max(
                        0.0,
                        float(
                            np.asarray(object_view.bbox_2d, dtype=np.float32).reshape(
                                4
                            )[3]
                            - np.asarray(object_view.bbox_2d, dtype=np.float32).reshape(
                                4
                            )[1]
                        ),
                    )
                ),
                reverse=True,
            )
            for object_view in ranked_object_views[:2]:
                object_view.status = "active"
            active_object_views = [
                object_view
                for object_view in candidate.object_view
                if getattr(object_view, "status", "active") == "active"
            ]

        for object_view in active_object_views:
            if object_view.mask_2d is not None:
                continue

            mask = self.segmenter.segment_from_box(
                object_view.view.rgb,
                np.asarray(object_view.bbox_2d, dtype=np.float32).reshape(4),
            )
            object_view.mask_2d = np.asarray(mask, dtype=np.uint8)

    def ensure_candidate_best_view_mask(self, candidate: CandidateObject) -> None:
        if self.segmenter is None:
            return
        if not candidate.object_view:
            return

        best_id = int(candidate.best_id)
        if best_id < 0 or best_id >= len(candidate.object_view):
            return

        best_object_view = candidate.object_view[best_id]
        if best_object_view.mask_2d is not None:
            return

        mask = self.segmenter.segment_from_box(
            best_object_view.view.rgb,
            np.asarray(best_object_view.bbox_2d, dtype=np.float32).reshape(4),
        )
        best_object_view.mask_2d = np.asarray(mask, dtype=np.uint8)

    def map_candidate_to_3d(
        self,
        candidate: CandidateObject,
        *,
        world_to_axis_align_matrix: Any = None,
        do_post_process: bool = True,
        use_best_only: bool = False,
    ) -> tuple[Any, Any]:
        if self.mapper_2d3d is None:
            raise ValueError("mapper_2d3d is not configured.")
        if self.intrinsic_matrix is None:
            raise ValueError("intrinsic_matrix is not configured.")
        align_matrix = world_to_axis_align_matrix
        if align_matrix is None:
            align_matrix = self.world_to_axis_align_matrix
        return self.mapper_2d3d.update_candidate_3d(
            candidate,
            intrinsic_matrix=self.intrinsic_matrix,
            world_to_axis_align_matrix=align_matrix,
            do_post_process=do_post_process,
            use_best_only=use_best_only,
        )

    def evaluate_candidates(self) -> tuple[CandidateObject | None, str]:
        saw_unsure = False
        for candidate in self.candidates.values():
            if getattr(candidate, "status", "active") != "active":
                continue
            decision = self.evaluate_candidate(candidate)
            if decision == "true":
                return candidate, "true"
            if decision == "unsure":
                saw_unsure = True
            if decision == "false":
                candidate.status = "false"
        return None, "unsure" if saw_unsure else "false"

    def initial_scan(self) -> list[View]:
        return self.motion.look_around()

    def consume_views(self, views: list[View]) -> None:
        if views:
            self.current_view = views[-1]
        for view in views:
            self.consume_view(view)

    def select_fallback_motion(self, decision: str):
        if decision == "unsure":
            return self.motion.yaw
        return self.motion.forward

    def step(self) -> CandidateObject | list[View] | None:
        views = self.initial_scan()
        self.consume_views(views)

        candidate, decision = self.evaluate_candidates()
        if candidate is not None:
            return candidate

        next_motion = self.select_fallback_motion(decision)
        next_views = next_motion()
        self.consume_views(next_views)
        return next_views
