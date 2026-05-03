from __future__ import annotations

import json
from typing import Any

import numpy as np

try:
    from .module.detector import GroundingDetection, YOLOWorldDetector
    from .agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View
    from .module.matcher import PATSMatcher
    from .motion import Motion
    from .prompt import build_candidate_judgement_prompt
    from .vlm_bridge import call_vlm_messages
except ImportError:
    from module.detector import GroundingDetection, YOLOWorldDetector  # type: ignore
    from agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View  # type: ignore
    from module.matcher import PATSMatcher  # type: ignore
    from motion import Motion  # type: ignore
    from prompt import build_candidate_judgement_prompt  # type: ignore
    from vlm_bridge import call_vlm_messages 
class Agent:
    def __init__(
        self,
        motion: Motion | Any,
        detector: YOLOWorldDetector | None = None,
        segmenter: Any = None,
        matcher: PATSMatcher | None = None,
        mapper_2d3d: Any = None,
        intrinsic_matrix: Any = None,
        view_selector: Any = None,
        debug: bool = True,
    ) -> None:
        self.detector = detector or YOLOWorldDetector()
        self.segmenter = segmenter
        self.matcher = matcher or PATSMatcher()
        self.mapper_2d3d = mapper_2d3d
        self.intrinsic_matrix = intrinsic_matrix
        self.view_selector = view_selector
        self.motion = motion
        self.debug = debug
        self.current_view: View | None = None
        self.query: Query | None = None
        self.candidates = CandidateMemory()

    def vlm(self, prompt, **_: Any) -> Any:
        return call_vlm_messages(prompt)

    def reset(self, query_text: str) -> None:
        self.query = Query(query_text)
        self.current_view = None
        self.candidates = CandidateMemory()

    def observe(self) -> View:
        return self.motion._current_view()

    def _require_query(self) -> Query:
        if self.query is None:
            raise ValueError("Agent query is not initialized. Call `reset(query_text)` first.")
        return self.query

    def detect_target_objects(self, view: View) -> list[GroundingDetection]:
        query = self._require_query()
        return self.detector.detect_detections(view.rgb, query.target_object)

    def detect_reference_objects(self, view: View) -> list[GroundingDetection]:
        query = self._require_query()
        if not query.reference_object:
            return []
        return self.detector.detect_detections(view.rgb, query.reference_object)

    def attach_reference(self, view: View) -> None:
        view.reference = self.detect_reference_objects(view)

    def build_object_view(self, view: View, detection: GroundingDetection, object_id: str | int) -> ObjectView:
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

    def collect_object_views(self, views: list[View]) -> list[ObjectView]:
        object_views: list[ObjectView] = []
        for view in views:
            detections = self.detect_target_objects(view)
            if not detections:
                continue
            self.attach_reference(view)
            for index, detection in enumerate(detections):
                object_views.append(self.build_object_view(view, detection, f"{view.view_id}_{index}"))
        return object_views

    def update_candidates(self, object_views: list[ObjectView]) -> None:
        for object_view in object_views:
            candidate, _ = self.candidates.add_ObjectView(object_view, self.matcher.match_object_view_to_candidate)
            self.ensure_candidate_best_view_mask(candidate)

    def _normalize_vlm_decision(self, result: Any) -> str:
        if isinstance(result, bool):
            return "true" if result else "false"
        if isinstance(result, dict):
            decision = result.get("decision") or result.get("answer") or result.get("result")
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
        prompt = build_candidate_judgement_prompt(self._require_query(), candidate)
        result = self._normalize_vlm_result(self.vlm(prompt, candidate=candidate))
        self._debug_print("vlm_raw_result", result)
        decision = self._normalize_vlm_decision(result)
        self._debug_print("vlm_normalized_decision", decision)
        return decision

    def complete_candidate_masks(self, candidate: CandidateObject) -> None:
        if self.segmenter is None:
            raise ValueError("segmenter is not configured.")

        for object_view in candidate.object_view:
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
        return self.mapper_2d3d.update_candidate_3d(
            candidate,
            intrinsic_matrix=self.intrinsic_matrix,
            world_to_axis_align_matrix=world_to_axis_align_matrix,
            do_post_process=do_post_process,
            use_best_only=use_best_only,
        )

    def evaluate_candidates(self) -> tuple[CandidateObject | None, str]:
        saw_unsure = False
        for candidate in self.candidates.values():
            decision = self.evaluate_candidate(candidate)
            if decision == "true":
                return candidate, "true"
            if decision == "unsure":
                saw_unsure = True
        return None, "unsure" if saw_unsure else "false"

    def initial_scan(self) -> list[View]:
        return self.motion.look_around()

    def consume_views(self, views: list[View]) -> None:
        if views:
            self.current_view = views[-1]
        object_views = self.collect_object_views(views)
        self.update_candidates(object_views)

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
