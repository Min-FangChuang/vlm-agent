from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .agent_modules import GroundingDetection, GroundingDinoDetector, PATSMatcher
    from .agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View
    from .motion import Motion
    from .prompt import build_candidate_judgement_prompt
except ImportError:
    from agent_modules import GroundingDetection, GroundingDinoDetector, PATSMatcher  # type: ignore
    from agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View  # type: ignore
    from motion import Motion  # type: ignore
    from prompt import build_candidate_judgement_prompt  # type: ignore
class Agent:
    def __init__(
        self,
        motion: Motion,
        detector: GroundingDinoDetector | None = None,
        matcher: PATSMatcher | None = None,
        mapper_2d3d: Any = None,
        view_selector: Any = None,
    ) -> None:
        self.detector = detector or GroundingDinoDetector()
        self.matcher = matcher or PATSMatcher()
        self.mapper_2d3d = mapper_2d3d
        self.view_selector = view_selector
        self.motion = motion
        self.current_view: View | None = None
        self.query: Query | None = None
        self.candidates = CandidateMemory()

    def vlm(self, prompt: str, **_: Any) -> Any:
        del prompt
        return None

    def reset(self, query_text: str) -> None:
        self.query = Query(query_text)
        self.current_view = None
        self.candidates = CandidateMemory()

    def _observation_to_view(self, observation: dict[str, Any]) -> View:
        view_id = observation.get("frame_index")
        if not isinstance(view_id, (str, int)):
            view_id = -1
        return View(
            rgb=observation["rgb"],
            depth=observation["depth"],
            camera_to_world=observation["camera_to_world"],
            view_id=view_id,
        )

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
            self.candidates.add_ObjectView(object_view, self.matcher.is_same_candidate)

    def _normalize_vlm_decision(self, result: Any) -> str:
        if isinstance(result, bool):
            return "true" if result else "false"
        if isinstance(result, str):
            lowered = result.strip().lower()
            if lowered in {"true", "false", "unsure"}:
                return lowered
        if isinstance(result, dict):
            decision = result.get("decision") or result.get("answer") or result.get("result")
            if isinstance(decision, str):
                lowered = decision.strip().lower()
                if lowered in {"true", "false", "unsure"}:
                    return lowered
        return "unsure"

    def evaluate_candidate(self, candidate: CandidateObject) -> str:
        prompt = build_candidate_judgement_prompt(self._require_query(), candidate)
        result = self.vlm(prompt, candidate=candidate)
        return self._normalize_vlm_decision(result)

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
