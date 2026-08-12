from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .module.detector_yoloe import GroundingDetection, YOLOEDetector, draw_bbox
    from .agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View
    from .module.matcher import PATSMatcher, ObjectViewMatchResult
    from .motion import Motion
    from .prompt import build_candidate_judgement_prompt
    from .read.scannet_more_view import project_bbox3d_to_view
    from .vlm_api_bridge import call_vlm_api_messages
    from .vlm_bridge import call_vlm_messages
except ImportError:
    from module.detector_yoloe import GroundingDetection, YOLOEDetector, draw_bbox  # type: ignore
    from agent_schema import CandidateMemory, CandidateObject, ObjectView, Query, View  # type: ignore
    from module.matcher import PATSMatcher, ObjectViewMatchResult  # type: ignore
    from motion import Motion  # type: ignore
    from prompt import build_candidate_judgement_prompt  # type: ignore
    from read.scannet_more_view import project_bbox3d_to_view  # type: ignore
    from vlm_bridge import call_vlm_messages  # type: ignore
    from vlm_api_bridge import call_vlm_api_messages  # type: ignore


class Agent:
    def __init__(
        self,
        motion: Motion | Any,
        detector: Any = None,
        segmenter: Any = None,
        matcher: PATSMatcher | None = None,
        mapper_2d3d: Any = None,
        intrinsic_matrix: Any = None,
        world_to_axis_align_matrix: Any = None,
        view_selector: Any = None,
        debug: bool = True,
    ) -> None:
        self.detector = detector or YOLOEDetector()
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
        self.max_verification_rounds = 2
        self.preapply_turn_around_before_verify = True
        self.bbox3d_precheck_center_distance_threshold = 200.0

    def vlm(self, prompt, **_: Any) -> Any:
        # OpenAI API route manual switch:
        # return call_vlm_api_messages(prompt)
        return call_vlm_messages(prompt)

    def reset(
        self, query_text: str, parsed_query: dict[str, Any] | None = None
    ) -> None:
        self.query = Query(query_text, parsed=parsed_query)
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

    # def detect_reference_objects(self, view: View) -> list[GroundingDetection]:
    #     query = self._require_query()
    #     if not query.reference_object:
    #         return []
    #     self.detector_call_count += 1
    #     if hasattr(self.detector, "set_view_context"):
    #         scene_id = getattr(self.motion, "scene_name", None)
    #         self.detector.set_view_context(
    #             scene_id="" if scene_id is None else str(scene_id),
    #             view_id=str(getattr(view, "view_id", "")),
    #         )
    #     return self.detector.detect_detections(view.rgb, query.reference_object)

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

    # def attach_reference(self, view: View) -> None:
    #     view.reference = self.detect_reference_objects(view)

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
            source="detected",
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

    def collect_view_object_views(
        self, view: View, detections: list[GroundingDetection]
    ) -> list[ObjectView]:
        object_views: list[ObjectView] = []
        for detection_index, detection in enumerate(detections):
            view.reference = self._other_detection_bboxes(detections, detection_index)
            object_view = self.build_object_view(
                view, detection, f"{view.view_id}_{detection_index}"
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

            def _match_with_optional_precheck(
                incoming_object_view: ObjectView,
                candidate_obj: CandidateObject,
            ) -> ObjectViewMatchResult:
                if getattr(candidate_obj, "bbox_3d", None) is None:
                    return self.matcher.match_object_view_to_candidate(
                        incoming_object_view,
                        candidate_obj,
                    )

                if self.intrinsic_matrix is not None:
                    best_view = candidate_obj.object_view[int(candidate_obj.best_id)]
                    incoming_bbox = np.asarray(
                        incoming_object_view.bbox_2d, dtype=np.float32
                    ).reshape(4)
                    projected_bbox = project_bbox3d_to_view(
                        candidate_obj.bbox_3d,
                        view=incoming_object_view.view,
                        intrinsic_matrix=np.asarray(
                            self.intrinsic_matrix, dtype=np.float64
                        ),
                        world_to_axis_align_matrix=None
                        if self.world_to_axis_align_matrix is None
                        else np.asarray(
                            self.world_to_axis_align_matrix, dtype=np.float64
                        ),
                    )
                    if projected_bbox is None:
                        return ObjectViewMatchResult(
                            total_matches=0,
                            num_bbox_matches=0,
                            num_mask_matches=0,
                            num_filtered_matches=0,
                            mask_back_project_coverage=0.0,
                            mask_back_project_support_ratio=0.0,
                            is_match=False,
                        )

                    projected_center = np.asarray(
                        [
                            (projected_bbox[0] + projected_bbox[2]) / 2.0,
                            (projected_bbox[1] + projected_bbox[3]) / 2.0,
                        ],
                        dtype=np.float32,
                    )
                    incoming_center = np.asarray(
                        [
                            (incoming_bbox[0] + incoming_bbox[2]) / 2.0,
                            (incoming_bbox[1] + incoming_bbox[3]) / 2.0,
                        ],
                        dtype=np.float32,
                    )
                    center_distance = float(
                        np.linalg.norm(projected_center - incoming_center)
                    )
                    overlap_x = min(projected_bbox[2], incoming_bbox[2]) - max(
                        projected_bbox[0], incoming_bbox[0]
                    )
                    overlap_y = min(projected_bbox[3], incoming_bbox[3]) - max(
                        projected_bbox[1], incoming_bbox[1]
                    )
                    has_overlap = overlap_x > 0 and overlap_y > 0
                    if (
                        not has_overlap
                        and center_distance
                        > self.bbox3d_precheck_center_distance_threshold
                    ):
                        return ObjectViewMatchResult(
                            total_matches=0,
                            num_bbox_matches=0,
                            num_mask_matches=0,
                            num_filtered_matches=0,
                            mask_back_project_coverage=0.0,
                            mask_back_project_support_ratio=0.0,
                            is_match=False,
                        )

                return self.matcher.match_object_view_to_candidate(
                    incoming_object_view,
                    candidate_obj,
                )

            candidate, _ = self.candidates.add_ObjectView(
                object_view,
                _match_with_optional_precheck,
            )
            self.ensure_candidate_best_view_mask(candidate)
            if (
                self.mapper_2d3d is not None
                and self.intrinsic_matrix is not None
                and getattr(candidate, "object_view", None)
            ):
                try:
                    best_id = int(getattr(candidate, "best_id", 0))
                    best_object_view = candidate.object_view[best_id]
                    if best_object_view.mask_2d is None:
                        continue
                    projection_input = (
                        self.mapper_2d3d.build_projection_input_from_object_view(
                            best_object_view,
                            intrinsic_matrix=np.asarray(
                                self.intrinsic_matrix, dtype=np.float64
                            ),
                            world_to_axis_align_matrix=None
                            if self.world_to_axis_align_matrix is None
                            else np.asarray(
                                self.world_to_axis_align_matrix, dtype=np.float64
                            ),
                            project_color=self.mapper_2d3d.project_color,
                        )
                    )
                    points_3d = self.mapper_2d3d.project_mask_to_3d(projection_input)
                    bbox_3d = self.mapper_2d3d.calculate_aabb(points_3d)
                    candidate.points_3d = points_3d
                    candidate.bbox_3d = bbox_3d
                except Exception:
                    pass

    def consume_view(self, view: View) -> None:
        detections = self.detect_target_objects(view)
        # self.attach_reference(view)
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

    def select_more_view_mode(
        self,
        decision: str,
        suggested_action: str | None = None,
    ) -> str:
        if decision != "unsure":
            return "forward"
        normalized_action = (suggested_action or "").strip().lower()
        if normalized_action == "yaw":
            return "yaw"
        if normalized_action == "backward":
            return "backward"
        if normalized_action == "turn_around":
            return "turn_around"
        return "forward"

    def complete_candidate_with_more_views_if_needed(
        self,
        candidate: CandidateObject,
        decision: str,
        suggested_action: str | None = None,
    ) -> CandidateObject:
        if decision not in {"new", "true", "unsure"}:
            return candidate
        action_mode = self.select_more_view_mode(decision, suggested_action)
        if action_mode and action_mode not in getattr(candidate, "done_actions", []):
            candidate.done_actions.append(action_mode)
        if not hasattr(self.motion, "complete_candidate_with_more_views"):
            return candidate
        return self.motion.complete_candidate_with_more_views(
            self,
            candidate,
            action_mode=action_mode,
        )

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

    def _extract_suggested_action(self, result: Any) -> str | None:
        if isinstance(result, dict):
            suggested_action = result.get("suggested_action")
            if isinstance(suggested_action, str) and suggested_action.strip():
                return suggested_action.strip().lower()
        return None

    def _extract_missing_conditions(self, result: Any) -> list[str]:
        if not isinstance(result, dict):
            return []
        missing_conditions = result.get("missing_conditions")
        if isinstance(missing_conditions, list):
            return [
                str(item).strip() for item in missing_conditions if str(item).strip()
            ]
        return []

    def _debug_print(self, title: str, payload: Any) -> None:
        if not self.debug or title not in {"vlm_raw_result", "vlm_normalized_decision"}:
            return
        print(f"[Agent] {title}")
        if isinstance(payload, (dict, list)):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload)

    def _evaluate_candidate_once(self, candidate: CandidateObject) -> str:
        prompt = build_candidate_judgement_prompt(self._require_query(), candidate)
        image_count = self._count_prompt_images(prompt)
        self.vlm_image_counts.append(image_count)
        print(f"[Agent] vlm_stitched_image_count={image_count}")
        result = self._normalize_vlm_result(self.vlm(prompt, candidate=candidate))
        self._debug_print("vlm_raw_result", result)
        decision = self._normalize_vlm_decision(result)
        self._debug_print("vlm_normalized_decision", decision)
        candidate.missing_conditions = self._extract_missing_conditions(result)
        candidate.last_suggested_action = self._extract_suggested_action(result)
        return decision

    def can_retry_candidate(self, candidate: CandidateObject) -> bool:
        return int(getattr(candidate, "verification_round", 0)) < int(
            self.max_verification_rounds
        )

    def pick_candidate_for_verification(self) -> CandidateObject | None:
        pending_candidates = [
            candidate
            for candidate in self.candidates.values()
            if getattr(candidate, "status", "new") in {"new", "expanded", "unsure"}
            and self.can_retry_candidate(candidate)
        ]
        if not pending_candidates:
            return None
        pending_candidates.sort(
            key=lambda candidate: len(getattr(candidate, "object_view", []) or []),
            reverse=True,
        )
        return pending_candidates[0]

    def verify_candidate_once(
        self,
        candidate: CandidateObject,
        suggested_action: str | None = None,
    ) -> tuple[CandidateObject, str]:
        should_preapply_turn_around = (
            self.preapply_turn_around_before_verify
            and "turn_around" not in getattr(candidate, "done_actions", [])
        )
        if (
            should_preapply_turn_around
            and getattr(candidate, "bbox_3d", None) is not None
        ):
            object_views = list(getattr(candidate, "object_view", []) or [])
            best_id = int(getattr(candidate, "best_id", 0))
            if 0 <= best_id < len(object_views):
                best_view = getattr(object_views[best_id], "view", None)
                best_camera_to_world = getattr(best_view, "camera_to_world", None)
                if best_camera_to_world is not None:
                    camera_position_h = np.ones((4,), dtype=np.float64)
                    camera_position_h[:3] = np.asarray(
                        best_camera_to_world[:3, 3], dtype=np.float64
                    )
                    if self.world_to_axis_align_matrix is not None:
                        camera_position_h = (
                            np.asarray(
                                self.world_to_axis_align_matrix, dtype=np.float64
                            )
                            @ camera_position_h
                        )
                    camera_position = camera_position_h[:3]
                    target_center = np.asarray(
                        candidate.bbox_3d, dtype=np.float64
                    ).reshape(-1)[:3]
                    distance_to_target = float(
                        np.linalg.norm(camera_position - target_center)
                    )
                    if distance_to_target > 2.0:
                        print(
                            "[Agent] skip_preapply_turn_around_distance "
                            f"candidate_id={getattr(candidate, 'object_id', '')} "
                            f"label={getattr(candidate, 'label', '')} "
                            f"distance={distance_to_target:.3f}"
                        )
                        should_preapply_turn_around = False
        if should_preapply_turn_around:
            print(
                "[Agent] preapply_turn_around "
                f"candidate_id={getattr(candidate, 'object_id', '')} "
                f"label={getattr(candidate, 'label', '')} "
                f"status={getattr(candidate, 'status', '')} "
                f"best_id={getattr(candidate, 'best_id', '')} "
                f"num_object_views={len(getattr(candidate, 'object_view', []) or [])} "
                f"done_actions={getattr(candidate, 'done_actions', [])}"
            )
            candidate = self.complete_candidate_with_more_views_if_needed(
                candidate,
                decision="unsure",
                suggested_action="turn_around",
            )
            print(
                "[Agent] preapply_turn_around_done "
                f"candidate_id={getattr(candidate, 'object_id', '')} "
                f"num_object_views={len(getattr(candidate, 'object_view', []) or [])} "
                f"done_actions={getattr(candidate, 'done_actions', [])}"
            )
        while True:
            decision = self._evaluate_candidate_once(candidate)
            candidate.verification_round = (
                int(getattr(candidate, "verification_round", 0)) + 1
            )
            candidate.status = decision
            effective_action = suggested_action or getattr(
                candidate, "last_suggested_action", None
            )

            if decision == "true":
                current_round = int(getattr(candidate, "verification_round", 0))
                candidate.verification_round = int(self.max_verification_rounds)
                if current_round != 1:
                    return candidate, decision
                candidate = self.complete_candidate_with_more_views_if_needed(
                    candidate,
                    decision=decision,
                    suggested_action="forward",
                )
                return candidate, decision

            if decision != "unsure":
                return candidate, decision

            if not self.can_retry_candidate(candidate):
                candidate.status = "unsure"
                return candidate, decision

            candidate = self.complete_candidate_with_more_views_if_needed(
                candidate,
                decision=decision,
                suggested_action=effective_action,
            )
            suggested_action = None

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

        pending_candidate = self.pick_candidate_for_verification()
        if pending_candidate is not None:
            decision = self.verify_candidate_once(pending_candidate)
            if decision == "true":
                return pending_candidate
        else:
            decision = "unsure"

        next_motion = self.select_fallback_motion(decision)
        next_views = next_motion()
        self.consume_views(next_views)
        return next_views
