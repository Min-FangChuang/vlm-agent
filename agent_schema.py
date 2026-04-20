from __future__ import annotations

from typing import Any

import numpy as np


class View:
    def __init__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        camera_to_world: np.ndarray | None,
        view_id: str | int,
        reference: Any = None,
    ) -> None:
        self.rgb = rgb
        self.depth = depth
        self.camera_to_world = camera_to_world
        self.view_id = view_id
        self.reference = reference


class Query:
    def __init__(self, query: str) -> None:
        self.query = query
        self.target_object = self.make_target_object(self.query)
        self.target_attributes = self.make_target_attributes(self.query)
        self.reference_object = self.make_reference_object(self.query)
        self.reference_attributes = self.make_reference_attributes(self.query)
        self.relation = self.make_relation(self.query)

    def make_target_object(self, query: str) -> str:
        return "the closet doors"

    def make_target_attributes(self, query: str) -> list[str]:
        return "in the corner of the room"

    def make_reference_object(self, query: str) -> str:
        return "bed"

    def make_reference_attributes(self, query: str) -> list[str]:
        return ""

    def make_relation(self, query: str) -> str:
        return "in front of"


class ObjectView:
    def __init__(
        self,
        object_id: str | int,
        label: str,
        score: float,
        view: View,
        bbox_2d: np.ndarray,
        mask_2d: np.ndarray | None = None,
        points_3d: Any = None,
        status: str = "active",
    ) -> None:
        self.object_id = object_id
        self.label = label
        self.score = float(score)
        self.view = view
        self.bbox_2d = np.asarray(bbox_2d, dtype=np.float32).reshape(-1)
        self.mask_2d = None if mask_2d is None else np.asarray(mask_2d)
        self.points_3d = points_3d
        self.status = status
        if self.bbox_2d.shape[0] != 4:
            raise ValueError(f"bbox_2d must have 4 values, got shape={self.bbox_2d.shape}")
        if self.mask_2d is not None and self.mask_2d.ndim != 2:
            raise ValueError(f"mask_2d must be 2D, got shape={self.mask_2d.shape}")

    @property
    def rgb(self) -> np.ndarray:
        return self.view.rgb

    @property
    def detection_2d(self) -> np.ndarray:
        return self.bbox_2d

    @property
    def mask(self) -> np.ndarray | None:
        return self.mask_2d


class CandidateObject:
    def __init__(
        self,
        object_id: str | int,
        label: str,
        score: float,
        view: ObjectView | None = None,
        detection_2d: np.ndarray | None = None,
        mask_2d: np.ndarray | None = None,
        points_3d: Any = None,
        status: str = "active",
        best_id: int = 0,
        object_view: list[ObjectView] | None = None,
    ) -> None:
        self.object_id = object_id
        self.label = label
        self.score = float(score)
        self.points_3d = points_3d
        self.status = status
        self.best_id = int(best_id)
        self.object_view = list(object_view or [])
        if view is not None:
            if isinstance(view, ObjectView):
                self.object_view.append(view)
            else:
                if detection_2d is None:
                    raise ValueError("detection_2d is required when `view` is not an ObjectView.")
                self.object_view.append(
                    ObjectView(
                        object_id=object_id,
                        label=label,
                        score=score,
                        view=view,
                        bbox_2d=detection_2d,
                        mask_2d=mask_2d,
                        points_3d=points_3d,
                        status=status,
                    )
                )
        if self.object_view:
            self.best_id = max(0, min(self.best_id, len(self.object_view) - 1))

    @property
    def object_views(self) -> list[ObjectView]:
        return self.object_view

    @property
    def views(self) -> list[View]:
        return [item.view for item in self.object_view]

    @property
    def detections_2d(self) -> list[np.ndarray]:
        return [item.bbox_2d for item in self.object_view]

    @property
    def masks_2d(self) -> list[np.ndarray | None]:
        return [item.mask_2d for item in self.object_view]

    def add_object_view(self, object_view: ObjectView) -> None:
        self.object_view.append(object_view)
        current_best_bbox = np.asarray(self.object_view[self.best_id].bbox_2d, dtype=np.float32).reshape(4)
        new_bbox = np.asarray(object_view.bbox_2d, dtype=np.float32).reshape(4)
        current_best_bbox_area = max(0.0, float(current_best_bbox[2] - current_best_bbox[0])) * max(
            0.0,
            float(current_best_bbox[3] - current_best_bbox[1]),
        )
        new_bbox_area = max(0.0, float(new_bbox[2] - new_bbox[0])) * max(
            0.0,
            float(new_bbox[3] - new_bbox[1]),
        )
        if new_bbox_area > current_best_bbox_area:
            self.best_id = len(self.object_view) - 1

    def current_best_view(self) -> ObjectView:
        if not self.object_view:
            raise ValueError("CandidateObject has no object views.")
        return self.object_view[self.best_id]


class CandidateMemory:
    def __init__(self) -> None:
        self.objects: list[CandidateObject] = []

    def add(self, candidate: CandidateObject) -> None:
        self.objects.append(candidate)

    def get(self, object_id: str | int) -> CandidateObject | None:
        for candidate in self.objects:
            if candidate.object_id == object_id:
                return candidate
        return None

    def values(self) -> list[CandidateObject]:
        return list(self.objects)

    def exist(self) -> bool:
        for candidate in self.objects:
            if candidate.status == "active":
                return True
        return False

    def add_ObjectView(self, object_view: ObjectView, match_fn: Any) -> tuple[CandidateObject, bool]:
        for candidate in self.objects:
            if candidate.status != "active":
                continue
            if match_fn(object_view, candidate):
                candidate.add_object_view(object_view)
                return candidate, True

        new_candidate = CandidateObject(
            object_id=len(self.objects),
            label=object_view.label,
            score=object_view.score,
            view=object_view,
            best_id=0,
            points_3d=object_view.points_3d,
        )
        self.add(new_candidate)
        return new_candidate, False

    def find_by_label(self, label: str) -> list[CandidateObject]:
        return [obj for obj in self.objects if obj.label == label]

    def remove(self, object_id: str | int) -> None:
        self.objects = [candidate for candidate in self.objects if candidate.object_id != object_id]
