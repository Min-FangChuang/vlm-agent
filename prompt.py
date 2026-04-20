from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from .module.detector import draw_bbox
    import cv2
except ImportError:
    from module.detector import draw_bbox  # type: ignore
    import cv2  # type: ignore


def _safe_getattr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _next_candidate_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    numeric_dirs: list[int] = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            numeric_dirs.append(int(child.name))
    next_index = max(numeric_dirs, default=-1) + 1
    output_dir = base_dir / str(next_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _draw_candidate_object_view(object_view: Any) -> np.ndarray:
    view = _safe_getattr(object_view, "view")
    if view is None:
        raise ValueError("object_view must provide `view`.")

    image = np.asarray(_safe_getattr(view, "rgb"), dtype=np.uint8).copy()
    image = draw_bbox(
        image,
        _safe_getattr(object_view, "bbox_2d"),
        str(_safe_getattr(object_view, "label", "object")),
        color=(0, 255, 0),
    )

    references = _safe_getattr(view, "reference", []) or []
    for reference in references:
        image = draw_bbox(
            image,
            _safe_getattr(reference, "bbox"),
            str(_safe_getattr(reference, "label", "reference")),
            color=(255, 0, 0),
        )
    return image


def _save_candidate_views(candidate: Any) -> Path | None:
    object_views = _safe_getattr(candidate, "object_view", []) or []
    if not object_views:
        return None

    output_dir = _next_candidate_output_dir(Path("output") / "test")
    for index, object_view in enumerate(object_views):
        image = _draw_candidate_object_view(object_view)
        view = _safe_getattr(object_view, "view")
        view_id = _safe_getattr(view, "view_id", index)
        file_path = output_dir / f"{index:03d}_{view_id}.png"
        cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return output_dir


def build_candidate_summary(candidate: Any) -> str:
    label = _safe_getattr(candidate, "label", "unknown")
    score = _safe_getattr(candidate, "score", 0.0)
    status = _safe_getattr(candidate, "status", "unknown")
    best_id = _safe_getattr(candidate, "best_id", -1)
    object_views = _safe_getattr(candidate, "object_view", [])
    num_views = len(object_views) if object_views is not None else 0
    saved_dir = _save_candidate_views(candidate)
    return (
        f"label={label}, score={float(score):.3f}, status={status}, "
        f"best_id={best_id}, num_object_views={num_views}, saved_dir={saved_dir}"
    )


def build_candidate_judgement_prompt(query: Any, candidate: Any) -> str:
    query_text = _safe_getattr(query, "query", "")
    target_object = _safe_getattr(query, "target_object", "")
    reference_object = _safe_getattr(query, "reference_object", "")
    relation = _safe_getattr(query, "relation", "")
    candidate_summary = build_candidate_summary(candidate)
    return (
        "You are checking whether a candidate object satisfies the search query.\n"
        f"Query: {query_text}\n"
        f"Target object: {target_object}\n"
        f"Reference object: {reference_object}\n"
        f"Relation: {relation}\n"
        f"Candidate summary: {candidate_summary}\n"
        "Reply with one of: true, false, unsure."
    )


def build_reference_detection_prompt(query: Any) -> str:
    reference_object = _safe_getattr(query, "reference_object", "")
    if not reference_object:
        return ""
    return f"Detect reference object: {reference_object}"
