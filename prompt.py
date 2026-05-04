from __future__ import annotations

from pathlib import Path
from typing import Any
import base64
import io
import json

import numpy as np
from PIL import Image

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

CANDIDATE_VERIFY_SYSTEM_PROMPT = """You are a visual grounding verifier for indoor environments.

You are given:
1. a structured query
2. one candidate object with multiple object views
3. optional reference object observations

Your task is to determine whether the candidate satisfies the query.

Rules:
- Use all object views as evidence.
- Treat the best view as the primary view.
- If reference object observations are provided, use them.
- If reference object observations are missing or incomplete, infer from the views when possible.
- Do not hallucinate unsupported details.
- If evidence is insufficient, return "unsure" instead of "true".

You must evaluate:
- target object category
- target attributes
- reference object existence if required
- spatial relation if required
- whether current evidence is sufficient

Return structured JSON only with this schema:
{
  "decision": "true",
  "confidence": "high",
  "reasoning": "brief evidence-based reasoning",
  "matched_conditions": [],
  "missing_conditions": [],
  "suggested_action": "forward"
}

Allowed values:
- decision: true / false / unsure
- confidence: high / medium / low
- suggested_action: forward / yaw / stop
"""


def _rgb_to_base64(rgb: np.ndarray) -> str:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def _resize_rgb_image(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.shape[-1] == 4:
        image = image[..., :3]

    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _stitch_candidate_object_views(
    object_views: list[Any],
    max_views: int = 6,
    tile_size: tuple[int, int] = (384, 288),
    columns: int = 3,
) -> tuple[np.ndarray | None, list[str]]:
    selected_views = list(object_views or [])[:max_views]
    if not selected_views:
        return None, []

    tile_width, tile_height = tile_size
    rows = int(np.ceil(len(selected_views) / columns))

    canvas = np.full(
        (rows * tile_height, columns * tile_width, 3),
        235,
        dtype=np.uint8,
    )

    tile_descriptions: list[str] = []

    for index, object_view in enumerate(selected_views):
        try:
            drawn_image = _draw_candidate_object_view(object_view)
        except Exception as exc:
            print(f"[Prompt] failed to draw object view {index}: {exc}")
            continue

        tile = _resize_rgb_image(drawn_image, tile_size)

        row = index // columns
        col = index % columns
        y1 = row * tile_height
        x1 = col * tile_width
        canvas[y1:y1 + tile_height, x1:x1 + tile_width] = tile

        view = _safe_getattr(object_view, "view")
        view_id = _safe_getattr(view, "view_id", index)
        label = _safe_getattr(object_view, "label", "object")
        score = float(_safe_getattr(object_view, "score", 0.0))

        tile_descriptions.append(
            f"tile {index}: view_index={index}, view_id={view_id}, label={label}, score={score:.3f}"
        )

    if not tile_descriptions:
        return None, []

    return canvas, tile_descriptions


def _normalize_reference(reference: Any) -> Any:
    if reference is None:
        return []

    if isinstance(reference, (str, int, float, bool)):
        return reference

    if isinstance(reference, np.ndarray):
        return reference.tolist()

    if isinstance(reference, list):
        normalized = []
        for item in reference:
            if isinstance(item, np.ndarray):
                normalized.append(item.tolist())
            elif hasattr(item, "__dict__"):
                normalized.append(vars(item))
            else:
                normalized.append(item)
        return normalized

    if hasattr(reference, "__dict__"):
        return vars(reference)

    return str(reference)


def build_candidate_text_input(query: Any, candidate: Any) -> str:
    object_views = _safe_getattr(candidate, "object_view", []) or []
    object_views_summary = []

    for idx, object_view in enumerate(object_views):
        view = _safe_getattr(object_view, "view")
        if view is None:
            continue

        ref_summary = _normalize_reference(_safe_getattr(view, "reference", None))

        object_views_summary.append(
            {
                "view_index": idx,
                "view_id": _safe_getattr(view, "view_id", idx),
                "label": _safe_getattr(object_view, "label", ""),
                "score": float(_safe_getattr(object_view, "score", 0.0)),
                "bbox_2d": np.asarray(
                    _safe_getattr(object_view, "bbox_2d", [0, 0, 0, 0]),
                    dtype=np.float32,
                ).tolist(),
                "has_mask": _safe_getattr(object_view, "mask_2d") is not None,
                "reference": ref_summary,
            }
        )

    payload = {
        "task": "candidate_verify",
        "query": {
            "raw_query": _safe_getattr(query, "query", ""),
            "target_object": _safe_getattr(query, "target_object", ""),
            "target_attributes": _safe_getattr(query, "target_attributes", []),
            "reference_object": _safe_getattr(query, "reference_object", ""),
            "reference_attributes": _safe_getattr(query, "reference_attributes", []),
            "relation": _safe_getattr(query, "relation", ""),
        },
        "candidate": {
            "object_id": _safe_getattr(candidate, "object_id", ""),
            "label": _safe_getattr(candidate, "label", ""),
            "score": float(_safe_getattr(candidate, "score", 0.0)),
            "status": _safe_getattr(candidate, "status", "unknown"),
            "best_id": int(_safe_getattr(candidate, "best_id", 0)),
            "num_views": len(object_views_summary),
            "object_views": object_views_summary,
        },
        #"debug_candidate_summary": build_candidate_summary(candidate),
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)

def build_candidate_judgement_prompt(query: Any, candidate: Any):
    object_views = _safe_getattr(candidate, "object_view", []) or []
    gpt_input = build_candidate_text_input(query, candidate)

    stitched_image, tile_descriptions = _stitch_candidate_object_views(
        object_views,
        max_views=6,
        tile_size=(384, 288),
        columns=3,
    )

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                gpt_input
                + "\n\nVisual evidence format:\n"
                + "- The attached image is a stitched grid of candidate object views.\n"
                + "- Green boxes indicate the candidate target object.\n"
                + "- Red boxes indicate detected reference objects, if any.\n"
                + "- Use tile labels and the JSON object_views view_index to refer to evidence.\n"
                + "- If a required reference object or spatial relation is not visually confirmed, return unsure.\n\n"
                + "Tile descriptions:\n"
                + ("\n".join(tile_descriptions) if tile_descriptions else "No stitched visual evidence available.")
            ),
        }
    ]

    if stitched_image is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_rgb_to_base64(stitched_image)}",
                    "detail": "high",
                },
            }
        )

    messages = [
        {"role": "system", "content": CANDIDATE_VERIFY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": content,
        },
    ]

    return messages

def build_reference_detection_prompt(query: Any) -> str:
    reference_object = _safe_getattr(query, "reference_object", "")
    if not reference_object:
        return ""
    return f"Detect reference object: {reference_object}"
