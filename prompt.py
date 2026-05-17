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
    # references = _safe_getattr(view, "reference", []) or []
    # for reference in references:
    #     image = draw_bbox(
    #         image,
    #         _safe_getattr(reference, "bbox"),
    #         str(_safe_getattr(reference, "label", "reference")),
    #         color=(0, 0, 255),
    #     )

    image = draw_bbox(
        image,
        _safe_getattr(object_view, "bbox_2d"),
        str(_safe_getattr(object_view, "label", "object")),
        color=(0, 255, 0),
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
    status = _safe_getattr(candidate, "status", "unknown")
    best_id = _safe_getattr(candidate, "best_id", -1)
    object_views = _safe_getattr(candidate, "object_view", [])
    num_views = len(object_views) if object_views is not None else 0
    saved_dir = _save_candidate_views(candidate)
    return (
        f"label={label}, status={status}, "
        f"best_id={best_id}, num_object_views={num_views}, saved_dir={saved_dir}"
    )

CANDIDATE_VERIFY_SYSTEM_PROMPT = """You are a visual grounding verifier for indoor environments.

You are given:
1. the full user request and a compact extraction of its key conditions
2. several photos of the same candidate object from different viewpoints
3. the surrounding scene context visible in those photos

The green boxes always indicate the object being evaluated. Your task is to determine whether this green-boxed object fully satisfies the request among the plausible objects in the scene.

Rules:
- Use all provided viewpoints as evidence.
- First verify that the green-boxed object itself matches the requested object category and stated attributes.
- Allow reasonable tolerance for attributes that may look slightly different because of occlusion, lighting, perspective, or other visual ambiguity, as long as the object could still plausibly satisfy the description. If an attribute is clearly contradicted by the evidence, return false.
- Then verify that any required nearby reference object or alternative object is visually supported and matches the requested role in the description.
- Then verify that the spatial relation, comparison, or relation chain is satisfied. This includes comparative or superlative requirements such as closest, farthest, leftmost, rightmost, biggest, smallest, or similar relative descriptions.
- For relative comparisons involving nearby objects, confirm the surrounding context is sufficient before deciding. Verify that the views contain enough nearby evidence to compare the green-boxed object against relevant alternative objects and reference objects, rather than judging from an isolated crop.
- Spatial relations are especially sensitive to incomplete viewpoints. Before deciding a relation, confirm that the views provide enough surrounding coverage of the green-boxed object and the relevant nearby objects needed for that judgment. If the relation cannot be judged reliably because the needed surrounding views are incomplete, return unsure.
- After checking the extracted conditions, do one final pass against the full original request and make sure no explicit requirement was missed.
- Do not hallucinate unsupported details.
- Only return true when the evidence is strong enough that the green-boxed object is the correct match, not just a partially matching object.
- If evidence is insufficient, return "unsure" instead of "true".

You must evaluate:
- object category
- object attributes
- reference-object existence if required
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

MULTI_CANDIDATE_SELECT_SYSTEM_PROMPT = """You are a visual grounding selector for indoor environments.

You are given:
1. the full user request and a compact extraction of its key conditions
2. multiple candidate objects, each represented by one stitched image made from several viewpoints of the same object

The green boxes in each stitched image indicate the object being evaluated for that choice. Your task is to choose the single best candidate that satisfies the query.

Rules:
- Compare all candidates against the full original request.
- Use the stitched image for each candidate as the primary visual evidence.
- For each choice, first check whether the green-boxed object itself matches the requested object and attributes, then check whether any needed nearby reference object or alternative object is supported, and then check whether the spatial relation or comparison is satisfied.
- After comparing all choices, do one final pass against the full original request and make sure no explicit requirement was missed.
- Do not rely on detector confidence or class names beyond what is visually supported.
- You must choose exactly one candidate, even if the evidence is imperfect. Select the candidate that best satisfies the query overall.

Return structured JSON only with this schema:
{
  "selected_index": 0,
  "reasoning": "brief evidence-based reasoning"
}
- selected_index must be a zero-based candidate index.
"""
QUERY_DECOMPOSE_SYSTEM_PROMPT = """
You are a query decomposition assistant for indoor visual grounding.

Analyze a natural language query and extract only explicitly stated information.

Return valid JSON only. Do not use markdown.

Rules:
1. Do not hallucinate missing details.
2. The target_object is the main object being searched for.
3. The reference_object is the object used as a spatial anchor.
4. Extract visual attributes such as color, material, size, state, or shape.
5. Extract spatial relation if explicitly stated.
6. If something is missing, use an empty string or an empty list.

Output schema:
{
  "raw_query": "<original query>",
  "target_object": "<main object>",
  "target_attributes": [],
  "reference_object": "",
  "reference_attributes": [],
  "relation": ""
}
""".strip()

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
    tile_size: tuple[int, int] = (384, 288),
    columns: int = 3,
) -> tuple[np.ndarray | None, list[str]]:
    selected_views = list(object_views or [])
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

    for local_index, object_view in enumerate(selected_views):
        try:
            drawn_image = _draw_candidate_object_view(object_view)
        except Exception as exc:
            print(f"[Prompt] failed to draw object view {local_index}: {exc}")
            continue

        tile = _resize_rgb_image(drawn_image, tile_size)

        row = local_index // columns
        col = local_index % columns
        y1 = row * tile_height
        x1 = col * tile_width
        canvas[y1:y1 + tile_height, x1:x1 + tile_width] = tile

        view = _safe_getattr(object_view, "view")
        view_id = _safe_getattr(view, "view_id", local_index)

        tile_descriptions.append(
            f"tile {local_index}: view_id={view_id}"
        )

    if not tile_descriptions:
        return None, []

    return canvas, tile_descriptions

def _stitch_candidate_object_view_batches(
    object_views: list[Any],
    views_per_stitched_image: int = 6,
    tile_size: tuple[int, int] = (384, 288),
    columns: int = 3,
) -> list[tuple[np.ndarray, list[str]]]:
    all_views = list(object_views or [])
    stitched_batches: list[tuple[np.ndarray, list[str]]] = []

    for start in range(0, len(all_views), views_per_stitched_image):
        batch_views = all_views[start:start + views_per_stitched_image]

        stitched_image, tile_descriptions = _stitch_candidate_object_views(
            batch_views,
            tile_size=tile_size,
            columns=columns,
        )

        if stitched_image is not None:
            stitched_batches.append((stitched_image, tile_descriptions))

    return stitched_batches

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
    payload = {
        "task": "verify whether the green-boxed object is the requested object",
        "request": {
            "full_text": _safe_getattr(query, "query", ""),
            "requested_object": _safe_getattr(query, "target_object", ""),
            "requested_object_attributes": _safe_getattr(query, "target_attributes", []),
            "reference_object": _safe_getattr(query, "reference_object", ""),
            "reference_object_attributes": _safe_getattr(query, "reference_attributes", []),
            "spatial_relation": _safe_getattr(query, "relation", ""),
        },
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)

def build_candidate_judgement_prompt(query: Any, candidate: Any):
    object_views = _safe_getattr(candidate, "object_view", []) or []
    gpt_input = build_candidate_text_input(query, candidate)

    stitched_batches = _stitch_candidate_object_view_batches(
        object_views,
        views_per_stitched_image=6,
        tile_size=(384, 288),
        columns=3,
    )

    all_tile_descriptions: list[str] = []
    image_contents: list[dict[str, Any]] = []

    for batch_index, (stitched_image, tile_descriptions) in enumerate(stitched_batches):
        all_tile_descriptions.append(
            f"stitched image {batch_index}: contains {', '.join(tile_descriptions)}"
        )

        image_contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_rgb_to_base64(stitched_image)}",
                    "detail": "high",
                },
            }
        )

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                gpt_input
                + "\n\nVisual evidence format:\n"
                + "- The attached images are stitched grids made from multiple photos of the same green-boxed object.\n"
                + "- Each stitched image contains up to 6 viewpoints.\n"
                + "- Green boxes indicate the object currently being evaluated.\n"
                + "- Tile numbers are local to each stitched image.\n"
                + "- If a required reference object or spatial relation is not visually confirmed, return unsure.\n\n"
                + "Tile descriptions:\n"
                + ("\n".join(all_tile_descriptions) if all_tile_descriptions else "No stitched visual evidence available.")
            ),
        },
        *image_contents,
    ]

    messages = [
        {"role": "system", "content": CANDIDATE_VERIFY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": content,
        },
    ]

    return messages


def build_multi_candidate_selection_prompt(query: Any, candidates: list[Any]):
    candidate_summaries: list[str] = []
    image_contents: list[dict[str, Any]] = []

    for candidate_index, candidate in enumerate(candidates):
        object_views = (_safe_getattr(candidate, "object_view", []) or [])[:6]
        stitched_image, tile_descriptions = _stitch_candidate_object_views(
            object_views,
            tile_size=(384, 288),
            columns=3,
        )
        candidate_summaries.append(f"choice {candidate_index}")
        if stitched_image is not None:
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_rgb_to_base64(stitched_image)}",
                        "detail": "high",
                    },
                }
            )

    payload = {
        "task": "choose the single best matching object",
        "request": {
            "full_text": _safe_getattr(query, "query", ""),
            "requested_object": _safe_getattr(query, "target_object", ""),
            "requested_object_attributes": _safe_getattr(query, "target_attributes", []),
            "reference_object": _safe_getattr(query, "reference_object", ""),
            "reference_object_attributes": _safe_getattr(query, "reference_attributes", []),
            "spatial_relation": _safe_getattr(query, "relation", ""),
        },
        "number_of_choices": len(candidates),
    }

    text = (
        json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nChoice images:\n"
        + "\n".join(candidate_summaries)
    )

    return [
        {"role": "system", "content": MULTI_CANDIDATE_SELECT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                *image_contents,
            ],
        },
    ]


def build_reference_detection_prompt(query: Any) -> str:
    reference_object = _safe_getattr(query, "reference_object", "")
    if not reference_object:
        return ""
    return f"Detect reference object: {reference_object}"
