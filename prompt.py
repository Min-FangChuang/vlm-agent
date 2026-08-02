from __future__ import annotations

from pathlib import Path
from typing import Any
import base64
import io
import json

import numpy as np
from PIL import Image

import cv2

try:
    from .module.detector_yoloe import draw_bbox
except ImportError:
    from module.detector_yoloe import draw_bbox  # type: ignore


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

    source = str(_safe_getattr(object_view, "source", ""))
    if source == "turn_around":
        image = np.asarray(_safe_getattr(view, "rgb"), dtype=np.uint8).copy()
        return image

    image = np.asarray(_safe_getattr(view, "rgb"), dtype=np.uint8).copy()
    references = _safe_getattr(view, "reference", []) or []
    for reference in references:
        image = draw_bbox(
            image,
            reference,
            "",
            color=(255, 0, 0),
        )

    image = draw_bbox(
        image,
        _safe_getattr(object_view, "bbox_2d"),
        "",
        color=(0, 255, 0),
    )
    return image


def _bbox_to_list(bbox: Any) -> list[float] | None:
    if bbox is None:
        return None
    try:
        return np.asarray(bbox, dtype=np.float32).reshape(4).tolist()
    except Exception:
        return None


def _infer_object_view_analysis(object_view: Any) -> dict[str, Any]:
    source = str(_safe_getattr(object_view, "source", "detected"))
    status = str(_safe_getattr(object_view, "status", "active"))
    mask = _safe_getattr(object_view, "mask_2d")
    mask_present = mask is not None

    support_only_sources = {
        "projected_yaw_support_only",
        "projected_bootstrap_support_only",
        "turn_around",
    }
    segmented_sources = {
        "projected_bootstrap_segmented",
        "projected_yaw_segmented",
    }
    refined_sources = {
        "projected_bootstrap_refined",
        "projected_yaw_refined",
    }

    if source == "detected":
        did_detect = True
        did_segment = bool(mask_present)
        refine_mode = "detected"
    elif source in segmented_sources:
        did_detect = True
        did_segment = True
        refine_mode = "segment_from_refined_bbox"
    elif source in refined_sources:
        did_detect = True
        did_segment = False
        refine_mode = "detection_or_projected_bbox"
    elif source in support_only_sources or status == "support_only":
        did_detect = False
        did_segment = False
        refine_mode = "none"
    else:
        did_detect = bool(mask_present)
        did_segment = bool(mask_present)
        refine_mode = "none"

    return {
        "source": source,
        "status": status,
        "mask_present": bool(mask_present),
        "did_detect": bool(did_detect),
        "did_segment": bool(did_segment),
        "refine_mode": refine_mode,
    }


def _save_candidate_views(candidate: Any) -> Path | None:
    object_views = _safe_getattr(candidate, "object_view", []) or []
    if not object_views:
        return None

    output_dir = _next_candidate_output_dir(Path("output") / "test")
    summary_payload = {
        "label": str(_safe_getattr(candidate, "label", "unknown")),
        "status": str(_safe_getattr(candidate, "status", "unknown")),
        "best_id": int(_safe_getattr(candidate, "best_id", -1)),
        "verification_round": int(_safe_getattr(candidate, "verification_round", 0)),
        "last_suggested_action": _safe_getattr(candidate, "last_suggested_action"),
        "num_object_views": len(object_views),
        "object_views": [],
    }
    for index, object_view in enumerate(object_views):
        image = _draw_candidate_object_view(object_view)
        view = _safe_getattr(object_view, "view")
        view_id = _safe_getattr(view, "view_id", index)
        file_path = output_dir / f"{index:03d}_{view_id}.png"
        cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        object_view_summary = {
            "index": int(index),
            "view_id": str(view_id),
            "bbox_2d": _bbox_to_list(
                _safe_getattr(
                    object_view,
                    "bbox_2d",
                    np.zeros((4,), dtype=np.float32),
                )
            ),
            "image_file": file_path.name,
            "label": str(_safe_getattr(object_view, "label", "object")),
            "score": float(_safe_getattr(object_view, "score", 0.0)),
            "is_best_view": bool(index == int(_safe_getattr(candidate, "best_id", -1))),
        }
        object_view_summary.update(_infer_object_view_analysis(object_view))
        summary_payload["object_views"].append(object_view_summary)
    (output_dir / "candidate_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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


CANDIDATE_VERIFY_SYSTEM_PROMPT = """You are a visual grounding verifier for indoor environments. The green boxes always indicate the candidate currently being evaluated.

The green-boxed candidate is only a coarse candidate, not a confirmed match. You must first verify whether it is genuinely the requested object category itself, not merely an object that loosely looks similar in shape, color, layout, or partial appearance.

This verifier operates inside a multi-round evidence-gathering loop. The currently provided views may be only part of the final evidence for this candidate. Your job is not only to judge the candidate from the current images, but also to decide whether the current evidence is already sufficient for a reliable final decision.

Use the full natural-language request as the main thing to verify. The request may involve object identity, attributes, nearby reference objects, spatial relations, comparisons, or relation chains. Natural-language descriptions may also be ambiguous, incomplete, or dependent on scene context, orientation, or nearby alternatives. You must interpret the request against the actual environment shown in the views, rather than treating each phrase as an isolated rigid rule. Your decision should reflect whether the boxed object satisfies the whole request in the scene, not only part of it.

 Pay special attention to left/right language in the query: determine whether it refers to the object's intrinsic left/right side or to observer/image-relative left/right, and do not treat those two reference frames as equivalent.

Visual evidence is provided as stitched images. Each regular stitched image is a grid made by combining up to 6 observed views of the same candidate object into one image. The green box indicates the candidate being evaluated in the corresponding tile.

Blue boxes indicate other detector results in the same RGB view. They are not the current candidate, but you may use them to judge relative relations, nearby alternatives, and whether the green-boxed candidate is really the best-matching target in that local scene.

If turn_around evidence is available, it is provided as one separate stitched image. Interpret it as a turn-in-place sequence around the currently evaluated candidate: imagine facing the candidate at 0 deg and then rotating around that same location while reconstructing the surrounding layout relative to the candidate. The 0 deg tile denotes the current best target view and the candidate currently being evaluated, and this 0 deg tile is the one where the candidate itself is marked by the green box. Non-zero-degree tiles show what lies around that candidate from other headings. For example, objects near +180 deg or -180 deg are roughly behind the current 0 deg viewing direction, so use those angles to imagine what lies around the candidate from the viewer's backside perspective. Do not read the angle labels as isolated tags; use them to mentally reconstruct the local spatial arrangement and the positions of nearby objects relative to the evaluated candidate. Turnaround tiles may emphasize surrounding context and orientation rather than always showing a complete target box, so use them mainly to understand nearby layout, reference objects, relative orientation, and whether additional evidence exists around the candidate. If another object seen in a non-zero-degree turnaround tile appears to satisfy the query better than the 0 deg candidate, treat that as evidence that the current candidate may not be the best match.

If the reference object is unclear, or if the relation cannot be judged reliably, do not overcommit.

Do not rationalize object identity, semantic meaning, or spatial relations around the green-boxed candidate just because it looks plausible. Judge the full scene evidence as evenly as possible, including nearby alternatives and the overall spatial layout.

Return unsure when the current candidate may still be correct but the current evidence is not yet sufficient to confirm the full request. In that case, use suggested_action to indicate what kind of additional evidence should be collected next. The purpose of suggested_action is to request the next kind of information that would most reduce the current uncertainty. It is not a candidate choice.

If a blue-boxed candidate in the same image could also satisfy the query and the current evidence is not sufficient to distinguish it reliably from the green-boxed candidate, return unsure.

If `history_missing_conditions` is provided, it is only a brief summary of what the previous verification round found missing. Use it as historical context, not as the sole basis for the current decision. The current decision must still be based primarily on the full request and the currently available visual evidence.

Use forward when more direct, target-centered views are needed to confirm the candidate itself.

Use yaw when more nearby viewpoints are needed to keep the same candidate in view and obtain a more complete or less partial view of that candidate.

Use backward when the current candidate likely needs a slightly wider and slightly more pulled-back local view so that more of the candidate itself can be seen together. Backward only provides limited extra space.

Use stop only when no additional view request is needed.

Return false only when the current candidate is clearly contradicted, clearly fails an essential requirement, or another nearby candidate clearly satisfies the request better. Return true only when the current evidence is already sufficient to confirm the full request with strong evidence and no meaningful condition-level doubt remains.

Do not hallucinate unsupported details. In reasoning, explicitly mention which stitched image and tile(s) provide the relevant evidence whenever visual evidence is cited.

Return structured JSON only with this schema:
{
  "decision": "true",
  "confidence": 95,
  "reasoning": "brief evidence-based reasoning",
  "matched_conditions": [],
  "missing_conditions": [],
  "suggested_action": "forward"
}

Rules for output:
- decision must be one of: true, false, unsure
- confidence must be an integer from 0 to 100
- confidence should reflect how certain you are that the decision is correct
- if evidence is insufficient, use decision="unsure" rather than forcing true or false
- suggested_action must be one of: forward, yaw, backward, stop
"""

MULTI_CANDIDATE_SELECT_SYSTEM_PROMPT = """You are selecting the best candidate object for a visual grounding query in an indoor environment.

Each stitched image corresponds to one candidate hypothesis. The green box in each tile indicates the candidate being evaluated in that stitched image. A candidate may look similar to the requested object but still be wrong. Different candidates are not guaranteed to correspond to different physical objects. Some candidates may be alternative boxes, alternative viewpoints, or alternative hypotheses for the same underlying object. Your task is to choose the single candidate that provides the most reliable and best-supported match to the full natural-language request.

Use the full query as the main thing to verify. The query may involve object identity, attributes, nearby reference objects, spatial relations, comparisons, or relation chains. Natural-language descriptions may also be ambiguous or depend on scene context, orientation, or nearby alternatives, so evaluate each candidate in the context of the whole scene description rather than isolated object appearance alone.

For each candidate, first verify whether the boxed object itself is genuinely the requested object category, not merely something that loosely looks similar. Then compare candidates by checking which one best satisfies the requested attributes, reference-object constraints, spatial relations, comparisons, and surrounding scene context. Similar target objects and similar reference objects may both appear multiple times in the scene, so make sure you are selecting the candidate that matches the correct target-reference pairing rather than any arbitrary similar object.

You must select exactly one candidate from the provided candidates. Do not reject all candidates and do not return an empty choice. Even if the evidence is imperfect, choose the candidate that is the strongest overall match relative to the other candidates. If several candidates appear to refer to the same underlying object, prefer the candidate whose views, framing, and surrounding evidence support the request more clearly and more completely.

Return structured JSON only with this schema:
{
  "selected_index": 0,
  "reasoning": "brief evidence-based comparison"
}

Rules for output:
- selected_index must be a zero-based candidate index.
- Return exactly one JSON object.
- Do not use markdown.
- Do not use code fences such as ```json.
- Do not output any text before or after the JSON.
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
        canvas[y1 : y1 + tile_height, x1 : x1 + tile_width] = tile

        view = _safe_getattr(object_view, "view")
        view_id = _safe_getattr(view, "view_id", local_index)

        tile_descriptions.append(f"tile {local_index}: view_id={view_id}")

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
        batch_views = all_views[start : start + views_per_stitched_image]

        stitched_image, tile_descriptions = _stitch_candidate_object_views(
            batch_views,
            tile_size=tile_size,
            columns=columns,
        )

        if stitched_image is not None:
            stitched_batches.append((stitched_image, tile_descriptions))

    return stitched_batches


def _collect_turn_around_batch(candidate: Any) -> tuple[Any, list[str]]:
    object_views = _safe_getattr(candidate, "object_view", []) or []
    for object_view in object_views:
        if str(_safe_getattr(object_view, "source", "")) != "turn_around":
            continue
        view = _safe_getattr(object_view, "view")
        stitched_image = _safe_getattr(view, "rgb")
        reference = _safe_getattr(view, "reference", {}) or {}
        if isinstance(reference, dict):
            tile_descriptions = list(reference.get("tile_descriptions", []) or [])
        else:
            tile_descriptions = list(
                _safe_getattr(reference, "tile_descriptions", []) or []
            )
        if stitched_image is None:
            continue
        image_array = np.asarray(stitched_image, dtype=np.uint8)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            continue
        return image_array, tile_descriptions
    return None, []


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
    request_payload = {
        "full_text": _safe_getattr(query, "query", ""),
        "requested_object": _safe_getattr(query, "target_object", ""),
        "requested_object_attributes": _safe_getattr(query, "target_attributes", []),
        "reference_object": _safe_getattr(query, "reference_object", ""),
        "reference_object_attributes": _safe_getattr(query, "reference_attributes", []),
        "done_actions": _safe_getattr(candidate, "done_actions", []),
        # "spatial_relation": _safe_getattr(query, "relation", ""),
    }
    history_missing_conditions = list(
        _safe_getattr(candidate, "missing_conditions", []) or []
    )
    if history_missing_conditions:
        request_payload["history_missing_conditions"] = history_missing_conditions

    payload = {
        "task": "verify whether the green-boxed object is the requested object",
        "request": request_payload,
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_candidate_judgement_prompt(query: Any, candidate: Any):
    object_views = _safe_getattr(candidate, "object_view", []) or []
    gpt_input = build_candidate_text_input(query, candidate)

    stitched_batches = _stitch_candidate_object_view_batches(
        [
            object_view
            for object_view in object_views
            if str(_safe_getattr(object_view, "source", "")) != "turn_around"
        ],
        views_per_stitched_image=6,
        tile_size=(384, 288),
        columns=3,
    )
    turn_around_image, turn_around_descriptions = _collect_turn_around_batch(candidate)
    if turn_around_image is not None:
        stitched_batches.append((turn_around_image, turn_around_descriptions))

    turn_around_text = ""
    if turn_around_image is not None:
        turn_around_text = (
            "- A turnaround image may appear as one separate stitched panel showing nearby environment/context views around the best view.\n"
            "- Tiles in the turnaround image may not contain the target object; use them mainly to inspect nearby context, reference objects, and relations.\n"
            "- Each turnaround tile has an angle label printed on the image, and 0 deg denotes the best current view.\n"
        )

    all_tile_descriptions: list[str] = []
    image_contents: list[dict[str, Any]] = []

    for batch_index, (stitched_image, tile_descriptions) in enumerate(stitched_batches):
        if batch_index == len(stitched_batches) - 1 and turn_around_image is not None:
            all_tile_descriptions.append(
                f"turnaround image {batch_index}: contains {', '.join(tile_descriptions)}"
            )
        else:
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
                + turn_around_text
                + "- Tile numbers are local to each stitched image.\n"
                + "- If a required reference object or spatial relation is not visually confirmed, return unsure.\n\n"
                + "Tile descriptions:\n"
                + (
                    "\n".join(all_tile_descriptions)
                    if all_tile_descriptions
                    else "No stitched visual evidence available."
                )
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
        object_views = [
            object_view
            for object_view in (_safe_getattr(candidate, "object_view", []) or [])
            if str(_safe_getattr(object_view, "source", "")) != "turn_around"
        ][:6]
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

    text = (
        f"Query: {_safe_getattr(query, 'query', '')}\n"
        f"Here are the images of {len(candidates)} possible objects.\n"
        "Choice images:\n" + "\n".join(candidate_summaries)
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
