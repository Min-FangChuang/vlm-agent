from __future__ import annotations

from typing import Any

try:
    from .json_utils import extract_first_json_object
    from .prompt import QUERY_DECOMPOSE_SYSTEM_PROMPT
    from .vlm_bridge import call_vlm_messages
except ImportError:
    from json_utils import extract_first_json_object  # type: ignore
    from prompt import QUERY_DECOMPOSE_SYSTEM_PROMPT  # type: ignore
    from vlm_bridge import call_vlm_messages  # type: ignore


def _fallback_parse(raw_query: str) -> dict[str, Any]:
    raw_query = str(raw_query or "").strip()
    return {
        "raw_query": raw_query,
        "target_object": raw_query.lower(),
        "target_attributes": [],
        "reference_object": "",
        "reference_attributes": [],
        "relation": "",
    }


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, str):
        value = value.strip().lower()
        return [value] if value else []
    return []


def _as_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize(raw_query: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_query": raw_query,
        "target_object": _as_text(data.get("target_object")),
        "target_attributes": _as_list(data.get("target_attributes")),
        "reference_object": _as_text(data.get("reference_object")),
        "reference_attributes": _as_list(data.get("reference_attributes")),
        "relation": _as_text(data.get("relation")),
    }


def parse_query_with_vlm(raw_query: str) -> dict[str, Any]:
    raw_query = str(raw_query or "").strip()

    if not raw_query:
        return _fallback_parse(raw_query)

    messages = [
        {
            "role": "system",
            "content": QUERY_DECOMPOSE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"Query: {raw_query}",
        },
    ]

    try:
        result = call_vlm_messages(messages)
        result_text = "" if result is None else str(result)

        print("[QueryParser] raw result")
        print(result_text if result_text.strip() else "<empty>")

        if isinstance(result, dict):
            parsed = result
        else:
            if not result_text.strip():
                raise ValueError("Empty VLM response.")
            parsed = extract_first_json_object(result_text)

        normalized = _normalize(raw_query, parsed)

        if not normalized["target_object"]:
            print("[QueryParser] empty target_object, fallback to raw query")
            return _fallback_parse(raw_query)

        return normalized

    except Exception as exc:
        print(f"[QueryParser] failed, fallback to raw query. error={exc}")
        return _fallback_parse(raw_query)
