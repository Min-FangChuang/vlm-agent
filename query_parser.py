from __future__ import annotations

import json
import re
from typing import Any

try:
    from .prompt import QUERY_DECOMPOSE_SYSTEM_PROMPT
    from .vlm_bridge import call_vlm_messages
except ImportError:
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


def _clean_json_text(text: str) -> str:
    text = str(text or "").strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text.strip()


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


def _normalize_object_name(value: Any) -> str:
    text = _as_text(value)
    if not text:
        return ""

    parts = [part for part in re.split(r"[^a-z]+", text) if part]
    if not parts:
        return text
    return parts[-1]


def _normalize(raw_query: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_query": raw_query,
        "target_object": _normalize_object_name(data.get("target_object")),
        "target_attributes": _as_list(data.get("target_attributes")),
        "reference_object": _normalize_object_name(data.get("reference_object")),
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

        print("[QueryParser] raw result")
        print(result)

        if isinstance(result, dict):
            parsed = result
        else:
            text = _clean_json_text(str(result))
            parsed = json.loads(text)

        normalized = _normalize(raw_query, parsed)

        if not normalized["target_object"]:
            print("[QueryParser] empty target_object, fallback to raw query")
            return _fallback_parse(raw_query)

        return normalized

    except Exception as exc:
        print(f"[QueryParser] failed, fallback to raw query. error={exc}")
        return _fallback_parse(raw_query)
