from __future__ import annotations

import json
from typing import Any

import requests


OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o"


def _normalize_messages_to_api_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")

        if isinstance(content, str):
            normalized.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": content,
                        }
                    ],
                }
            )
            continue

        if isinstance(content, list):
            normalized_content: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    normalized_content.append(
                        {
                            "type": "input_text",
                            "text": str(item),
                        }
                    )
                    continue

                if item.get("type") == "text":
                    normalized_content.append(
                        {
                            "type": "input_text",
                            "text": str(item.get("text", "")),
                        }
                    )
                    continue

                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    normalized_content.append(
                        {
                            "type": "input_image",
                            "image_url": image_url.get("url"),
                            "detail": image_url.get("detail", "high"),
                        }
                    )
                    continue

                normalized_content.append(dict(item))

            normalized.append(
                {
                    "role": role,
                    "content": normalized_content,
                }
            )
            continue

        normalized.append(
            {
                "role": role,
                "content": [
                    {
                        "type": "input_text",
                        "text": str(content or ""),
                    }
                ],
            }
        )

    return normalized


def _extract_output_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = data.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        if texts:
            return "\n".join(texts)

    raise RuntimeError(f"OpenAI API response did not contain output text: {json.dumps(data, ensure_ascii=False)}")


def _extract_json_text(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""

    if stripped.startswith("```json"):
        stripped = stripped[len("```json"):].strip()
    elif stripped.startswith("```"):
        stripped = stripped[len("```"):].strip()

    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start:end + 1].strip()

    return ""


def _unwrap_nested_reasoning_json(text: str) -> str:
    outer_json_text = _extract_json_text(text)
    if not outer_json_text:
        return text

    try:
        outer_data = json.loads(outer_json_text)
    except json.JSONDecodeError:
        return outer_json_text

    if not isinstance(outer_data, dict):
        return outer_json_text

    reasoning = outer_data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return outer_json_text

    inner_json_text = _extract_json_text(reasoning)
    if not inner_json_text:
        return outer_json_text

    try:
        inner_data = json.loads(inner_json_text)
    except json.JSONDecodeError:
        return outer_json_text

    if not isinstance(inner_data, dict):
        return outer_json_text

    if "decision" in inner_data:
        return inner_json_text

    return outer_json_text


def call_vlm_api_messages(messages: list[dict[str, Any]]) -> Any:
    api_key = OPENAI_API_KEY.strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is empty in vlm_api_bridge.py.")

    payload = {
        "model": OPENAI_MODEL,
        "input": _normalize_messages_to_api_input(messages),
        "max_output_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(f"OpenAI API request failed: status={response.status_code} body={response.text}")

    data = response.json()
    text = _extract_output_text(data)
    cleaned_text = _unwrap_nested_reasoning_json(text)
    if cleaned_text == text:
        cleaned_text = _extract_json_text(text)
    return cleaned_text or text
