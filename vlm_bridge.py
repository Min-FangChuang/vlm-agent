from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _is_retryable_vlm_error(message: str) -> bool:
    lowered = str(message or "").strip().lower()
    retryable_markers = [
        "fetch failed",
        "empty stdout",
        "empty vlm result",
        "failed to parse vlm backend stdout as json",
        "did not return a response object",
        "unexpected token '<'",
        "unexpected end of json input",
        "is not valid json",
        "<html>",
        "terminated",
        "und_err_socket",
        "other side closed",
        "socketerror",
    ]
    return any(marker in lowered for marker in retryable_markers)


def _is_retryable_vlm_result_text(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    retryable_markers = [
        "upstream",
        "upstream connect error",
        "bad gateway",
        "gateway",
        "temporarily unavailable",
        "connection reset",
    ]
    return any(marker in lowered for marker in retryable_markers)


def call_vlm_messages(messages: list[dict[str, Any]]) -> Any:
    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "backend" / "vlm_messages.js"

    if not script_path.exists():
        raise FileNotFoundError(f"VLM JS backend not found: {script_path}")

    payload = {
        "messages": messages,
        "model": "gpt-5.4",
        "max_output_tokens": 300,
    }

    max_attempts = 3
    retry_delays = [1.0, 3.0]
    last_error: RuntimeError | None = None

    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(
            ["node", str(script_path)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
            cwd=str(repo_root),
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        try:
            if not stdout:
                raise RuntimeError(
                    f"VLM backend returned empty stdout. stderr={stderr}"
                )

            try:
                data = json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Failed to parse VLM backend stdout as JSON. "
                    f"stdout={stdout} stderr={stderr}"
                ) from exc

            if not data.get("success", False):
                raise RuntimeError(data.get("error", "Unknown VLM backend error."))

            result_text = data.get("result")
            if result_text is None or not str(result_text).strip():
                raise RuntimeError("Empty VLM result.")
            if isinstance(result_text, str) and _is_retryable_vlm_result_text(
                result_text
            ):
                raise RuntimeError(result_text)

            return result_text
        except RuntimeError as exc:
            last_error = exc
            if attempt >= max_attempts or not _is_retryable_vlm_error(str(exc)):
                raise
            sleep_seconds = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
            error_text = str(exc).replace("\n", " ").strip()
            if len(error_text) > 180:
                error_text = error_text[:177] + "..."
            print(
                f"[vlm_bridge] retry attempt={attempt}/{max_attempts} "
                f"sleep={sleep_seconds:.1f}s error={error_text}"
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unknown VLM backend failure.")
