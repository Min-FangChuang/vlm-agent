from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def call_vlm_messages(messages: list[dict[str, Any]]) -> Any:
    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "backend" / "vlm_messages.js"

    if not script_path.exists():
        raise FileNotFoundError(f"VLM JS backend not found: {script_path}")

    payload = {
        "messages": messages,
        "model": "gpt-5.2",
        "max_output_tokens": 300,
    }

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

    if stderr:
        print("[vlm_bridge] backend stderr:")
        print(stderr)

    if not stdout:
        raise RuntimeError(f"VLM backend returned empty stdout. stderr={stderr}")

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse VLM backend stdout as JSON. stdout={stdout} stderr={stderr}"
        ) from exc

    if not data.get("success", False):
        raise RuntimeError(data.get("error", "Unknown VLM backend error."))

    return data.get("result")
