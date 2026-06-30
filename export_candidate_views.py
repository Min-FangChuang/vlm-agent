from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from agent import Agent
    from prompt import build_candidate_summary
    from read import Read
except ImportError:
    from .agent import Agent
    from .prompt import build_candidate_summary
    from .read import Read


def _bbox_to_list(bbox: Any) -> list[float]:
    array = np.asarray(bbox, dtype=np.float32).reshape(-1)
    return [round(float(value), 2) for value in array.tolist()]


def _build_candidate_payload(scene: str, query: str, candidate: Any) -> dict[str, Any]:
    object_views = []
    for object_view in candidate.object_view:
        view = object_view.view
        object_views.append(
            {
                "object_view_id": str(object_view.object_id),
                "view_id": str(view.view_id),
                "image_file": f"{view.view_id}.jpg",
                "bbox_2d": _bbox_to_list(object_view.bbox_2d),
                "label": str(object_view.label),
                "score": round(float(object_view.score), 4),
                "status": str(getattr(object_view, "status", "active")),
            }
        )
    return {
        "scene": scene,
        "query": query,
        "candidate_id": int(candidate.object_id),
        "label": str(candidate.label),
        "status": str(getattr(candidate, "status", "active")),
        "best_id": int(candidate.best_id),
        "num_object_views": len(candidate.object_view),
        "summary": build_candidate_summary(candidate),
        "object_views": object_views,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-candidate JSON with all candidate views and bbox metadata."
    )
    parser.add_argument(
        "--scene", required=True, help="Scene name under scannet/posed_images"
    )
    parser.add_argument("--query", required=True, help="Query passed to Agent.reset()")
    parser.add_argument(
        "--max-frames", type=int, default=40, help="Maximum frames per read unit"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Sample every Nth frame when building views",
    )
    parser.add_argument(
        "--max-units", type=int, default=3, help="Maximum read units to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "candidate_exports",
        help="Directory to save per-candidate JSON files",
    )
    args = parser.parse_args()

    reader = Read(
        args.scene, max_frames_per_find=args.max_frames, frame_skip=args.frame_skip
    )
    agent = Agent(motion=reader)
    agent.reset(args.query)

    total_views = 0
    for unit_index in range(args.max_units):
        views = reader.find()
        if not views:
            break
        total_views += len(views)
        agent.consume_views(views)
        print(
            f"unit={unit_index} views={len(views)} candidates={len(agent.candidates.values())}"
        )

    output_dir = args.output_dir / args.scene / "_".join(args.query.split())
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scene": args.scene,
        "query": args.query,
        "frame_skip": args.frame_skip,
        "max_frames": args.max_frames,
        "max_units": args.max_units,
        "total_views": total_views,
        "num_candidates": len(agent.candidates.values()),
        "candidates": [],
    }

    for candidate in agent.candidates.values():
        payload = _build_candidate_payload(args.scene, args.query, candidate)
        file_name = f"candidate_{int(candidate.object_id):03d}.json"
        (output_dir / file_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["candidates"].append(
            {
                "candidate_id": int(candidate.object_id),
                "file": file_name,
                "num_object_views": len(candidate.object_view),
                "label": str(candidate.label),
                "summary": build_candidate_summary(candidate),
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"scene={args.scene}")
    print(f"query={args.query}")
    print(f"total_views={total_views}")
    print(f"num_candidates={len(agent.candidates.values())}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
