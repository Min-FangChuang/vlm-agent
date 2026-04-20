from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from agent import Agent
from motion import Motion
from prompt import build_candidate_summary
from scene_controller import SceneControlModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal agent smoke test.")
    parser.add_argument("--scene", default="scene0207_00", help="Scene name under vlm-agent/scannet/init")
    parser.add_argument("--query", default="this is the closet doors in the corner of the room . a bed is present in front of the closet .", help="Search query passed to Agent.reset()")
    parser.add_argument("--max-translation-step", type=float, default=0.1, help="Max translation step in meters")
    parser.add_argument("--max-rotation-step-deg", type=float, default=5.0, help="Max rotation step in degrees")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save the collected detections and candidates as JSON",
    )
    args = parser.parse_args()

    controller = SceneControlModule(
        args.scene,
        max_translation_step=args.max_translation_step,
        max_rotation_step_deg=args.max_rotation_step_deg,
    )
    controller.look_up(10.0)
    agent = Agent(motion=Motion(controller))

    try:
        agent.reset(args.query)

        initial_view = agent.observe()
        print(f"scene={args.scene}")
        print(f"query={args.query}")
        print(f"initial_view_id={initial_view.view_id}")

        views = agent.initial_scan()
        print(f"look_around_views={len(views)}")

        object_views = agent.collect_object_views(views)
        print(f"object_views={len(object_views)}")

        agent.update_candidates(object_views)
        candidates = agent.candidates.values()
        print(f"candidates={len(candidates)}")

        for index, candidate in enumerate(candidates, start=1):
            print(f"candidate[{index}] {build_candidate_summary(candidate)}")

        decision_candidate, decision = agent.evaluate_candidates()
        print(f"decision={decision}")
        if decision_candidate is not None:
            print(f"selected_candidate={build_candidate_summary(decision_candidate)}")

        if args.output_json:
            payload = {
                "scene": args.scene,
                "query": args.query,
                "initial_view_id": initial_view.view_id,
                "look_around_views": len(views),
                "object_views": len(object_views),
                "initial_view_detections": [
                    {
                        "label": str(item.label),
                        "score": float(item.score),
                        "bbox": np.asarray(item.bbox, dtype=np.float32).tolist(),
                    }
                    for item in agent.detect_target_objects(initial_view)
                ],
                "candidates": [
                    {
                        "object_id": item.object_id,
                        "label": item.label,
                        "score": float(item.score),
                        "best_id": int(item.best_id),
                        "num_object_views": len(item.object_view),
                        "best_bbox": np.asarray(item.current_best_view().bbox_2d, dtype=np.float32).tolist(),
                    }
                    for item in candidates
                ],
                "decision": decision,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"saved_json={args.output_json}")
    finally:
        controller.close()
