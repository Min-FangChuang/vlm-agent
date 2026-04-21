from __future__ import annotations

import argparse

try:
    from agent import Agent
    from prompt import build_candidate_summary
    from read import Read
except ImportError:
    from .agent import Agent
    from .prompt import build_candidate_summary
    from .read import Read


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read posed images in chunks and run agent detection.")
    parser.add_argument("--scene", default="scene0207_00", help="Scene name under vlm-agent/scannet/posed_images")
    parser.add_argument("--query", default="chair", help="Search query passed to Agent.reset()")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum frames per read unit")
    parser.add_argument("--max-units", type=int, default=1, help="Maximum read units to process")
    args = parser.parse_args()

    reader = Read(args.scene, max_frames_per_find=args.max_frames)
    agent = Agent(motion=reader)
    agent.reset(args.query)

    total_views = 0
    total_object_views = 0
    for unit_index in range(args.max_units):
        views = reader.find()
        if not views:
            break

        total_views += len(views)
        object_views = agent.collect_object_views(views)
        total_object_views += len(object_views)
        agent.update_candidates(object_views)

        print(f"unit={unit_index}")
        print(f"views={len(views)}")
        print(f"object_views={len(object_views)}")
        print(f"candidates={len(agent.candidates.values())}")

    print(f"scene={args.scene}")
    print(f"query={args.query}")
    print(f"total_views={total_views}")
    print(f"total_object_views={total_object_views}")
    print(f"total_candidates={len(agent.candidates.values())}")
    for index, candidate in enumerate(agent.candidates.values(), start=1):
        print(f"candidate[{index}] {build_candidate_summary(candidate)}")
