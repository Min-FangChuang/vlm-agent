from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    import open3d as o3d
except ImportError:  # optional dependency
    o3d = None

try:
    from benchmark.utils import BENCHMARK_DIR, load_pc
except ImportError:
    from .benchmark.utils import BENCHMARK_DIR, load_pc


def _load_target_points(scene_id: str, target_id: int) -> np.ndarray:
    pcd_path = Path(BENCHMARK_DIR) / "pcd_with_global_alignment" / f"{scene_id}.pth"
    pcds, _, _, instance_labels = torch.load(str(pcd_path))
    pcds = np.asarray(pcds)
    instance_labels = np.asarray(instance_labels)
    mask = instance_labels == target_id
    if np.count_nonzero(mask) == 0:
        raise ValueError(f"No GT points found for target_id={target_id} in scene {scene_id}")
    return pcds[mask][:, :3].astype(np.float64)


def _compute_aabb(points: np.ndarray) -> np.ndarray:
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    center = (min_corner + max_corner) / 2.0
    size = max_corner - min_corner
    return np.concatenate([center, size])


def _visualize_points(points: np.ndarray) -> None:
    if o3d is None:
        raise ImportError("open3d is required for 3D visualization.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.1, 0.7, 0.1])

    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1.0, 0.0, 0.0)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([pcd, aabb, frame])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the GT 3D point cloud for one ScanRefer target.")
    parser.add_argument("--data-path", default="benchmark/scanrefer_250.json", help="Path to ScanRefer benchmark json")
    parser.add_argument("--case-index", type=int, required=True, help="0-based case index in benchmark json")
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "inspect_scanrefer_target_3d", help="Directory to save GT target point cloud")
    parser.add_argument("--no-visualize", action="store_true", help="Skip Open3D visualization and only save files")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    cases = json.loads(data_path.read_text(encoding="utf-8"))
    case = cases[args.case_index]

    scene_id = str(case["scan_id"])
    target_id = int(case["target_id"])
    query = str(case.get("caption", ""))

    obj_ids, obj_labels, obj_locs = load_pc(scene_id)
    if target_id not in obj_ids:
        raise ValueError(f"target_id={target_id} not found in load_pc(scene_id={scene_id})")
    target_index = obj_ids.index(target_id)
    target_label = obj_labels[target_index]
    gt_bbox = np.asarray(obj_locs[target_index], dtype=np.float64)

    points = _load_target_points(scene_id, target_id)
    bbox = _compute_aabb(points)

    output_dir = args.output_dir / f"{args.case_index:04d}_{scene_id}_target_{target_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "points_xyz.npy", points)
    np.save(output_dir / "bbox_from_points.npy", bbox)

    summary = {
        "case_index": args.case_index,
        "scene_id": scene_id,
        "target_id": target_id,
        "target_label": target_label,
        "query": query,
        "num_points": int(points.shape[0]),
        "bbox_from_load_pc": gt_bbox.tolist(),
        "bbox_from_points": bbox.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"scene_id={scene_id}")
    print(f"target_id={target_id}")
    print(f"target_label={target_label}")
    print(f"query={query}")
    print(f"num_points={points.shape[0]}")
    print(f"bbox_from_load_pc={gt_bbox.tolist()}")
    print(f"bbox_from_points={bbox.tolist()}")
    print(f"output_dir={output_dir}")

    if not args.no_visualize:
        _visualize_points(points)


if __name__ == "__main__":
    main()
