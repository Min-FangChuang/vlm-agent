from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from benchmark.utils import BENCHMARK_DIR, load_pc
    from module.detector import draw_bbox
    from read import Read
except ImportError:
    from .benchmark.utils import BENCHMARK_DIR, load_pc
    from .module.detector import draw_bbox
    from .read import Read


def _load_scene_points(scene_id: str):
    pcds, _, _, instance_labels = torch.load(
        os.path.join(BENCHMARK_DIR, "pcd_with_global_alignment", f"{scene_id}.pth")
    )
    return np.asarray(pcds), np.asarray(instance_labels)


def _project_points(
    points_xyz_aligned: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    world_to_axis_align_matrix: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    points_xyz = np.asarray(points_xyz_aligned, dtype=np.float64)
    if world_to_axis_align_matrix is not None:
        axis_align = np.asarray(world_to_axis_align_matrix, dtype=np.float64)
        aligned_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
        raw_world_h = (np.linalg.inv(axis_align) @ aligned_h.T).T
        points_xyz = raw_world_h[:, :3]

    world_to_camera = np.linalg.inv(camera_to_world)
    points_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (world_to_camera @ points_h.T).T
    valid = cam[:, 2] > 1e-6
    cam = cam[valid]
    if cam.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), valid

    proj = (intrinsic @ cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    return uv, valid


def _bbox_from_projected_points(uv: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray | None:
    if uv.shape[0] == 0:
        return None
    h, w = image_shape[:2]
    x = uv[:, 0]
    y = uv[:, 1]
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if np.count_nonzero(inside) < 20:
        return None

    x = x[inside]
    y = y[inside]
    x1 = max(0, int(np.floor(np.min(x))))
    y1 = max(0, int(np.floor(np.min(y))))
    x2 = min(w - 1, int(np.ceil(np.max(x))))
    y2 = min(h - 1, int(np.ceil(np.max(y))))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _slugify(text: str) -> str:
    chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        elif ch in {" ", "/", "\\", ":", "."}:
            chars.append("_")
    return "".join(chars).strip("_") or "case"


def main() -> None:
    parser = argparse.ArgumentParser(description="Save posed views with the GT ScanRefer object highlighted.")
    parser.add_argument("--data-path", default="benchmark/scanrefer_250.json", help="Path to ScanRefer benchmark json")
    parser.add_argument("--case-index", type=int, required=True, help="0-based case index in benchmark json")
    parser.add_argument("--max-saved-views", type=int, default=30, help="Maximum number of visible GT views to save")
    parser.add_argument("--frame-skip", type=int, default=1, help="Sample every Nth posed frame")
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "inspect_scanrefer_case", help="Directory to save highlighted views")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    cases = json.loads(data_path.read_text(encoding="utf-8"))
    case = cases[args.case_index]

    scene_id = str(case["scan_id"])
    target_id = int(case["target_id"])
    query = str(case.get("caption", ""))

    pcd_path = Path(BENCHMARK_DIR) / "pcd_with_global_alignment" / f"{scene_id}.pth"
    pcds, _, _, instance_labels = torch.load(str(pcd_path))
    pcds = np.asarray(pcds)
    instance_labels = np.asarray(instance_labels)
    mask = instance_labels == target_id
    if np.count_nonzero(mask) == 0:
        raise ValueError(f"No GT points found for target_id={target_id} in scene {scene_id}")
    target_points = pcds[mask][:, :3].astype(np.float64)

    reader = Read(scene_id, max_frames_per_find=999999, frame_skip=args.frame_skip)
    views = reader.find()

    output_dir = args.output_dir / f"{args.case_index:04d}_{scene_id}_{_slugify(query)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "case_index": args.case_index,
        "scene_id": scene_id,
        "target_id": target_id,
        "query": query,
        "saved_views": [],
    }

    saved = 0
    for view in views:
        uv, _ = _project_points(
            target_points,
            reader.intrinsic_matrix.astype(np.float64),
            np.asarray(view.camera_to_world, dtype=np.float64),
            None if reader.world_to_axis_align_matrix is None else np.asarray(reader.world_to_axis_align_matrix, dtype=np.float64),
        )
        bbox = _bbox_from_projected_points(uv, view.rgb.shape)
        if bbox is None:
            continue

        vis = draw_bbox(view.rgb, bbox, f"GT target {target_id}", color=(0, 255, 0))
        file_path = output_dir / f"{saved:03d}_{view.view_id}.png"
        cv2.imwrite(str(file_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        manifest["saved_views"].append(
            {
                "view_id": view.view_id,
                "bbox_2d": bbox.tolist(),
                "file": file_path.name,
            }
        )
        saved += 1
        if saved >= args.max_saved_views:
            break

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"scene_id={scene_id}")
    print(f"target_id={target_id}")
    print(f"query={query}")
    print(f"saved_views={saved}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    import os

    main()
