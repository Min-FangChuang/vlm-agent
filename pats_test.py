from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
PATS_ROOT = REPO_ROOT / "pats"
if str(PATS_ROOT) not in sys.path:
    sys.path.append(str(PATS_ROOT))

from models.pats import PATS  # type: ignore
from utils.utils import Resize_img  # type: ignore


def load_config(config_path: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PATS matching on a single image pair."
    )
    parser.add_argument(
        "--config", type=str, default=config_path, help="PATS config yaml path"
    )
    parser.add_argument("--image0", required=True, help="First image path")
    parser.add_argument("--image1", required=True, help="Second image path")
    parser.add_argument(
        "--output",
        type=str,
        help="Output visualization path. Default: output/pats_test/<img0>_<img1>.jpg",
    )
    parser.add_argument(
        "--draw-matches",
        action="store_true",
        help="Draw connecting lines between matched keypoints",
    )
    parser.add_argument(
        "--max-vis-matches",
        type=int,
        default=500,
        help="Maximum number of matches to visualize",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override seed from config"
    )
    args = parser.parse_args()
    seed_override = args.seed

    with open(args.config, "r", encoding="utf-8") as file:
        yaml_dict = yaml.safe_load(file)
    for key, value in yaml_dict.items():
        args.__dict__[key] = value

    args.seed = yaml_dict.get("seed", 0) if seed_override is None else seed_override
    return args


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def preprocess_image(
    image_path: str,
    size: int = 640,
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image_rgb = image_bgr[:, :, [2, 1, 0]]
    ori_h, ori_w = image_rgb.shape[:2]
    max_shape = max(ori_h, ori_w)
    scale_factor = size / max_shape
    resized = Resize_img(
        image_rgb,
        np.array([int(ori_w * scale_factor), int(ori_h * scale_factor)]),
    )
    resized_h, resized_w = resized.shape[:2]
    if resized_h > 480 or resized_w > 640:
        raise ValueError(
            f"Image becomes larger than 480x640 after resizing: {image_path}"
        )

    padded = cv2.copyMakeBorder(
        resized,
        0,
        480 - resized_h,
        0,
        640 - resized_w,
        cv2.BORDER_CONSTANT,
        None,
        0,
    )
    return image_bgr, padded, float(scale_factor), (ori_h, ori_w)


def build_model(args: argparse.Namespace) -> PATS:
    model = PATS(args)
    model.config.checkpoint = str(PATS_ROOT / model.config.checkpoint)
    model.config.checkpoint2 = str(PATS_ROOT / model.config.checkpoint2)
    model.config.checkpoint3 = str(PATS_ROOT / model.config.checkpoint3)
    model.load_state_dict()
    return model.cuda().eval()


def filter_matches(
    kp0: torch.Tensor,
    kp1: torch.Tensor,
    scale_factor: float,
    ori_image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    kp0 = torch.round(kp0 / scale_factor).to(torch.int16)
    kp1 = torch.round(kp1 / scale_factor).to(torch.int16)
    ori_h, ori_w = ori_image_shape

    mask0 = torch.logical_and(
        torch.logical_and(kp0[:, 1] >= 0, kp0[:, 1] < ori_w),
        torch.logical_and(kp0[:, 0] >= 0, kp0[:, 0] < ori_h),
    )
    mask1 = torch.logical_and(
        torch.logical_and(kp1[:, 1] >= 0, kp1[:, 1] < ori_w),
        torch.logical_and(kp1[:, 0] >= 0, kp1[:, 0] < ori_h),
    )
    mask = torch.logical_and(mask0, mask1)
    return kp0[mask].cpu().numpy(), kp1[mask].cpu().numpy()


def sample_matches(
    kp0: np.ndarray,
    kp1: np.ndarray,
    max_vis_matches: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(kp0) <= max_vis_matches:
        return kp0, kp1

    indices = np.linspace(0, len(kp0) - 1, max_vis_matches, dtype=int)
    return kp0[indices], kp1[indices]


def get_output_paths(
    args: argparse.Namespace,
    image0_path: str,
    image1_path: str,
) -> tuple[str, str, str]:
    output_path = args.output
    if output_path is None:
        output_dir = REPO_ROOT / "output" / "pats_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        image0_stem = Path(image0_path).stem
        image1_stem = Path(image1_path).stem
        output_path = str(output_dir / f"{image0_stem}_{image1_stem}.jpg")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_root = str(Path(output_path).with_suffix(""))
    return output_path, f"{output_root}.json", f"{output_root}.txt"


def to_xy(points: np.ndarray) -> np.ndarray:
    return np.stack([points[:, 1], points[:, 0]], axis=1)


def _kp_color(x: np.ndarray, y: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    colors = np.stack(
        [
            np.clip(255.0 * x / max(width - 1, 1), 0, 255),
            np.clip(255.0 * y / max(height - 1, 1), 0, 255),
            np.full_like(x, 180.0, dtype=np.float32),
        ],
        axis=1,
    )
    return colors.astype(np.uint8)


def _draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    for index, keypoint in enumerate(keypoints):
        output = cv2.circle(
            output,
            (int(keypoint[1]), int(keypoint[0])),
            1,
            colors[index].tolist(),
            -1,
        )
    return output


def _draw_match_lines(
    image: np.ndarray,
    keypoints0: np.ndarray,
    keypoints1: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    for keypoint0, keypoint1 in zip(keypoints0, keypoints1):
        cv2.line(
            output,
            (int(keypoint0[1]), int(keypoint0[0])),
            (int(keypoint1[1]), int(keypoint1[0])),
            (0, 255, 0),
            1,
        )
    return output


def visualize_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    kp0: np.ndarray,
    kp1: np.ndarray,
    *,
    draw_matches: bool,
) -> np.ndarray:
    left_h, left_w = image0.shape[:2]
    right_h, right_w = image1.shape[:2]
    mask0 = np.logical_and.reduce(
        np.array(
            (kp0[:, 1] >= 0, kp0[:, 1] < left_w, kp0[:, 0] >= 0, kp0[:, 0] < left_h)
        )
    )
    mask1 = np.logical_and.reduce(
        np.array(
            (kp1[:, 1] >= 0, kp1[:, 1] < right_w, kp1[:, 0] >= 0, kp1[:, 0] < right_h)
        )
    )
    keep = np.logical_and(mask0, mask1)
    kp0 = kp0[keep]
    kp1 = kp1[keep]

    colors = _kp_color(kp0[:, 1], kp0[:, 0], (left_h, left_w))
    image0_vis = _draw_keypoints(image0, kp0, colors)
    image1_vis = _draw_keypoints(image1, kp1, colors)

    pad_width = 5
    spacer = np.zeros((left_h, pad_width, 3), dtype=image0_vis.dtype)
    vis = np.concatenate([image0_vis, spacer, image1_vis], axis=1)

    if draw_matches:
        kp1_shifted = kp1.copy()
        kp1_shifted[:, 1] += left_w + pad_width
        vis = _draw_match_lines(vis, kp0, kp1_shifted)
    return vis


def save_match_metadata(
    json_path: str,
    txt_path: str,
    image0_path: str,
    image1_path: str,
    kp0: np.ndarray,
    kp1: np.ndarray,
) -> None:
    kp0_xy = to_xy(kp0)
    kp1_xy = to_xy(kp1)
    match_records = []
    for point0, point1 in zip(kp0_xy.tolist(), kp1_xy.tolist()):
        match_records.append(
            {
                "image0_xy": [int(point0[0]), int(point0[1])],
                "image1_xy": [int(point1[0]), int(point1[1])],
            }
        )

    payload = {
        "image0": image0_path,
        "image1": image1_path,
        "num_matches": len(match_records),
        "matches": match_records,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as file:
        for index, (point0, point1) in enumerate(
            zip(kp0_xy.tolist(), kp1_xy.tolist()), start=1
        ):
            file.write(
                f"[{index}] image0_xy=({int(point0[0])}, {int(point0[1])}) image1_xy=({int(point1[0])}, {int(point1[1])})\n"
            )


def main() -> None:
    default_config = str(PATS_ROOT / "configs" / "test_scannet.yaml")
    args = load_config(default_config)
    set_random_seed(int(args.seed))

    image0_path = str(Path(args.image0).resolve())
    image1_path = str(Path(args.image1).resolve())
    image0_bgr, image0_input, scale_factor0, ori_shape0 = preprocess_image(image0_path)
    image1_bgr, image1_input, scale_factor1, ori_shape1 = preprocess_image(image1_path)

    if abs(scale_factor0 - scale_factor1) > 1e-6:
        raise ValueError(
            "The two images must produce the same scale factor, matching PATS preprocessing behavior."
        )
    if ori_shape0 != ori_shape1:
        print(
            f"[Warning] Original image shapes differ: {ori_shape0} vs {ori_shape1}. Filtering uses image0 shape."
        )

    model = build_model(args)
    data = {
        "image0_name": [Path(image0_path).name],
        "image0": torch.from_numpy(image0_input).unsqueeze(0).float().cuda(),
        "image1_name": [Path(image1_path).name],
        "image1": torch.from_numpy(image1_input).unsqueeze(0).float().cuda(),
    }

    with torch.no_grad():
        with autocast(enabled=torch.cuda.is_available()):
            result = model(data)

    kp0 = result["matches_l"]
    kp1 = result["matches_r"]
    print(f"Raw matches: {len(kp0)}")

    if len(kp0) == 0 or len(kp1) == 0:
        print("No matches found.")
        return

    kp0, kp1 = filter_matches(kp0, kp1, scale_factor0, ori_shape0)
    print(f"Filtered matches: {len(kp0)}")
    if len(kp0) == 0:
        print("No valid matches remain after bounds filtering.")
        return

    kp0_xy = to_xy(kp0)
    kp1_xy = to_xy(kp1)
    preview_count = min(10, len(kp0))
    for index in range(preview_count):
        print(
            f"[{index + 1}] image0_xy={kp0_xy[index].tolist()} image1_xy={kp1_xy[index].tolist()}"
        )

    vis_kp0, vis_kp1 = sample_matches(kp0, kp1, int(args.max_vis_matches))
    vis_image = visualize_matches(
        image0_bgr.copy(),
        image1_bgr.copy(),
        vis_kp0,
        vis_kp1,
        draw_matches=bool(args.draw_matches),
    )

    output_path, json_path, txt_path = get_output_paths(args, image0_path, image1_path)
    cv2.imwrite(output_path, vis_image)
    print(f"Saved visualization to: {output_path}")
    save_match_metadata(json_path, txt_path, image0_path, image1_path, kp0, kp1)
    print(f"Saved match json to: {json_path}")
    print(f"Saved match txt to: {txt_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
