from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from ..agent_schema import View
except ImportError:
    from agent_schema import View  # type: ignore


class Read:
    def __init__(self, scene_name: str, max_frames_per_find: int = 10, frame_skip: int = 1) -> None:
        self.scene_name = scene_name
        self.max_frames_per_find = int(max_frames_per_find)
        self.frame_skip = max(1, int(frame_skip))
        scannet_root = Path(__file__).resolve().parent.parent / "scannet"
        self.scene_dir = scannet_root / "posed_images" / scene_name
        self.alignment_dir = scannet_root / "alignment" / scene_name
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"Posed image scene directory does not exist: {self.scene_dir}")

        self.intrinsic_matrix = self._read_intrinsic_matrix()
        self.world_to_axis_align_matrix = self._read_world_to_axis_align_matrix()

        self.frame_ids = self._discover_frame_ids()
        if not self.frame_ids:
            raise FileNotFoundError(f"No posed frames found under: {self.scene_dir}")
        self._cursor = 0

    def _discover_frame_ids(self) -> list[str]:
        frame_ids: list[str] = []
        for jpg_path in sorted(self.scene_dir.glob("*.jpg")):
            frame_id = jpg_path.stem
            txt_path = self.scene_dir / f"{frame_id}.txt"
            if txt_path.is_file():
                frame_ids.append(frame_id)
        frame_ids.sort(key=int)
        return frame_ids

    def reset(self) -> None:
        self._cursor = 0

    def _read_rgb(self, frame_id: str) -> np.ndarray:
        rgb_path = self.scene_dir / f"{frame_id}.jpg"
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read RGB image: {rgb_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _read_depth(self, frame_id: str) -> np.ndarray | None:
        depth_path = self.scene_dir / f"{frame_id}.png"
        if not depth_path.is_file():
            return None
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
        return depth

    def _read_camera_to_world(self, frame_id: str) -> np.ndarray:
        pose_path = self.scene_dir / f"{frame_id}.txt"
        matrix = np.loadtxt(str(pose_path), dtype=np.float32)
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 pose matrix in {pose_path}, got {matrix.shape}")
        return matrix

    def _read_intrinsic_matrix(self) -> np.ndarray:
        intrinsic_path = self.scene_dir / "intrinsic.txt"
        matrix = np.loadtxt(str(intrinsic_path), dtype=np.float32)
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 intrinsic matrix in {intrinsic_path}, got {matrix.shape}")
        return matrix

    def _read_world_to_axis_align_matrix(self) -> np.ndarray | None:
        alignment_path = self.alignment_dir / f"{self.scene_name}.txt"
        if not alignment_path.is_file():
            return None

        axis_values: list[float] | None = None
        for line in alignment_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("axisAlignment"):
                continue
            _, _, raw_values = stripped.partition("=")
            tokens = raw_values.strip().split()
            axis_values = [float(token) for token in tokens]
            break

        if axis_values is None:
            raise ValueError(f"axisAlignment not found in {alignment_path}")
        if len(axis_values) != 16:
            raise ValueError(f"Expected 16 axisAlignment values in {alignment_path}, got {len(axis_values)}")

        matrix = np.asarray(axis_values, dtype=np.float32).reshape(4, 4)
        return matrix

    def _build_view(self, frame_id: str) -> View:
        return View(
            rgb=self._read_rgb(frame_id),
            depth=self._read_depth(frame_id),
            camera_to_world=self._read_camera_to_world(frame_id),
            view_id=frame_id,
        )

    def _current_view(self) -> View:
        if self._cursor >= len(self.frame_ids):
            raise IndexError("No more frames available.")
        return self._build_view(self.frame_ids[self._cursor])

    def find(self, max_frames: int | None = None) -> list[View]:
        if self._cursor >= len(self.frame_ids):
            return []

        limit = self.max_frames_per_find if max_frames is None else min(int(max_frames), self.max_frames_per_find)
        limit = max(0, limit)
        selected_ids = self.frame_ids[self._cursor : self._cursor + limit * self.frame_skip : self.frame_skip]
        self._cursor += limit * self.frame_skip
        self._cursor = min(self._cursor, len(self.frame_ids))
        return [self._build_view(frame_id) for frame_id in selected_ids]

    def look_around(self) -> list[View]:
        return self.find()
