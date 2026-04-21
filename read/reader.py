from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from ..agent_schema import View
except ImportError:
    from agent_schema import View  # type: ignore


class Read:
    def __init__(self, scene_name: str, max_frames_per_find: int = 10) -> None:
        self.scene_name = scene_name
        self.max_frames_per_find = int(max_frames_per_find)
        self.scene_dir = Path(__file__).resolve().parent.parent / "scannet" / "posed_images" / scene_name
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"Posed image scene directory does not exist: {self.scene_dir}")

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
        selected_ids = self.frame_ids[self._cursor : self._cursor + limit]
        self._cursor += len(selected_ids)
        return [self._build_view(frame_id) for frame_id in selected_ids]

    def look_around(self) -> list[View]:
        return self.find()
