from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import imageio.v2 as imageio
import numpy as np

try:
    from scannet.open3d_scene_agent import (
        ActionValidator,
        SceneObservationScene,
        VLMSceneBridge,
    )
except ModuleNotFoundError:
    from open3d_scene_agent import (  # type: ignore
        ActionValidator,
        SceneObservationScene,
        VLMSceneBridge,
    )


REQUIRED_SCENE_FILES = (
    "00000.txt",
    "intrinsic.txt",
    "depth_intrinsic.txt",
)
DEFAULT_SCANNET_ROOT = Path(__file__).resolve().parent / "scannet" / "init"


def _normalize_scene_name(scene_name: str) -> str:
    scene_name = scene_name.strip()
    if not scene_name:
        raise ValueError("scene_name must not be empty.")
    return scene_name


def _rotation_x(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]],
        dtype=np.float64,
    )


def _rotation_z(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _translation_transform(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return transform


def _rotation_transform(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _world_z_rotation(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return _rotation_transform(rotation)


def _depth_to_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return preview
    min_depth = float(np.min(depth[valid]))
    max_depth = float(np.max(depth[valid]))
    if max_depth <= min_depth:
        preview[valid] = 255
        return preview
    normalized = (depth[valid] - min_depth) / (max_depth - min_depth)
    preview[valid] = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return preview


@dataclass
class ControlStepResult:
    action_type: str
    command: str
    value: float
    transform: np.ndarray
    observation: Dict[str, Any]
    substeps: list[Dict[str, Any]] = field(default_factory=list)


class SceneControlModule:
    def __init__(
        self,
        scene_name: str,
        scene_root: str | Path = DEFAULT_SCANNET_ROOT,
        validator: Optional[ActionValidator] = None,
        max_translation_step: float = 0.1,
        max_rotation_step_deg: float = 5.0,
        **scene_kwargs: Any,
    ) -> None:
        self.scene_name = _normalize_scene_name(scene_name)
        self.scene_root = Path(scene_root)
        self.scene_dir = self.resolve_scene_dir(self.scene_name, self.scene_root)
        self._validate_scene_files(self.scene_dir)
        if max_translation_step <= 0:
            raise ValueError("max_translation_step must be positive.")
        if max_rotation_step_deg <= 0:
            raise ValueError("max_rotation_step_deg must be positive.")
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step_deg = float(max_rotation_step_deg)
        self.scene = SceneObservationScene(self.scene_dir, validator=validator, **scene_kwargs)
        self.bridge = VLMSceneBridge(self.scene)

    @staticmethod
    def resolve_scene_dir(scene_name: str, scene_root: str | Path = DEFAULT_SCANNET_ROOT) -> Path:
        scene_root = Path(scene_root)
        scene_dir = scene_root / _normalize_scene_name(scene_name)
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
        return scene_dir

    @staticmethod
    def _validate_scene_files(scene_dir: Path) -> None:
        missing = [name for name in REQUIRED_SCENE_FILES if not (scene_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Scene {scene_dir.name} is missing required files: {missing}")
        metadata_path = scene_dir / f"{scene_dir.name}.txt"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Scene {scene_dir.name} is missing metadata/alignment file: {metadata_path.name}"
            )
        if not any(scene_dir.glob("*.ply")):
            raise FileNotFoundError(f"Scene {scene_dir.name} does not contain a .ply file.")

    @staticmethod
    def available_scenes(scene_root: str | Path = DEFAULT_SCANNET_ROOT) -> list[str]:
        root = Path(scene_root)
        if not root.exists():
            return []
        return sorted(path.name for path in root.iterdir() if path.is_dir())

    def observe(self) -> Dict[str, Any]:
        return self.bridge.observe()

    def reset(self) -> Dict[str, Any]:
        self.scene.reset()
        return self.observe()

    def current_pose(self) -> np.ndarray:
        return self.scene.current_pose()

    def close(self) -> None:
        self.scene.close()

    def save_observation(
        self,
        output_dir: str | Path,
        observation: Dict[str, Any],
        prefix: Optional[str] = None,
        save_depth_preview: bool = True,
    ) -> Dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame_index = int(observation.get("frame_index", 0))
        file_prefix = prefix or f"frame_{frame_index:05d}"

        rgb_path = output_path / f"{file_prefix}_rgb.png"
        depth_path = output_path / f"{file_prefix}_depth.npy"
        pose_path = output_path / f"{file_prefix}_camera_to_world.txt"
        intrinsic_path = output_path / f"{file_prefix}_intrinsic.txt"
        depth_intrinsic_path = output_path / f"{file_prefix}_depth_intrinsic.txt"

        imageio.imwrite(rgb_path, np.asarray(observation["rgb"], dtype=np.uint8))
        np.save(depth_path, np.asarray(observation["depth"], dtype=np.float32))
        np.savetxt(pose_path, np.asarray(observation["camera_to_world"], dtype=np.float64), fmt="%.6f")
        np.savetxt(
            intrinsic_path,
            np.asarray(observation["color_camera"]["intrinsic_matrix"], dtype=np.float64),
            fmt="%.6f",
        )
        np.savetxt(
            depth_intrinsic_path,
            np.asarray(observation["depth_camera"]["intrinsic_matrix"], dtype=np.float64),
            fmt="%.6f",
        )

        saved_paths = {
            "frame_index": str(frame_index),
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "camera_to_world": str(pose_path),
            "intrinsic": str(intrinsic_path),
            "depth_intrinsic": str(depth_intrinsic_path),
        }

        if save_depth_preview:
            depth_preview_path = output_path / f"{file_prefix}_depth_preview.png"
            imageio.imwrite(depth_preview_path, _depth_to_preview(np.asarray(observation["depth"], dtype=np.float32)))
            saved_paths["depth_preview"] = str(depth_preview_path)

        return saved_paths

    def save_current_observation(
        self,
        output_dir: str | Path,
        prefix: Optional[str] = None,
        save_depth_preview: bool = True,
    ) -> Dict[str, str]:
        return self.save_observation(
            output_dir=output_dir,
            observation=self.observe(),
            prefix=prefix,
            save_depth_preview=save_depth_preview,
        )

    def save_step_result(
        self,
        step_result: ControlStepResult,
        output_dir: str | Path,
        prefix: Optional[str] = None,
        save_depth_preview: bool = True,
    ) -> Dict[str, str]:
        return self.save_observation(
            output_dir=output_dir,
            observation=step_result.observation,
            prefix=prefix or step_result.command,
            save_depth_preview=save_depth_preview,
        )

    def save_step_sequence(
        self,
        step_result: ControlStepResult,
        output_dir: str | Path,
        prefix: Optional[str] = None,
        save_depth_preview: bool = True,
    ) -> list[Dict[str, str]]:
        saved: list[Dict[str, str]] = []
        for index, observation in enumerate(step_result.substeps or [step_result.observation]):
            current_prefix = None if prefix is None else f"{prefix}_{index + 1:03d}"
            saved.append(
                self.save_observation(
                    output_dir=output_dir,
                    observation=observation,
                    prefix=current_prefix,
                    save_depth_preview=save_depth_preview,
                )
            )
        return saved

    def _step(self, action_type: str, command: str, value: float, transform: np.ndarray) -> ControlStepResult:
        observation = self.bridge.step({"transform": transform, "relative_to": "camera"})
        return ControlStepResult(
            action_type=action_type,
            command=command,
            value=float(value),
            transform=transform,
            observation=observation,
            substeps=[observation],
        )

    def _step_with_relative_frame(
        self,
        action_type: str,
        command: str,
        value: float,
        transform: np.ndarray,
        relative_to: str,
    ) -> ControlStepResult:
        observation = self.bridge.step({"transform": transform, "relative_to": relative_to})
        return ControlStepResult(
            action_type=action_type,
            command=command,
            value=float(value),
            transform=transform,
            observation=observation,
            substeps=[observation],
        )

    @staticmethod
    def _split_value(total_value: float, max_step: float) -> list[float]:
        magnitude = abs(float(total_value))
        if magnitude == 0.0:
            return [0.0]
        sign = 1.0 if total_value >= 0 else -1.0
        steps: list[float] = []
        remaining = magnitude
        while remaining > max_step:
            steps.append(sign * max_step)
            remaining -= max_step
        if remaining > 0:
            steps.append(sign * remaining)
        return steps

    @staticmethod
    def _compose_transforms(transforms: list[np.ndarray]) -> np.ndarray:
        total = np.eye(4, dtype=np.float64)
        for transform in transforms:
            total = total @ transform
        return total

    def _move_by_translation(self, command: str, translation_xyz: Iterable[float], distance: float) -> ControlStepResult:
        translation = np.asarray(tuple(translation_xyz), dtype=np.float64)
        norm = float(np.linalg.norm(translation))
        if norm == 0.0:
            transform = np.eye(4, dtype=np.float64)
            return self._step("move", command, distance, transform)

        direction = translation / norm
        step_values = self._split_value(distance, self.max_translation_step)
        transforms: list[np.ndarray] = []
        substeps: list[Dict[str, Any]] = []
        last_result: Optional[ControlStepResult] = None
        for step_distance in step_values:
            step_translation = direction * step_distance
            transform = _translation_transform(
                float(step_translation[0]),
                float(step_translation[1]),
                float(step_translation[2]),
            )
            transforms.append(transform)
            last_result = self._step("move", command, step_distance, transform)
            substeps.extend(last_result.substeps)

        if last_result is None:
            transform = np.eye(4, dtype=np.float64)
            return self._step("move", command, distance, transform)

        return ControlStepResult(
            action_type="move",
            command=command,
            value=float(distance),
            transform=self._compose_transforms(transforms),
            observation=last_result.observation,
            substeps=substeps,
        )

    def _pose_by_rotation(self, command: str, angle_deg: float) -> ControlStepResult:
        abs_angle = abs(float(angle_deg))
        if abs_angle == 0.0:
            transform = np.eye(4, dtype=np.float64)
            return self._step("look", command, angle_deg, transform)

        step_values = self._split_value(angle_deg, self.max_rotation_step_deg)
        transforms: list[np.ndarray] = []
        substeps: list[Dict[str, Any]] = []
        last_result: Optional[ControlStepResult] = None
        for step_angle in step_values:
            if command in {"look_up", "look_down"}:
                step_rotation = _rotation_x(step_angle if command == "look_down" else -step_angle)
            elif command in {"look_left", "look_right"}:
                step_rotation = _rotation_z(step_angle if command == "look_right" else -step_angle)
            else:
                raise ValueError(f"Unsupported pose command: {command}")
            transform = _rotation_transform(step_rotation)
            transforms.append(transform)
            last_result = self._step("look", command, step_angle, transform)
            substeps.extend(last_result.substeps)

        if last_result is None:
            transform = np.eye(4, dtype=np.float64)
            return self._step("look", command, angle_deg, transform)

        return ControlStepResult(
            action_type="look",
            command=command,
            value=float(angle_deg),
            transform=self._compose_transforms(transforms),
            observation=last_result.observation,
            substeps=substeps,
        )

    def _yaw_by_world_up(self, command: str, angle_deg: float) -> ControlStepResult:
        abs_angle = abs(float(angle_deg))
        if abs_angle == 0.0:
            transform = np.eye(4, dtype=np.float64)
            return self._step_with_relative_frame("look", command, angle_deg, transform, "world")

        signed_total = float(angle_deg) if command == "look_right" else -float(angle_deg)
        step_values = self._split_value(signed_total, self.max_rotation_step_deg)
        transforms: list[np.ndarray] = []
        substeps: list[Dict[str, Any]] = []
        last_result: Optional[ControlStepResult] = None
        for step_angle in step_values:
            current_pose = self.current_pose()
            position = current_pose[:3, 3]
            world_rotation = _world_z_rotation(step_angle)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = world_rotation[:3, :3]
            transform[:3, 3] = position - world_rotation[:3, :3] @ position
            transforms.append(transform)
            last_result = self._step_with_relative_frame("look", command, step_angle, transform, "world")
            substeps.extend(last_result.substeps)

        if last_result is None:
            transform = np.eye(4, dtype=np.float64)
            return self._step_with_relative_frame("look", command, angle_deg, transform, "world")

        return ControlStepResult(
            action_type="look",
            command=command,
            value=float(angle_deg),
            transform=self._compose_transforms(transforms),
            observation=last_result.observation,
            substeps=substeps,
        )

    def forward(self, distance: float) -> ControlStepResult:
        return self._move_by_translation("forward", (0.0, 0.0, -float(distance)), distance)

    def backward(self, distance: float) -> ControlStepResult:
        return self._move_by_translation("backward", (0.0, 0.0, float(distance)), distance)

    def left(self, distance: float) -> ControlStepResult:
        return self._move_by_translation("left", (-float(distance), 0.0, 0.0), distance)

    def right(self, distance: float) -> ControlStepResult:
        return self._move_by_translation("right", (float(distance), 0.0, 0.0), distance)

    def move_command(self, direction: str, distance: float) -> ControlStepResult:
        actions = {
            "forward": self.forward,
            "backward": self.backward,
            "left": self.left,
            "right": self.right,
        }
        key = direction.strip().lower()
        if key not in actions:
            raise ValueError(f"Unsupported move direction: {direction}")
        return actions[key](distance)

    def look_up(self, angle_deg: float) -> ControlStepResult:
        return self._pose_by_rotation("look_up", angle_deg)

    def look_down(self, angle_deg: float) -> ControlStepResult:
        return self._pose_by_rotation("look_down", angle_deg)

    def look_left(self, angle_deg: float) -> ControlStepResult:
        return self._yaw_by_world_up("look_left", angle_deg)

    def look_right(self, angle_deg: float) -> ControlStepResult:
        return self._yaw_by_world_up("look_right", angle_deg)

    def view_pose(self, direction: str, angle_deg: float) -> ControlStepResult:
        actions = {
            "up": self.look_up,
            "down": self.look_down,
            "left": self.look_left,
            "right": self.look_right,
        }
        key = direction.strip().lower()
        if key not in actions:
            raise ValueError(f"Unsupported view direction: {direction}")
        return actions[key](angle_deg)


if __name__ == "__main__":
    controller = SceneControlModule("scene0207_00")
    print("available scenes:", SceneControlModule.available_scenes())
    print("current pose:\n", controller.current_pose())
    result = controller.forward(0.25)
    print("forward transform:\n", result.transform)
