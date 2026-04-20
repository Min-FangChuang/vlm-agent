from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None


MatrixLike = np.ndarray
ActionValidator = Callable[["SceneObservationScene", Mapping[str, Any], MatrixLike], "ActionValidationResult"]
DEFAULT_SCANNET_ROOT = Path(__file__).resolve().parent / "scannet" / "init"


def _read_matrix(matrix_path: Path) -> np.ndarray:
    matrix = np.loadtxt(matrix_path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix from {matrix_path}, got {matrix.shape}.")
    return matrix


def _require_open3d() -> Any:
    if o3d is None:
        raise ModuleNotFoundError(
            "open3d is required for rendering. Install it with `pip install open3d`."
        )
    return o3d


def _read_scannet_metadata(metadata_path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _axis_alignment_from_metadata(metadata: Mapping[str, str], metadata_path: Path) -> np.ndarray:
    value = metadata.get("axisAlignment")
    if value is None:
        raise ValueError(f"axisAlignment is missing from {metadata_path}.")
    numbers = np.fromstring(value, sep=" ", dtype=np.float64)
    if numbers.shape[0] != 16:
        raise ValueError(f"axisAlignment in {metadata_path} must contain 16 values, got {numbers.shape[0]}.")
    return numbers.reshape(4, 4)


def _int_from_metadata(metadata: Mapping[str, str], key: str, default: int) -> int:
    value = metadata.get(key)
    if value is None:
        return default
    return int(float(value))


def _build_transform(
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_xyz_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    rx, ry, rz = np.deg2rad(rotation_xyz_deg)
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0], [-np.sin(ry), 0.0, np.cos(ry)]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot_z @ rot_y @ rot_x
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _normalize_vector(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"Cannot normalize zero-length vector for {name}.")
    return vector / norm


def _level_camera_pose(camera_to_world: np.ndarray) -> np.ndarray:
    leveled = np.asarray(camera_to_world, dtype=np.float64).copy()
    position = leveled[:3, 3].copy()
    rotation = leveled[:3, :3]

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_world = -rotation[:, 2]
    forward_world[2] = 0.0
    if np.linalg.norm(forward_world) <= 1e-12:
        forward_world = rotation[:, 1].copy()
        forward_world[2] = 0.0
    forward_world = _normalize_vector(forward_world, "leveled forward")
    right_world = _normalize_vector(np.cross(forward_world, world_up), "leveled right")
    up_world = _normalize_vector(np.cross(right_world, forward_world), "leveled up")

    leveled[:3, 0] = right_world
    leveled[:3, 1] = up_world
    leveled[:3, 2] = -forward_world
    leveled[:3, 3] = position
    return leveled


def _to_vec3(value: Any, name: str) -> Tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape[0] != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {array.shape[0]}.")
    return (float(array[0]), float(array[1]), float(array[2]))


@dataclass
class CameraSpec:
    width: int
    height: int
    intrinsic_matrix: np.ndarray
    depth_scale: float = 1000.0
    depth_max: float = 20.0

    @classmethod
    def from_4x4(
        cls,
        intrinsic_matrix: MatrixLike,
        width: int,
        height: int,
        depth_scale: float = 1000.0,
        depth_max: float = 20.0,
    ) -> "CameraSpec":
        matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 intrinsic matrix, got {matrix.shape}.")
        return cls(
            width=width,
            height=height,
            intrinsic_matrix=matrix,
            depth_scale=depth_scale,
            depth_max=depth_max,
        )

    def to_open3d_intrinsic(self) -> Any:
        open3d = _require_open3d()
        return open3d.camera.PinholeCameraIntrinsic(
            self.width,
            self.height,
            float(self.intrinsic_matrix[0, 0]),
            float(self.intrinsic_matrix[1, 1]),
            float(self.intrinsic_matrix[0, 2]),
            float(self.intrinsic_matrix[1, 2]),
        )


@dataclass
class ActionValidationResult:
    ok: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneObservation:
    frame_index: int
    rgb: np.ndarray
    depth: np.ndarray
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    axis_alignment_matrix: np.ndarray
    color_camera: CameraSpec
    depth_camera: CameraSpec
    validation: ActionValidationResult


class SceneObservationScene:
    def __init__(
        self,
        scene_dir: str | Path,
        ply_name: Optional[str] = None,
        pose_name: str = "00000.txt",
        metadata_name: Optional[str] = None,
        color_intrinsic_name: str = "intrinsic.txt",
        depth_intrinsic_name: str = "depth_intrinsic.txt",
        color_size: Tuple[int, int] = (1296, 968),
        depth_size: Tuple[int, int] = (640, 480),
        point_size: float = 2.0,
        background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        validator: Optional[ActionValidator] = None,
    ) -> None:
        self.scene_dir = Path(scene_dir)
        self.metadata_path = self.scene_dir / (metadata_name or f"{self.scene_dir.name}.txt")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"ScanNet metadata file does not exist: {self.metadata_path}")
        self.metadata = _read_scannet_metadata(self.metadata_path)
        self.axis_alignment_matrix = _axis_alignment_from_metadata(self.metadata, self.metadata_path)
        color_size = (
            _int_from_metadata(self.metadata, "colorWidth", color_size[0]),
            _int_from_metadata(self.metadata, "colorHeight", color_size[1]),
        )
        depth_size = (
            _int_from_metadata(self.metadata, "depthWidth", depth_size[0]),
            _int_from_metadata(self.metadata, "depthHeight", depth_size[1]),
        )
        self.color_camera = CameraSpec.from_4x4(
            _read_matrix(self.scene_dir / color_intrinsic_name),
            width=color_size[0],
            height=color_size[1],
        )
        self.depth_camera = CameraSpec.from_4x4(
            _read_matrix(self.scene_dir / depth_intrinsic_name),
            width=depth_size[0],
            height=depth_size[1],
        )
        raw_camera_to_world = _read_matrix(self.scene_dir / pose_name)
        self.initial_camera_to_world = self.axis_alignment_matrix @ raw_camera_to_world
        self.camera_to_world = self.initial_camera_to_world.copy()
        self.frame_index = 0
        self.validator = validator or self.default_validator
        self.background_color = background_color
        self.point_size = point_size
        self.geometry_name = "scene"

        if ply_name is None:
            ply_candidates = sorted(self.scene_dir.glob("*.ply"))
            if not ply_candidates:
                raise FileNotFoundError(f"No .ply file found in {self.scene_dir}.")
            self.ply_path = ply_candidates[0]
        else:
            self.ply_path = self.scene_dir / ply_name

        self.geometry, self.geometry_kind = self._load_geometry(self.ply_path)

        self._color_renderer: Any = self._build_renderer(self.color_camera)
        self._depth_renderer: Any = self._build_renderer(self.depth_camera)

    def _load_geometry(self, geometry_path: Path) -> tuple[Any, str]:
        open3d = _require_open3d()

        triangle_mesh = open3d.io.read_triangle_mesh(str(geometry_path))
        if not triangle_mesh.is_empty() and len(triangle_mesh.triangles) > 0:
            triangle_mesh.transform(self.axis_alignment_matrix)
            if not triangle_mesh.has_vertex_normals():
                triangle_mesh.compute_vertex_normals()
            return triangle_mesh, "mesh"

        point_cloud = open3d.io.read_point_cloud(str(geometry_path))
        if point_cloud.is_empty():
            raise ValueError(f"Loaded geometry is empty: {geometry_path}")
        point_cloud.transform(self.axis_alignment_matrix)
        if not point_cloud.has_normals():
            point_cloud.estimate_normals()
        return point_cloud, "point_cloud"

    def _build_renderer(self, camera: CameraSpec) -> Any:
        open3d = _require_open3d()
        renderer = open3d.visualization.rendering.OffscreenRenderer(camera.width, camera.height)
        renderer.scene.set_background(np.asarray(self.background_color, dtype=np.float32))
        material = open3d.visualization.rendering.MaterialRecord()
        if self.geometry_kind == "mesh":
            material.shader = "defaultLit"
        else:
            material.shader = "defaultUnlit"
            material.point_size = self.point_size
        renderer.scene.add_geometry(self.geometry_name, self.geometry, material)
        renderer.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 75000)
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.scene.enable_indirect_light(True)
        return renderer

    @staticmethod
    def default_validator(
        scene: "SceneObservationScene",
        action: Mapping[str, Any],
        candidate_camera_to_world: MatrixLike,
    ) -> ActionValidationResult:
        del scene, action, candidate_camera_to_world
        return ActionValidationResult(ok=True)

    def reset(self) -> None:
        self.camera_to_world = self.initial_camera_to_world.copy()
        self.frame_index = 0

    def current_pose(self) -> np.ndarray:
        return self.camera_to_world.copy()

    def validate_action(
        self,
        action: Mapping[str, Any],
        candidate_camera_to_world: MatrixLike,
    ) -> ActionValidationResult:
        return self.validator(self, action, np.asarray(candidate_camera_to_world, dtype=np.float64))

    def move(
        self,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_xyz_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        transform: Optional[MatrixLike] = None,
        relative_to: str = "camera",
    ) -> SceneObservation:
        if transform is None:
            transform = _build_transform(translation=translation, rotation_xyz_deg=rotation_xyz_deg)
        else:
            transform = np.asarray(transform, dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}.")

        if relative_to == "camera":
            candidate_camera_to_world = self.camera_to_world @ transform
        elif relative_to == "world":
            candidate_camera_to_world = transform @ self.camera_to_world
        else:
            raise ValueError("relative_to must be either 'camera' or 'world'.")

        action = {
            "translation": tuple(float(v) for v in translation),
            "rotation_xyz_deg": tuple(float(v) for v in rotation_xyz_deg),
            "relative_to": relative_to,
            "transform": transform.copy(),
        }
        validation = self.validate_action(action, candidate_camera_to_world)
        if not validation.ok:
            raise ValueError(validation.reason or "Action rejected by validator.")

        self.camera_to_world = candidate_camera_to_world
        self.frame_index += 1
        return self.get_observation(validation=validation)

    def _render_color(self) -> np.ndarray:
        if self._color_renderer is None:
            raise RuntimeError("Renderer is closed.")
        world_to_camera = np.linalg.inv(self.camera_to_world)
        self._color_renderer.setup_camera(
            self.color_camera.to_open3d_intrinsic(),
            world_to_camera,
        )
        image = self._color_renderer.render_to_image()
        return np.asarray(image)

    def _render_depth(self) -> np.ndarray:
        if self._depth_renderer is None:
            raise RuntimeError("Renderer is closed.")
        world_to_camera = np.linalg.inv(self.camera_to_world)
        self._depth_renderer.setup_camera(
            self.depth_camera.to_open3d_intrinsic(),
            world_to_camera,
        )
        image = self._depth_renderer.render_to_depth_image(z_in_view_space=True)
        return np.asarray(image, dtype=np.float32)

    def get_observation(
        self,
        validation: Optional[ActionValidationResult] = None,
    ) -> SceneObservation:
        validation = validation or ActionValidationResult(ok=True)
        return SceneObservation(
            frame_index=self.frame_index,
            rgb=self._render_color(),
            depth=self._render_depth(),
            camera_to_world=self.camera_to_world.copy(),
            world_to_camera=np.linalg.inv(self.camera_to_world),
            axis_alignment_matrix=self.axis_alignment_matrix.copy(),
            color_camera=self.color_camera,
            depth_camera=self.depth_camera,
            validation=validation,
        )

    @classmethod
    def from_scannet_init(
        cls,
        scene_id: str,
        root: str | Path = DEFAULT_SCANNET_ROOT,
        **kwargs: Any,
    ) -> "SceneObservationScene":
        return cls(Path(root) / scene_id, **kwargs)

    def close(self) -> None:
        for renderer_name in ("_color_renderer", "_depth_renderer"):
            renderer = getattr(self, renderer_name, None)
            if renderer is None:
                continue
            try:
                renderer.scene.clear_geometry()
            except Exception:
                pass
            try:
                del renderer
            except Exception:
                pass
            setattr(self, renderer_name, None)
        gc.collect()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def sample_floor_guard(
    min_height: float,
    max_height: Optional[float] = None,
) -> ActionValidator:
    def _validator(
        scene: SceneObservationScene,
        action: Mapping[str, Any],
        candidate_camera_to_world: MatrixLike,
    ) -> ActionValidationResult:
        del scene, action
        height = float(candidate_camera_to_world[2, 3])
        if height < min_height:
            return ActionValidationResult(ok=False, reason=f"Camera z={height:.3f} is below {min_height:.3f}.")
        if max_height is not None and height > max_height:
            return ActionValidationResult(ok=False, reason=f"Camera z={height:.3f} is above {max_height:.3f}.")
        return ActionValidationResult(ok=True, metadata={"z": height})

    return _validator


class VLMSceneBridge:
    def __init__(self, scene: SceneObservationScene) -> None:
        self.scene = scene

    @staticmethod
    def _pack_observation(observation: SceneObservation) -> Dict[str, Any]:
        return {
            "frame_index": observation.frame_index,
            "rgb": observation.rgb,
            "depth": observation.depth,
            "camera_to_world": observation.camera_to_world,
            "world_to_camera": observation.world_to_camera,
            "axis_alignment_matrix": observation.axis_alignment_matrix,
            "color_camera": {
                "width": observation.color_camera.width,
                "height": observation.color_camera.height,
                "intrinsic_matrix": observation.color_camera.intrinsic_matrix.copy(),
            },
            "depth_camera": {
                "width": observation.depth_camera.width,
                "height": observation.depth_camera.height,
                "intrinsic_matrix": observation.depth_camera.intrinsic_matrix.copy(),
                "depth_scale": observation.depth_camera.depth_scale,
                "depth_max": observation.depth_camera.depth_max,
            },
            "validation": {
                "ok": observation.validation.ok,
                "reason": observation.validation.reason,
                "metadata": dict(observation.validation.metadata),
            },
        }

    def observe(self) -> Dict[str, Any]:
        return self._pack_observation(self.scene.get_observation())

    def step(self, action: Mapping[str, Any]) -> Dict[str, Any]:
        observation = self.scene.move(
            translation=_to_vec3(action.get("translation"), "translation"),
            rotation_xyz_deg=_to_vec3(action.get("rotation_xyz_deg"), "rotation_xyz_deg"),
            transform=action.get("transform"),
            relative_to=str(action.get("relative_to", "camera")),
        )
        return self._pack_observation(observation)


if __name__ == "__main__":
    scene = SceneObservationScene.from_scannet_init("scene0207_00")
    observation = scene.get_observation()
    moved = scene.move(translation=(0.0, 0.0, -0.25), rotation_xyz_deg=(0.0, 15.0, 0.0))
    print("initial rgb:", observation.rgb.shape, observation.rgb.dtype)
    print("initial depth:", observation.depth.shape, observation.depth.dtype)
    print("moved pose:\n", moved.camera_to_world)
