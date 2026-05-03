from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:  # optional dependency
    o3d = None

ArrayLike = np.ndarray


@dataclass
class MorphologyConfig:
    erosion: bool = True
    dilation: bool = False
    kernel_size: int = 3
    keep_largest_components: bool = True
    num_components: int = 1


@dataclass
class PointFilterConfig:
    filter_type: str = "statistical"  # statistical | truncated | none
    nb_neighbors: int = 20
    std_ratio: float = 1.0
    tx: float = 0.05
    ty: float = 0.05
    tz: float = 0.05


@dataclass
class ProjectionInput:
    depth_image: Union[str, ArrayLike]
    intrinsic_matrix: ArrayLike
    extrinsic_matrix: ArrayLike
    world_to_axis_align_matrix: Optional[ArrayLike] = None
    mask: Optional[ArrayLike] = None
    color_image: Optional[Union[str, ArrayLike]] = None


class TwoDToThreeDTool:
    """
    Standalone 2D->3D estimation tool.

    Main flow:
        2D mask -> 3D points -> optional filtering -> 3D AABB
    """

    def __init__(
        self,
        morphology: Optional[MorphologyConfig] = None,
        point_filter: Optional[PointFilterConfig] = None,
        project_color: bool = False,
        depth_scale: float = 0.001,
    ) -> None:
        self.morphology = morphology or MorphologyConfig()
        self.point_filter = point_filter or PointFilterConfig()
        self.project_color = project_color
        self.depth_scale = depth_scale

    def post_process_mask(self, mask: ArrayLike) -> ArrayLike:
        """Clean a binary mask with morphology and connected-components filtering."""
        if mask is None:
            raise ValueError("mask must not be None")

        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape={mask.shape}")

        img = (mask > 0).astype(np.uint8) * 255
        k = self.morphology.kernel_size
        kernel = np.ones((k * 2 + 1, k * 2 + 1), np.uint8)

        if self.morphology.erosion:
            img = cv2.erode(img, kernel, iterations=1)
        if self.morphology.dilation:
            img = cv2.dilate(img, kernel, iterations=1)

        if self.morphology.keep_largest_components:
            num_labels, labels_im = cv2.connectedComponents(img)
            if num_labels > 1:
                component_areas = [
                    (label, int(np.sum(labels_im == label)))
                    for label in range(1, num_labels)
                ]
                component_areas.sort(key=lambda x: x[1], reverse=True)
                kept = [label for label, _ in component_areas[: self.morphology.num_components]]
                img = np.isin(labels_im, kept).astype(np.uint8) * 255

        return img.astype(bool)

    @staticmethod
    def build_projection_input_from_object_view(
        object_view: Any,
        *,
        intrinsic_matrix: ArrayLike,
        world_to_axis_align_matrix: Optional[ArrayLike] = None,
        project_color: bool = False,
    ) -> ProjectionInput:
        view = getattr(object_view, "view", None)
        if view is None:
            raise ValueError("object_view.view must not be None")
        if getattr(object_view, "mask_2d", None) is None:
            raise ValueError("object_view.mask_2d must not be None")
        if getattr(view, "depth", None) is None:
            raise ValueError("object_view.view.depth must not be None")
        if getattr(view, "camera_to_world", None) is None:
            raise ValueError("object_view.view.camera_to_world must not be None")

        return ProjectionInput(
            depth_image=np.asarray(view.depth),
            intrinsic_matrix=np.asarray(intrinsic_matrix, dtype=np.float64),
            extrinsic_matrix=np.asarray(view.camera_to_world, dtype=np.float64),
            world_to_axis_align_matrix=(
                None
                if world_to_axis_align_matrix is None
                else np.asarray(world_to_axis_align_matrix, dtype=np.float64)
            ),
            mask=np.asarray(object_view.mask_2d),
            color_image=np.asarray(view.rgb) if project_color else None,
        )

    def build_projection_inputs_from_candidate(
        self,
        candidate: Any,
        *,
        intrinsic_matrix: ArrayLike,
        world_to_axis_align_matrix: Optional[ArrayLike] = None,
        use_best_only: bool = False,
    ) -> list[ProjectionInput]:
        object_views = list(getattr(candidate, "object_view", []) or [])
        if use_best_only and object_views:
            best_id = int(getattr(candidate, "best_id", 0))
            if 0 <= best_id < len(object_views):
                object_views = [object_views[best_id]]
            else:
                object_views = []

        inputs: list[ProjectionInput] = []
        for object_view in object_views:
            if getattr(object_view, "mask_2d", None) is None:
                continue
            view = getattr(object_view, "view", None)
            if view is None or getattr(view, "depth", None) is None or getattr(view, "camera_to_world", None) is None:
                continue
            inputs.append(
                self.build_projection_input_from_object_view(
                    object_view,
                    intrinsic_matrix=intrinsic_matrix,
                    world_to_axis_align_matrix=world_to_axis_align_matrix,
                    project_color=self.project_color,
                )
            )
        return inputs

    def project_candidate_to_3d(
        self,
        candidate: Any,
        *,
        intrinsic_matrix: ArrayLike,
        world_to_axis_align_matrix: Optional[ArrayLike] = None,
        do_post_process: bool = True,
        use_best_only: bool = False,
    ) -> Tuple[ArrayLike, ArrayLike]:
        inputs = self.build_projection_inputs_from_candidate(
            candidate,
            intrinsic_matrix=intrinsic_matrix,
            world_to_axis_align_matrix=world_to_axis_align_matrix,
            use_best_only=use_best_only,
        )
        if not inputs:
            raise ValueError("Candidate does not contain any projectable object views with mask_2d.")
        return self.run_multi_view(inputs, do_post_process=do_post_process)

    def update_candidate_3d(
        self,
        candidate: Any,
        *,
        intrinsic_matrix: ArrayLike,
        world_to_axis_align_matrix: Optional[ArrayLike] = None,
        do_post_process: bool = True,
        use_best_only: bool = False,
    ) -> Tuple[ArrayLike, ArrayLike]:
        points, bbox = self.project_candidate_to_3d(
            candidate,
            intrinsic_matrix=intrinsic_matrix,
            world_to_axis_align_matrix=world_to_axis_align_matrix,
            do_post_process=do_post_process,
            use_best_only=use_best_only,
        )
        candidate.points_3d = points
        return points, bbox

    @staticmethod
    def visualize_points_and_aabb(points: ArrayLike, bbox: ArrayLike) -> None:
        if o3d is None:
            raise ImportError("open3d is required for visualization.")

        xyz = np.asarray(points, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] < 3:
            raise ValueError("points must be a non-empty Nx3 or Nx6 array.")

        bbox_array = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if bbox_array.shape[0] != 6:
            raise ValueError("bbox must contain 6 values: center_x, center_y, center_z, dx, dy, dz.")

        center = bbox_array[:3]
        dimensions = bbox_array[3:]
        min_corner = center - dimensions / 2.0
        max_corner = center + dimensions / 2.0

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz[:, :3])
        if xyz.shape[1] >= 6:
            point_cloud.colors = o3d.utility.Vector3dVector(np.clip(xyz[:, 3:6] / 255.0, 0.0, 1.0))

        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_corner, max_bound=max_corner)
        aabb.color = (1.0, 0.0, 0.0)
        o3d.visualization.draw_geometries([point_cloud, aabb])

    def project_mask_to_3d(self, data: ProjectionInput) -> ArrayLike:
        """
        Project a 2D mask into 3D using depth + intrinsics + extrinsics.

        Returns:
            Nx3 array or Nx6 array if color is enabled.
        """
        depth_image = self._load_depth(data.depth_image)
        color_image = self._load_color(data.color_image) if self.project_color else None

        if data.mask is None:
            ref_h, ref_w = color_image.shape[:2] if color_image is not None else depth_image.shape[:2]
            mask = np.ones((ref_h, ref_w), dtype=bool)
        else:
            mask = np.asarray(data.mask).astype(bool)

        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape={mask.shape}")

        intrinsic = np.asarray(data.intrinsic_matrix, dtype=np.float64)
        extrinsic = np.asarray(data.extrinsic_matrix, dtype=np.float64)
        axis_align = None if data.world_to_axis_align_matrix is None else np.asarray(data.world_to_axis_align_matrix, dtype=np.float64)

        if intrinsic.shape != (4, 4):
            raise ValueError(f"intrinsic_matrix must be 4x4, got {intrinsic.shape}")
        if extrinsic.shape != (4, 4):
            raise ValueError(f"extrinsic_matrix must be 4x4, got {extrinsic.shape}")
        if axis_align is not None and axis_align.shape != (4, 4):
            raise ValueError(f"world_to_axis_align_matrix must be 4x4, got {axis_align.shape}")

        scale_y = depth_image.shape[0] / mask.shape[0]
        scale_x = depth_image.shape[1] / mask.shape[1]

        mask_y, mask_x = np.where(mask)
        if len(mask_x) == 0:
            feature_dim = 6 if color_image is not None else 3
            return np.empty((0, feature_dim), dtype=np.float64)

        depth_y = np.round(mask_y * scale_y).astype(int)
        depth_x = np.round(mask_x * scale_x).astype(int)
        depth_y = np.clip(depth_y, 0, depth_image.shape[0] - 1)
        depth_x = np.clip(depth_x, 0, depth_image.shape[1] - 1)

        depth_values = depth_image[depth_y, depth_x].astype(np.float64) * self.depth_scale
        valid = depth_values > 0
        depth_values = depth_values[valid]
        mask_x = mask_x[valid]
        mask_y = mask_y[valid]

        if len(depth_values) == 0:
            feature_dim = 6 if color_image is not None else 3
            return np.empty((0, feature_dim), dtype=np.float64)

        homogeneous_pixels = np.vstack([
            mask_x * depth_values,
            mask_y * depth_values,
            depth_values,
            np.ones_like(depth_values),
        ])

        cam_coords = np.linalg.inv(intrinsic) @ homogeneous_pixels
        world_coords = extrinsic @ cam_coords

        if axis_align is not None:
            world_coords = axis_align @ world_coords

        xyz = world_coords[:3].T
        if color_image is not None:
            rgb = color_image[mask_y, mask_x]
            return np.hstack((xyz, rgb))
        return xyz

    def project_views_to_3d(self, views: Sequence[ProjectionInput]) -> ArrayLike:
        """Project multiple views and concatenate their 3D points."""
        if not views:
            feature_dim = 6 if self.project_color else 3
            return np.empty((0, feature_dim), dtype=np.float64)

        all_points: List[ArrayLike] = []
        for view in views:
            pts = self.project_mask_to_3d(view)
            if pts.shape[0] > 0:
                all_points.append(pts)

        if not all_points:
            feature_dim = 6 if self.project_color else 3
            return np.empty((0, feature_dim), dtype=np.float64)
        return np.concatenate(all_points, axis=0)

    def filter_points(self, points: ArrayLike) -> ArrayLike:
        if points.size == 0:
            return points
        cfg = self.point_filter
        if cfg.filter_type == "none":
            return points
        if cfg.filter_type == "statistical":
            return self.remove_statistical_outliers(points, cfg.nb_neighbors, cfg.std_ratio)
        if cfg.filter_type == "truncated":
            return self.remove_truncated_outliers(points, cfg.tx, cfg.ty, cfg.tz)
        raise NotImplementedError(f"Unknown filter_type: {cfg.filter_type}")

    @staticmethod
    def remove_statistical_outliers(point_cloud_data: ArrayLike, nb_neighbors: int = 20, std_ratio: float = 1.0) -> ArrayLike:
        if point_cloud_data.shape[0] == 0:
            return point_cloud_data
        if o3d is None:
            raise ImportError(
                "open3d is required for statistical outlier removal. "
                "Install it or switch filter_type to 'truncated' or 'none'."
            )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
        _, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        return point_cloud_data[ind, :]

    @staticmethod
    def remove_truncated_outliers(point_cloud_data: ArrayLike, tx: float, ty: float, tz: float) -> ArrayLike:
        if point_cloud_data.shape[0] == 0:
            return point_cloud_data
        if not (0 <= tx < 0.5 and 0 <= ty < 0.5 and 0 <= tz < 0.5):
            raise ValueError("tx, ty, tz must each be in [0, 0.5)")
        if tx == 0 and ty == 0 and tz == 0:
            return point_cloud_data

        n_points = len(point_cloud_data)
        nx, ny, nz = int(tx * n_points), int(ty * n_points), int(tz * n_points)
        x_sorted = np.argsort(point_cloud_data[:, 0])
        y_sorted = np.argsort(point_cloud_data[:, 1])
        z_sorted = np.argsort(point_cloud_data[:, 2])
        valid_x = x_sorted[nx:-nx] if 2 * nx < n_points and nx > 0 else x_sorted
        valid_y = y_sorted[ny:-ny] if 2 * ny < n_points and ny > 0 else y_sorted
        valid_z = z_sorted[nz:-nz] if 2 * nz < n_points and nz > 0 else z_sorted
        valid_idx = np.intersect1d(valid_x, valid_y)
        valid_idx = np.intersect1d(valid_idx, valid_z)
        return point_cloud_data[valid_idx]

    @staticmethod
    def calculate_aabb(point_cloud_data: ArrayLike) -> ArrayLike:
        if point_cloud_data.shape[0] == 0:
            raise ValueError("Cannot calculate AABB from empty point cloud")
        min_corner = np.min(point_cloud_data[:, :3], axis=0)
        max_corner = np.max(point_cloud_data[:, :3], axis=0)
        center = (max_corner + min_corner) / 2.0
        dimensions = max_corner - min_corner
        return np.concatenate([center, dimensions])

    def run_single_view(
        self,
        *,
        mask: ArrayLike,
        depth_image: Union[str, ArrayLike],
        intrinsic_matrix: ArrayLike,
        extrinsic_matrix: ArrayLike,
        world_to_axis_align_matrix: Optional[ArrayLike] = None,
        color_image: Optional[Union[str, ArrayLike]] = None,
        do_post_process: bool = True,
    ) -> Tuple[ArrayLike, ArrayLike]:
        clean_mask = self.post_process_mask(mask) if do_post_process else np.asarray(mask).astype(bool)
        points = self.project_mask_to_3d(
            ProjectionInput(
                depth_image=depth_image,
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
                world_to_axis_align_matrix=world_to_axis_align_matrix,
                mask=clean_mask,
                color_image=color_image,
            )
        )
        filtered_points = self.filter_points(points)
        bbox = self.calculate_aabb(filtered_points)
        return filtered_points, bbox

    def run_multi_view(self, views: Sequence[ProjectionInput], do_post_process: bool = True) -> Tuple[ArrayLike, ArrayLike]:
        processed_views: List[ProjectionInput] = []
        for view in views:
            if view.mask is None:
                processed_views.append(view)
            else:
                clean_mask = self.post_process_mask(view.mask) if do_post_process else np.asarray(view.mask).astype(bool)
                processed_views.append(
                    ProjectionInput(
                        depth_image=view.depth_image,
                        intrinsic_matrix=view.intrinsic_matrix,
                        extrinsic_matrix=view.extrinsic_matrix,
                        world_to_axis_align_matrix=view.world_to_axis_align_matrix,
                        mask=clean_mask,
                        color_image=view.color_image,
                    )
                )
        points = self.project_views_to_3d(processed_views)
        filtered_points = self.filter_points(points)
        bbox = self.calculate_aabb(filtered_points)
        return filtered_points, bbox

    @staticmethod
    def _load_depth(depth_image: Union[str, ArrayLike]) -> ArrayLike:
        if isinstance(depth_image, str):
            loaded = cv2.imread(depth_image, -1)
            if loaded is None:
                raise FileNotFoundError(f"Failed to load depth image: {depth_image}")
            return loaded
        return np.asarray(depth_image)

    @staticmethod
    def _load_color(color_image: Optional[Union[str, ArrayLike]]) -> Optional[ArrayLike]:
        if color_image is None:
            return None
        if isinstance(color_image, str):
            loaded = cv2.imread(color_image)
            if loaded is None:
                raise FileNotFoundError(f"Failed to load color image: {color_image}")
            return cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        return np.asarray(color_image)
