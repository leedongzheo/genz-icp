from __future__ import annotations

import importlib
from abc import ABC
from typing import Optional

import numpy as np


class StubVisualizer(ABC):
    def update(self, source: np.ndarray, local_map: np.ndarray, pose: np.ndarray):
        return

    def close(self):
        return


class RegistrationVisualizer(StubVisualizer):
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'open3d is required for visualization. Install with: pip install "genz-icp[visualizer]"'
            ) from exc

        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="GenZ-ICP Visualizer", width=1600, height=900)

        self._map = self.o3d.geometry.PointCloud()
        self._scan = self.o3d.geometry.PointCloud()
        self._trajectory = self.o3d.geometry.LineSet()

        self._poses_xyz = []
        self._initialized = False

    def _set_points(self, pcd, points: np.ndarray, color: Optional[np.ndarray] = None):
        pcd.points = self.o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        if color is not None:
            pcd.paint_uniform_color(np.asarray(color, dtype=np.float64))

    def update(self, source: np.ndarray, local_map: np.ndarray, pose: np.ndarray):
        source = np.asarray(source, dtype=np.float64)
        local_map = np.asarray(local_map, dtype=np.float64)
        pose = np.asarray(pose, dtype=np.float64)

        self._set_points(self._map, local_map, color=np.array([1.0, 1.0, 0.0]))

        if source.size:
            scan_global = source @ pose[:3, :3].T + pose[:3, 3]
        else:
            scan_global = source
        self._set_points(self._scan, scan_global, color=np.array([0.1, 0.8, 0.1]))

        self._poses_xyz.append(pose[:3, 3].copy())
        if len(self._poses_xyz) >= 2:
            points = np.asarray(self._poses_xyz)
            lines = [[idx, idx + 1] for idx in range(len(points) - 1)]
            self._trajectory.points = self.o3d.utility.Vector3dVector(points)
            self._trajectory.lines = self.o3d.utility.Vector2iVector(lines)
            self._trajectory.colors = self.o3d.utility.Vector3dVector(
                np.tile(np.array([[1.0, 0.1, 0.1]]), (len(lines), 1))
            )

        if not self._initialized:
            self.vis.add_geometry(self._map)
            self.vis.add_geometry(self._scan)
            self.vis.add_geometry(self._trajectory)
            self.vis.get_render_option().background_color = [0.0, 0.0, 0.0]
            self.vis.get_render_option().point_size = 2.0
            self._initialized = True
        else:
            self.vis.update_geometry(self._map)
            self.vis.update_geometry(self._scan)
            self.vis.update_geometry(self._trajectory)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        if hasattr(self, "vis"):
            self.vis.destroy_window()
