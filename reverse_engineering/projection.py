"""Reusable pinhole projection helpers for the 3D reverse-engineering workspace."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from reverse_engineering.geometry import CameraIntrinsics, CameraModel, CameraExtrinsics
from reverse_engineering.scene import SceneModel, SceneSubject


@dataclass(frozen=True)
class ProjectionPreviewResult:
    points: np.ndarray
    bbox: tuple[float, float, float, float] | None
    in_front_count: int
    total_count: int
    person_index: int = 0

    @property
    def visible_fraction(self) -> float:
        return 0.0 if self.total_count <= 0 else self.in_front_count / self.total_count


def _camera_basis(scene: SceneModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = scene.camera
    position = scene.camera_position()
    forward = camera.forward()
    world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-9)
    roll = math.radians(camera.roll)
    c, s = math.cos(roll), math.sin(roll)
    rolled_right = right * c + up * s
    rolled_up = -right * s + up * c
    return position, rolled_right, rolled_up, forward


def build_camera_model(scene: SceneModel, width: int, height: int) -> CameraModel:
    intrinsics = CameraIntrinsics.from_focal_mm(
        scene.camera.focal_length_mm,
        int(width), int(height),
        scene.camera.sensor_width_mm,
        scene.camera.sensor_width_mm * int(height) / max(int(width), 1),
    )
    position, right, up, forward = _camera_basis(scene)
    rotation = np.vstack([right, up, forward])
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = -rotation @ position
    return CameraModel(
        intrinsics,
        CameraExtrinsics(
            rvec.reshape(3), tvec.reshape(3), position,
            scene.camera.pitch, scene.camera.yaw, scene.camera.roll,
        ),
    )


def subject_world_points_for(scene: SceneModel, subject: SceneSubject) -> np.ndarray:
    points = subject.proxy_points().copy()
    points[:, 0] += subject.center_x
    points[:, 1] += subject.center_y
    points[:, 2] += subject.center_z
    return points


def subject_world_points(scene: SceneModel) -> np.ndarray:
    """Backward-compatible world points for the primary subject."""
    return subject_world_points_for(scene, scene.subject)


def _project_points(scene: SceneModel, subject: SceneSubject, width: int, height: int) -> ProjectionPreviewResult:
    points_3d = subject_world_points_for(scene, subject)
    camera = build_camera_model(scene, width, height)
    points_2d = camera.project_points(points_3d)
    assert camera.extrinsics is not None
    rotation = cv2.Rodrigues(camera.extrinsics.rvec)[0]
    camera_points = (rotation @ points_3d.T + camera.extrinsics.tvec.reshape(3, 1)).T
    front = (camera_points[:, 2] > 1e-6) & np.isfinite(points_2d).all(axis=1)
    visible = points_2d[front]
    bbox = None
    if len(visible):
        x0, y0 = visible.min(axis=0)
        x1, y1 = visible.max(axis=0)
        bbox = (float(x0), float(y0), float(x1), float(y1))
    return ProjectionPreviewResult(
        points=points_2d,
        bbox=bbox,
        in_front_count=int(front.sum()),
        total_count=len(points_3d),
        person_index=int(getattr(subject, "person_index", 0)),
    )


def project_subject(scene: SceneModel, width: int, height: int) -> ProjectionPreviewResult:
    """Project the primary subject; kept for existing callers/tests."""
    return _project_points(scene, scene.subject, width, height)


def project_scene(scene: SceneModel, width: int, height: int) -> tuple[ProjectionPreviewResult, ...]:
    """Project every scene subject using the same calibrated camera model."""
    subjects = list(scene.subjects) if scene.subjects else [scene.subject]
    return tuple(_project_points(scene, subject, int(width), int(height)) for subject in subjects)
