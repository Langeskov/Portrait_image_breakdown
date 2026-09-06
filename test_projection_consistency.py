from __future__ import annotations

import cv2
import numpy as np

from reverse_engineering.geometry import CameraIntrinsics, CameraModel, _camera_pose_from_params


def test_camera_pose_is_proper_rotation():
    _, extrinsics = _camera_pose_from_params(5.0, 1.4, 12.0, -20.0, 6.0)
    rotation = cv2.Rodrigues(extrinsics.rvec)[0]
    assert np.linalg.det(rotation) > 0.99
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)


def test_world_up_projects_above_image_center():
    width, height = 2400, 1600
    intr = CameraIntrinsics.from_focal_mm(50.0, width, height, 36.0, 24.0)
    _, extrinsics = _camera_pose_from_params(5.0, 1.4, 0.0, 0.0, 0.0)
    model = CameraModel(intr, extrinsics)

    camera_level = np.array([0.0, extrinsics.position[1], 0.0])
    above = camera_level + np.array([0.0, 0.5, 0.0])
    center_px = np.asarray(model.project_point(camera_level))
    above_px = np.asarray(model.project_point(above))

    assert np.isfinite(center_px).all()
    assert np.isfinite(above_px).all()
    assert abs(center_px[0] - intr.cx) < 1e-6
    assert abs(center_px[1] - intr.cy) < 1e-6
    assert above_px[1] < center_px[1]


def test_projection_and_unprojection_share_image_y_convention():
    width, height = 2000, 1333
    intr = CameraIntrinsics.from_focal_mm(70.0, width, height)
    _, extrinsics = _camera_pose_from_params(4.0, 1.2, 8.0, -15.0, 4.0)
    model = CameraModel(intr, extrinsics)
    point = np.array([0.2, 0.4, 0.5])
    pixel = np.asarray(model.project_point(point))

    rotation = cv2.Rodrigues(extrinsics.rvec)[0]
    camera_point = rotation @ point + extrinsics.tvec
    recovered_camera_point = model.unproject_point(
        float(pixel[0]), float(pixel[1]), depth=float(camera_point[2])
    )
    recovered_world = np.linalg.inv(rotation) @ recovered_camera_point + extrinsics.position
    assert np.allclose(recovered_world - point, np.zeros(3), atol=1e-5)
