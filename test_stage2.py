"""Contract tests for geometry, camera fitting, cache, and visualization."""
from __future__ import annotations

import math
import cv2
import numpy as np


def test_geometry_intrinsics_and_camera_axis():
    from reverse_engineering.geometry import CameraIntrinsics, CameraModel
    from reverse_engineering.scene import SceneCamera
    k = CameraIntrinsics.from_focal_mm(50.0, 4000, 3000)
    assert k.fx > 0 and 35 < k.fov_x < 45 and 25 < k.fov_y < 40
    ray = CameraModel(k).unproject_point(2000, 1500, 2.0)
    assert np.isfinite(ray).all() and abs(ray[2] - 2.0) < 1e-6
    cam = SceneCamera(distance=4.0, height=1.5)
    assert np.isclose(np.linalg.norm(cam.forward()), 1.0)
    cam.pitch = 10.0
    assert cam.forward()[1] < 0


def test_camera_fit_does_not_collapse_to_bounds():
    from reverse_engineering.geometry import CameraIntrinsics, CameraModel, CameraExtrinsics, canonical_person_points, PoseSolver
    width, height = 1800, 1200
    focal = 70.0
    intr = CameraIntrinsics.from_focal_mm(focal, width, height)
    object_points = canonical_person_points()
    position = np.array([0.35, 1.05, -6.0])
    target = np.array([0.0, 0.0, 0.0])
    forward = target - position; forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward); right /= np.linalg.norm(right)
    true_up = np.cross(forward, right); true_up /= np.linalg.norm(true_up)
    R = np.vstack([right, -true_up, forward])
    rvec, _ = cv2.Rodrigues(R); tvec = -R @ position
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, intr.to_matrix(), None)
    image_points = projected.reshape(-1,2)
    kp = np.c_[image_points, np.ones(len(image_points))]
    bbox = tuple(np.round([image_points[:,0].min(),image_points[:,1].min(),image_points[:,0].max(),image_points[:,1].max()]).astype(int))
    candidates = PoseSolver.fit_camera_to_pose(kp, width, height, bbox, focal_seeds=(35,50,70,85,105), num_candidates=5)
    assert candidates
    best=candidates[0]
    assert best.score > .35
    assert best.losses["mean_reprojection_px"] < 20.0
    assert .9 < best.distance < 15.0
    assert .4 < best.height < 2.1
    assert 28.0 <= best.focal_equiv_35mm <= 180.0


def test_candidate_solution_family():
    from reverse_engineering.geometry import PoseSolver
    kp=np.zeros((17,3),dtype=float)
    for i in range(17): kp[i]=[900+(i%2)*80,200+i*55,.95]
    kp[5,:2],kp[6,:2]=[850,500],[950,500]
    candidates=PoseSolver.solve_from_body_geometry(kp,1800,1200)
    assert len(candidates)>=3
    assert len({round(c.focal_equiv_35mm) for c in candidates})>=3
    assert all(c.distance>0 for c in candidates)


def test_3d_to_2d_projection_sync():
    from reverse_engineering.scene import SceneModel
    from reverse_engineering.projection import build_camera_model, project_subject
    scene=SceneModel(); model=build_camera_model(scene,4000,3000)
    assert model.extrinsics is not None and model.intrinsics.focal_length_mm>0
    projected=project_subject(scene,4000,3000)
    assert projected.points.shape==(8,2) and projected.bbox is not None and projected.visible_fraction==1.0
    before=projected.bbox; scene.camera.focal_length_mm=100
    assert before!=project_subject(scene,4000,3000).bbox


def test_pose_rescaling_preserves_normalized_position():
    from core.pose_detector import PoseLandmark, PoseResult
    pose=PoseResult([PoseLandmark(0,400,300,0,1.0,.4,.3,0.0)],1000,1000,.9,(300,200,500,600))
    scaled=pose.rescaled(2000,1000)
    assert scaled.landmarks[0].x==800 and scaled.landmarks[0].y==300
    assert np.isclose(scaled.landmarks[0].world_x,.4) and np.isclose(scaled.landmarks[0].world_y,.3)
    assert scaled.bbox==(600,200,1000,600)


def test_roll_estimation_from_tilted_scene_lines():
    from reverse_engineering.camera_pose import estimate_image_roll
    image=np.zeros((400,600,3),dtype=np.uint8); angle=math.radians(6)
    for y in (70,170,270): cv2.line(image,(50,y),(550,int(y+math.tan(angle)*500)),(255,255,255),2)
    roll,confidence,line_count=estimate_image_roll(image)
    assert 5<=roll<=7 and confidence>.2 and line_count>=2


def test_analysis_cache():
    from gui.cache import AnalysisCache,image_cache_key
    image=np.zeros((32,48,3),dtype=np.uint8); key=image_cache_key(image); cache=AnalysisCache(capacity=2); cache[key]="bundle"
    assert key in cache and cache[key]=="bundle"


def test_3d_workspace_contract():
    from gui.reverse_3d import Reverse3DWorkspace,SceneView,ProjectionPreview
    workspace=Reverse3DWorkspace()
    assert hasattr(workspace,"set_result") and hasattr(workspace,"update_results") and hasattr(workspace,"set_image")
    assert isinstance(workspace._view,SceneView) and isinstance(workspace._preview,ProjectionPreview)


if __name__ == "__main__":
    for test in (test_geometry_intrinsics_and_camera_axis,test_camera_fit_does_not_collapse_to_bounds,test_candidate_solution_family,test_3d_to_2d_projection_sync,test_pose_rescaling_preserves_normalized_position,test_roll_estimation_from_tilted_scene_lines,test_analysis_cache,test_3d_workspace_contract):
        test(); print(f"OK: {test.__name__}")
    print("Stage 2 contract tests passed")
