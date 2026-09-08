"""3D scene model for photography reverse engineering."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional
import numpy as np
from reverse_engineering.geometry import PoseCandidate, pose_driven_person_points
from reverse_engineering.data_types import ReverseEngineeringResult


@dataclass
class SceneCamera:
    distance: float = 4.0
    height: float = 1.5
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    focal_length_mm: float = 50.0
    sensor_width_mm: float = 36.0

    @property
    def horizontal_fov_deg(self):
        return math.degrees(2.0 * math.atan((self.sensor_width_mm * .5) / max(self.focal_length_mm, 1e-6)))

    @property
    def vertical_fov_deg(self):
        return math.degrees(2.0 * math.atan(((self.sensor_width_mm * 2 / 3) * .5) / max(self.focal_length_mm, 1e-6)))

    def position(self, target=None):
        target = np.asarray(target if target is not None else [0.0, 0.0, 0.0], dtype=float)
        y = math.radians(self.yaw)
        return target + np.array([math.sin(y) * self.distance, self.height, -math.cos(y) * self.distance])

    def forward(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        v = np.array([-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)], dtype=float)
        return v / max(np.linalg.norm(v), 1e-9)

    def aim_point(self, target=None):
        """Return the point on the optical axis nearest the intended subject target."""
        target = np.asarray(target if target is not None else [0.0, 0.0, 0.0], dtype=float)
        position = self.position(target)
        forward = self.forward()
        along = float(np.dot(target - position, forward))
        along = max(0.1, along)
        return position + forward * along

    def aim_error(self, target=None):
        target = np.asarray(target if target is not None else [0.0, 0.0, 0.0], dtype=float)
        return float(np.linalg.norm(self.aim_point(target) - target))


@dataclass
class SceneSubject:
    height: float = 1.70
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    keypoints: Optional[np.ndarray] = None
    fitted_points_3d: Optional[np.ndarray] = None
    person_index: int = 0
    depth_is_relative: bool = False
    depth_confidence: float = 0.0

    def proxy_points(self):
        if self.fitted_points_3d is not None:
            return self.fitted_points_3d.copy()
        h = self.height
        return np.array([
            [-.16, -.85, 0], [.16, -.85, 0], [-.14, -.05, 0], [.14, -.05, 0],
            [-.18, .37, 0], [.18, .37, 0], [0, .81, .08], [0, .85, .08],
        ], dtype=float) * np.array([1.0, h, 1.0])


@dataclass
class SceneModel:
    camera: SceneCamera = field(default_factory=SceneCamera)
    subject: SceneSubject = field(default_factory=SceneSubject)
    ground_size: float = 24.0
    candidate_solutions: list[PoseCandidate] = field(default_factory=list)
    selected_candidate: int = 0
    subjects: list[SceneSubject] = field(default_factory=list)
    relative_layout: bool = False

    def __post_init__(self):
        if not self.subjects:
            self.subjects = [self.subject]
        elif self.subject not in self.subjects:
            self.subjects.insert(0, self.subject)

    @classmethod
    def from_reverse_result(cls, result: Optional[ReverseEngineeringResult]) -> "SceneModel":
        scene = cls()
        if result is None:
            return scene
        pose = getattr(result, "subject_keypoints", None)
        if pose:
            kp = np.array([[lm.x, lm.y, lm.visibility] for lm in pose[:17]], dtype=float)
            scene.subject.keypoints = kp
            scene.subject.fitted_points_3d = pose_driven_person_points(kp, result.image_size[0], result.image_size[1], scene.subject.height)
        scene.candidate_solutions = list(getattr(result, "candidate_solutions", None) or getattr(result, "_sim_candidates", []) or [])
        if scene.candidate_solutions:
            scene.selected_candidate = max(range(len(scene.candidate_solutions)), key=lambda i: scene.candidate_solutions[i].score)
            scene.set_candidate(scene.selected_candidate)
        else:
            cp = result.camera_pose
            fl = result.focal_length.equivalent_35mm
            scene.camera = SceneCamera(
                float(cp.camera_distance.value or 4), float(cp.camera_height.value or 1.5),
                float(cp.camera_yaw.value or 0), float(cp.camera_pitch.value or 0),
                float(cp.camera_roll.value or 0), float(fl.value or 50),
            )

        layout = getattr(result, "multi_person_layout", None)
        if layout is not None and getattr(layout, "people", None):
            people = list(layout.people)
            image_w, image_h = result.image_size
            primary = people[0]
            scene.subjects = []
            base_cx, base_cy = primary.center
            for person in people:
                kp_rows = np.asarray(person.keypoints, dtype=float)
                if kp_rows.ndim == 2 and kp_rows.shape[0] >= 17:
                    kp_pixels = kp_rows[:17].copy()
                    kp_pixels[:, 0] *= image_w
                    kp_pixels[:, 1] *= image_h
                    fitted = pose_driven_person_points(kp_pixels, image_w, image_h, scene.subject.height)
                else:
                    kp_pixels = None
                    fitted = None
                # The x/y conversion keeps the observed image arrangement in the
                # same camera-facing coordinate frame. It is deliberately scaled
                # from FOV and camera distance instead of pretending to be a room
                # survey. Relative z is used only when the layout has independent
                # depth evidence.
                nx, ny = person.center
                lateral = (nx - base_cx) * 2.0 * math.tan(math.radians(scene.camera.horizontal_fov_deg) * 0.5) * scene.camera.distance
                vertical = (base_cy - ny) * 2.0 * math.tan(math.radians(scene.camera.vertical_fov_deg) * 0.5) * scene.camera.distance
                depth_offset = float(person.relative_z) * max(0.5, scene.camera.distance * 0.35) if person.usable_3d else 0.0
                subject = SceneSubject(
                    height=scene.subject.height,
                    center_x=float(lateral),
                    center_y=float(vertical),
                    center_z=float(depth_offset),
                    keypoints=kp_pixels,
                    fitted_points_3d=fitted,
                    person_index=int(person.person_index),
                    depth_is_relative=bool(layout.independent_depth and person.usable_3d),
                    depth_confidence=float(person.depth_confidence),
                )
                scene.subjects.append(subject)
            if scene.subjects:
                scene.subject = next((s for s in scene.subjects if s.person_index == primary.person_index), scene.subjects[0])
            scene.relative_layout = bool(layout.independent_depth)
        else:
            scene.subjects = [scene.subject]
        return scene

    def camera_position(self):
        return self.camera.position(self.camera_target())

    def camera_target(self):
        return np.array([self.subject.center_x, self.subject.center_y, self.subject.center_z], dtype=float)

    def camera_aim_target(self):
        return self.camera.aim_point(self.camera_target())

    def camera_aim_error(self):
        return self.camera.aim_error(self.camera_target())

    def set_candidate(self, index):
        if not self.candidate_solutions:
            return
        index = max(0, min(index, len(self.candidate_solutions) - 1))
        c = self.candidate_solutions[index]
        self.selected_candidate = index
        self.camera.distance = float(c.distance)
        self.camera.height = float(c.height)
        self.camera.focal_length_mm = float(c.focal_equiv_35mm)
        self.camera.pitch = float(getattr(c.extrinsics, 'pitch', self.camera.pitch))
        self.camera.yaw = float(getattr(c.extrinsics, 'yaw', self.camera.yaw))
        self.camera.roll = float(getattr(c.extrinsics, 'roll', self.camera.roll))

    def candidate_summary(self):
        return [
            {"index": i, "focal_length_mm": c.focal_equiv_35mm, "distance_m": c.distance,
             "height_m": c.height, "score": c.score}
            for i, c in enumerate(self.candidate_solutions)
        ]
