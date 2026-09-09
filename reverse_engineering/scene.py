"""3D scene model for photography reverse engineering."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional
import numpy as np
from reverse_engineering.geometry import PoseCandidate, pose_driven_person_points
from reverse_engineering.data_types import ReverseEngineeringResult
from reverse_engineering.scene_anchors import SceneAnchor, default_scene_anchors, next_anchor_id
from reverse_engineering.camera_semantics import CameraOrbit, CameraOpticalAim

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
    @property
    def orbit(self) -> CameraOrbit:
        return CameraOrbit(float(self.distance), float(self.height), float(self.yaw), float(self.pitch))
    def set_orbit(self, orbit: CameraOrbit) -> None:
        self.distance = float(orbit.distance_m); self.height = float(orbit.height_m); self.yaw = float(orbit.yaw_deg); self.pitch = float(orbit.pitch_deg)
    @property
    def optical_aim(self) -> CameraOpticalAim:
        return CameraOpticalAim(reframe_yaw_deg=0.0, reframe_pitch_deg=0.0, roll_deg=float(self.roll))
    def position(self, target=None):
        target = np.asarray(target if target is not None else [0.0, 0.0, 0.0], dtype=float)
        y = math.radians(self.yaw)
        return target + np.array([math.sin(y) * self.distance, self.height, -math.cos(y) * self.distance])
    def forward(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        v = np.array([-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)], dtype=float)
        return v / max(np.linalg.norm(v), 1e-9)
    def aim_point(self, target=None):
        target = np.asarray(target if target is not None else [0.0, 0.0, 0.0], dtype=float)
        position = self.position(target); forward = self.forward(); along = float(np.dot(target - position, forward))
        return position + forward * max(0.1, along)
    def aim_error(self, target=None):
        target = np.asarray(target if target is not None else [0.0, 0.0, 0.0], dtype=float)
        return float(np.linalg.norm(self.aim_point(target) - target))

@dataclass
class SceneSubject:
    height: float = 1.70; center_x: float = 0.0; center_y: float = 0.0; center_z: float = 0.0
    keypoints: Optional[np.ndarray] = None; fitted_points_3d: Optional[np.ndarray] = None
    person_index: int = 0; depth_is_relative: bool = False; depth_confidence: float = 0.0
    def proxy_points(self):
        if self.fitted_points_3d is not None: return self.fitted_points_3d.copy()
        h = self.height
        return np.array([[-.16, -.85, 0], [.16, -.85, 0], [-.14, -.05, 0], [.14, -.05, 0], [-.18, .37, 0], [.18, .37, 0], [0, .81, .08], [0, .85, .08]], dtype=float) * np.array([1.0, h, 1.0])

class SceneModel:
    """Canonical scene state; ``subjects`` is the sole stored people collection."""
    def __init__(self, camera=None, subject=None, ground_size=24.0, candidate_solutions=None, selected_candidate=0, subjects=None, relative_layout=False, anchors=None):
        self.camera = camera if camera is not None else SceneCamera(); self.ground_size = float(ground_size)
        self.candidate_solutions = list(candidate_solutions or []); self.selected_candidate = int(selected_candidate)
        self.subjects = list(subjects or []); self.relative_layout = bool(relative_layout)
        self.anchors = list(anchors) if anchors is not None else default_scene_anchors(); self._primary_subject_person_index = 0
        if subject is not None:
            existing = next((s for s in self.subjects if s.person_index == subject.person_index), None)
            if existing is None: self.subjects.insert(0, subject)
            self._primary_subject_person_index = int(subject.person_index)
        if not self.subjects: self.subjects = [SceneSubject()]
    @property
    def subject(self) -> SceneSubject:
        if not self.subjects:
            self.subjects.append(SceneSubject(person_index=self._primary_subject_person_index))
        return next((s for s in self.subjects if s.person_index == self._primary_subject_person_index), self.subjects[0])
    @subject.setter
    def subject(self, value: SceneSubject):
        if value is None: raise ValueError("primary subject cannot be None")
        for index, existing in enumerate(self.subjects):
            if existing.person_index == value.person_index:
                self.subjects[index] = value; self._primary_subject_person_index = int(value.person_index); return
        self.subjects.insert(0, value); self._primary_subject_person_index = int(value.person_index)
    @classmethod
    def from_reverse_result(cls, result: Optional[ReverseEngineeringResult]) -> "SceneModel":
        scene = cls()
        if result is None: return scene
        pose = getattr(result, "subject_keypoints", None)
        if pose:
            kp = np.array([[lm.x, lm.y, lm.visibility] for lm in pose[:17]], dtype=float)
            scene.subject.keypoints = kp; scene.subject.fitted_points_3d = pose_driven_person_points(kp, result.image_size[0], result.image_size[1], scene.subject.height)
        scene.candidate_solutions = list(getattr(result, "candidate_solutions", None) or getattr(result, "_sim_candidates", []) or [])
        if scene.candidate_solutions:
            scene.selected_candidate = max(range(len(scene.candidate_solutions)), key=lambda i: scene.candidate_solutions[i].score); scene.set_candidate(scene.selected_candidate)
        else:
            cp = result.camera_pose; fl = result.focal_length.equivalent_35mm
            scene.camera = SceneCamera(float(cp.camera_distance.value or 4), float(cp.camera_height.value or 1.5), float(cp.camera_yaw.value or 0), float(cp.camera_pitch.value or 0), float(cp.camera_roll.value or 0), float(fl.value or 50))
        layout = getattr(result, "multi_person_layout", None)
        if layout is not None and getattr(layout, "people", None):
            people = list(layout.people); image_w, image_h = result.image_size; primary = people[0]
            subject_height = float(scene.subject.height)
            scene.subjects = []; scene._primary_subject_person_index = int(primary.person_index); base_cx, base_cy = primary.center
            for person in people:
                kp_rows = np.asarray(person.keypoints, dtype=float)
                if kp_rows.ndim == 2 and kp_rows.shape[0] >= 17:
                    kp_pixels = kp_rows[:17].copy(); kp_pixels[:, 0] *= image_w; kp_pixels[:, 1] *= image_h; fitted = pose_driven_person_points(kp_pixels, image_w, image_h, subject_height)
                else: kp_pixels, fitted = None, None
                nx, ny = person.center
                lateral = (nx - base_cx) * 2.0 * math.tan(math.radians(scene.camera.horizontal_fov_deg) * 0.5) * scene.camera.distance
                vertical = (base_cy - ny) * 2.0 * math.tan(math.radians(scene.camera.vertical_fov_deg) * 0.5) * scene.camera.distance
                depth_offset = float(person.relative_z) * max(0.5, scene.camera.distance * 0.35) if person.usable_3d else 0.0
                scene.subjects.append(SceneSubject(height=subject_height, center_x=float(lateral), center_y=float(vertical), center_z=float(depth_offset), keypoints=kp_pixels, fitted_points_3d=fitted, person_index=int(person.person_index), depth_is_relative=bool(layout.independent_depth and person.usable_3d), depth_confidence=float(person.depth_confidence)))
            scene.relative_layout = bool(layout.independent_depth)
        return scene
    def camera_position(self): return self.camera.position(self.camera_target())
    def camera_target(self): return np.array([self.subject.center_x, self.subject.center_y, self.subject.center_z], dtype=float)
    def camera_aim_target(self): return self.camera.aim_point(self.camera_target())
    def camera_aim_error(self): return self.camera.aim_error(self.camera_target())
    def set_candidate(self, index):
        if not self.candidate_solutions: return
        index = max(0, min(index, len(self.candidate_solutions) - 1)); c = self.candidate_solutions[index]; self.selected_candidate = index
        self.camera.distance = float(c.distance); self.camera.height = float(c.height); self.camera.focal_length_mm = float(c.focal_equiv_35mm)
        self.camera.pitch = float(getattr(c.extrinsics, 'pitch', self.camera.pitch)); self.camera.yaw = float(getattr(c.extrinsics, 'yaw', self.camera.yaw)); self.camera.roll = float(getattr(c.extrinsics, 'roll', self.camera.roll))
    def add_anchor(self, anchor: SceneAnchor) -> None:
        errors = anchor.validate()
        if errors: raise ValueError("invalid scene anchor: " + "; ".join(errors))
        if any(existing.anchor_id == anchor.anchor_id for existing in self.anchors): raise ValueError(f"duplicate scene anchor id: {anchor.anchor_id}")
        self.anchors.append(anchor)
    def create_anchor(self, name: str = "Scene anchor", kind=None) -> SceneAnchor:
        from reverse_engineering.scene_anchors import AnchorKind
        anchor = SceneAnchor(anchor_id=next_anchor_id(self.anchors), name=name, kind=kind or AnchorKind.POINT); self.add_anchor(anchor); return anchor
    def remove_anchor(self, anchor_id: str) -> bool:
        if anchor_id == "ground": return False
        before = len(self.anchors); self.anchors = [anchor for anchor in self.anchors if anchor.anchor_id != anchor_id]; return len(self.anchors) != before
    def anchor_by_id(self, anchor_id: str) -> Optional[SceneAnchor]: return next((anchor for anchor in self.anchors if anchor.anchor_id == anchor_id), None)
    def enabled_planes(self) -> list[SceneAnchor]: return [anchor for anchor in self.anchors if anchor.enabled and anchor.kind.value == "plane"]
    def anchor_summary(self) -> list[dict]: return [anchor.to_dict() for anchor in self.anchors]
    def candidate_summary(self): return [{"index": i, "focal_length_mm": c.focal_equiv_35mm, "distance_m": c.distance, "height_m": c.height, "score": c.score} for i, c in enumerate(self.candidate_solutions)]
