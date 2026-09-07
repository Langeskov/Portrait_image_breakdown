"""Weak support-plane evidence for camera pitch coherence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class SupportPlaneEvidence:
    active: bool
    confidence: float
    contact_world_y: float
    contact_fraction: float
    visible_ankles: int
    basis: tuple[str, ...]
    @property
    def usable(self) -> bool:
        return self.active and self.confidence >= 0.35 and self.visible_ankles >= 1
    def to_dict(self) -> dict:
        return {"active": bool(self.active), "confidence": round(float(self.confidence), 3), "contact_world_y": round(float(self.contact_world_y), 4), "contact_fraction": round(float(self.contact_fraction), 3), "visible_ankles": int(self.visible_ankles), "basis": list(self.basis)}

def estimate_support_plane(pose_keypoints: np.ndarray, image_width: int, image_height: int) -> SupportPlaneEvidence:
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
        return SupportPlaneEvidence(False, 0.0, 0.0, 0.0, 0, ("insufficient pose evidence",))
    contacts = []
    for ankle_i, knee_i in ((15, 13), (16, 14)):
        if kp[ankle_i, 2] <= 0.45 or kp[knee_i, 2] <= 0.35 or not np.isfinite(kp[ankle_i, :2]).all():
            continue
        if kp[ankle_i, 1] > kp[knee_i, 1] + 0.04 * image_height:
            contacts.append(float(kp[ankle_i, 1]))
    if not contacts:
        return SupportPlaneEvidence(False, 0.0, 0.0, 0.0, 0, ("no reliable ankle/knee contact geometry",))
    max_y = max(contacts)
    contact_fraction = float(np.clip(max_y / max(image_height, 1), 0.0, 1.0))
    confidence = float(np.clip(0.42 + (0.16 if len(contacts) == 2 else 0.0) + (0.10 if contact_fraction > 0.65 else 0.0), 0.0, 0.75))
    basis = ["visible ankle/knee geometry", "human-scale support hypothesis"]
    if len(contacts) == 2:
        basis.append("bilateral contact consistency")
    return SupportPlaneEvidence(True, confidence, 0.0, contact_fraction, len(contacts), tuple(basis))

def expected_support_pitch_deg(candidate_distance_m: float, candidate_height_m: float, support_plane: SupportPlaneEvidence) -> float:
    if not support_plane.usable:
        return float("nan")
    return float(np.degrees(np.arctan2(support_plane.contact_world_y - float(candidate_height_m), max(float(candidate_distance_m), 1e-6))))

def candidate_support_plane_score(candidate, support_plane: Optional[SupportPlaneEvidence]) -> Optional[float]:
    if support_plane is None or not support_plane.usable:
        return None
    expected = expected_support_pitch_deg(candidate.distance, candidate.height, support_plane)
    if not np.isfinite(expected):
        return None
    delta = abs(float(candidate.extrinsics.pitch) - expected)
    return float(np.clip(np.exp(-delta / 18.0), 0.0, 1.0))
