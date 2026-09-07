"""Reference-photo reconstruction helpers for v3.

The first v3 layer is deliberately deterministic: it turns a reference photo's
existing analysis into editable 2D composition anchors and a camera/pose target.
It does not claim metric room reconstruction from a single image.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import math
import numpy as np

from core.pose_detector import LandmarkIndex as LI, PoseResult


@dataclass(frozen=True)
class SceneAnchor:
    name: str
    x: float
    y: float
    confidence: float
    source: str


@dataclass(frozen=True)
class PoseDelta:
    landmark: str
    dx: float
    dy: float
    distance: float
    instruction: str
    target_x: float = 0.0
    target_y: float = 0.0


@dataclass(frozen=True)
class ReferenceComposition:
    width: int
    height: int
    anchors: tuple[SceneAnchor, ...]
    subject_bbox: Optional[tuple[float, float, float, float]]
    subject_center: tuple[float, float]
    subject_scale: float

    def to_dict(self) -> dict:
        data = asdict(self)
        data["anchors"] = [asdict(a) for a in self.anchors]
        data["subject_bbox"] = list(self.subject_bbox) if self.subject_bbox else None
        return data


def _visible(pose: PoseResult, idx: int, threshold: float = 0.35) -> bool:
    return 0 <= int(idx) < len(pose.landmarks) and pose.landmarks[int(idx)].visibility >= threshold


def choose_semantic_anchor(pose: PoseResult | None) -> SceneAnchor | None:
    if pose is None:
        return None
    # Prefer a stable body anchor over the face. This is especially useful for
    # seated/reclined references where the head is not a reliable spatial base.
    candidates = []
    if _visible(pose, LI.LEFT_HIP) and _visible(pose, LI.RIGHT_HIP):
        a, b = pose.landmarks[LI.LEFT_HIP], pose.landmarks[LI.RIGHT_HIP]
        candidates.append(SceneAnchor("hip_center", (a.x + b.x) / 2, (a.y + b.y) / 2, min(a.visibility, b.visibility), "bilateral_hips"))
    if _visible(pose, LI.LEFT_SHOULDER) and _visible(pose, LI.RIGHT_SHOULDER):
        a, b = pose.landmarks[LI.LEFT_SHOULDER], pose.landmarks[LI.RIGHT_SHOULDER]
        candidates.append(SceneAnchor("shoulder_center", (a.x + b.x) / 2, (a.y + b.y) / 2, min(a.visibility, b.visibility) * 0.9, "bilateral_shoulders"))
    if _visible(pose, LI.NOSE):
        lm = pose.landmarks[LI.NOSE]
        candidates.append(SceneAnchor("head", lm.x, lm.y, lm.visibility * 0.75, "nose"))
    return max(candidates, key=lambda a: a.confidence) if candidates else None


def build_reference_composition(pose: PoseResult, width: int, height: int) -> ReferenceComposition:
    visible = [lm for lm in pose.landmarks[:17] if lm.visibility > 0.35]
    if visible:
        bbox = (min(lm.x for lm in visible), min(lm.y for lm in visible), max(lm.x for lm in visible), max(lm.y for lm in visible))
    else:
        bbox = None
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) if bbox else (width * 0.5, height * 0.5)
    scale = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / max(width * height, 1)) if bbox else 0.0
    anchors: list[SceneAnchor] = []
    if bbox:
        anchors.extend([
            SceneAnchor("bbox_tl", bbox[0], bbox[1], 0.9, "pose_bbox"),
            SceneAnchor("bbox_center", center[0], center[1], 0.95, "pose_bbox"),
            SceneAnchor("bbox_br", bbox[2], bbox[3], 0.9, "pose_bbox"),
        ])
    semantic = choose_semantic_anchor(pose)
    if semantic:
        anchors.append(semantic)
    return ReferenceComposition(width, height, tuple(anchors), bbox, center, float(scale))


def compare_pose_to_reference(reference: PoseResult, current: PoseResult, width: int, height: int, threshold: float = 0.35) -> list[PoseDelta]:
    labels = {
        0: "nose", 5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip", 13: "left_knee",
        14: "right_knee", 15: "left_ankle", 16: "right_ankle",
    }
    deltas: list[PoseDelta] = []
    diag = max(math.hypot(width, height), 1.0)
    for idx, name in labels.items():
        if not (_visible(reference, idx, threshold) and _visible(current, idx, threshold)):
            continue
        r, c = reference.landmarks[idx], current.landmarks[idx]
        dx, dy = (r.x - c.x) / max(width, 1), (r.y - c.y) / max(height, 1)
        distance = math.hypot(dx * width, dy * height) / diag
        if distance < 0.015:
            continue
        horizontal = "右" if dx > 0 else "左"
        vertical = "下" if dy > 0 else "上"
        if abs(dx) > abs(dy) * 1.6:
            instruction = f"{name} 向{horizontal}移动 {abs(dx):.0%} 画面宽度"
        elif abs(dy) > abs(dx) * 1.6:
            instruction = f"{name} 向{vertical}移动 {abs(dy):.0%} 画面高度"
        else:
            instruction = f"{name} 向{horizontal}{vertical}移动"
        deltas.append(PoseDelta(
            name, float(dx), float(dy), float(distance), instruction,
            target_x=float(r.x / max(reference.image_width if hasattr(reference, 'image_width') else width, 1)),
            target_y=float(r.y / max(reference.image_height if hasattr(reference, 'image_height') else height, 1)),
        ))
    deltas.sort(key=lambda d: d.distance, reverse=True)
    return deltas[:8]


def composition_delta(reference: ReferenceComposition, current: ReferenceComposition) -> dict[str, float]:
    rx, ry = reference.subject_center
    cx, cy = current.subject_center
    return {
        "center_dx": float((rx - cx) / max(reference.width, 1)),
        "center_dy": float((ry - cy) / max(reference.height, 1)),
        "scale_ratio": float(current.subject_scale / reference.subject_scale) if reference.subject_scale > 1e-9 else 1.0,
        "target_scale": float(reference.subject_scale),
        "current_scale": float(current.subject_scale),
    }


def reference_summary(reference: ReferenceComposition) -> str:
    x, y = reference.subject_center
    return f"Reference {reference.width}×{reference.height} · subject center ({x / max(reference.width, 1):.0%}, {y / max(reference.height, 1):.0%}) · scale {reference.subject_scale:.0%}"
