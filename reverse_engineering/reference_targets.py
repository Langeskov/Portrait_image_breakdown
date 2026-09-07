"""Turn v3 reference deltas into a reproducible shooting target plan."""
from __future__ import annotations

from dataclasses import dataclass

from reverse_engineering.reference_reconstruction import PoseDelta, ReferenceComposition


@dataclass(frozen=True)
class ReferenceTargetPlan:
    headline: str
    framing_actions: tuple[str, ...]
    pose_actions: tuple[str, ...]

    def as_text(self) -> str:
        parts = [self.headline]
        if self.framing_actions:
            parts.append("构图：" + "；".join(self.framing_actions))
        if self.pose_actions:
            parts.append("姿态：" + "；".join(self.pose_actions[:3]))
        return "\n".join(parts)


def build_reference_target_plan(
    reference: ReferenceComposition,
    current: ReferenceComposition,
    pose_deltas: list[PoseDelta],
) -> ReferenceTargetPlan:
    """Build conservative framing + pose actions from image-space deltas."""
    rx, ry = reference.subject_center
    cx, cy = current.subject_center
    dx = (rx - cx) / max(reference.width, 1)
    dy = (ry - cy) / max(reference.height, 1)
    scale_ratio = current.subject_scale / reference.subject_scale if reference.subject_scale > 1e-9 else 1.0

    framing: list[str] = []
    if abs(scale_ratio - 1.0) >= 0.08:
        framing.append("靠近人物，增大人物占比" if scale_ratio < 1.0 else "后退一点，减小人物占比")
    if abs(dx) >= 0.04:
        framing.append(f"人物整体向{'右' if dx > 0 else '左'}调整")
    if abs(dy) >= 0.04:
        framing.append(f"人物整体向{'下' if dy > 0 else '上'}调整")
    if not framing:
        framing.append("构图比例和主体位置已接近参考图")

    pose_actions = [d.instruction for d in pose_deltas[:5]]
    if pose_actions:
        headline = "先对齐构图，再用少量动作修正对齐度。"
    else:
        headline = "构图和可见姿态已经接近参考状态。"
    return ReferenceTargetPlan(headline, tuple(framing), tuple(pose_actions))
