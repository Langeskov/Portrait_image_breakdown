"""Photographer-ready verbal cues derived from pose and composition analysis.

This module now delegates to the unified guidance engine in core.guidance
and translates GuidanceActions into PhotographerCues for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.action_classifier import ActionResult
from core.camera_analyzer import CameraResult
from core.composition import CompositionResult
from core.guidance import GuidancePriority, GuidanceState, generate_guidance
from core.orientation import OrientationResult


class CuePriority(Enum):
    PRIMARY = "主提示"
    SECONDARY = "辅助提示"
    OPTIONAL = "可选"


@dataclass(frozen=True)
class PhotographerCue:
    priority: CuePriority
    cue: str
    reason: str
    category: str


def generate_photographer_cues(
    action: ActionResult,
    orientation: OrientationResult,
    camera: CameraResult,
    composition: CompositionResult,
) -> list[PhotographerCue]:
    """Generate concise verbal cues via the unified guidance engine."""
    result = generate_guidance(action, orientation, camera, composition)

    cues: list[PhotographerCue] = []

    if result.state == GuidanceState.HOLD:
        cues.append(PhotographerCue(
            CuePriority.PRIMARY,
            result.hold_message,
            "当前状态良好，建议保持。",
            "hold",
        ))
        # Even in HOLD, include any remaining adjust actions as secondary
        for action in result.all_actions[1:]:
            cues.append(PhotographerCue(
                CuePriority.SECONDARY,
                action.cue,
                action.reason,
                action.category,
            ))
        return cues[:6]

    # Map all actions to cues, not just primary/secondary
    for i, action in enumerate(result.all_actions):
        if i == 0:
            cues.append(PhotographerCue(CuePriority.PRIMARY, action.cue, action.reason, action.category))
        elif i == 1:
            cues.append(PhotographerCue(CuePriority.SECONDARY, action.cue, action.reason, action.category))
        else:
            cues.append(PhotographerCue(CuePriority.OPTIONAL, action.cue, action.reason, action.category))

    return cues[:6]


def speakable_summary(cues: list[PhotographerCue]) -> str:
    """Return the primary cue plus one follow-up as a compact field instruction."""
    if not cues:
        return "先保持自然，我会根据画面继续调整。"
    lines = [cues[0].cue]
    if len(cues) > 1:
        lines.append(cues[1].cue)
    return " ".join(lines)

