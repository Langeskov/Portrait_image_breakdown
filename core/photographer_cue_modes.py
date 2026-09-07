"""Field-oriented presentation of existing photographer cues.

The underlying cue generation remains unchanged. This module only controls how
much of the existing professional guidance is exposed to a photographer in a
field situation, keeping technical analysis available for later review.
"""
from __future__ import annotations

from enum import Enum

from core.photographer_cues import PhotographerCue


class CueMode(Enum):
    CONCISE = "简洁"
    NORMAL = "普通"
    TECHNICAL = "专业"


def format_cues(cues: list[PhotographerCue], mode: CueMode = CueMode.NORMAL) -> list[str]:
    """Return speakable cue text without changing the underlying analysis."""
    if not cues:
        return ["先保持自然，我根据画面继续调整。"]

    if mode is CueMode.CONCISE:
        return [cues[0].cue]

    if mode is CueMode.TECHNICAL:
        return [f"{cue.cue}（原因：{cue.reason}）" for cue in cues[:4]]

    return [cue.cue for cue in cues[:3]]


def primary_cue(cues: list[PhotographerCue]) -> str:
    """Return one line suitable for a photographer's immediate attention."""
    return format_cues(cues, CueMode.CONCISE)[0]
