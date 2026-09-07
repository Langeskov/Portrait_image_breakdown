"""Photographer-ready verbal cues derived from pose and composition analysis.

The analysis layer remains rich and technical; this module translates useful
findings into short, speakable instructions a photographer can say on set.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.action_classifier import ActionResult
from core.camera_analyzer import CameraResult
from core.composition import CompositionResult
from core.orientation import FacingDirection, OrientationResult


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


def _feature(action: ActionResult, key: str, default: float = 0.0) -> float:
    return float(action.features.get(key, default))


def generate_photographer_cues(
    action: ActionResult,
    orientation: OrientationResult,
    camera: CameraResult,
    composition: CompositionResult,
) -> list[PhotographerCue]:
    """Generate concise verbal cues while preserving analysis-driven reasoning."""
    cues: list[PhotographerCue] = []

    knee_avg = _feature(action, "knee_angle_avg", 150.0)
    knee_diff = _feature(action, "knee_angle_diff", 0.0)
    stance = _feature(action, "stance_width", 0.1)
    hands_up = bool(_feature(action, "hands_above_shoulders", 0.0))
    ankle_diff = _feature(action, "ankle_y_diff", 0.0)

    if knee_diff < 10 and ankle_diff < 0.08 and stance < 0.12:
        cues.append(PhotographerCue(
            CuePriority.PRIMARY,
            "重心放到一条腿上，另一条腿放松一点。",
            "当前下肢接近对称，容易形成僵硬的站姿。",
            "pose",
        ))
    elif knee_diff > 20 or ankle_diff > 0.10:
        cues.append(PhotographerCue(
            CuePriority.PRIMARY,
            "很好，先别把两条腿收齐，保持这个高低差。",
            "当前已经有明显重心差，继续校正反而可能损失自然感。",
            "pose",
        ))

    elbow_l = action.joint_angles.get("left_elbow", 180.0)
    elbow_r = action.joint_angles.get("right_elbow", 180.0)
    if not hands_up and elbow_l > 155 and elbow_r > 155:
        cues.append(PhotographerCue(
            CuePriority.PRIMARY,
            "一只手离开身体一点，肘部别夹死。",
            "双臂贴近躯干会让轮廓粘在一起。",
            "pose",
        ))

    if knee_avg < 125:
        cues.append(PhotographerCue(
            CuePriority.SECONDARY,
            "两条腿前后错一点，别让膝盖和脚踝落在同一条线上。",
            "弯曲腿部已经形成动作，可通过前后层次让轮廓更清楚。",
            "pose",
        ))
    elif stance < 0.08 and knee_avg > 145:
        cues.append(PhotographerCue(
            CuePriority.SECONDARY,
            "两只脚错开半步，脚尖稍微转出去一点。",
            "当前站姿较窄且对齐，可以用很小的脚位变化增加层次。",
            "pose",
        ))

    if orientation.facing in (
        FacingDirection.FRONT,
        FacingDirection.FRONT_LEFT,
        FacingDirection.FRONT_RIGHT,
    ):
        cues.append(PhotographerCue(
            CuePriority.SECONDARY,
            "身体先不动，下巴往一侧偏一点，看我。",
            "轻微头部偏转通常比整个人转身更适合改善正面平面感。",
            "head",
        ))
    elif orientation.facing in (
        FacingDirection.BACK,
        FacingDirection.BACK_LEFT,
        FacingDirection.BACK_RIGHT,
    ):
        cues.append(PhotographerCue(
            CuePriority.SECONDARY,
            "身体保持背面，只回一点头，别整个人转回来。",
            "可以保留背部轮廓，同时增加面部信息。",
            "head",
        ))

    px, _ = composition.subject_position
    if px < 0.38:
        cues.append(PhotographerCue(
            CuePriority.SECONDARY,
            "动作往你的右边打开一点。",
            "主体偏画面左侧，把肢体或视线向中央/空白方向展开更容易保持平衡。",
            "composition",
        ))
    elif px > 0.62:
        cues.append(PhotographerCue(
            CuePriority.SECONDARY,
            "动作往你的左边打开一点。",
            "主体偏画面右侧，需要把视觉方向留给空白区域。",
            "composition",
        ))

    if camera.subject_ratio > 0.65:
        cues.append(PhotographerCue(
            CuePriority.OPTIONAL,
            "动作幅度小一点，先把手和脸留在画面里。",
            "人物已经占据较大画面，复杂动作更容易出框。",
            "camera",
        ))

    return cues[:6]


def speakable_summary(cues: list[PhotographerCue]) -> str:
    """Return the primary cue plus one follow-up as a compact field instruction."""
    if not cues:
        return "先保持自然，我会根据画面继续调整。"
    lines = [cues[0].cue]
    if len(cues) > 1:
        lines.append(cues[1].cue)
    return " ".join(lines)
