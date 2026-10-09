"""Photography guidance engine.

This module delegates core guidance decisions to the unified engine in
core.guidance and adds higher-level suggestion aggregation for the UI.
"""
from __future__ import annotations

import dataclasses
from enum import Enum

from core.action_classifier import ActionCategory, ActionResult
from core.camera_analyzer import ShotType, CameraAngle, CameraResult
from core.composition import CompositionType, CompositionResult
from core.guidance import (
    GuidanceAction,
    GuidancePriority,
    GuidanceResult,
    GuidanceState,
    generate_guidance,
)
from core.orientation import FacingDirection, TiltDirection, OrientationResult
from core.photographer_cues import PhotographerCue, generate_photographer_cues


class SuggestionPriority(Enum):
    HIGH = "高优先"
    MEDIUM = "中优先"
    LOW = "参考"


@dataclasses.dataclass
class Suggestion:
    priority: SuggestionPriority
    category: str
    title: str
    description: str
    icon: str = ""


@dataclasses.dataclass
class SuggestionResult:
    suggestions: list[Suggestion]
    next_actions: list[str]
    creative_direction: str
    photographer_cues: list[PhotographerCue] = dataclasses.field(default_factory=list)

    @property
    def summary(self) -> str:
        return "\n".join(f"• {s.title}: {s.description}" for s in self.suggestions[:3])


ACTION_TRANSITIONS: dict[ActionCategory, list[tuple[str, str]]] = {
    ActionCategory.STANDING: [("移动重心", "释放对称感"), ("转身", "增加侧面线条")],
    ActionCategory.WALKING: [("停步回望", "保留动作感")],
    ActionCategory.RUNNING: [("急停", "利用身体惯性")],
    ActionCategory.JUMPING: [("展开", "打开空中轮廓")],
    ActionCategory.SQUATTING: [("抬起上身", "保持低姿态但打开胸口")],
    ActionCategory.SITTING: [("交错双腿", "增加腿部层次"), ("侧靠", "形成非对称轮廓")],
    ActionCategory.LYING: [("侧转", "让肩胯错位")],
    ActionCategory.ARMS_RAISED: [("放松手腕", "避免手臂僵硬")],
    ActionCategory.BALANCING: [("延长轴线", "强化纵向线条")],
    ActionCategory.BOWING: [("抬头", "恢复面部信息")],
    ActionCategory.FIGHTING_STANCE: [("前后错步", "明确动作方向")],
    ActionCategory.DANCING: [("停在延伸点", "抓住线条最佳瞬间")],
}

_DEFAULT_TRANSITIONS = ["调整重心", "打开身体轮廓", "改变头部方向"]


def _guidance_to_suggestions(result: GuidanceResult) -> list[Suggestion]:
    """Convert GuidanceActions to Suggestion objects for the UI panel."""
    suggestions = []
    priority_map = {
        GuidancePriority.VISIBILITY: SuggestionPriority.HIGH,
        GuidancePriority.SILHOUETTE: SuggestionPriority.HIGH,
        GuidancePriority.BALANCE: SuggestionPriority.MEDIUM,
        GuidancePriority.BODY_CURVE: SuggestionPriority.MEDIUM,
        GuidancePriority.HEAD_GAZE: SuggestionPriority.MEDIUM,
        GuidancePriority.COMPOSITION: SuggestionPriority.MEDIUM,
    }

    for action in result.all_actions[:5]:
        pri = priority_map.get(action.priority, SuggestionPriority.LOW)
        suggestions.append(Suggestion(
            pri, action.category, action.cue[:20], action.reason,
        ))

    return suggestions


def generate_suggestions(
    action: ActionResult,
    orientation: OrientationResult,
    camera: CameraResult,
    composition: CompositionResult,
) -> SuggestionResult:
    """Generate suggestions using the unified guidance engine."""
    # Get unified guidance
    guidance = generate_guidance(action, orientation, camera, composition)

    # Convert to UI suggestions
    suggestions = _guidance_to_suggestions(guidance)

    # Get photographer cues (also uses guidance engine)
    photographer_cues = generate_photographer_cues(action, orientation, camera, composition)

    # Add action-category context (not for guidance, just for UI reference)
    if action.confidence < 0.40:
        suggestions.append(Suggestion(
            SuggestionPriority.LOW, "pose", "动作类别仅供参考",
            f'当前更适合根据身体线条和构图调整，而不是追求"{action.category.value}"这个标签本身。',
        ))

    # Add camera-specific suggestions
    if camera.shot_type in (ShotType.LONG, ShotType.EXTREME_LONG):
        suggestions.append(Suggestion(
            SuggestionPriority.HIGH, "camera", "先把人物放大到可读范围",
            "人物占画面比例偏小。优先考虑靠近或使用更长焦段，再决定动作。",
        ))
    elif camera.shot_type == ShotType.EXTREME_CLOSEUP:
        suggestions.append(Suggestion(
            SuggestionPriority.MEDIUM, "camera", "给肢体留一点空间",
            "当前构图已经很紧。若希望动作本身成为重点，可以稍微拉远。",
        ))

    if camera.camera_angle == CameraAngle.HIGH_ANGLE:
        suggestions.append(Suggestion(
            SuggestionPriority.LOW, "camera", "俯拍下优先扩大身体轮廓",
            "俯拍会压缩身体的纵向高度；可以让四肢稍微展开。",
        ))
    elif camera.camera_angle == CameraAngle.LOW_ANGLE:
        suggestions.append(Suggestion(
            SuggestionPriority.LOW, "camera", "仰拍下保持四肢清楚",
            "仰拍容易让近端肢体占比过大，避免手脚直接贴边。",
        ))

    if abs(camera.dutch_angle_deg) > 5:
        suggestions.append(Suggestion(
            SuggestionPriority.LOW, "camera", "检查画面倾斜是否有意",
            f"当前估计倾斜约 {camera.dutch_angle_deg:.1f}°。如果不是刻意制造失衡感，优先校平相机。",
        ))

    for comp_suggestion in composition.suggestions:
        suggestions.append(Suggestion(SuggestionPriority.MEDIUM, "composition", "构图优化", comp_suggestion))

    if composition.headroom < 0.08:
        suggestions.append(Suggestion(
            SuggestionPriority.HIGH, "composition", "头部需要呼吸空间",
            "人物顶部过于贴边。先抬高取景或稍微后退。",
        ))
    elif composition.headroom > 0.40:
        suggestions.append(Suggestion(
            SuggestionPriority.MEDIUM, "composition", "减少顶部留白",
            "上方空间较大。优先重新构图。",
        ))

    # Creative direction
    goals: list[str] = []
    if guidance.state == GuidanceState.HOLD:
        goals.append("当前状态良好，保持即可")
    elif guidance.primary:
        goals.append(guidance.primary.cue[:20])
    if orientation.facing in (FacingDirection.BACK, FacingDirection.BACK_LEFT, FacingDirection.BACK_RIGHT):
        goals.append("保留背面轮廓并增加头部信息")
    if not goals:
        goals.append("以自然线条为优先，不必追求明确的动作标签")
    creative_direction = "；".join(goals) + "。"

    # Sort and return
    priority_order = {SuggestionPriority.HIGH: 0, SuggestionPriority.MEDIUM: 1, SuggestionPriority.LOW: 2}
    category_order = {"pose": 0, "camera": 1, "composition": 2, "action": 3}
    suggestions.sort(key=lambda s: (priority_order[s.priority], category_order.get(s.category, 9)))

    return SuggestionResult(
        suggestions=suggestions[:16],
        next_actions=list(_DEFAULT_TRANSITIONS),
        creative_direction=creative_direction,
        photographer_cues=photographer_cues,
    )

