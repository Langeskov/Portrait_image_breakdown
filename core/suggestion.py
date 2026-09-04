"""
下一动作建议引擎

综合骨架、朝向、动作、镜头、构图分析结果, 生成具体的下一动作建议。
支持摄影指导和动画/游戏两个场景。
"""

from __future__ import annotations

import dataclasses
from enum import Enum

from core.action_classifier import ActionCategory, ActionResult
from core.camera_analyzer import ShotType, CameraAngle, CameraResult
from core.composition import CompositionType, CompositionResult
from core.orientation import FacingDirection, TiltDirection, OrientationResult


class SuggestionPriority(Enum):
    HIGH = "高优先"
    MEDIUM = "中优先"
    LOW = "参考"


@dataclasses.dataclass
class Suggestion:
    priority: SuggestionPriority
    category: str           # 分类: "pose" / "camera" / "composition" / "action"
    title: str              # 建议标题
    description: str        # 详细描述
    icon: str               # 表情图标


@dataclasses.dataclass
class SuggestionResult:
    suggestions: list[Suggestion]
    next_actions: list[str]       # 推荐的下一动作(动作名)
    creative_direction: str       # 创意方向总结

    @property
    def summary(self) -> str:
        top3 = self.suggestions[:3]
        return "\n".join(f"• {s.title}: {s.description}" for s in top3)


# ── 动作转换矩阵: 当前动作 → 推荐下一动作 ──
ACTION_TRANSITIONS: dict[ActionCategory, list[tuple[str, str]]] = {
    ActionCategory.STANDING: [
        ("行走", "从站立过渡到行走, 增加动感"),
        ("转身", "自然的身体旋转, 适合侧面/背面拍摄"),
        ("抬手", "手臂自然上举, 展现优雅线条"),
        ("坐下", "切换到坐姿, 改变画面重心"),
        ("跳跃", "从站立直接跳起, 抓拍腾空瞬间"),
    ],
    ActionCategory.WALKING: [
        ("跑步", "加速到跑步, 增加速度感"),
        ("停步回望", "行走中突然停下回望, 增加故事感"),
        ("转身", "行走中转身, 衣摆飘动"),
        ("跳跃", "行走中的跳跃, 动态感更强"),
    ],
    ActionCategory.RUNNING: [
        ("跳跃", "跑步起跳, 最具动感的瞬间"),
        ("急停", "突然停止, 衣物惯性飘动"),
        ("转弯", "改变方向, 展现侧面线条"),
    ],
    ActionCategory.JUMPING: [
        ("落地", "跳跃后的落地瞬间, 展现平衡感"),
        ("二次跳跃", "连续跳跃, 适合连拍"),
        ("空中姿态", "腾空中展开四肢, 展现力量感"),
    ],
    ActionCategory.SQUATTING: [
        ("站起", "从蹲姿站起, 展现爆发力"),
        ("蹲走", "蹲姿移动, 适合低角度拍摄"),
        ("伸展", "从蹲姿向上伸展, 对比明显"),
    ],
    ActionCategory.SITTING: [
        ("站起", "从坐姿站起的过渡动作"),
        ("翘腿", "变换坐姿, 展现不同线条"),
        ("前倾", "坐姿前倾, 增加亲和力"),
        ("侧靠", "向一侧倾斜, 展现放松感"),
    ],
    ActionCategory.LYING: [
        ("翻身", "从仰卧到侧卧/俯卧"),
        ("坐起", "从躺卧到坐起的过渡"),
        ("伸展", "躺姿伸展, 展现身体线条"),
    ],
    ActionCategory.ARMS_RAISED: [
        ("放下手臂", "自然放下, 展现放松过渡"),
        ("交叉手臂", "手臂交叉于胸前, 展现自信"),
        ("侧展", "手臂向两侧展开, 展现平衡"),
    ],
    ActionCategory.BALANCING: [
        ("单脚站立", "保持平衡, 展现优雅"),
        ("旋转", "单脚旋转, 动态感强"),
        ("跳跃", "从平衡态跳起"),
    ],
    ActionCategory.BOWING: [
        ("直起身", "从弯腰到直立的过渡"),
        ("转身", "弯腰后转身, 增加故事性"),
    ],
    ActionCategory.FIGHTING_STANCE: [
        ("出拳", "格斗姿态的下一步动作"),
        ("踢腿", "腿部攻击动作"),
        ("闪避", "向侧面闪避, 展现敏捷"),
        ("后撤步", "防御性后退, 保持紧张感"),
    ],
    ActionCategory.DANCING: [
        ("旋转", "舞蹈中的旋转动作"),
        ("跳跃", "舞蹈跳跃"),
        ("伸展", "伸展手臂/腿部, 展现线条"),
    ],
}

# 对没有特定转换的类别, 使用默认建议
_DEFAULT_TRANSITIONS = [
    ("站立", "回到基础站姿"),
    ("行走", "加入移动元素"),
    ("转身", "身体旋转带来新角度"),
]


def generate_suggestions(
    action: ActionResult,
    orientation: OrientationResult,
    camera: CameraResult,
    composition: CompositionResult,
) -> SuggestionResult:
    """
    生成综合建议

    参数:
        action: 动作分析结果
        orientation: 朝向分析结果
        camera: 镜头分析结果
        composition: 构图分析结果

    返回:
        SuggestionResult
    """
    suggestions: list[Suggestion] = []

    # ══════════════════════════════════════════════════════════════
    # 1. 动作建议
    # ══════════════════════════════════════════════════════════════
    transitions = ACTION_TRANSITIONS.get(action.category, _DEFAULT_TRANSITIONS)
    next_actions = [name for name, _ in transitions[:3]]

    # 当前动作的评价
    if action.confidence < 0.4:
        suggestions.append(Suggestion(
            priority=SuggestionPriority.MEDIUM,
            category="pose",
            title="姿态不够明确",
            description=f"当前识别为「{action.category.value}」但置信度较低({action.confidence:.0%}), "
                        f"建议调整姿态使动作更清晰",
            icon="🎭",
        ))

    # 基于当前动作的具体建议
    for action_name, reason in transitions[:2]:
        suggestions.append(Suggestion(
            priority=SuggestionPriority.MEDIUM,
            category="action",
            title=f"下一动作: {action_name}",
            description=reason,
            icon="🏃",
        ))

    # ══════════════════════════════════════════════════════════════
    # 2. 朝向建议
    # ══════════════════════════════════════════════════════════════
    if orientation.facing in (FacingDirection.BACK, FacingDirection.BACK_LEFT, FacingDirection.BACK_RIGHT):
        suggestions.append(Suggestion(
            priority=SuggestionPriority.LOW,
            category="pose",
            title="背面拍摄",
            description="当前为背面视角, 适合营造意境感。如需更多表情信息, 考虑让人物转头或侧身",
            icon="🔄",
        ))

    if orientation.tilt == TiltDirection.LEANING_FORWARD:
        suggestions.append(Suggestion(
            priority=SuggestionPriority.LOW,
            category="pose",
            title="前倾姿态",
            description="身体前倾增加动感和亲和力, 但注意不要过度以免画面失衡",
            icon="↗️",
        ))

    # ══════════════════════════════════════════════════════════════
    # 3. 镜头建议
    # ══════════════════════════════════════════════════════════════
    if camera.shot_type in (ShotType.LONG, ShotType.EXTREME_LONG):
        suggestions.append(Suggestion(
            priority=SuggestionPriority.HIGH,
            category="camera",
            title="拉近镜头",
            description="人物在画面中占比过小, 建议靠近或使用长焦, 让人物成为视觉焦点",
            icon="📷",
        ))
    elif camera.shot_type in (ShotType.EXTREME_CLOSEUP,):
        suggestions.append(Suggestion(
            priority=SuggestionPriority.MEDIUM,
            category="camera",
            title="适当拉远",
            description="大特写虽然冲击力强, 但适当拉远可以展现更多肢体语言",
            icon="📷",
        ))

    if camera.camera_angle == CameraAngle.HIGH_ANGLE:
        suggestions.append(Suggestion(
            priority=SuggestionPriority.LOW,
            category="camera",
            title="俯拍视角",
            description="俯拍使人物显得较小, 适合表现脆弱感或环境关系。如需力量感, 尝试平视或仰拍",
            icon="📐",
        ))
    elif camera.camera_angle == CameraAngle.LOW_ANGLE:
        suggestions.append(Suggestion(
            priority=SuggestionPriority.LOW,
            category="camera",
            title="仰拍视角",
            description="仰拍增强人物气势, 适合表现力量感和权威感",
            icon="📐",
        ))

    if abs(camera.dutch_angle_deg) > 5:
        suggestions.append(Suggestion(
            priority=SuggestionPriority.LOW,
            category="camera",
            title="画面倾斜",
            description=f"画面倾斜{camera.dutch_angle_deg:.1f}°, 如非刻意营造不安感, 建议保持水平",
            icon="📐",
        ))

    # ══════════════════════════════════════════════════════════════
    # 4. 构图建议
    # ══════════════════════════════════════════════════════════════
    for comp_suggestion in composition.suggestions:
        suggestions.append(Suggestion(
            priority=SuggestionPriority.MEDIUM,
            category="composition",
            title="构图优化",
            description=comp_suggestion,
            icon="🖼️",
        ))

    # 头部空间
    if composition.headroom < 0.08:
        suggestions.append(Suggestion(
            priority=SuggestionPriority.HIGH,
            category="composition",
            title="头顶空间不足",
            description="人物头顶紧贴画面上缘, 给人压迫感。下移镜头或后退, 留出呼吸空间",
            icon="⬆️",
        ))
    elif composition.headroom > 0.4:
        suggestions.append(Suggestion(
            priority=SuggestionPriority.MEDIUM,
            category="composition",
            title="头顶空间过多",
            description="画面上方留白过多, 重心偏低。上移镜头让人物更突出",
            icon="⬇️",
        ))

    # 三分法
    if composition.thirds_alignment < 0.4:
        px, py = composition.subject_position
        suggestions.append(Suggestion(
            priority=SuggestionPriority.MEDIUM,
            category="composition",
            title="偏离三分法交叉点",
            description=f"主体位于({px:.0%}, {py:.0%}), 建议调整到三分法交叉点附近, 增强视觉张力",
            icon="🎯",
        ))

    # ══════════════════════════════════════════════════════════════
    # 5. 综合创意方向
    # ══════════════════════════════════════════════════════════════
    creative_parts = []

    # 基于动作类型的创意方向
    if action.category in (ActionCategory.STANDING, ActionCategory.SITTING):
        creative_parts.append("当前姿态较为静态, 可以尝试增加肢体动感")
    elif action.category in (ActionCategory.WALKING, ActionCategory.RUNNING):
        creative_parts.append("动态抓拍, 可以用连拍捕捉最佳瞬间")
    elif action.category == ActionCategory.JUMPING:
        creative_parts.append("跳跃瞬间充满活力, 注意快门速度确保清晰")

    # 基于构图的创意方向
    if composition.thirds_alignment > 0.7:
        creative_parts.append("构图符合三分法, 可以尝试打破规则寻找新视角")
    elif composition.primary_type == CompositionType.CENTER:
        creative_parts.append("居中构图给人稳定感, 可以尝试偏移构图增加动感")

    # 基于镜头的创意方向
    if camera.shot_type == ShotType.MEDIUM:
        creative_parts.append("中景适合展现肢体语言, 可以尝试不同景别丰富画面")

    if not creative_parts:
        creative_parts.append("综合表现良好, 尝试更多创意角度和动作变化")

    creative_direction = " | ".join(creative_parts)

    # 按优先级排序
    priority_order = {
        SuggestionPriority.HIGH: 0,
        SuggestionPriority.MEDIUM: 1,
        SuggestionPriority.LOW: 2,
    }
    suggestions.sort(key=lambda s: priority_order[s.priority])

    return SuggestionResult(
        suggestions=suggestions,
        next_actions=next_actions,
        creative_direction=creative_direction,
    )
