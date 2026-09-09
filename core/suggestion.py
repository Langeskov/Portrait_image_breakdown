"""Photography guidance engine.

The action classifier answers "what is happening now". This module answers
"how can this frame be improved" and "what can the photographer say next".
Guidance is driven primarily by body geometry, silhouette, balance, framing,
and visual intent; action classification remains contextual.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

from core.action_classifier import ActionCategory, ActionResult
from core.camera_analyzer import ShotType, CameraAngle, CameraResult
from core.composition import CompositionType, CompositionResult
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


def _feature(action: ActionResult, key: str, default: float = 0.0) -> float:
    return float(action.features.get(key, default))


def _generate_pose_guidance(action: ActionResult, orientation: OrientationResult, composition: CompositionResult) -> list[Suggestion]:
    """Generate pose coaching from body geometry rather than action labels."""
    angles = action.joint_angles
    guidance: list[Suggestion] = []
    knee_avg = _feature(action, "knee_angle_avg", 150.0)
    knee_diff = _feature(action, "knee_angle_diff", 0.0)
    stance = _feature(action, "stance_width", 0.1)
    shoulder_y = _feature(action, "shoulder_y", 0.5)
    hip_y = _feature(action, "hip_y", 0.5)
    hands_up = bool(_feature(action, "hands_above_shoulders", 0.0))
    wrist_y_avg = _feature(action, "wrist_y_avg", 0.5)
    ankle_diff = _feature(action, "ankle_y_diff", 0.0)

    if knee_diff < 10 and ankle_diff < 0.08 and stance < 0.12:
        guidance.append(Suggestion(SuggestionPriority.HIGH, "pose", "先释放对称站姿", "把重心轻轻移到一条腿，另一条腿只负责支撑平衡；身体不要刻意挺直。这样通常比直接要求“做一个动作”更自然。"))
    elif knee_diff > 20 or ankle_diff > 0.10:
        guidance.append(Suggestion(SuggestionPriority.MEDIUM, "pose", "保留当前重心差", "左右腿已经形成明显的高低或弯曲差，先不要纠正成对称；微调骨盆方向，让这条不对称线保持干净。"))

    elbow_l = angles.get("left_elbow", 180.0)
    elbow_r = angles.get("right_elbow", 180.0)
    if not hands_up and elbow_l > 155 and elbow_r > 155:
        guidance.append(Suggestion(SuggestionPriority.HIGH, "pose", "把手臂从躯干上分开", "任选一只手离开身体几厘米，手肘留一点空间；另一只手可以放在腰侧、腿侧或轻触服装。重点不是“抬手”，而是让轮廓不要粘在一起。"))
    elif hands_up and wrist_y_avg < shoulder_y - 0.12:
        guidance.append(Suggestion(SuggestionPriority.MEDIUM, "pose", "上举后先放松手腕", "手臂已经打开，不需要继续抬高；让手腕、手指自然弯曲，避免手臂形成僵硬直线。"))

    if knee_avg < 125:
        guidance.append(Suggestion(SuggestionPriority.HIGH, "pose", "让双腿产生前后层次", "不要让两个膝盖、脚踝落在同一条线上。把其中一条腿略微前伸或后收，画面会更容易读出腿部轮廓。"))
    elif stance < 0.08 and knee_avg > 145:
        guidance.append(Suggestion(SuggestionPriority.MEDIUM, "pose", "双脚不要完全并齐", "让一只脚稍微前后错开，或把脚尖转出一点角度，保持小幅度即可。"))

    if abs(shoulder_y - hip_y) < 0.08:
        guidance.append(Suggestion(SuggestionPriority.MEDIUM, "pose", "给躯干加一点弧线", "肩和髋现在比较接近同一高度，试着让一侧肩稍高或身体略向一侧弯，不要主动做大幅度扭转。"))

    if orientation.facing in (FacingDirection.FRONT, FacingDirection.FRONT_LEFT, FacingDirection.FRONT_RIGHT):
        guidance.append(Suggestion(SuggestionPriority.MEDIUM, "pose", "头部不要完全锁在正中", "身体可以保持当前方向，只把头或下巴轻轻偏向一侧；这样既保留正面交流，又能减少证件照式的平面感。"))
    elif orientation.facing in (FacingDirection.BACK, FacingDirection.BACK_LEFT, FacingDirection.BACK_RIGHT):
        guidance.append(Suggestion(SuggestionPriority.MEDIUM, "pose", "保留背面，同时增加头部信息", "不必整个人转回来；只让头部或肩线回一点，通常就能同时得到背部轮廓和人物交流感。"))

    px, _ = composition.subject_position
    if px < 0.38:
        guidance.append(Suggestion(SuggestionPriority.MEDIUM, "pose", "把视觉动作朝画面中央打开", "主体偏左，手臂、膝盖或视线可以略向右打开，避免身体把可用的负空间堵住。"))
    elif px > 0.62:
        guidance.append(Suggestion(SuggestionPriority.MEDIUM, "pose", "把视觉动作朝画面中央打开", "主体偏右，尽量让手臂、膝盖或视线向左打开，为人物留出呼吸空间。"))

    if orientation.tilt == TiltDirection.LEANING_FORWARD:
        guidance.append(Suggestion(SuggestionPriority.MEDIUM, "pose", "前倾可以保留，但从髋部发力", "保持前倾的视觉感觉，同时避免只折腰；让髋部先移动，胸口和头部随后跟上，会更像一个主动姿态。"))

    priority_order = {SuggestionPriority.HIGH: 0, SuggestionPriority.MEDIUM: 1, SuggestionPriority.LOW: 2}
    guidance.sort(key=lambda s: priority_order[s.priority])
    return guidance[:5]


def _movement_options() -> list[str]:
    return list(_DEFAULT_TRANSITIONS)


def generate_suggestions(action: ActionResult, orientation: OrientationResult, camera: CameraResult, composition: CompositionResult) -> SuggestionResult:
    suggestions: list[Suggestion] = []
    pose_guidance = _generate_pose_guidance(action, orientation, composition)
    suggestions.extend(pose_guidance)
    photographer_cues = generate_photographer_cues(action, orientation, camera, composition)

    if action.confidence < 0.40:
        suggestions.append(Suggestion(SuggestionPriority.LOW, "pose", "动作类别仅供参考", f"当前更适合根据身体线条和构图调整，而不是追求“{action.category.value}”这个标签本身。"))
    if camera.shot_type in (ShotType.LONG, ShotType.EXTREME_LONG):
        suggestions.append(Suggestion(SuggestionPriority.HIGH, "camera", "先把人物放大到可读范围", "人物占画面比例偏小。优先考虑靠近或使用更长焦段，再决定动作；小人物的复杂姿态很难被读出来。"))
    elif camera.shot_type == ShotType.EXTREME_CLOSEUP:
        suggestions.append(Suggestion(SuggestionPriority.MEDIUM, "camera", "给肢体留一点空间", "当前构图已经很紧。若希望动作本身成为重点，可以稍微拉远，让肩、手或上半身进入画面。"))
    if camera.camera_angle == CameraAngle.HIGH_ANGLE:
        suggestions.append(Suggestion(SuggestionPriority.LOW, "camera", "俯拍下优先扩大身体轮廓", "俯拍会压缩身体的纵向高度；可以让四肢稍微展开，再利用地面或环境线条辅助构图。"))
    elif camera.camera_angle == CameraAngle.LOW_ANGLE:
        suggestions.append(Suggestion(SuggestionPriority.LOW, "camera", "仰拍下保持四肢清楚", "仰拍容易让近端肢体占比过大，避免手脚直接贴边；先保证主要轮廓完整。"))
    if abs(camera.dutch_angle_deg) > 5:
        suggestions.append(Suggestion(SuggestionPriority.LOW, "camera", "检查画面倾斜是否有意", f"当前估计倾斜约 {camera.dutch_angle_deg:.1f}°。如果不是刻意制造失衡感，优先校平相机，再调整人物姿态。"))
    for comp_suggestion in composition.suggestions:
        suggestions.append(Suggestion(SuggestionPriority.MEDIUM, "composition", "构图优化", comp_suggestion))
    if composition.headroom < 0.08:
        suggestions.append(Suggestion(SuggestionPriority.HIGH, "composition", "头部需要呼吸空间", "人物顶部过于贴边。先抬高取景或稍微后退，再要求人物做更夸张的动作。"))
    elif composition.headroom > 0.40:
        suggestions.append(Suggestion(SuggestionPriority.MEDIUM, "composition", "减少顶部留白", "上方空间较大。优先重新构图，而不是让人物主动“抬高”姿态去填空。"))
    if composition.thirds_alignment < 0.4:
        px, py = composition.subject_position
        suggestions.append(Suggestion(SuggestionPriority.MEDIUM, "composition", "让动作服务于画面方向", f"主体位于约 ({px:.0%}, {py:.0%})。不要为了三分法硬搬人物，而是让手、腿或视线朝空余方向展开。"))

    goals: list[str] = []
    titles = {s.title for s in pose_guidance}
    if "先释放对称站姿" in titles: goals.append("优先做非对称重心")
    if "把手臂从躯干上分开" in titles: goals.append("打开身体轮廓")
    if composition.subject_position[0] < 0.38 or composition.subject_position[0] > 0.62: goals.append("让动作朝负空间展开")
    if camera.shot_type in (ShotType.LONG, ShotType.EXTREME_LONG): goals.append("先保证人物可读性")
    if orientation.facing in (FacingDirection.BACK, FacingDirection.BACK_LEFT, FacingDirection.BACK_RIGHT): goals.append("保留背面轮廓并增加头部信息")
    if not goals: goals.append("以自然线条为优先，不必追求明确的动作标签")
    creative_direction = "；".join(goals) + "。"

    priority_order = {SuggestionPriority.HIGH: 0, SuggestionPriority.MEDIUM: 1, SuggestionPriority.LOW: 2}
    category_order = {"pose": 0, "camera": 1, "composition": 2, "action": 3}
    suggestions.sort(key=lambda s: (priority_order[s.priority], category_order.get(s.category, 9)))
    # Keep all pose-guidance contracts visible; the GUI already scrolls the list.
    return SuggestionResult(suggestions=suggestions[:16], next_actions=_movement_options(), creative_direction=creative_direction, photographer_cues=photographer_cues)
