"""
焦段估计 - FocalLengthEstimator

明确: 焦段不能从透视唯一确定。
输出候选解列表而非单一值。
"""
from __future__ import annotations
from reverse_engineering.data_types import EstimatedValue, FocalLengthResult
from reverse_engineering.geometry import CameraIntrinsics, PoseCandidate


def estimate_focal_length(
    perspective_strength: float,
    perspective_type: str,
    subject_scale: float,
    candidates: list[PoseCandidate] = None,
) -> FocalLengthResult:
    """
    估计焦段

    如果有几何求解的候选解, 优先使用。
    否则基于透视和主体比例给出类别估计(不给出虚假精确值)。
    """
    # 如果有几何候选解, 使用最佳解
    if candidates:
        best = candidates[0]
        # 候选解列表
        candidate_info = []
        for c in candidates[:3]:
            candidate_info.append(f"{c.focal_equiv_35mm}mm/d={c.distance}m/score={c.score:.2f}")

        equiv = best.focal_equiv_35mm
        if equiv < 35: cat = "wide"
        elif equiv < 60: cat = "normal"
        elif equiv < 105: cat = "short_telephoto"
        else: cat = "telephoto"

        margin = equiv * 0.25
        return FocalLengthResult(
            category=EstimatedValue(
                cat, confidence=best.score,
                basis=["geometric constraint"] + candidate_info),
            equivalent_35mm=EstimatedValue(
                round(equiv), unit="mm",
                range_min=max(18, round(equiv - margin)),
                range_max=min(300, round(equiv + margin)),
                confidence=best.score * 0.9,
                basis=["body geometry + projection model"]))

    # Fallback: 类别估计(不给出虚假精确值)
    scores = {"wide": 0, "normal": 0, "short_telephoto": 0, "telephoto": 0}
    if perspective_strength > 0.55: scores["wide"] += 0.4
    elif perspective_strength > 0.35: scores["normal"] += 0.3; scores["short_telephoto"] += 0.2
    else: scores["telephoto"] += 0.3; scores["short_telephoto"] += 0.3

    if subject_scale > 0.6: scores["telephoto"] += 0.3
    elif subject_scale > 0.3: scores["short_telephoto"] += 0.3; scores["normal"] += 0.2
    elif subject_scale > 0.15: scores["normal"] += 0.3
    else: scores["wide"] += 0.3

    if perspective_type == "wide": scores["wide"] += 0.2
    elif perspective_type == "telephoto": scores["telephoto"] += 0.2

    cat = max(scores, key=scores.get)
    conf = min(scores[cat] + 0.15, 0.7)

    fmap = {"wide": (24, 18, 35), "normal": (45, 35, 60),
            "short_telephoto": (85, 70, 105), "telephoto": (150, 100, 300)}
    est, lo, hi = fmap[cat]

    return FocalLengthResult(
        category=EstimatedValue(
            cat, confidence=conf,
            basis=[f"persp={perspective_strength:.2f}", f"type={perspective_type}",
                   f"scale={subject_scale:.2f}"]),
        equivalent_35mm=EstimatedValue(
            est, unit="mm", range_min=lo, range_max=hi,
            confidence=conf * 0.6,
            basis=["perspective + subject scale heuristic",
                   "NOTE: cannot be uniquely determined from single image"]))
