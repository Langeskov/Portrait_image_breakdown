"""
摄影手法识别 - ShootingTechniqueAnalyzer
"""
from __future__ import annotations
from reverse_engineering.data_types import ShootingTechniqueResult

def classify_techniques(perspective, camera_pose, focal_length, dof, motion, composition, subject_scale):
    techs = []
    def add(name, conf, desc=""):
        if conf > 0.2: techs.append({"name": name, "confidence": round(min(conf, 0.95), 2), "description": desc})
    if dof.dof_type.value == "shallow":
        add("shallow_depth_of_field", dof.dof_type.confidence, "background blur isolates subject")
    if dof.background_blur > 0.5 and subject_scale > 0.3:
        add("subject_isolation", min(dof.background_blur, 0.9))
    if perspective.perspective_type.value == "wide":
        add("wide_angle_exaggeration", perspective.perspective_type.confidence)
    if perspective.perspective_type.value in ("telephoto", "short_telephoto"):
        add("telephoto_compression", perspective.perspective_type.confidence)
    pitch = camera_pose.camera_pitch.value
    if pitch is not None:
        if pitch > 10: add("low_angle", 0.6 + abs(pitch)/60, "shooting upward")
        elif pitch < -10: add("high_angle", 0.6 + abs(pitch)/60, "shooting downward")
        else: add("eye_level", 0.7, "natural perspective")
    for s in composition.styles:
        sn = s["name"]; sc = s["confidence"]
        if sn == "rule_of_thirds": add("rule_of_thirds_composition", sc)
        elif sn == "center": add("static_composition", sc)
        elif sn == "leading_lines": add("leading_line", sc)
        elif sn == "frame_within_frame": add("foreground_framing", sc)
    if motion.blur_type.value == "panning": add("panning", motion.blur_type.confidence, "tracking blur")
    elif motion.blur_type.value == "long_exposure": add("long_exposure", motion.blur_type.confidence)
    if 0.15 < subject_scale < 0.5:
        add("environmental_portrait", 0.6 + (0.3 - abs(subject_scale - 0.3)) * 2, "subject in environment")
    elif subject_scale > 0.5:
        add("portrait", 0.7, "close-up portrait")
    techs.sort(key=lambda t: t["confidence"], reverse=True)
    return ShootingTechniqueResult(techniques=techs[:10])
