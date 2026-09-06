"""Perspective analysis backed by shared scene-geometry evidence."""
from __future__ import annotations

import math

import cv2
import numpy as np

from reverse_engineering.data_types import EstimatedValue, PerspectiveResult
from reverse_engineering.scene_geometry import analyze_scene_geometry


def _line_angle(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def analyze_perspective(image):
    h, w = image.shape[:2]
    scene = analyze_scene_geometry(image)
    normalized_lines = [
        (round(line.x1 / w, 5), round(line.y1 / h, 5),
         round(line.x2 / w, 5), round(line.y2 / h, 5))
        for line in scene.lines
    ]

    angles = np.array([line.angle_deg for line in scene.lines], dtype=float)
    strength = float(np.clip(np.std(angles) / 45.0, 0.0, 1.0)) if len(angles) >= 2 else 0.0
    if strength > 0.6:
        ptype, pc = "strong_perspective", 0.65
    elif strength > 0.35:
        ptype, pc = "moderate_perspective", 0.50
    elif len(scene.lines) >= 5:
        ptype, pc = "flat_or_limited_perspective", 0.45
    else:
        ptype, pc = "unknown", 0.10

    vertical_support = 0
    horizontal_support = 0
    if scene.vertical_cluster is not None and scene.vertical_cluster < len(scene.clusters):
        vertical_support = len(scene.clusters[scene.vertical_cluster])
    if scene.horizontal_clusters:
        horizontal_support = sum(len(scene.clusters[i]) for i in scene.horizontal_clusters if i < len(scene.clusters))

    vp_values = [(round(vp.x / w, 5), round(vp.y / h, 5)) for vp in scene.vanishing_points]
    return PerspectiveResult(
        perspective_strength=EstimatedValue(
            round(strength, 3),
            confidence=min(0.75, 0.20 + 0.55 * scene.confidence),
            basis=[f"{len(scene.lines)} detected line segments", "orientation spread used qualitatively"],
        ),
        perspective_type=EstimatedValue(ptype, confidence=pc, basis=["scene line orientation distribution"]),
        vanishing_points=vp_values,
        vertical_convergence=EstimatedValue(
            round(min(1.0, vertical_support / 12.0), 3),
            confidence=min(0.8, 0.20 + vertical_support / 15.0) if vertical_support else 0.1,
            basis=["Manhattan vertical-direction cluster"],
        ),
        horizontal_convergence=EstimatedValue(
            round(min(1.0, horizontal_support / 20.0), 3),
            confidence=min(0.8, 0.20 + horizontal_support / 24.0) if horizontal_support else 0.1,
            basis=["two horizontal Manhattan-direction clusters"],
        ),
        line_segments=normalized_lines,
    )
