"""Perspective analysis using OpenCV line detection and vanishing-point evidence."""
from __future__ import annotations

import math
import cv2
import numpy as np
from reverse_engineering.data_types import EstimatedValue, PerspectiveResult


def _detect_lines(image, max_lines=80):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = max(24, max(image.shape[0], image.shape[1]) // 15)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60,
                            minLineLength=min_len, maxLineGap=20)
    if lines is None:
        return np.empty((0, 4), dtype=int)
    lines = lines.reshape(-1, 4)
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    order = np.argsort(-lengths)
    return lines[order[:max_lines]]


def _line_angle(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _line_intersection(l1, l2):
    x1, y1, x2, y2 = map(float, l1)
    x3, y3, x4, y4 = map(float, l2)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def _intersection_candidates(lines, w, h):
    points = []
    margin = max(w, h) * 3
    for i in range(len(lines)):
        for j in range(i + 1, min(i + 30, len(lines))):
            p = _line_intersection(lines[i], lines[j])
            if p is None:
                continue
            x, y = p
            if -margin <= x <= w + margin and -margin <= y <= h + margin:
                points.append((x / w, y / h))
    return points


def _find_vanishing_points(lines, w, h):
    if len(lines) < 4:
        return []
    points = _intersection_candidates(lines, w, h)
    if not points:
        return []
    # Robust median is intentionally used as an evidence point, not a claim of
    # exact camera calibration.
    return [(float(np.median([p[0] for p in points])),
             float(np.median([p[1] for p in points])))]


def _perspective_strength(lines):
    if len(lines) < 2:
        return 0.0
    # Use angular spread only as a qualitative scene complexity signal. It is
    # deliberately not interpreted as focal length.
    angles = np.array([_line_angle(l) for l in lines], dtype=float)
    folded = ((angles + 90.0) % 180.0) - 90.0
    return float(np.clip(np.std(folded) / 45.0, 0.0, 1.0))


def analyze_perspective(image):
    h, w = image.shape[:2]
    lines = _detect_lines(image)
    normalized_lines = [
        (round(float(x1 / w), 5), round(float(y1 / h), 5),
         round(float(x2 / w), 5), round(float(y2 / h), 5))
        for x1, y1, x2, y2 in lines
    ]

    if len(lines) < 5:
        return PerspectiveResult(
            perspective_strength=EstimatedValue(0.0, confidence=0.15, basis=["few reliable line segments"]),
            perspective_type=EstimatedValue("unknown", confidence=0.1, basis=["insufficient line evidence"]),
            vanishing_points=[],
            vertical_convergence=EstimatedValue(0.0, confidence=0.1),
            horizontal_convergence=EstimatedValue(0.0, confidence=0.1),
            line_segments=normalized_lines,
        )

    strength = _perspective_strength(lines)
    vps = _find_vanishing_points(lines, w, h)
    vert_lines = [l for l in lines if abs(abs(_line_angle(l)) - 90) < 12]
    horiz_lines = [l for l in lines if min(abs(_line_angle(l)), abs(abs(_line_angle(l)) - 180)) < 12]

    vert_conv = min(1.0, max(0.0, len(vert_lines) / max(len(lines) * 0.4, 1)))
    horiz_conv = min(1.0, max(0.0, len(horiz_lines) / max(len(lines) * 0.4, 1)))

    # Qualitative only. Do not call this a lens measurement.
    if strength > 0.6:
        ptype, pc = "strong_perspective", 0.65
    elif strength > 0.35:
        ptype, pc = "moderate_perspective", 0.5
    else:
        ptype, pc = "flat_or_limited_perspective", 0.45

    return PerspectiveResult(
        perspective_strength=EstimatedValue(
            round(strength, 3),
            confidence=min(0.7, 0.2 + 0.5 * (len(lines) / 80.0)),
            basis=[f"{len(lines)} detected line segments", "angular spread used qualitatively"],
        ),
        perspective_type=EstimatedValue(ptype, confidence=pc,
                                         basis=["line orientation distribution"]),
        vanishing_points=[(round(x, 3), round(y, 3)) for x, y in vps],
        vertical_convergence=EstimatedValue(round(float(vert_conv), 3), confidence=0.35 if vert_lines else 0.1,
                                            basis=["vertical line evidence"]),
        horizontal_convergence=EstimatedValue(round(float(horiz_conv), 3), confidence=0.35 if horiz_lines else 0.1,
                                              basis=["horizontal line evidence"]),
        line_segments=normalized_lines,
    )
