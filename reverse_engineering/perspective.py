"""
透视分析模块 - PerspectiveAnalyzer
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from reverse_engineering.data_types import EstimatedValue, PerspectiveResult

def _detect_lines(image, max_lines=80):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60,
                            minLineLength=max(image.shape[0], image.shape[1])//15, maxLineGap=20)
    return lines.reshape(-1, 4) if lines is not None else None

def _line_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def _intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

def _find_vanishing_points(lines, w, h, angle_thresh=15):
    diag = []
    for line in lines:
        angle = _line_angle(*line)
        if not (abs(angle) < angle_thresh or abs(angle-180) < angle_thresh or abs(angle+180) < angle_thresh):
            if not (abs(angle-90) < angle_thresh or abs(angle+90) < angle_thresh):
                diag.append(line)
    vps = []
    if len(diag) >= 2:
        intersections = []
        for i in range(min(len(diag), 20)):
            for j in range(i+1, min(len(diag), 20)):
                pt = _intersection(diag[i], diag[j])
                if pt is not None:
                    margin = max(w, h) * 2
                    if -margin < pt[0] < w+margin and -margin < pt[1] < h+margin:
                        intersections.append((pt[0]/w, pt[1]/h))
        if intersections:
            vps.append((float(np.median([p[0] for p in intersections])),
                        float(np.median([p[1] for p in intersections]))))
    return vps[:3]

def _perspective_strength(lines):
    angles = [_line_angle(*l) for l in lines]
    return min(float(np.std(angles)) / 45.0, 1.0)

def analyze_perspective(image):
    h, w = image.shape[:2]
    lines = _detect_lines(image)
    if lines is None or len(lines) < 5:
        return PerspectiveResult(
            perspective_strength=EstimatedValue(0.3, confidence=0.2, basis=["few lines"]),
            perspective_type=EstimatedValue("normal", confidence=0.2, basis=["insufficient data"]),
            vanishing_points=[], vertical_convergence=EstimatedValue(0.0, confidence=0.1),
            horizontal_convergence=EstimatedValue(0.0, confidence=0.1))
    strength = _perspective_strength(lines)
    basis = ["strong diagonal convergence" if strength > 0.6 else "moderate convergence" if strength > 0.35 else "limited convergence"]
    vps = _find_vanishing_points(lines, w, h)
    vert_lines = [l for l in lines if abs(abs(_line_angle(*l)) - 90) < 15]
    vert_conv = 0.0
    if len(vert_lines) >= 2:
        top_ys = [min(l[1], l[3]) for l in vert_lines]
        vert_conv = 1.0 - min(float(np.std(top_ys)) / (h + 1e-6) * 5, 1.0)
    if strength > 0.55: ptype, pc, pb = "wide", min(strength*1.2, 0.9), ["strong perspective"]
    elif strength > 0.3: ptype, pc, pb = "normal", 0.5, ["moderate perspective"]
    else: ptype, pc, pb = "telephoto", min((1-strength)*1.2, 0.85), ["flat perspective"]
    return PerspectiveResult(
        perspective_strength=EstimatedValue(round(strength,3), confidence=min(strength*1.5+0.2, 0.9), basis=basis),
        perspective_type=EstimatedValue(ptype, confidence=pc, basis=pb),
        vanishing_points=[(round(vp[0],3), round(vp[1],3)) for vp in vps],
        vertical_convergence=EstimatedValue(round(vert_conv,3), confidence=0.5 if vert_lines else 0.1),
        horizontal_convergence=EstimatedValue(round(strength*0.8,3), confidence=0.4, basis=["horizontal line spread"]))
