"""Scene-geometry evidence for camera rotation recovery.

This module deliberately separates scene geometry from human pose.  It uses
standard OpenCV line detection plus deterministic robust fitting to estimate
Manhattan-style vanishing points and the image-space horizon.  The result is
evidence, not a claim that every photo has a Manhattan world.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    cluster: int = -1

    def homogeneous(self) -> np.ndarray:
        p1 = np.array([self.x1, self.y1, 1.0], dtype=np.float64)
        p2 = np.array([self.x2, self.y2, 1.0], dtype=np.float64)
        line = np.cross(p1, p2)
        norm = math.hypot(float(line[0]), float(line[1]))
        return line / max(norm, 1e-12)


@dataclass(frozen=True)
class VanishingPoint:
    x: float
    y: float
    cluster: int
    support: int
    confidence: float
    mean_line_residual_px: float


@dataclass(frozen=True)
class SceneGeometryEvidence:
    width: int
    height: int
    lines: tuple[LineSegment, ...]
    clusters: tuple[tuple[int, ...], ...]
    vanishing_points: tuple[VanishingPoint, ...]
    vertical_cluster: int | None
    horizontal_clusters: tuple[int, ...]
    horizon_angle_deg: float | None
    confidence: float

    @property
    def has_three_directions(self) -> bool:
        return len(self.vanishing_points) >= 3 and len(self.horizontal_clusters) >= 2 and self.vertical_cluster is not None


def _angle_deg(dx: float, dy: float) -> float:
    angle = math.degrees(math.atan2(dy, dx))
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def detect_line_segments(image: np.ndarray, max_lines: int = 160) -> list[LineSegment]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    h, w = gray.shape[:2]
    min_len = max(30, int(min(w, h) * 0.08))
    raw = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 1800.0,
        threshold=max(30, int(min(w, h) * 0.05)),
        minLineLength=min_len,
        maxLineGap=max(12, int(min(w, h) * 0.03)),
    )
    if raw is None:
        return []
    segments: list[LineSegment] = []
    for x1, y1, x2, y2 in np.asarray(raw).reshape(-1, 4):
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        if length < min_len:
            continue
        segments.append(LineSegment(float(x1), float(y1), float(x2), float(y2), length, _angle_deg(x2 - x1, y2 - y1)))
    segments.sort(key=lambda s: s.length, reverse=True)
    return segments[:max_lines]


def _angle_distance(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _cluster_orientations(lines: list[LineSegment], max_clusters: int = 3) -> list[list[int]]:
    if not lines:
        return []
    k = min(max_clusters, len(lines))
    features = np.array([[math.cos(math.radians(2.0 * line.angle_deg)), math.sin(math.radians(2.0 * line.angle_deg))] for line in lines], dtype=np.float64)

    centers: list[np.ndarray] = [features[0].copy()]
    while len(centers) < k:
        dist = np.min([np.sum((features - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(features[int(np.argmax(dist))].copy())

    centers_arr = np.asarray(centers, dtype=np.float64)
    labels = np.zeros(len(lines), dtype=int)
    for _ in range(25):
        d = ((features[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d, axis=1)
        new_centers = centers_arr.copy()
        for cluster in range(k):
            members = features[new_labels == cluster]
            if len(members):
                m = members.mean(axis=0)
                n = np.linalg.norm(m)
                if n > 1e-9:
                    new_centers[cluster] = m / n
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers_arr):
            labels = new_labels
            break
        labels, centers_arr = new_labels, new_centers

    clusters = [[i for i, label in enumerate(labels) if label == cluster] for cluster in range(k)]
    clusters = [c for c in clusters if len(c) >= 2]
    clusters.sort(key=lambda c: -sum(lines[i].length for i in c))
    return clusters


def _estimate_vanishing_point(lines: list[LineSegment], indices: list[int], width: int, height: int, cluster_id: int) -> VanishingPoint | None:
    if len(indices) < 2:
        return None
    A = []
    weights = []
    for idx in indices:
        line = lines[idx].homogeneous()
        A.append(line)
        weights.append(math.sqrt(max(lines[idx].length, 1.0)))
    A = np.asarray(A, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)[:, None]
    _, _, vt = np.linalg.svd(A * weights, full_matrices=False)
    v = vt[-1]
    if abs(v[2]) < 1e-9:
        return None
    x, y = (v[:2] / v[2]).tolist()
    if not np.isfinite([x, y]).all():
        return None

    # Compute support in pixels against the fitted VP.  Large-distance VPs
    # are allowed because room vanishing points commonly lie outside the image.
    residuals = []
    for idx in indices:
        line = lines[idx].homogeneous()
        residuals.append(abs(float(line @ np.array([x, y, 1.0]))))
    median_residual = float(np.median(residuals))
    image_diag = math.hypot(width, height)
    support_conf = min(1.0, len(indices) / 12.0)
    residual_conf = math.exp(-median_residual / max(2.0, 0.012 * image_diag))
    distance = math.hypot(x - width * 0.5, y - height * 0.5)
    distance_penalty = 1.0 / (1.0 + distance / max(image_diag * 6.0, 1.0))
    confidence = float(np.clip(0.15 + 0.55 * support_conf + 0.25 * residual_conf + 0.05 * distance_penalty, 0.0, 0.95))
    return VanishingPoint(x, y, cluster_id, len(indices), confidence, median_residual)


def analyze_scene_geometry(image: np.ndarray) -> SceneGeometryEvidence:
    h, w = image.shape[:2]
    lines = detect_line_segments(image)
    clusters = _cluster_orientations(lines)
    vps: list[VanishingPoint] = []
    cluster_records: list[tuple[int, ...]] = []
    for cluster_id, indices in enumerate(clusters):
        vp = _estimate_vanishing_point(lines, indices, w, h, cluster_id)
        if vp is None:
            continue
        # Write cluster id after filtering so it always indexes the returned tuple.
        vps.append(VanishingPoint(vp.x, vp.y, cluster_id, vp.support, vp.confidence, vp.mean_line_residual_px))
        cluster_records.append(tuple(indices))

    vertical_cluster: int | None = None
    if clusters:
        mean_angles = [
            float(np.average([lines[i].angle_deg for i in cluster], weights=[lines[i].length for i in cluster]))
            for cluster in clusters
        ]
        vertical_cluster = int(np.argmin([_angle_distance(angle, 90.0) for angle in mean_angles]))

    horizontal_clusters = tuple(i for i in range(len(clusters)) if i != vertical_cluster)
    horizontal_vps = [vp for vp in vps if vp.cluster in horizontal_clusters]
    horizon_angle = None
    if len(horizontal_vps) >= 2:
        a, b = horizontal_vps[0], horizontal_vps[1]
        horizon_angle = _angle_deg(b.x - a.x, b.y - a.y)

    if vps:
        confidence = float(np.mean([vp.confidence for vp in vps]))
        if len(vps) >= 3:
            confidence = min(0.95, confidence + 0.10)
    else:
        confidence = 0.0

    return SceneGeometryEvidence(
        width=w,
        height=h,
        lines=tuple(lines),
        clusters=tuple(cluster_records),
        vanishing_points=tuple(vps),
        vertical_cluster=vertical_cluster,
        horizontal_clusters=horizontal_clusters,
        horizon_angle_deg=horizon_angle,
        confidence=confidence,
    )
