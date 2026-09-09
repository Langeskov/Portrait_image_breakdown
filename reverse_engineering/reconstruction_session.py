"""Versioned, deterministic serialization for editable v3 reconstruction sessions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reverse_engineering.scene import SceneCamera, SceneModel, SceneSubject
from reverse_engineering.scene_anchors import SceneAnchor, default_scene_anchors

SCHEMA = "portrait-image-breakdown.reconstruction"
SCHEMA_VERSION = 1


def _finite_tuple(values, size, default):
    try:
        result = tuple(float(v) for v in values)
        if len(result) == size and np.isfinite(result).all():
            return result
    except (TypeError, ValueError):
        pass
    return tuple(default)


def _camera_to_dict(camera: SceneCamera) -> dict[str, Any]:
    return {
        "distance": float(camera.distance),
        "height": float(camera.height),
        "yaw": float(camera.yaw),
        "pitch": float(camera.pitch),
        "roll": float(camera.roll),
        "focal_length_mm": float(camera.focal_length_mm),
        "sensor_width_mm": float(camera.sensor_width_mm),
    }


def _camera_from_dict(data: dict[str, Any]) -> SceneCamera:
    return SceneCamera(
        distance=float(data.get("distance", 4.0)),
        height=float(data.get("height", 1.5)),
        yaw=float(data.get("yaw", 0.0)),
        pitch=float(data.get("pitch", 0.0)),
        roll=float(data.get("roll", 0.0)),
        focal_length_mm=float(data.get("focal_length_mm", 50.0)),
        sensor_width_mm=float(data.get("sensor_width_mm", 36.0)),
    )


def _subject_to_dict(subject: SceneSubject) -> dict[str, Any]:
    data: dict[str, Any] = {
        "height": float(subject.height),
        "center_x": float(subject.center_x),
        "center_y": float(subject.center_y),
        "center_z": float(subject.center_z),
        "person_index": int(subject.person_index),
        "depth_is_relative": bool(subject.depth_is_relative),
        "depth_confidence": float(subject.depth_confidence),
    }
    if subject.keypoints is not None:
        data["keypoints"] = np.asarray(subject.keypoints, dtype=float).tolist()
    if subject.fitted_points_3d is not None:
        data["fitted_points_3d"] = np.asarray(subject.fitted_points_3d, dtype=float).tolist()
    return data


def _subject_from_dict(data: dict[str, Any]) -> SceneSubject:
    keypoints = data.get("keypoints")
    fitted = data.get("fitted_points_3d")
    return SceneSubject(
        height=float(data.get("height", 1.7)),
        center_x=float(data.get("center_x", 0.0)),
        center_y=float(data.get("center_y", 0.0)),
        center_z=float(data.get("center_z", 0.0)),
        keypoints=np.asarray(keypoints, dtype=float) if keypoints is not None else None,
        fitted_points_3d=np.asarray(fitted, dtype=float) if fitted is not None else None,
        person_index=int(data.get("person_index", 0)),
        depth_is_relative=bool(data.get("depth_is_relative", False)),
        depth_confidence=float(data.get("depth_confidence", 0.0)),
    )


def _validate_anchor(anchor: SceneAnchor) -> None:
    errors = anchor.validate()
    if errors:
        raise ValueError(f"invalid scene anchor {anchor.anchor_id}: {'; '.join(errors)}")


def scene_to_dict(
    scene: SceneModel,
    *,
    image_path: Optional[str] = None,
    image_shape: Optional[tuple[int, int]] = None,
    metadata: Optional[dict[str, Any]] = None,
    plane_constraints: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    anchors = []
    for anchor in scene.anchors:
        _validate_anchor(anchor)
        anchors.append(anchor.to_dict())
    subjects = [_subject_to_dict(subject) for subject in scene.subjects]
    primary_index = next(
        (i for i, subject in enumerate(subjects) if int(subject["person_index"]) == int(scene.subject.person_index)),
        0,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": {
            "path": str(image_path) if image_path else None,
            "width": int(image_shape[0]) if image_shape else None,
            "height": int(image_shape[1]) if image_shape else None,
        },
        "camera": _camera_to_dict(scene.camera),
        "subjects": subjects,
        "primary_subject_index": primary_index,
        "ground_size": float(scene.ground_size),
        "candidate_selected": int(scene.selected_candidate),
        "relative_layout": bool(scene.relative_layout),
        "anchors": anchors,
        "plane_constraints": list(plane_constraints or getattr(scene, "plane_constraints", [])),
        "metadata": dict(metadata or {}),
    }
    return result


def scene_from_dict(data: dict[str, Any]) -> tuple[SceneModel, list[dict[str, Any]]]:
    if data.get("schema") != SCHEMA:
        raise ValueError("unsupported reconstruction session schema")
    version = int(data.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported reconstruction session version: {version}")

    subjects = [_subject_from_dict(item) for item in data.get("subjects", [])]
    if subjects:
        primary_index = min(max(int(data.get("primary_subject_index", 0)), 0), len(subjects) - 1)
        primary = subjects[primary_index]
    else:
        primary = SceneSubject()
        subjects = [primary]

    anchors = []
    for item in data.get("anchors", []):
        anchor = SceneAnchor.from_dict(item)
        _validate_anchor(anchor)
        anchors.append(anchor)
    if not anchors:
        anchors = default_scene_anchors()

    scene = SceneModel(
        camera=_camera_from_dict(dict(data.get("camera", {}))),
        subject=primary,
        ground_size=float(data.get("ground_size", 24.0)),
        candidate_solutions=[],
        selected_candidate=int(data.get("candidate_selected", 0)),
        subjects=subjects,
        relative_layout=bool(data.get("relative_layout", False)),
        anchors=anchors,
    )
    scene.plane_constraints = list(data.get("plane_constraints", []))
    return scene, scene.plane_constraints


def save_session(
    path: str | Path,
    scene: SceneModel,
    *,
    image_path: Optional[str] = None,
    image_shape: Optional[tuple[int, int]] = None,
    metadata: Optional[dict[str, Any]] = None,
    plane_constraints: Optional[list[dict[str, Any]]] = None,
) -> Path:
    target = Path(path)
    payload = scene_to_dict(
        scene,
        image_path=image_path,
        image_shape=image_shape,
        metadata=metadata,
        plane_constraints=plane_constraints,
    )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    checksum = hashlib.sha256(encoded).hexdigest()
    payload["integrity"] = {"algorithm": "sha256", "payload_sha256": checksum}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(target)
    return target


def load_session(path: str | Path) -> tuple[SceneModel, dict[str, Any]]:
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    integrity = dict(data.pop("integrity", {}))
    expected = integrity.get("payload_sha256")
    if expected:
        encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        actual = hashlib.sha256(encoded).hexdigest()
        if actual != expected:
            raise ValueError("reconstruction session integrity check failed")
    scene, _ = scene_from_dict(data)
    return scene, data
