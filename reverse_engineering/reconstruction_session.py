"""Versioned deterministic serialization for editable reconstruction sessions."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reverse_engineering.scene import SceneCamera, SceneModel, SceneSubject
from reverse_engineering.scene_anchors import SceneAnchor, default_scene_anchors

SCHEMA = "portrait-image-breakdown.reconstruction"
SCHEMA_VERSION = 2
SUPPORTED_VERSIONS = {1, 2}


def _subject_to_dict(subject: SceneSubject) -> dict[str, Any]:
    data = {
        "height": float(subject.height), "center_x": float(subject.center_x), "center_y": float(subject.center_y), "center_z": float(subject.center_z),
        "person_index": int(subject.person_index), "depth_is_relative": bool(subject.depth_is_relative), "depth_confidence": float(subject.depth_confidence),
    }
    if subject.keypoints is not None: data["keypoints"] = np.asarray(subject.keypoints, dtype=float).tolist()
    if subject.fitted_points_3d is not None: data["fitted_points_3d"] = np.asarray(subject.fitted_points_3d, dtype=float).tolist()
    return data


def _subject_from_dict(data: dict[str, Any]) -> SceneSubject:
    return SceneSubject(
        height=float(data.get("height", 1.7)), center_x=float(data.get("center_x", 0.0)), center_y=float(data.get("center_y", 0.0)), center_z=float(data.get("center_z", 0.0)),
        keypoints=np.asarray(data["keypoints"], dtype=float) if data.get("keypoints") is not None else None,
        fitted_points_3d=np.asarray(data["fitted_points_3d"], dtype=float) if data.get("fitted_points_3d") is not None else None,
        person_index=int(data.get("person_index", 0)), depth_is_relative=bool(data.get("depth_is_relative", False)), depth_confidence=float(data.get("depth_confidence", 0.0)),
    )


def _camera_to_dict(camera: SceneCamera) -> dict[str, Any]:
    return {"distance": float(camera.distance), "height": float(camera.height), "yaw": float(camera.yaw), "pitch": float(camera.pitch), "roll": float(camera.roll), "focal_length_mm": float(camera.focal_length_mm), "sensor_width_mm": float(camera.sensor_width_mm)}


def _camera_from_dict(data: dict[str, Any]) -> SceneCamera:
    return SceneCamera(distance=float(data.get("distance", 4.0)), height=float(data.get("height", 1.5)), yaw=float(data.get("yaw", 0.0)), pitch=float(data.get("pitch", 0.0)), roll=float(data.get("roll", 0.0)), focal_length_mm=float(data.get("focal_length_mm", 50.0)), sensor_width_mm=float(data.get("sensor_width_mm", 36.0)))


def scene_to_dict(scene: SceneModel, *, image_path: Optional[str] = None, image_shape: Optional[tuple[int, int]] = None, metadata: Optional[dict[str, Any]] = None, plane_constraints: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    anchors = []
    image_evidence = []
    for anchor in scene.anchors:
        errors = anchor.validate()
        if errors: raise ValueError(f"invalid scene anchor {anchor.anchor_id}: {'; '.join(errors)}")
        anchors.append(anchor.to_dict())
        if anchor.image_points:
            image_evidence.append({"anchor_id": anchor.anchor_id, "points": [[float(x), float(y)] for x, y in anchor.image_points], "constraint": str(anchor.reference_line_constraint)})
    subjects = [_subject_to_dict(subject) for subject in scene.subjects]
    primary_index = next((i for i, subject in enumerate(scene.subjects) if int(subject.person_index) == int(scene.subject.person_index)), 0)
    return {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "image": {"path": str(image_path) if image_path else None, "width": int(image_shape[0]) if image_shape else None, "height": int(image_shape[1]) if image_shape else None},
        "camera": _camera_to_dict(scene.camera), "subjects": subjects, "primary_subject_index": primary_index,
        "ground_size": float(scene.ground_size), "candidate_selected": int(scene.selected_candidate), "relative_layout": bool(scene.relative_layout),
        "anchors": anchors, "image_evidence": image_evidence, "plane_constraints": list(plane_constraints or getattr(scene, "plane_constraints", [])), "metadata": dict(metadata or {}),
    }


def migrate_session(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != SCHEMA: raise ValueError("unsupported reconstruction session schema")
    version = int(data.get("schema_version", 0))
    if version not in SUPPORTED_VERSIONS: raise ValueError(f"unsupported reconstruction session version: {version}")
    migrated = dict(data)
    if version == 1:
        evidence = []
        for anchor in migrated.get("anchors", []):
            points = anchor.get("image_points", [])
            if points:
                evidence.append({"anchor_id": anchor.get("anchor_id"), "points": points, "constraint": anchor.get("reference_line_constraint", "free")})
        migrated["image_evidence"] = evidence
        migrated["schema_version"] = 2
    return migrated


def scene_from_dict(data: dict[str, Any]) -> tuple[SceneModel, list[dict[str, Any]]]:
    migrated = migrate_session(data)
    subjects = [_subject_from_dict(item) for item in migrated.get("subjects", [])]
    if subjects:
        primary_index = min(max(int(migrated.get("primary_subject_index", 0)), 0), len(subjects) - 1)
        primary = subjects[primary_index]
    else:
        primary = SceneSubject(); subjects = [primary]
    anchors = []
    for item in migrated.get("anchors", []):
        anchor = SceneAnchor.from_dict(item)
        errors = anchor.validate()
        if errors: raise ValueError(f"invalid scene anchor {anchor.anchor_id}: {'; '.join(errors)}")
        anchors.append(anchor)
    if not anchors: anchors = default_scene_anchors()
    scene = SceneModel(camera=_camera_from_dict(dict(migrated.get("camera", {}))), subject=primary, ground_size=float(migrated.get("ground_size", 24.0)), selected_candidate=int(migrated.get("candidate_selected", 0)), subjects=subjects, relative_layout=bool(migrated.get("relative_layout", False)), anchors=anchors)
    scene.plane_constraints = list(migrated.get("plane_constraints", []))
    for evidence in migrated.get("image_evidence", []):
        anchor = scene.anchor_by_id(str(evidence.get("anchor_id", "")))
        if anchor is None: continue
        anchor.image_points = tuple((float(p[0]), float(p[1])) for p in evidence.get("points", []) if len(p) >= 2)
        anchor.reference_line_constraint = str(evidence.get("constraint", anchor.reference_line_constraint))
    return scene, scene.plane_constraints


def save_session(path: str | Path, scene: SceneModel, *, image_path: Optional[str] = None, image_shape: Optional[tuple[int, int]] = None, metadata: Optional[dict[str, Any]] = None, plane_constraints: Optional[list[dict[str, Any]]] = None) -> Path:
    target = Path(path)
    payload = scene_to_dict(scene, image_path=image_path, image_shape=image_shape, metadata=metadata, plane_constraints=plane_constraints)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    payload["integrity"] = {"algorithm": "sha256", "payload_sha256": hashlib.sha256(encoded).hexdigest()}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(target)
    return target


def load_session(path: str | Path) -> tuple[SceneModel, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    integrity = dict(data.pop("integrity", {}))
    expected = integrity.get("payload_sha256")
    if expected:
        actual = hashlib.sha256(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")).hexdigest()
        if actual != expected: raise ValueError("reconstruction session integrity check failed")
    migrated = migrate_session(data)
    scene, _ = scene_from_dict(migrated)
    return scene, migrated
