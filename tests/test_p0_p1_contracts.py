from reverse_engineering.reference_anchor import ReferenceImageAnchor
from reverse_engineering.reference_reconstruction import ReferenceComposition
from reverse_engineering.reconstruction_session import SCHEMA, SCHEMA_VERSION, migrate_session, scene_from_dict, scene_to_dict
from reverse_engineering.scene import SceneModel, SceneSubject


def test_reference_anchor_is_distinct_from_world_anchor():
    anchor = ReferenceImageAnchor("rail", 10.0, 20.0, 0.9, "manual")
    composition = ReferenceComposition(100, 100, (anchor,), None, (50.0, 50.0), 0.1)
    assert type(anchor).__name__ == "ReferenceImageAnchor"
    assert composition.anchors[0] is anchor


def test_subject_collection_is_canonical():
    first = SceneSubject(person_index=1)
    second = SceneSubject(person_index=2)
    scene = SceneModel(subjects=[first, second], subject=second)
    scene.subject.center_x = 2.0
    assert scene.subjects[1].center_x == 2.0
    replacement = SceneSubject(person_index=2, center_x=3.0)
    scene.subject = replacement
    assert scene.subject is replacement
    assert len(scene.subjects) == 2


def test_v1_session_migration_is_idempotent():
    legacy = {
        "schema": SCHEMA,
        "schema_version": 1,
        "subjects": [{"person_index": 0}],
        "primary_subject_index": 0,
        "anchors": [{"anchor_id": "wall", "name": "Wall", "kind": "plane", "position": [0, 1, 0], "normal": [0, 0, 1], "size": [4, 3], "image_points": [[10, 20], [100, 20]], "reference_line_constraint": "horizontal"}],
    }
    migrated = migrate_session(legacy)
    assert migrated["schema_version"] == SCHEMA_VERSION
    assert migrated["image_evidence"][0]["constraint"] == "horizontal"
    assert migrate_session(migrated) == migrated
    scene, _ = scene_from_dict(migrated)
    assert scene.anchor_by_id("wall").image_points == ((10.0, 20.0), (100.0, 20.0))


def test_new_session_contains_independent_image_evidence():
    scene = SceneModel()
    anchor = scene.anchor_by_id("ground")
    anchor.image_points = ((1.0, 2.0), (11.0, 2.0))
    anchor.reference_line_constraint = "horizontal"
    payload = scene_to_dict(scene)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["image_evidence"][0]["anchor_id"] == "ground"
