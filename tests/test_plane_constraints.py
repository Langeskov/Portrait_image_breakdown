from reverse_engineering.plane_constraints import (
    PlaneConstraint,
    PlaneRelation,
    apply_position_constraint,
    evaluate_constraint,
    signed_distance_to_plane,
)
from reverse_engineering.scene_anchors import AnchorKind, SceneAnchor


def test_plane_constraint_projects_point_and_reports_residual():
    plane = SceneAnchor("ground", "Ground", AnchorKind.PLANE, (0, 0, 0), (0, 1, 0), (4, 4))
    constraint = PlaneConstraint("c1", "ground", relation=PlaneRelation.ON_PLANE)
    point = (1.0, 1.2, 2.0)
    assert signed_distance_to_plane(point, plane) == 1.2
    corrected = apply_position_constraint(point, constraint, [plane])
    assert corrected == (1.0, 0.0, 2.0)
    result = evaluate_constraint(point, constraint, [plane])
    assert result.residual_m == 1.2
    assert not result.satisfied


def test_angular_plane_constraint_requires_target_direction():
    plane = SceneAnchor("wall", "Wall", AnchorKind.PLANE, (0, 0, 0), (0, 0, 1), (4, 4))
    constraint = PlaneConstraint("c2", "wall", relation=PlaneRelation.PARALLEL)
    result = evaluate_constraint((0, 0, 0), constraint, [plane], direction=None)
    assert not result.satisfied
    assert result.residual_m == float("inf")
    assert "requires a target direction" in result.message
