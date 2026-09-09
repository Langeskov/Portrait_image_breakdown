from reverse_engineering.reference_line_calibration import (
    ReferenceLineConstraint,
    ReferenceLineEvidence,
)


def test_horizontal_reference_line_exposes_roll_correction():
    evidence = ReferenceLineEvidence(
        (100.0, 100.0),
        (260.0, 114.0),
        ReferenceLineConstraint.HORIZONTAL,
    )

    assert evidence.observed_angle_deg > 0.0
    assert evidence.target_angle_deg == 0.0
    assert evidence.correction_deg < 0.0
    assert evidence.supports_roll


def test_free_reference_line_does_not_offer_roll_correction():
    evidence = ReferenceLineEvidence(
        (100.0, 100.0),
        (260.0, 114.0),
        ReferenceLineConstraint.FREE,
    )

    assert evidence.target_angle_deg is None
    assert evidence.correction_deg is None
    assert not evidence.supports_roll
