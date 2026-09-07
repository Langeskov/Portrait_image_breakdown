"""v2.5 evidence-state presentation tests."""

from reverse_engineering.data_types import EstimatedValue, ConfidenceLevel


def test_estimated_value_exposes_explicit_evidence_state():
    observed = EstimatedValue(50.0, unit="mm", confidence=1.0, is_observed=True)
    estimated = EstimatedValue(50.0, unit="mm", confidence=0.72)
    unknown = EstimatedValue(0.0, unit="mm", confidence=0.0)

    assert observed.evidence_state == "observed"
    assert estimated.evidence_state == "estimated"
    assert unknown.evidence_state == "unknown"
    assert observed.to_dict()["evidence_state"] == "observed"
    assert estimated.to_dict()["evidence_state"] == "estimated"
    assert unknown.to_dict()["confidence_level"] == ConfidenceLevel.UNKNOWN.value
