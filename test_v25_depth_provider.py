"""Depth provider backend and normalization tests."""
import numpy as np

from reverse_engineering.depth_provider import MonocularDepthProvider


def test_default_depth_provider_returns_normalized_map():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:60, 30:70] = 180
    p = MonocularDepthProvider()
    d = p.estimate_relative_depth(image)
    assert d.shape == (80, 100)
    assert np.isfinite(d).all()
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0
    assert p.last_backend == "structure_prior"


def test_local_model_adapter_is_used_when_valid():
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    expected = np.linspace(0, 1, 32 * 40, dtype=np.float32).reshape(32, 40)
    p = MonocularDepthProvider(model_fn=lambda _: expected)
    d = p.estimate_depth(image)
    assert d.shape == expected.shape
    assert p.last_backend == "local_model"
    assert p.last_confidence > 0.7


def test_failed_local_model_falls_back_without_breaking_pipeline():
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    p = MonocularDepthProvider(model_fn=lambda _: (_ for _ in ()).throw(RuntimeError("offline model unavailable")))
    d = p.estimate_depth(image)
    assert d.shape == (32, 40)
    assert p.last_backend == "structure_prior"
