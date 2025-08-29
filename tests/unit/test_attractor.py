import pytest

from bcp.core.attractor import delay_embedding


def test_delay_embedding_basic():
    sig = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    out = delay_embedding(sig, delay_ms=10, sample_rate_hz=1000.0, embed_dim=3)
    assert set(out.keys()) >= {"x", "y", "z"}
    # Length should be original minus window
    assert len(out["x"]) > 0


def test_delay_embedding_invalid():
    with pytest.raises(ValueError):
        delay_embedding([], 10, 1000.0)

