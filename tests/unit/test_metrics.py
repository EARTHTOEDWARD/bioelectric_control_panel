from bcp.core.metrics import summary_metrics


def test_summary_metrics():
    sig = [1.0, 2.0, 3.0]
    m = summary_metrics(sig)
    assert set(m.keys()) == {"mean", "std", "min", "max", "rms"}
    assert m["min"] == 1.0 and m["max"] == 3.0

