import numpy as np

from bcp.core.quality import quality_gate
from bcp.core.guardrails import guardrail_action_cardiac


def test_quality_gate_blocks_on_ci_overlap_and_stationarity():
    ok, msgs = quality_gate(lam1_ci=(-0.1, 0.2), D2_ci=(2.0, 2.5), stationarity_ok=False)
    assert ok is False
    assert any("Stationarity" in m or "stationarity" in m for m in msgs)
    assert any("λ1" in m or "CI" in m for m in msgs)


def test_guardrail_blocks_shock_when_not_strongly_chaotic_and_clamps_energy():
    action = {"type": "shock", "energy_J": 500.0}
    out = guardrail_action_cardiac(action, metrics={"lambda1": 0.1, "D2": 2.0}, quality_ok=True)
    assert out["allowed"] is False
    assert out["action"]["energy_J"] <= 200.0


def test_guardrail_clamps_pacing_params():
    action = {"type": "pace", "rate_bpm": 10.0, "amp_mA": 100.0, "pulse_width_ms": 100.0}
    out = guardrail_action_cardiac(action, metrics={"lambda1": 0.0, "D2": 1.0}, quality_ok=True)
    assert out["allowed"] is True
    assert 60.0 <= out["action"]["rate_bpm"] <= 300.0
    assert 0.1 <= out["action"]["amp_mA"] <= 25.0
    assert 0.1 <= out["action"]["pulse_width_ms"] <= 20.0

