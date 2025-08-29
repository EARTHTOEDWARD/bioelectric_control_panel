from __future__ import annotations

from typing import Any, Dict


CARDIAC_LIMITS = {
    "shock_energy_J": (1.0, 200.0),
    "pace_rate_bpm": (60.0, 300.0),
    "pace_amp_mA": (0.1, 25.0),
    "pulse_width_ms": (0.1, 20.0),
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def guardrail_action_cardiac(
    action: Dict[str, Any], metrics: Dict[str, float], quality_ok: bool
) -> Dict[str, Any]:
    """Clamp parameters and gate actions based on chaos metrics and quality.

    action: {"type": "shock"|"pace", ...params}
    metrics: {"lambda1": float, "D2": float}
    Returns: {allowed: bool, action: dict, messages: list[str]}
    """
    out: Dict[str, Any] = {"allowed": True, "action": dict(action), "messages": []}
    if not quality_ok:
        out["allowed"] = False
        out["messages"].append("Blocked: analysis quality insufficient (stationarity/CI).")

    lam1 = float(metrics.get("lambda1", 0.0))
    D2 = float(metrics.get("D2", 0.0))

    typ = str(action.get("type", "")).lower()
    if typ == "shock":
        if lam1 < 0.3 or D2 < 3.0:
            out["allowed"] = False
            out["messages"].append(
                "Blocked: shock reserved for strongly chaotic state (λ1≥0.3 & D2≥3)."
            )
        E = float(action.get("energy_J", 0.0))
        lo, hi = CARDIAC_LIMITS["shock_energy_J"]
        adj = _clamp(E, lo, hi)
        out["action"]["energy_J"] = float(adj)
        if adj != E:
            out["messages"].append(f"Energy clamped to safe range [{lo},{hi}] J.")
    elif typ == "pace":
        r = float(action.get("rate_bpm", 0.0))
        a = float(action.get("amp_mA", 0.0))
        w = float(action.get("pulse_width_ms", 0.0))
        rlo, rhi = CARDIAC_LIMITS["pace_rate_bpm"]
        alo, ahi = CARDIAC_LIMITS["pace_amp_mA"]
        wlo, whi = CARDIAC_LIMITS["pulse_width_ms"]
        rr = _clamp(r, rlo, rhi)
        aa = _clamp(a, alo, ahi)
        ww = _clamp(w, wlo, whi)
        out["action"]["rate_bpm"] = float(rr)
        out["action"]["amp_mA"] = float(aa)
        out["action"]["pulse_width_ms"] = float(ww)
        if (rr != r) or (aa != a) or (ww != w):
            out["messages"].append("Pacing parameters were clamped to safety limits.")
        if lam1 >= 0.3 and D2 >= 3.0:
            out["messages"].append(
                "Note: strongly chaotic; pacing may be insufficient → consider shock if pacing fails."
            )
    else:
        out["allowed"] = False
        out["messages"].append("Unknown action type.")
    return out

