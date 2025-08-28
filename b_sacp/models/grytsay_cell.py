# Grytsay 10D metabolic model (Arthrobacter globiformis)
# Core equations (1)-(10) and defaults extracted from the paper.
# Strange attractors analyzed via invariant measures (Grytsay 2018, 2025).
from dataclasses import dataclass
import numpy as np
from typing import Dict
from scipy.integrate import solve_ivp

def V(x): return x / (1.0 + x)
def V1(psi): return 1.0 / (1.0 + psi*psi)

@dataclass
class GryParams:
    # canonical defaults collated from the papers
    l: float = 0.2; l1: float = 0.2; k1: float = 0.2
    l2: float = 0.27; l10: float = 0.27; l5: float = 0.6
    l4: float = 0.5; l6: float = 0.5; l7: float = 1.2; l8: float = 2.4
    k2: float = 1.5; E10: float = 3.0; beta1: float = 2.0; N1: float = 0.03
    m: float = 2.5; alpha: float = 0.033; a1: float = 0.007; alpha1: float = 0.0068
    E20: float = 1.2; beta: float = 0.01; beta2: float = 1.0; N2: float = 0.03
    alpha2: float = 0.02; G0: float = 0.019; N3: float = 2.0; gamma2: float = 0.2
    alpha5: float = 0.014; alpha3: float = 0.001; alpha4: float = 0.001
    alpha6: float = 0.001; alpha7: float = 0.001; O20: float = 0.015
    N5: float = 0.1; N0: float = 0.003; N4: float = 1.0; K10: float = 0.7

def gry_rhs(t, s, p: GryParams):
    # State: [G, P, B, E1, e1, Q, O2, E2, N, psi]
    G, P, B, E1, e1, Q, O2, E2, N, psi = s
    dG  = p.G0/(p.N3 + G + p.gamma2*psi) - p.l1*V(E1)*V(G) - p.alpha3*G
    dP  = p.l1*V(E1)*V(G) - p.l2*V(E2)*V(N)*V(P) - p.alpha4*P
    dB  = p.l2*V(E2)*V(N)*V(P) - p.k1*V(psi)*V(B) - p.alpha5*B
    dE1 = p.E10*(G*G)/(p.beta1 + G*G)*(1.0 - (P + p.m*N)/(p.N1 + P + p.m*N)) \
          - p.l1*V(E1)*V(G) + p.l4*V(e1)*V(Q) - p.a1*E1
    de1 = -p.l4*V(e1)*V(Q) + p.l1*V(E1)*V(G) - p.alpha1*e1
    dQ  = 6.0*p.l*V(2.0 - Q)*V(O2)*V1(psi) - p.l6*V(e1)*V(Q) - p.l7*V(Q)*V(N)
    dO2 = p.O20/(p.N5 + O2) - p.l*V(2.0 - Q)*V(O2)*V1(psi) - p.alpha7*O2
    dE2 = p.E20*(P*P)/(p.beta2 + P*P) * (N/(p.beta + N)) * (1.0 - B/(p.N2 + B)) \
          - p.l10*V(E2)*V(N)*V(P) - p.alpha2*E2
    dN  = -p.l2*V(E2)*V(N)*V(P) - p.l7*V(Q)*V(N) + p.k2*V(B)*psi/(p.K10 + psi) \
          + p.N0/(p.N4 + N) - p.alpha6*N
    dpsi= p.l5*V(E1)*V(G) + p.l8*V(N)*V(Q) - p.alpha*psi
    return np.array([dG,dP,dB,dE1,de1,dQ,dO2,dE2,dN,dpsi])

def integrate_gry(p: GryParams,
                  s0=None,
                  t_span=(0.0, 2.0e6),
                  dt=10.0,
                  transient=1.0e6,
                  rtol=1e-9, atol=1e-12) -> Dict[str, np.ndarray]:
    if s0 is None:
        # mild nonzero start
        s0 = np.array([0.02,0.02,0.02, 0.5,0.1, 0.5,0.02, 0.4,0.05, 0.1])
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lambda t, s: gry_rhs(t, s, p),
                    t_span, s0, t_eval=t_eval, rtol=rtol, atol=atol, method="RK45")
    t = sol.t; X = sol.y.T
    keep = t >= transient
    return {"t": t[keep], "X": X[keep]}

