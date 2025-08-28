SACP–Bioelectric (Grytsay/Karatetskaia) Architecture

Purpose. Extend the Strange Attractor Control Panels (SACPs) with a Bioelectric Control Panel (B‑SACP) that (i) respects Grytsay’s requirement to analyze chaotic modes via invariant measures (not kinetic curves) and (ii) exploits Karatetskaia’s organizing bifurcations (degenerate Bogdanov–Takens, zero‑Hopf) and Shilnikov chaos to map controllable transitions between tumor/immune/normal‑cell regimes and bioelectric states.

Conceptual alignment

Why invariant measures? Grytsay shows chaotic (“strange attractor”) modes cannot be reliably compared using kinetic curves because of exponential trajectory run‑off and hypersensitivity to initial conditions; instead, estimation of the invariant measure gives the probability distribution of the trajectory over phase‑space regions, and its time‑convergence serves as an adaptation indicator. We make this a first‑class primitive (“Invariant Measure Engine”).

Where do the chaotic regimes come from? In the Itik–Banks cancer model variant studied by Karatetskaia, spiral/Shilnikov attractors organize the richest dynamics; key regimes are born near degenerate BT and zero‑Hopf codimension‑two points, with chaotic regions and routes (period‑doubling cascades to Shilnikov chaos) laid out across (a21, a31) and related planes (see Lyapunov diagram and curves in Fig. 1 / pp. 5–6; zero‑Hopf unfolding in Fig. 3 / p. 8; Shilnikov scenario in Fig. 8 / p. 13). We expose these as “Bifurcation Maps” and “Shilnikov Watchpoints” in the panel.

Bioelectric data and models. The BCP ingests ENG/optical mapping/EEG‑MEG (NWB/OME‑Zarr), computes LLE, D₂, MSE, RQA, and supports reservoir compression. We route those features into the same invariant‑measure and attractor‑state analytics to compare biological recordings with model‑space attractors.

System blocks

Data I/O & Curation (BCP)

Ingest NWB/OME‑Zarr datasets; standardize channels and metadata. (Re‑use data_interface.py hooks.)

Model Suite

IB‑3D Cancer (Itik–Banks form used in Karatetskaia):

    ẋ1 = x1(1 − x1) − a12 x1 x2 − a13 x1 x3,
    ẋ2 = r2 x2(1 − x2) − a21 x1 x2,
    ẋ3 = (r3 x1 x3)/(x1 + k3) − a31 x1 x3 − d3 x3

Defaults follow (1.2): a12=1, a13=2.5, r2=0.6, r3=6, d3=0.5, k3=1, while sweeping a21, a31 or r2.

Grytsay–Cell 10D metabolic model (Eqs. (1)–(10) with V(X)=X/(1+X), V(1)(ψ)=1/(1+ψ²)), including α (membrane‑potential dissipation) as the bifurcation knob for torus/period‑doubling/intermittency to chaos.

Attractor Analysis Core

Invariant Measure Engine (IME)

Build Poincaré sections or full‑space histograms; estimate invariant density; monitor convergence vs. sample length (Krylov–Bogolyubov existence assumptions satisfied per Grytsay; convergence ∝ 1/t).

Lyapunov & Bifurcation Engine (LBE)

LLE (Wolf/Rosenstein), Jacobian eigenanalysis at equilibria, heuristic flags for zero‑Hopf (1 real zero + ±iω) and BT (two zeros), “distance‑to‑O₁” diagnostic (à la Karatetskaia’s distance diagram, Fig. 5).

Recurrence & Entropy

RQA (RR, DET, LAM), MSE, correlation dimension D₂.

Perturbation & Control

Parameter nudges (e.g., a31 as immune inactivation by tumor cells), weak periodic/impulse inputs to test attractor‑basin shifts and spiral attractor capture—bridging to electroceutical hypotheses.

UI: SACP Panels

Live attractor orbit + Poincaré cloud, invariant measure heatmap, Lyapunov bar, bifurcation locator, Shilnikov/torus flags, and BCP tabs for real recordings.

Files & Packages

b_sacp/ (below) implements models, invariant measure, sweeps, and BCP adapters.

Integrates with BCP pipelines (NWB/OME‑Zarr; ESN features).

Acceptance tests (minimal)

Kar‑line sweep: a21=1.6, a31↓ from 3.0 → ~1.86 to reproduce focus → Hopf → period‑doubling cascade → Rössler‑like chaos → Shilnikov (Fig. 8) with LLE>0 and IME convergence.

Gry‑α sweep: replicate torus/period‑doubling/intermittency bands around α≈0.03217–0.03255 and show invariant measure convergence and density peaks in mixing funnels.

roadmap.md
Milestones

M1 — Core analytics online (this drop)

IB‑3D & Grytsay models; invariant measure; LLE; Poincaré; Jacobian eigen probes; single/bi‑param sweeps.

M2 — BCP wiring

NWB/OME‑Zarr ingestion + per‑channel embedding → IME/RQA/LLE; align feature schema with data_interface.py.

M3 — Bifurcation map UX

Overlays: estimated SN/AH/PD/NS curves via continuation heuristics; Shilnikov watch lanes (HomO₁, HomO₁² proxies).

M4 — Control sandbox

Perturbation protocols; basin‑shift quantification; “nudge‑to‑health” demos for cardiac/EEG motifs.

M5 — Docs & tests

Repro notebooks; datasets manifest; unit tests; CI.

Sprints (2–3 weeks each)

S1: Implement + validate Kar‑line and Gry‑α sweeps; CSV/Parquet outputs + plots.

S2: NWB hook‑up, RQA batch for ENG/EEG; ESN compression benchmark (vs PCA/UMAP).

S3: UX polish, parameter planes, slice explorers; alert badges for zero‑Hopf/BT heuristics.

S4: Control & intervention; minimal closed‑loop demo (sim → metric threshold → parameter nudge).

S5: Docs, examples, test coverage.

b_sacp package (code)

How to use: copy these files into your repo under b_sacp/. They are self‑contained Python modules relying on numpy, scipy, and pandas (optional for CSV). The CLI examples at the bottom run the Kar‑line sweep and Gry‑α sweep and write results to CSV.

