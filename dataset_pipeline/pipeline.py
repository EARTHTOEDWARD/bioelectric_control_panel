import csv
import json
import os
from collections import defaultdict
from typing import Dict, Tuple, List, Any
import numpy as np

from .io_load import SeriesMap
from .signal_ops import (
    welch_psd, magnitude_squared_coherence, robust_beta_loglog, band_mask,
    compute_step_metrics, sliding_windows, higuchi_fd, katz_fd, rqa_metrics,
    phase_slope_index, lagged_xcorr_peak,
)
from .stats_ops import mann_whitney_u, cliffs_delta, bootstrap_ci_diff_median, fdr_bh


Pair = Tuple[str, str]


def _write_csv(path: str, rows: List[Dict[str, Any]], header: List[str]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


essential_headers = {
    'spectral_beta_per_cell': [
        'condition','channel','cell_id','beta','r2','fmin','fmax','fs_Hz','n','source_id'
    ],
    'spectral_beta_group': [
        'channel','condition','median_beta','delta_vs_baseline','mw_p','cliffs_delta','ci_lo','ci_hi','fdr_q'
    ],
    'coherence_pairs': [
        'condition','pair','cell_id','coh_median','fmin','fmax','fs_Hz','n'
    ],
    'coherence_group': [
        'pair','condition','median_coh','delta_vs_baseline','mw_p','cliffs_delta','ci_lo','ci_hi','fdr_q'
    ],
    'delta_coherence_pre_post': [
        'condition','pair','cell_id','pre_coh','post_coh','delta_coh','fs_Hz','fmin','fmax'
    ],
    'step_metrics': [
        'condition','channel','cell_id','amp','tau63_s','pre_mean','post_mean'
    ],
    'windowed_metrics': [
        'condition','channel','cell_id','win_start_s','win_end_s','beta','r2'
    ],
    'features_fractal_rqa': [
        'condition','channel','cell_id','higuchi_fd','katz_fd','rqa_det','rqa_lam','rqa_tt'
    ],
    'directed_metrics': [
        'condition','pair','cell_id','psi','xcorr_lag_samples','xcorr_peak'
    ]
}


def run_pipeline(series: SeriesMap, meta: Dict[str, Any], events: Dict[str, float], cfg: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # QC sidecar
    qc = {
        'beta_band_Hz': cfg['spectral']['beta_band_Hz'],
        'coh_band_Hz': cfg['spectral']['coh_band_Hz'],
        'welch': cfg['spectral']['welch'],
        'step': cfg['step'],
        'windowed': cfg['windowed'],
        'rqa': cfg['rqa'],
        'bootstrap': cfg['bootstrap'],
        'notes': 'All coherence recomputed via Welch; no post-hoc rescaling.'
    }
    with open(os.path.join(out_dir, 'qc_sidecar.json'), 'w') as f:
        json.dump(qc, f, indent=2)

    beta_band = tuple(cfg['spectral']['beta_band_Hz'])
    coh_band = tuple(cfg['spectral']['coh_band_Hz'])
    welch_cfg = cfg['spectral']['welch']

    # 1) Per-cell β
    beta_rows: List[Dict[str, Any]] = []
    for (cond, chan, cell), (t, x, fs) in series.items():
        if not np.isfinite(fs) or fs <= 0:
            fs = cfg['spectral'].get('fs_default_Hz', 0.2)
        try:
            f, Pxx = welch_psd(x, fs, welch_cfg['segment_len_s'], welch_cfg['overlap'], welch_cfg['detrend'])
            beta, r2 = robust_beta_loglog(f, Pxx, beta_band)
        except Exception:
            beta, r2 = float('nan'), float('nan')
            f = np.array([0.0, 1.0]); Pxx = np.array([np.nan, np.nan])
        beta_rows.append({
            'condition': cond,
            'channel': chan,
            'cell_id': cell,
            'beta': beta,
            'r2': r2,
            'fmin': beta_band[0],
            'fmax': beta_band[1],
            'fs_Hz': fs,
            'n': t.size,
            'source_id': ''
        })
    _write_csv(os.path.join(out_dir, 'spectral_beta_per_cell.csv'), beta_rows, essential_headers['spectral_beta_per_cell'])

    # 2) Group β stats and Δ vs baseline
    # Collect by channel×condition
    by_cc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in beta_rows:
        if np.isfinite(r['beta']):
            by_cc[(r['channel'], r['condition'])].append(r['beta'])
    channels = sorted({ch for (ch, _) in by_cc.keys()})
    conditions = sorted({c for (_, c) in by_cc.keys()})

    group_rows: List[Dict[str, Any]] = []
    pvals_for_fdr: List[float] = []
    idx_to_row = []
    for ch in channels:
        base = np.array(by_cc.get((ch, 'baseline'), []), dtype=float)
        for cond in conditions:
            vals = np.array(by_cc.get((ch, cond), []), dtype=float)
            if vals.size == 0:
                continue
            med = float(np.median(vals))
            delta = float(med - np.median(base)) if base.size else float('nan')
            U, p = mann_whitney_u(base, vals) if base.size else (float('nan'), float('nan'))
            delta_eff = cliffs_delta(vals, base) if base.size else float('nan')
            ci_lo, ci_hi = bootstrap_ci_diff_median(base, vals, iters=cfg['bootstrap']['iterations'], method=cfg['bootstrap']['method']) if base.size else (float('nan'), float('nan'))
            row = {
                'channel': ch,
                'condition': cond,
                'median_beta': med,
                'delta_vs_baseline': delta,
                'mw_p': p,
                'cliffs_delta': delta_eff,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
                'fdr_q': float('nan')
            }
            idx_to_row.append(len(group_rows))
            pvals_for_fdr.append(p if np.isfinite(p) else 1.0)
            group_rows.append(row)
    if pvals_for_fdr:
        qvals = fdr_bh(pvals_for_fdr)
        for i, q in zip(idx_to_row, qvals):
            group_rows[i]['fdr_q'] = q
    _write_csv(os.path.join(out_dir, 'spectral_beta_group.csv'), group_rows, essential_headers['spectral_beta_group'])

    # 3) Coherence per pair per cell (matched channels within same (cond, cell))
    pairs = [('ATP/ADP','dpsi'), ('ATP/ADP','NAD(P)H'), ('dpsi','NAD(P)H')]
    coh_rows: List[Dict[str, Any]] = []
    # Build quick index
    idx: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, float]] = series
    for cond in meta.get('conditions', []):
        # gather cell IDs that have all required channels
        chan_to_cells: Dict[str, set] = defaultdict(set)
        for (c, ch, cell) in idx.keys():
            if c == cond:
                chan_to_cells[ch].add(cell)
        for (a, b) in pairs:
            common_cells = sorted(list(chan_to_cells.get(a, set()) & chan_to_cells.get(b, set())))
            for cell in common_cells:
                tA, xA, fsA = idx[(cond, a, cell)]
                tB, xB, fsB = idx[(cond, b, cell)]
                fs = fsA if np.isfinite(fsA) and fsA>0 else (fsB if np.isfinite(fsB) and fsB>0 else cfg['spectral'].get('fs_default_Hz',0.2))
                n = int(min(tA.size, tB.size))
                xA2 = xA[:n]; xB2 = xB[:n]
                f, coh_f, _ = magnitude_squared_coherence(xA2, xB2, fs, welch_cfg['segment_len_s'], welch_cfg['overlap'], welch_cfg['detrend'])
                m = band_mask(f, coh_band)
                coh_med = float(np.nanmedian(coh_f[m])) if m.sum() else float('nan')
                coh_rows.append({
                    'condition': cond,
                    'pair': f"{a}<->{b}",
                    'cell_id': cell,
                    'coh_median': coh_med,
                    'fmin': coh_band[0],
                    'fmax': coh_band[1],
                    'fs_Hz': fs,
                    'n': n
                })
    _write_csv(os.path.join(out_dir, 'coherence_pairs.csv'), coh_rows, essential_headers['coherence_pairs'])

    # 4) Coherence group stats and Δ vs baseline
    coh_by_pc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in coh_rows:
        coh_by_pc[(r['pair'], r['condition'])].append(r['coh_median'])
    coh_group_rows: List[Dict[str, Any]] = []
    pvals = []
    idx_rows = []
    for pair in sorted({p for (p, _) in coh_by_pc.keys()}):
        base = np.array(coh_by_pc.get((pair, 'baseline'), []), dtype=float)
        for cond in sorted({c for (_, c) in coh_by_pc.keys()}):
            vals = np.array(coh_by_pc.get((pair, cond), []), dtype=float)
            if vals.size == 0:
                continue
            med = float(np.median(vals))
            delta = float(med - np.median(base)) if base.size else float('nan')
            U, p = mann_whitney_u(base, vals) if base.size else (float('nan'), float('nan'))
            delta_eff = cliffs_delta(vals, base) if base.size else float('nan')
            ci_lo, ci_hi = bootstrap_ci_diff_median(base, vals, iters=cfg['bootstrap']['iterations'], method=cfg['bootstrap']['method']) if base.size else (float('nan'), float('nan'))
            row = {
                'pair': pair,
                'condition': cond,
                'median_coh': med,
                'delta_vs_baseline': delta,
                'mw_p': p,
                'cliffs_delta': delta_eff,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
                'fdr_q': float('nan')
            }
            idx_rows.append(len(coh_group_rows))
            pvals.append(p if np.isfinite(p) else 1.0)
            coh_group_rows.append(row)
    if pvals:
        qvals = fdr_bh(pvals)
        for i, q in zip(idx_rows, qvals):
            coh_group_rows[i]['fdr_q'] = q
    _write_csv(os.path.join(out_dir, 'coherence_group.csv'), coh_group_rows, essential_headers['coherence_group'])

    # 5) Δ-coherence pre→post recomputation
    delta_coh_rows: List[Dict[str, Any]] = []
    for cond in meta.get('conditions', []):
        if cond == 'baseline':
            continue
        event_t = float(events.get(cond, np.nan))
        if not np.isfinite(event_t):
            continue
        # cells with pairs
        chan_to_cells: Dict[str, set] = defaultdict(set)
        for (c, ch, cell) in idx.keys():
            if c == cond:
                chan_to_cells[ch].add(cell)
        for (a, b) in pairs:
            common = sorted(list(chan_to_cells.get(a, set()) & chan_to_cells.get(b, set())))
            for cell in common:
                tA, xA, fsA = idx[(cond, a, cell)]
                tB, xB, fsB = idx[(cond, b, cell)]
                fs = fsA if np.isfinite(fsA) and fsA>0 else (fsB if np.isfinite(fsB) and fsB>0 else cfg['spectral'].get('fs_default_Hz',0.2))
                # pre window
                pre_amp, _, _, _ = compute_step_metrics(tA, xA, event_t, **{
                    'pre_win': cfg['step']['pre_win_s'], 'pre_gap': cfg['step']['pre_gap_s'],
                    'post_start': cfg['step']['post_start_s'], 'post_win': cfg['step']['post_win_s'], 'search_win': cfg['step']['tau63_search_s']
                })
                # Build masks consistently for A and B
                pre_start = event_t - cfg['step']['pre_gap_s'] - cfg['step']['pre_win_s']
                pre_end = event_t - cfg['step']['pre_gap_s']
                pre_mask_A = (tA >= pre_start) & (tA < pre_end)
                pre_mask_B = (tB >= pre_start) & (tB < pre_end)
                if pre_mask_A.sum() < 8 or pre_mask_B.sum() < 8:
                    # fallback to compact windows
                    idx_pre_A = np.where(tA < (event_t - cfg['step']['pre_gap_s']))[0]
                    idx_pre_B = np.where(tB < (event_t - cfg['step']['pre_gap_s']))[0]
                    if idx_pre_A.size >= 8 and idx_pre_B.size >= 8:
                        k = min(idx_pre_A.size, idx_pre_B.size, max(8, int(0.2*min(tA.size,tB.size))))
                        pre_mask_A = np.zeros_like(tA, dtype=bool); pre_mask_A[idx_pre_A[-k:]] = True
                        pre_mask_B = np.zeros_like(tB, dtype=bool); pre_mask_B[idx_pre_B[-k:]] = True
                    else:
                        continue
                post_start_abs = event_t + cfg['step']['post_start_s']
                post_end_abs = post_start_abs + cfg['step']['post_win_s']
                post_mask_A = (tA >= post_start_abs) & (tA < post_end_abs)
                post_mask_B = (tB >= post_start_abs) & (tB < post_end_abs)
                if post_mask_A.sum() < 8 or post_mask_B.sum() < 8:
                    idx_post_A = np.where(tA >= event_t)[0]
                    idx_post_B = np.where(tB >= event_t)[0]
                    if idx_post_A.size >= 8 and idx_post_B.size >= 8:
                        k = min(idx_post_A.size, idx_post_B.size, max(8, int(0.2*min(tA.size,tB.size))))
                        post_mask_A = np.zeros_like(tA, dtype=bool); post_mask_A[idx_post_A[:k]] = True
                        post_mask_B = np.zeros_like(tB, dtype=bool); post_mask_B[idx_post_B[:k]] = True
                    else:
                        continue
                xA_pre = xA[pre_mask_A]; xB_pre = xB[pre_mask_B]
                xA_post = xA[post_mask_A]; xB_post = xB[post_mask_B]
                # equalize lengths
                npre = min(xA_pre.size, xB_pre.size); npost = min(xA_post.size, xB_post.size)
                if npre < 8 or npost < 8:
                    continue
                f_pre, coh_pre, _ = magnitude_squared_coherence(xA_pre[:npre], xB_pre[:npre], fs, welch_cfg['segment_len_s'], welch_cfg['overlap'], welch_cfg['detrend'])
                f_post, coh_post, _ = magnitude_squared_coherence(xA_post[:npost], xB_post[:npost], fs, welch_cfg['segment_len_s'], welch_cfg['overlap'], welch_cfg['detrend'])
                bm_pre = band_mask(f_pre, coh_band)
                bm_post = band_mask(f_post, coh_band)
                pre_val = float(np.nanmedian(coh_pre[bm_pre])) if bm_pre.sum() else float('nan')
                post_val = float(np.nanmedian(coh_post[bm_post])) if bm_post.sum() else float('nan')
                delta = post_val - pre_val if (np.isfinite(pre_val) and np.isfinite(post_val)) else float('nan')
                delta_coh_rows.append({
                    'condition': cond,
                    'pair': f"{a}<->{b}",
                    'cell_id': cell,
                    'pre_coh': pre_val,
                    'post_coh': post_val,
                    'delta_coh': delta,
                    'fs_Hz': fs,
                    'fmin': coh_band[0],
                    'fmax': coh_band[1]
                })
    _write_csv(os.path.join(out_dir, 'delta_coherence_pre_post.csv'), delta_coh_rows, essential_headers['delta_coherence_pre_post'])

    # 6) Step metrics (amplitude, tau63) per channel
    step_rows: List[Dict[str, Any]] = []
    for (cond, chan, cell), (t, x, fs) in series.items():
        if cond == 'baseline':
            continue
        event_t = float(events.get(cond, np.nan))
        if not np.isfinite(event_t):
            continue
        amp, tau, pre_mean, post_mean = compute_step_metrics(t, x, event_t, cfg['step']['pre_win_s'], cfg['step']['pre_gap_s'], cfg['step']['post_start_s'], cfg['step']['post_win_s'], cfg['step']['tau63_search_s'])
        step_rows.append({
            'condition': cond,
            'channel': chan,
            'cell_id': cell,
            'amp': amp,
            'tau63_s': tau,
            'pre_mean': pre_mean,
            'post_mean': post_mean
        })
    _write_csv(os.path.join(out_dir, 'step_metrics.csv'), step_rows, essential_headers['step_metrics'])

    # 7) Windowed β per series
    win_rows: List[Dict[str, Any]] = []
    for (cond, chan, cell), (t, x, fs) in series.items():
        if not np.isfinite(fs) or fs <= 0:
            fs = cfg['spectral'].get('fs_default_Hz', 0.2)
        win_len = int(round(cfg['windowed']['win_len_s'] * fs))
        step = int(max(1, round(win_len * (1.0 - cfg['windowed']['overlap']))))
        idxs = sliding_windows(x.size, win_len, step)
        for i0, i1 in idxs:
            if i1 - i0 < 8:
                continue
            xi = x[i0:i1]
            ti = t[i0:i1]
            f, Pxx = welch_psd(xi, fs, welch_cfg['segment_len_s'], welch_cfg['overlap'], welch_cfg['detrend'])
            beta, r2 = robust_beta_loglog(f, Pxx, beta_band)
            win_rows.append({
                'condition': cond,
                'channel': chan,
                'cell_id': cell,
                'win_start_s': float(ti[0]),
                'win_end_s': float(ti[-1]),
                'beta': beta,
                'r2': r2
            })
    _write_csv(os.path.join(out_dir, 'windowed_metrics.csv'), win_rows, essential_headers['windowed_metrics'])

    # 8) Fractal & RQA per series
    feat_rows: List[Dict[str, Any]] = []
    for (cond, chan, cell), (t, x, fs) in series.items():
        hfd = higuchi_fd(x)
        kfd = katz_fd(x)
        det, lam, tt = rqa_metrics(x, cfg['rqa']['embed_dim'], cfg['rqa']['delay'], cfg['rqa']['threshold_quantile'], cfg['rqa']['diag_min'], cfg['rqa']['vert_min'])
        feat_rows.append({
            'condition': cond,
            'channel': chan,
            'cell_id': cell,
            'higuchi_fd': hfd,
            'katz_fd': kfd,
            'rqa_det': det,
            'rqa_lam': lam,
            'rqa_tt': tt
        })
    _write_csv(os.path.join(out_dir, 'features_fractal_rqa.csv'), feat_rows, essential_headers['features_fractal_rqa'])

    # 9) Directed metrics (PSI and lagged xcorr) for pairs within same (cond, cell)
    dir_rows: List[Dict[str, Any]] = []
    for cond in meta.get('conditions', []):
        chan_to_cells: Dict[str, set] = defaultdict(set)
        for (c, ch, cell) in idx.keys():
            if c == cond:
                chan_to_cells[ch].add(cell)
        for (a, b) in pairs:
            for cell in sorted(list(chan_to_cells.get(a, set()) & chan_to_cells.get(b, set()))):
                tA, xA, fsA = idx[(cond, a, cell)]
                tB, xB, fsB = idx[(cond, b, cell)]
                fs = fsA if np.isfinite(fsA) and fsA>0 else (fsB if np.isfinite(fsB) and fsB>0 else cfg['spectral'].get('fs_default_Hz',0.2))
                n = min(xA.size, xB.size)
                xA2 = xA[:n]; xB2 = xB[:n]
                psi = phase_slope_index(xA2, xB2, fs, welch_cfg['segment_len_s'], welch_cfg['overlap'], welch_cfg['detrend'], beta_band)
                lag, pk = lagged_xcorr_peak(xA2, xB2, max_lag=int(round(60*fs)))
                dir_rows.append({
                    'condition': cond,
                    'pair': f"{a}->{b}",
                    'cell_id': cell,
                    'psi': psi,
                    'xcorr_lag_samples': lag,
                    'xcorr_peak': pk
                })
    _write_csv(os.path.join(out_dir, 'directed_metrics.csv'), dir_rows, essential_headers['directed_metrics'])

