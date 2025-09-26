import csv
from collections import defaultdict
from typing import Dict, Tuple, List, Any
import numpy as np

SeriesMap = Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, float]]
MetaMap = Dict[str, Any]


def _infer_fs(t: np.ndarray) -> float:
    if len(t) < 3:
        return float('nan')
    diffs = np.diff(t)
    med = float(np.median(diffs))
    if med <= 0:
        return float('nan')
    return 1.0 / med


def load_long_csv(path: str) -> Tuple[SeriesMap, MetaMap]:
    """
    Read long-format CSV into a mapping {(condition, channel, cell_id) -> (t, x, fs)}
    Returns (series_map, meta)
    """
    groups: Dict[Tuple[str, str, str], List[Tuple[float, float, float]]] = defaultdict(list)
    meta: MetaMap = {"source_ids": set(), "channels": set(), "conditions": set(), "cells": set()}
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        needed = {'channel', 'condition', 'cell_id', 't_s', 'value'}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")
        for row in reader:
            chan = row['channel'].strip()
            cond = row['condition'].strip()
            cell = row['cell_id'].strip()
            try:
                t = float(row['t_s'])
                v = float(row['value'])
            except Exception:
                # skip non-numeric
                continue
            fs = row.get('fs_Hz', '')
            try:
                fs_val = float(fs) if fs != '' else float('nan')
            except Exception:
                fs_val = float('nan')
            groups[(cond, chan, cell)].append((t, v, fs_val))
            # Collect some meta
            sid = row.get('source_id')
            if sid:
                meta['source_ids'].add(sid)
            meta['channels'].add(chan)
            meta['conditions'].add(cond)
            meta['cells'].add(cell)
    series: SeriesMap = {}
    for key, triplets in groups.items():
        triplets.sort(key=lambda x: x[0])
        t = np.array([a[0] for a in triplets], dtype=float)
        x = np.array([a[1] for a in triplets], dtype=float)
        fss = np.array([a[2] for a in triplets], dtype=float)
        if np.isfinite(fss).any():
            fs = float(np.nanmedian(fss[np.isfinite(fss)]))
        else:
            fs = _infer_fs(t)
        series[key] = (t, x, fs)
    # finalize meta
    for k in list(meta.keys()):
        if isinstance(meta[k], set):
            meta[k] = sorted(list(meta[k]))
    return series, meta


def load_event_times(path: str) -> Dict[str, float]:
    import json
    with open(path, 'r') as f:
        m = json.load(f)
    # normalize keys to lower for robust matching
    return {str(k): float(v) for k, v in m.items()}
