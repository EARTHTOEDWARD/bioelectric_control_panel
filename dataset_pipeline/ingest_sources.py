import os
import glob
import json
from typing import Optional
import csv


def _try_imports():
    mods = {}
    errors = []
    for name in ("pandas", "imageio", "imageio.v3", "openpyxl", "numpy", "skimage", "skimage.draw"):
        try:
            module = __import__(name, fromlist=['*'])
            mods[name] = module
        except Exception as e:
            errors.append((name, str(e)))
    return mods, errors


def ingest_elife(elife_dir: str, out_rows: list) -> int:
    mods, _ = _try_imports()
    if "pandas" not in mods:
        return 0
    pd = mods["pandas"]
    import numpy as np

    def sheet_feature(sheet_name: str) -> str:
        s = sheet_name.lower()
        if "bound" in s:
            return "NAD(P)H_bound_frac"
        if "short" in s and "life" in s:
            return "NAD(P)H_tau_short"
        if "long" in s and "life" in s:
            return "NAD(P)H_tau_long"
        if "intensity" in s:
            return "NAD(P)H_intensity"
        if "nadhf" in s:
            return "NAD(P)H_free"
        if "nadhb" in s:
            return "NAD(P)H_bound"
        if "rox" in s:
            return "rox_rate"
        if "jox" in s:
            return "jox_rate"
        if "ocr" in s:
            return "ocr_rate"
        return sheet_name.strip().lower().replace(" ", "_")

    def sheet_unit(sheet_name: str) -> str:
        s = sheet_name.lower()
        if any(k in s for k in ("ratio", "frac")):
            return "fraction"
        if any(k in s for k in ("lifetime", "tau")) or " ns" in s:
            return "ns"
        if "u" in s and "per second" in s:
            return "uM/s"
        if "fmol" in s and "per" in s:
            return "fmol/s"
        if "per second" in s:
            return "1/s"
        return "a.u."

    baseline_labels = {"baseline", "control", "aksom", "gal", "glu", "akson", "untreated", "0mm k"}
    perturbation_labels = {
        "fccp", "oligomycin", "rotenone", "rot", "oxamate", "cccp", "gal+rot", "glu+rot",
        "aoa", "pyr", "oua", "lat", "k", "oxa"
    }

    files = sorted(glob.glob(os.path.join(elife_dir, "*.xlsx")))
    if not files:
        return 0
    count = 0
    for path in files:
        try:
            xl = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet_name=sheet)
            except Exception:
                continue
            if df.empty:
                continue
            feature = sheet_feature(sheet)
            unit = sheet_unit(sheet)
            cell_col = next((c for c in df.columns if isinstance(c, str) and any(k in c.lower() for k in ("cell", "id", "roi", "sample"))), None)
            value_columns = [c for c in df.columns if c != cell_col]
            for idx, row in df.iterrows():
                raw_id = None
                if cell_col:
                    raw_id = row.get(cell_col)
                    if isinstance(raw_id, str):
                        raw_id = raw_id.strip()
                if raw_id is None or (isinstance(raw_id, float) and not np.isfinite(raw_id)) or raw_id == "":
                    cell_id = f"row{idx}"
                else:
                    cell_id = str(raw_id)
                for col in value_columns:
                    condition = str(col).strip()
                    if not condition:
                        continue
                    try:
                        val = float(row[col])
                    except Exception:
                        continue
                    if not np.isfinite(val):
                        continue
                    cond_lower = condition.lower()
                    drug = condition if cond_lower in perturbation_labels else "none"
                    t_s = 0.0 if cond_lower in baseline_labels else 1200.0
                    out_rows.append({
                        "source_id": os.path.basename(path) + ":" + sheet,
                        "doi": "10.7554/eLife.73808",
                        "channel": feature,
                        "condition": condition,
                        "drug": drug,
                        "cell_id": cell_id,
                        "t_s": t_s,
                        "value": val,
                        "unit": unit,
                        "fs_Hz": "",
                        "imaging_modality": "FLIM",
                        "file_origin": path
                    })
                    count += 1
    return count


def ingest_plos_tmrm(plos_dir: str, roi_json: Optional[str], out_rows: list) -> int:
    mods, _ = _try_imports()
    if "imageio.v3" not in mods:
        return 0
    iio = mods["imageio.v3"]
    import numpy as np
    try:
        from skimage.draw import polygon
    except Exception:
        polygon = None
    mp4s = sorted(glob.glob(os.path.join(plos_dir, "*.mp4")))
    if not mp4s:
        return 0
    rois = {}
    if roi_json and os.path.exists(roi_json):
        try:
            with open(roi_json) as f:
                rois = json.load(f)
        except Exception:
            rois = {}
    count = 0
    for path in mp4s:
        try:
            frame0 = iio.imread(path, index=0)
        except Exception:
            continue
        H, W = frame0.shape[:2]
        roi_list = rois.get(os.path.basename(path), [])
        if not roi_list:
            # use full frame as a placeholder ROI; user should provide real ROIs
            roi_list = [{"roi_id":"frame_mean","poly_xy":[[0,0],[W,0],[W,H],[0,H]]}]
        # precompute masks
        masks = []
        for r in roi_list:
            if polygon is None:
                masks.append((r["roi_id"], None))
            else:
                poly = np.array(r["poly_xy"], dtype=float)
                rr, cc = polygon(poly[:,1], poly[:,0], (H, W))
                mask = np.zeros((H, W), bool); mask[rr, cc] = True
                masks.append((r["roi_id"], mask))
        # fps inference
        fs = 0.2  # default if metadata missing
        try:
            meta = iio.imopen(path, "r").properties()
            fs = float(meta.get("fps", fs))
        except Exception:
            pass
        t = 0.0
        dt = 1.0/float(fs)
        for frame in iio.imiter(path):
            if frame.ndim == 3:
                frame = frame[..., 0]
            frame = frame.astype("float32")
            for roi_id, mask in masks:
                if mask is None:
                    val = float(frame.mean())
                else:
                    val = float(frame[mask].mean())
                out_rows.append({
                    "source_id": os.path.basename(path),
                    "doi": "10.1371/journal.pone.0058059",
                    "channel": "dpsi_TMRM",
                    "condition": "baseline",
                    "drug": "none",
                    "cell_id": roi_id,
                    "t_s": t,
                    "value": val,
                    "unit": "a.u.",
                    "fs_Hz": fs,
                    "imaging_modality": "widefield_timelapse",
                    "file_origin": path
                })
                count += 1
            t += dt
    return count


def ingest_mosaic(mosaic_dir: str, out_rows: list) -> int:
    mods, _ = _try_imports()
    if "pandas" not in mods:
        return 0
    pd = mods["pandas"]
    files = sorted(glob.glob(os.path.join(mosaic_dir, "*.xlsx")))
    if not files:
        return 0
    count = 0
    for path in files:
        try:
            xl = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet_name=sheet)
            except Exception:
                continue
            # Heuristics: find time, value, condition columns
            cols = {c.lower(): c for c in df.columns}
            time_col = next((v for k, v in cols.items() if "time" in k), None)
            val_col = next((v for k, v in cols.items() if "perceval" in k or "ratio" in k), None)
            cond_col = next((v for k, v in cols.items() if "condition" in k or "drug" in k), None)
            cell_col = next((v for k, v in cols.items() if "cell" in k or "island" in k or "roi" in k), None)
            if not (time_col and val_col and cond_col and cell_col):
                continue
            for _, r in df.iterrows():
                try:
                    t = float(r[time_col])
                    v = float(r[val_col])
                except Exception:
                    continue
                cond = str(r[cond_col]).strip().lower()
                out_rows.append({
                    "source_id": os.path.basename(path)+":"+sheet,
                    "doi": "10.1038/ncommsXXXX",  # placeholder DOI tag
                    "channel": "ATP/ADP",
                    "condition": cond,
                    "drug": cond,
                    "cell_id": str(r[cell_col]),
                    "t_s": t,
                    "value": v,
                    "unit": "ratio",
                    "fs_Hz": "",
                    "imaging_modality": "widefield_timelapse",
                    "file_origin": path
                })
                count += 1
    return count


def ingest_to_long(elife_dir: str, plos_dir: str, mosaic_dir: str, roi_json: Optional[str], out_csv: str) -> int:
    rows = []
    n1 = ingest_elife(elife_dir, rows)
    n2 = ingest_plos_tmrm(plos_dir, roi_json, rows)
    n3 = ingest_mosaic(mosaic_dir, rows)
    if not rows:
        # write a header-only file with schema so users can see what to provide
        fieldnames = [
            "source_id","doi","channel","condition","drug","cell_id","t_s","value","unit","fs_Hz","imaging_modality","file_origin"
        ]
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        print("[ingest] No source files found. Place files under the provided directories or authorize fetching.")
        return 0
    # Normalize and write
    fieldnames = [
        "source_id","doi","channel","condition","drug","cell_id","t_s","value","unit","fs_Hz","imaging_modality","file_origin"
    ]
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # ensure string-ification for optional fields
            r = {**{k: r.get(k, "") for k in fieldnames}, **r}
            w.writerow(r)
    return len(rows)
