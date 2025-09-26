import argparse
import json
import os
from typing import Dict, Tuple, List

from .config_load import load_config
from .io_load import load_long_csv, load_event_times
from .pipeline import run_pipeline
from .ingest_sources import ingest_to_long


def main() -> int:
    parser = argparse.ArgumentParser(prog="dataset_pipeline", description="Bioelectric dataset analysis toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Build long-format CSV from real sources if present")
    p_ingest.add_argument("--elife-dir", default="data/elife_nadh", help="Dir containing eLife Excel files")
    p_ingest.add_argument("--plos-dir", default="data/plos_tmrm", help="Dir containing PLOS MP4 movies")
    p_ingest.add_argument("--mosaic-dir", default="data/mosaic_percevalhr", help="Dir containing MOSAIC source data (Excel)")
    p_ingest.add_argument("--roi-json", default=None, help="ROI JSON for PLOS movies (optional)")
    p_ingest.add_argument("--out", required=True, help="Output CSV path for long-format data")

    p_build = sub.add_parser("build", help="Run analyses and export CSV summaries")
    p_build.add_argument("--input", required=True, help="Path to long-format CSV input")
    p_build.add_argument("--out", required=True, help="Output directory")
    p_build.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_default.json"), help="Config JSON path")
    p_build.add_argument("--event-times", default=None, help="JSON mapping of condition->event_t_s (optional)")

    args = parser.parse_args()

    if args.cmd == "build":
        os.makedirs(args.out, exist_ok=True)
        cfg = load_config(args.config)
        series, meta = load_long_csv(args.input)
        events = load_event_times(args.event_times) if args.event_times else {}
        run_pipeline(series, meta, events, cfg, args.out)
        return 0
    elif args.cmd == "ingest":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        count = ingest_to_long(
            elife_dir=args.elife_dir,
            plos_dir=args.plos_dir,
            mosaic_dir=args.mosaic_dir,
            roi_json=args.roi_json,
            out_csv=args.out
        )
        print(f"[ingest] Wrote {count} rows to {args.out}")
        return 0
        os.makedirs(args.out, exist_ok=True)
        cfg = load_config(args.config)
        series, meta = load_long_csv(args.input)
        events = load_event_times(args.event_times) if args.event_times else {}
        run_pipeline(series, meta, events, cfg, args.out)
        return 0

    return 0
