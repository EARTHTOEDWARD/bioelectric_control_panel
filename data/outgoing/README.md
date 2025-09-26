# Curated Bioelectric Long-Format Dataset

This folder contains the consolidated dataset generated via `dataset_pipeline ingest` from the raw eLife NAD(P)H spreadsheets and the PLOS TMRM movies.

## Files

- `long.csv` — normalized long-format table ready for analytics or LLM ingestion.
  - Columns: see `manifest.json` or `dataset_pipeline/README.md` for schema details.
  - Each row corresponds to a single `(condition, channel, cell_id, time)` observation.
- `manifest.json` — lightweight summary (row counts, per-channel/condition totals, source provenance).

## Regeneration

To rebuild after updating sources:

```bash
real_bioelectric_data/.venv/bin/python -m dataset_pipeline ingest \
  --elife-dir real_bioelectric_data/data_raw/elife_nadh \
  --plos-dir real_bioelectric_data/data_raw/plos_tmrm \
  --mosaic-dir real_bioelectric_data/data_raw/mosaic_percevalhr \
  --out data/outgoing/long.csv
```

The pipeline writes results here; rerun `dataset_pipeline build` if you need the derived analytics under `real_bioelectric_data/data_processed/`.
