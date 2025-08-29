#!/usr/bin/env python
import pathlib
import pytest


EXPECTED_DATASETS = {
    "vagus_eng": {"min_files": 1, "extensions": [".mat", ".json"], "min_size_mb": 10},
    "lang_optical": {"min_files": 1, "extensions": [".tif", ".tiff", ".csv", ".json"], "min_size_mb": 100},
    "hd_eeg": {"min_files": 1, "extensions": [".vhdr", ".eeg", ".vmrk", ".fif", ".tsv"], "min_size_mb": 50},
}


RAW_DIR = pathlib.Path(__file__).parents[2] / "data/raw"


@pytest.mark.integration
@pytest.mark.parametrize("dataset", EXPECTED_DATASETS.keys())
def test_dataset_exists(dataset):
    dataset_path = RAW_DIR / dataset
    if not dataset_path.exists():
        pytest.skip(f"Dataset {dataset} not downloaded yet")
    assert dataset_path.exists()
    assert dataset_path.is_dir()

