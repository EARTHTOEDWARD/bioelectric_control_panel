#!/usr/bin/env python
import pathlib
import pytest

zarr = pytest.importorskip("zarr")
xr = pytest.importorskip("xarray")


PROCESSED_DIR = pathlib.Path(__file__).parents[2] / "data/processed"


def get_all_zarr_files():
    return list(PROCESSED_DIR.rglob("*.zarr"))


@pytest.mark.integration
@pytest.mark.parametrize("zarr_file", get_all_zarr_files())
def test_zarr_valid_format(zarr_file):
    store = zarr.DirectoryStore(str(zarr_file))
    root = zarr.group(store=store)
    assert root is not None
    ds = xr.open_zarr(str(zarr_file))
    assert len(ds.data_vars) > 0

