#!/usr/bin/env python
import pathlib
import pytest

pynwb = pytest.importorskip("pynwb")
from pynwb import NWBHDF5IO  # type: ignore


PROCESSED_DIR = pathlib.Path(__file__).parents[2] / "data/processed"


def get_all_nwb_files():
    return list(PROCESSED_DIR.rglob("*.nwb"))


@pytest.mark.integration
@pytest.mark.parametrize("nwb_file", get_all_nwb_files())
def test_nwb_valid_format(nwb_file):
    with NWBHDF5IO(str(nwb_file), "r") as io:
        nwbfile = io.read()
        assert nwbfile.identifier is not None
        assert nwbfile.session_description is not None
        assert nwbfile.session_start_time is not None

