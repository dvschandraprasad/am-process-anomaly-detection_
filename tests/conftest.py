"""Shared test data: a very small fake dataset, made once per test run."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import PartSpec, make_dataset, write_part  # noqa: E402

SMALL = PartSpec(lx=3.0, ly=2.0, n_layers=8, layer_mm=0.1, overhang_start_mm=0.2,
                 overhang_max_mm=0.6, hatch_mm=0.2, n_pores=4)


@pytest.fixture(scope="session")
def dataset():
    """Point tables of three small fake parts: {1: table, 2: table, 3: table}."""
    return make_dataset(3, SMALL, seed=3)


@pytest.fixture(scope="session")
def data_dir(dataset, tmp_path_factory):
    """The same parts written to disk the way NIST stores them."""
    root = tmp_path_factory.mktemp("data")
    for part, points in dataset.items():
        write_part(points, root / f"part{part}")
    return root
