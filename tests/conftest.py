"""Shared fixtures: synthetic workbooks in the real MCMRT layout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_synthetic_data import make_workbook  # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three synthetic experiments of 20 compounds each."""
    directory = tmp_path_factory.mktemp("synthetic")
    for i in range(3):
        make_workbook(directory / f"Dataset_{9000 + i}.xlsx", i, 20, seed=0)
    return directory


@pytest.fixture(scope="session")
def many_methods_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Seven synthetic experiments: enough methods for 5-fold grouped CV after the split."""
    directory = tmp_path_factory.mktemp("synthetic_methods")
    for i in range(7):
        make_workbook(directory / f"Dataset_{9100 + i}.xlsx", i, 12, seed=1)
    return directory
