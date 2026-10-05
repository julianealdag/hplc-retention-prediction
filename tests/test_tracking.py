"""Tests for the optional W&B logging, run offline so no account is needed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hplc_rt import cli, config, tracking

wandb = pytest.importorskip("wandb")


@pytest.fixture(autouse=True)
def offline_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")


def test_headline_metrics_separate_pooled_and_per_method_scores() -> None:
    summary = pd.DataFrame({
        "Dataset": [config.POOLED_DATASET_KEY, "Dataset 01", "Dataset 02"],
        "Model": ["Ridge", "Ridge", "Ridge"],
        "MAE_test": [3.0, 1.0, 2.0],
        "MSE_test": [9.0, 1.0, 4.0],
        "R2_test": [0.7, 0.8, 0.6],
    })
    metrics = tracking.headline_metrics(summary)
    assert metrics["pooled/Ridge/MAE_test"] == 3.0
    assert metrics["per_method_mean/Ridge/MAE_test"] == pytest.approx(1.5)
    assert metrics["per_method_mean/Ridge/R2_test"] == pytest.approx(0.7)


def test_evaluate_and_train_log_to_wandb(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert cli.main([
        "evaluate", "--data-dir", str(data_dir), "--pooled-only", "--models", "Ridge",
        "--output", str(out), "--wandb", "--wandb-project", "hplc-rt-tests",
    ]) == 0
    assert cli.main([
        "train", "--data-dir", str(data_dir), "--model-type", "Ridge",
        "--out", str(tmp_path / "model.joblib"), "--wandb",
        "--wandb-project", "hplc-rt-tests",
    ]) == 0
    runs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(runs) == 2
