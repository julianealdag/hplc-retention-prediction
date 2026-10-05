"""Optional experiment tracking with Weights & Biases.

Nothing here runs unless ``--wandb`` is passed, and ``wandb`` is only imported
then, so the package works without it. Install with ``pip install -e ".[wandb]"``
and log in once with ``wandb login``.

Each ``evaluate`` run logs its settings, the CV and test tables, headline metrics
and figures. Each ``train`` run also uploads the model file as an artifact.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__, config

DEFAULT_PROJECT: str = "hplc-retention-prediction"


def _wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "--wandb needs the wandb package: pip install -e \".[wandb]\""
        ) from exc
    return wandb


def git_commit() -> str | None:
    """Short hash of the checked-out commit, or ``None`` outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def start_run(
    job_type: str,
    settings: dict[str, Any],
    project: str = DEFAULT_PROJECT,
    name: str | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Start a W&B run with the package version and git commit added to its config.

    Args:
        job_type: ``"evaluate"`` or ``"train"``, used to group runs.
        settings: Run settings to record, such as the split and models.
        project: W&B project name.
        name: Run name shown in the W&B interface.
        tags: Labels for filtering runs.

    Returns:
        The ``wandb`` run.
    """
    run_config = {"version": __version__, "git_commit": git_commit(), **settings}
    return _wandb().init(
        project=project, job_type=job_type, name=name, tags=tags, config=run_config
    )


def headline_metrics(test_summary: pd.DataFrame) -> dict[str, float]:
    """Flat metric dict for the run summary.

    Pooled-model scores are reported per model, and per-method scores as the
    mean over the per-method datasets, so 30 methods do not produce 360 keys.

    Args:
        test_summary: Output of :func:`hplc_rt.evaluate.summarize`.

    Returns:
        Keys like ``pooled/RandomForest/R2_test`` and
        ``per_method_mean/Ridge/MAE_test``.
    """
    metrics: dict[str, float] = {}
    columns = ["MAE_test", "MSE_test", "R2_test"]
    pooled = test_summary[test_summary["Dataset"] == config.POOLED_DATASET_KEY]
    for _, row in pooled.iterrows():
        for column in columns:
            metrics[f"pooled/{row['Model']}/{column}"] = float(row[column])
    per_method = test_summary[test_summary["Dataset"] != config.POOLED_DATASET_KEY]
    if not per_method.empty:
        means = per_method.groupby("Model")[columns].mean()
        for model, row in means.iterrows():
            for column in columns:
                metrics[f"per_method_mean/{model}/{column}"] = float(row[column])
    return metrics


def log_results(
    run: Any,
    cv_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    figures_dir: Path | None = None,
) -> None:
    """Log result tables, headline metrics and figures to a run.

    Args:
        run: An active run from :func:`start_run`.
        cv_summary: Nested-CV summary table.
        test_summary: Held-out test summary table.
        figures_dir: Directory of ``.png`` figures to upload, if any.
    """
    wandb = _wandb()
    tables = {"cv_summary": cv_summary, "test_summary": test_summary}
    run.log({key: wandb.Table(dataframe=_stringify(t)) for key, t in tables.items()})
    run.summary.update(headline_metrics(test_summary))
    if figures_dir is not None and figures_dir.is_dir():
        images = {
            f"figures/{path.stem}": wandb.Image(str(path))
            for path in sorted(figures_dir.glob("*.png"))
        }
        if images:
            run.log(images)


def log_model(run: Any, path: Path, metadata: dict[str, Any]) -> None:
    """Upload a saved model file as a versioned W&B artifact.

    Args:
        run: An active run from :func:`start_run`.
        path: The ``.joblib`` file written by ``hplc-rt train``.
        metadata: Shown with the artifact, e.g. model type and training size.
    """
    artifact = _wandb().Artifact("hplc-rt-model", type="model", metadata=metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def _stringify(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy with dict-valued cells (hyperparameters) as text, which W&B tables need."""
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].map(lambda v: isinstance(v, dict)).any():
            frame[column] = frame[column].map(str)
    return frame
