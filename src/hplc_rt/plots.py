"""Figures.

Every function takes data and returns a Matplotlib ``Figure`` rather than calling
``plt.show()``, so the same code works in a notebook, in a script that saves to
disk, and in a test that never renders anything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe default; notebooks override this themselves
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402

MODEL_PALETTE: dict[str, str] = {
    "Ridge": "royalblue",
    "Lasso": "lightcoral",
    "RandomForest": "mediumseagreen",
    "Dummy": "darkgray",
}


def model_comparison(
    summary: pd.DataFrame,
    metric: str = "R2_test",
    include_dummy: bool = True,
    title: str | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing models across datasets.

    Args:
        summary: Output of :func:`hplc_rt.evaluate.summarize`.
        metric: Column to plot, e.g. ``"R2_test"`` or ``"MAE_test"``.
        include_dummy: Whether to show the baseline. Omit it for R², where the
            baseline is 0 by construction and adds nothing.
        title: Plot title; a sensible default is derived from ``metric``.

    Returns:
        The figure.
    """
    data = summary if include_dummy else summary[summary["Model"] != "Dummy"]

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.barplot(
        data=data, x="Dataset", y=metric, hue="Model", palette=MODEL_PALETTE, ax=ax
    )
    ax.set_title(title or f"Model comparison ({metric}) on held-out test sets")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def predicted_vs_true(
    y_true: pd.Series, y_pred, title: str = "Predicted vs. true retention time"
) -> plt.Figure:
    """Scatter of predictions against measurements, with the ideal y = x line.

    Args:
        y_true: Measured retention times.
        y_pred: Predicted retention times.
        title: Plot title.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=14, color="royalblue", edgecolor="none")
    lo, hi = float(min(y_true)), float(max(y_true))
    ax.plot([lo, hi], [lo, hi], "--", color="firebrick", linewidth=1.5, label="ideal")
    ax.set_xlabel("Measured RT (min)")
    ax.set_ylabel("Predicted RT (min)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def feature_importance(
    importances: pd.Series, top_n: int = 15, title: str = "Feature importance"
) -> plt.Figure:
    """Horizontal bar chart of the strongest features.

    Args:
        importances: Output of :meth:`hplc_rt.models.CVResult.importance_series`.
        top_n: How many features to show.
        title: Plot title.

    Returns:
        The figure.
    """
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top))))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="Blues_r",
                legend=False, ax=ax)
    ax.set_xlabel("Mean absolute importance across outer folds")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def importance_heatmap(
    per_dataset: pd.DataFrame, title: str = "Feature importance per dataset"
) -> plt.Figure:
    """Heatmap of feature importance across datasets.

    Note:
        Each column must come from that dataset's own folds. The original coursework code
        built this figure with a slice index that was never advanced, so all 30
        columns showed the same five folds — see ``docs/corrections.md``.

    Args:
        per_dataset: Features as rows, datasets as columns.
        title: Plot title.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * per_dataset.shape[1]), 8))
    sns.heatmap(
        per_dataset, cmap="Blues", annot=False, linewidths=0.4,
        cbar_kws={"label": "Mean absolute importance"}, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, name: str, directory: Path | None = None) -> Path:
    """Write a figure to the figures directory.

    Args:
        fig: Figure to save.
        name: Filename stem; ``.png`` is appended.
        directory: Destination; defaults to :data:`config.FIGURES_DIR`.

    Returns:
        The path written.
    """
    directory = directory or config.FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
