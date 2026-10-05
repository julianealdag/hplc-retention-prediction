"""Held-out test evaluation and a mean-predicting dummy baseline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .splits import Split

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Performance of one fitted model on one held-out test set.

    ``predictions`` holds one row per test measurement (method, measured and
    predicted retention time) so errors can be broken down and plotted.
    """

    dataset: str
    model: str
    mae: float
    mse: float
    r2: float
    best_params: dict[str, Any] | None = None
    predictions: pd.DataFrame | None = field(default=None, repr=False)

    def to_row(self) -> dict[str, Any]:
        """Flatten to a dict suitable for a summary DataFrame."""
        return {
            "Dataset": self.dataset,
            "Model": self.model,
            "MAE_test": self.mae,
            "MSE_test": self.mse,
            "R2_test": self.r2,
            "BestParams": self.best_params,
        }


def evaluate(
    estimator: BaseEstimator,
    split: Split,
    model_name: str,
    best_params: dict[str, Any] | None = None,
) -> TestResult:
    """Score a fitted estimator on the held-out test set.

    Args:
        estimator: Model already fitted on ``split.X_train``.
        split: The dataset split.
        model_name: Label for the result.
        best_params: Hyperparameters used, recorded for the summary table.

    Returns:
        The :class:`TestResult`.
    """
    predictions = estimator.predict(split.X_test)
    return TestResult(
        dataset=split.name,
        model=model_name,
        mae=mean_absolute_error(split.y_test, predictions),
        mse=mean_squared_error(split.y_test, predictions),
        r2=r2_score(split.y_test, predictions),
        best_params=best_params,
        predictions=prediction_table(split, model_name, predictions),
    )


def prediction_table(split: Split, model_name: str, predicted: np.ndarray) -> pd.DataFrame:
    """Measured and predicted retention time for every test row.

    Args:
        split: The dataset split.
        model_name: Label for the rows.
        predicted: Predictions for ``split.X_test``, in order.

    Returns:
        Columns ``Dataset``, ``Model``, ``Method``, ``Measured``, ``Predicted``.
        ``Method`` is the dataset name when the split does not record methods.
    """
    methods = split.methods_test if split.methods_test is not None else split.name
    return pd.DataFrame({
        "Dataset": split.name,
        "Model": model_name,
        "Method": methods,
        "Measured": split.y_test.to_numpy(),
        "Predicted": np.asarray(predicted, dtype=float),
    }).reset_index(drop=True)


def dummy_baseline(split: Split, on: str = "test") -> TestResult:
    """Score a mean-predicting baseline.

    The mean is always taken from the *training* set — taking it from the test set
    would be using information the model is not allowed to have, and would make the
    baseline artificially strong.

    Args:
        split: The dataset split.
        on: ``"test"`` or ``"train"``, selecting which set to score against.

    Returns:
        The :class:`TestResult` for the baseline.

    Raises:
        ValueError: If ``on`` is not ``"test"`` or ``"train"``.
    """
    if on not in {"test", "train"}:
        raise ValueError(f"on must be 'test' or 'train', got {on!r}")

    y_true = split.y_test if on == "test" else split.y_train
    prediction = float(np.mean(split.y_train))
    predicted = np.full(len(y_true), prediction, dtype=float)

    return TestResult(
        dataset=split.name,
        model="Dummy",
        mae=mean_absolute_error(y_true, predicted),
        mse=mean_squared_error(y_true, predicted),
        r2=r2_score(y_true, predicted),
    )


def summarize(results: list[TestResult]) -> pd.DataFrame:
    """Collect test results into a sorted summary table.

    Args:
        results: Results to combine.

    Returns:
        DataFrame sorted by dataset then descending R², so each dataset's models
        appear together best-first.
    """
    frame = pd.DataFrame([r.to_row() for r in results])
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["Dataset", "R2_test"], ascending=[True, False]
    ).reset_index(drop=True)
