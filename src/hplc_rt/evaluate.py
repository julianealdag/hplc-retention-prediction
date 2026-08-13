"""Held-out test evaluation and the baseline it is judged against.

A model is only as impressive as the baseline it beats. The dummy regressor here
predicts the training-set mean retention time for every compound, which by
construction gives R² = 0 on the data it was fitted to. Its MAE is the number to
beat: for the pooled dataset it is 7.25 min, so a model with an MAE of 3.16 min is
useful but far from precise, and one at 1.09 min is genuinely good.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .splits import Split

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Performance of one fitted model on one held-out test set."""

    dataset: str
    model: str
    mae: float
    mse: float
    r2: float
    best_params: dict[str, Any] | None = None

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
    )


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
