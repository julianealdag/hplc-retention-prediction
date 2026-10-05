"""Model definitions and nested cross-validation.

Inner 5-fold CV selects hyperparameters. Outer 5-fold CV scores the
tune-and-fit procedure on data the inner loop did not see. Mean and standard
deviation across outer folds are both reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .splits import Split

logger = logging.getLogger(__name__)


def make_ridge() -> tuple[Pipeline, dict[str, Any]]:
    """Ridge regression behind a standard scaler.

    Scaling is required: L2 shrinkage penalises coefficients equally, so
    unscaled features are penalised in proportion to their units. Scaled
    coefficients can also be compared for feature importance.

    Returns:
        ``(estimator, param_grid)``.
    """
    pipeline = make_pipeline(
        StandardScaler(),
        Ridge(max_iter=config.LINEAR_MAX_ITER, random_state=config.RANDOM_STATE),
    )
    return pipeline, config.PARAM_GRID_RIDGE


def make_lasso() -> tuple[Pipeline, dict[str, Any]]:
    """Lasso regression behind a standard scaler.

    Returns:
        ``(estimator, param_grid)``.
    """
    pipeline = make_pipeline(
        StandardScaler(),
        Lasso(max_iter=config.LINEAR_MAX_ITER, random_state=config.RANDOM_STATE),
    )
    return pipeline, config.PARAM_GRID_LASSO


def make_random_forest() -> tuple[RandomForestRegressor, dict[str, Any]]:
    """Random forest regressor.

    No scaler: trees split on thresholds, so monotone rescaling changes nothing.

    Returns:
        ``(estimator, param_grid)``.
    """
    return (
        RandomForestRegressor(random_state=config.RANDOM_STATE),
        config.PARAM_GRID_RF,
    )


MODEL_FACTORIES = {
    "Ridge": make_ridge,
    "Lasso": make_lasso,
    "RandomForest": make_random_forest,
}


@dataclass
class CVResult:
    """Outcome of nested cross-validation for one model on one dataset.

    Attributes:
        dataset: Dataset name.
        model: Model name.
        best_params: Hyperparameters chosen by refitting on the full training set.
        mae, mse, r2: Outer-fold means. MAE and MSE are positive (scikit-learn
            reports them negated so that "higher is better"; the sign is flipped
            back here).
        mae_std, mse_std, r2_std: Standard deviations across the outer folds.
        fold_importances: Per-outer-fold coefficient or feature-importance vectors.
        feature_names: Names matching ``fold_importances`` columns.
        fold_scores: One row per outer fold with its MAE, MSE and R², and for a
            grouped split the groups it held out.
    """

    dataset: str
    model: str
    best_params: dict[str, Any]
    mae: float
    mae_std: float
    mse: float
    mse_std: float
    r2: float
    r2_std: float
    fold_importances: list[np.ndarray] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    fold_scores: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_row(self) -> dict[str, Any]:
        """Flatten to a dict suitable for a summary DataFrame."""
        return {
            "Dataset": self.dataset,
            "Model": self.model,
            "MAE": self.mae,
            "MAE_std": self.mae_std,
            "MSE": self.mse,
            "MSE_std": self.mse_std,
            "R2": self.r2,
            "R2_std": self.r2_std,
            "BestParams": self.best_params,
        }

    def importance_series(self) -> pd.Series:
        """Mean absolute importance per feature across the outer folds.

        For linear models these are absolute standardised coefficients; for the
        forest they are impurity-based importances.

        Returns:
            Series indexed by feature name, sorted descending. Empty if no
            importances were collected.
        """
        if not self.fold_importances:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(self.fold_importances, columns=self.feature_names)
        return frame.abs().mean().sort_values(ascending=False)


def _held_out(split: Split, indices: np.ndarray, max_listed: int = 10) -> str:
    """Describe the groups in one validation fold: names if few, else a count."""
    if split.groups_train is None:
        return ""
    groups = sorted(split.groups_train.iloc[indices].unique())
    if len(groups) > max_listed:
        return f"{len(groups)} groups"
    return ", ".join(groups)


def _extract_importances(estimator: BaseEstimator) -> np.ndarray | None:
    """Pull coefficients or feature importances out of a fitted estimator."""
    target = estimator
    if isinstance(estimator, Pipeline):
        target = estimator[-1]
    if hasattr(target, "coef_"):
        return np.asarray(target.coef_).ravel()
    if hasattr(target, "feature_importances_"):
        return np.asarray(target.feature_importances_)
    return None


def outer_splitter(split: Split, n_folds: int) -> KFold | GroupKFold:
    """Folds for the outer loop of nested CV.

    A grouped split (by compound or by method) needs grouped folds too: plain
    k-fold would put the same molecule or method on both sides of a fold and
    make the cross-validation scores as optimistic as a row-wise split.

    Args:
        split: The dataset split; its ``groups_train`` decide the fold type.
        n_folds: Number of folds.

    Returns:
        ``GroupKFold`` when the split is grouped, otherwise ``KFold``.

    Raises:
        ValueError: If there are fewer groups than folds.
    """
    if split.groups_train is None:
        return KFold(n_splits=n_folds)
    n_groups = split.groups_train.nunique()
    if n_groups < n_folds:
        raise ValueError(
            f"{split.name}: {n_groups} groups in training, need at least {n_folds} "
            "for grouped cross-validation"
        )
    return GroupKFold(n_splits=n_folds)


def nested_cv(
    split: Split,
    model_name: str,
    inner_folds: int = config.INNER_CV_FOLDS,
    outer_folds: int = config.OUTER_CV_FOLDS,
    n_jobs: int = -1,
) -> CVResult:
    """Run nested cross-validation for one model on one dataset's training set.

    Only the training half of ``split`` is used; the test half is reserved for
    :mod:`hplc_rt.evaluate`. For a grouped split the outer folds are grouped
    as well (see :func:`outer_splitter`). The inner loop stays ungrouped: it only
    picks hyperparameters, and the outer folds still score them on unseen groups.

    Args:
        split: The dataset split.
        model_name: One of :data:`MODEL_FACTORIES`.
        inner_folds: Folds in the hyperparameter-selection loop.
        outer_folds: Folds in the evaluation loop.
        n_jobs: Parallelism passed to the inner grid search.

    Returns:
        The :class:`CVResult`.

    Raises:
        KeyError: If ``model_name`` is not a known model.
    """
    if model_name not in MODEL_FACTORIES:
        raise KeyError(
            f"unknown model {model_name!r}; expected one of {list(MODEL_FACTORIES)}"
        )

    estimator, param_grid = MODEL_FACTORIES[model_name]()
    X, y = split.X_train, split.y_train

    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=inner_folds,
        scoring=config.SELECTION_METRIC,
        n_jobs=n_jobs,
    )
    cv = cross_validate(
        grid, X, y, groups=split.groups_train, cv=outer_splitter(split, outer_folds),
        scoring=config.SCORING, return_estimator=True, return_indices=True,
    )

    fold_importances: list[np.ndarray] = []
    for fold_estimator in cv["estimator"]:
        importances = _extract_importances(fold_estimator.best_estimator_)
        if importances is not None:
            fold_importances.append(importances)

    # Refit on the full training set to record the chosen hyperparameters.
    # Performance numbers above come from the outer CV, not from this fit.
    grid.fit(X, y)

    fold_scores = pd.DataFrame({
        "Dataset": split.name,
        "Model": model_name,
        "Fold": range(1, len(cv["test_R2"]) + 1),
        "MAE": -cv["test_MAE"],
        "MSE": -cv["test_MSE"],
        "R2": cv["test_R2"],
        "HeldOut": [_held_out(split, idx) for idx in cv["indices"]["test"]],
    })

    result = CVResult(
        dataset=split.name,
        model=model_name,
        best_params=grid.best_params_,
        mae=-cv["test_MAE"].mean(),
        mae_std=cv["test_MAE"].std(),
        mse=-cv["test_MSE"].mean(),
        mse_std=cv["test_MSE"].std(),
        r2=cv["test_R2"].mean(),
        r2_std=cv["test_R2"].std(),
        fold_importances=fold_importances,
        feature_names=split.feature_names,
        fold_scores=fold_scores,
    )
    logger.info(
        "%s / %s: R2=%.3f+/-%.3f MAE=%.3f", split.name, model_name, result.r2,
        result.r2_std, result.mae,
    )
    return result


def zero_coefficient_frequency(results: list[CVResult]) -> pd.DataFrame:
    """How often Lasso eliminated each feature, across every fold supplied.

    The denominator is counted from the folds actually present in ``results``.
    The original coursework code computed it separately as ``len(all_data) * 5``, which
    by that point included the pooled dataset the loop had not visited — 155
    instead of 150 — understating every percentage. Deriving the denominator from
    the data removes the possibility of the two disagreeing.

    Args:
        results: Lasso :class:`CVResult` objects. Non-Lasso results are ignored.

    Returns:
        Columns ``Feature``, ``Zero_Count``, ``Total_Folds``,
        ``Zero_Frequency (%)``, sorted most-eliminated first. Empty if no Lasso
        results were supplied.
    """
    lasso_results = [r for r in results if r.model == "Lasso" and r.fold_importances]
    if not lasso_results:
        return pd.DataFrame(
            columns=["Feature", "Zero_Count", "Total_Folds", "Zero_Frequency (%)"]
        )

    counts: dict[str, int] = {}
    total_folds = 0
    for result in lasso_results:
        for coefficients in result.fold_importances:
            total_folds += 1
            for name, value in zip(result.feature_names, coefficients, strict=True):
                if abs(value) < config.ZERO_COEF_TOLERANCE:
                    counts[name] = counts.get(name, 0) + 1

    frame = pd.DataFrame(
        {
            "Feature": list(counts),
            "Zero_Count": list(counts.values()),
        }
    )
    frame["Total_Folds"] = total_folds
    frame["Zero_Frequency (%)"] = frame["Zero_Count"] / total_folds * 100
    return frame.sort_values("Zero_Frequency (%)", ascending=False).reset_index(drop=True)


def fit_final(split: Split, model_name: str, best_params: dict[str, Any]) -> BaseEstimator:
    """Fit the final model on the full training set with chosen hyperparameters.

    Args:
        split: The dataset split.
        model_name: One of :data:`MODEL_FACTORIES`.
        best_params: Hyperparameters, as returned in :attr:`CVResult.best_params`.

    Returns:
        The fitted estimator.
    """
    estimator, _ = MODEL_FACTORIES[model_name]()
    estimator.set_params(**best_params)
    estimator.fit(split.X_train, split.y_train)
    return estimator
