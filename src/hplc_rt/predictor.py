"""Training a reusable model and predicting retention times for new molecules.

:mod:`hplc_rt.pipeline` evaluates models; this module produces one that can be
saved, shared and used. :func:`train` tunes the pooled model on *all* rows and
returns a :class:`TrainedModel`, which holds everything needed to predict
without the original workbooks:

- the fitted estimator and the exact feature order it expects,
- the method features of every training experiment, so a prediction only needs
  a SMILES string and a method name,
- the descriptor ranges seen in training, to flag molecules unlike any of them.

Predictions are limited to the methods in the training data. The model has no
way to describe a method it has not seen, and the one-hot column identity cannot
represent a new column.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from rdkit import Chem
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from . import __version__, config, descriptors, models, pipeline, splits

logger = logging.getLogger(__name__)

PREDICTION_COLUMN: str = "Predicted RT (min)"


def canonical_smiles(smiles: str | None) -> str | None:
    """RDKit canonical SMILES, or ``None`` if the input does not parse.

    Args:
        smiles: Any SMILES string.

    Returns:
        A canonical form, so that two spellings of one molecule compare equal.
    """
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


@dataclass
class TrainedModel:
    """A fitted pooled model plus what it needs to predict on its own.

    Attributes:
        estimator: The fitted regressor.
        model_name: One of :data:`hplc_rt.models.MODEL_FACTORIES`.
        best_params: Hyperparameters chosen by cross-validation.
        feature_names: Columns of the training matrix, in order.
        method_features: One row per training method, indexed by method name,
            with every non-descriptor feature.
        descriptor_ranges: Minimum and maximum of each molecular descriptor in
            the training data (rows ``min`` and ``max``).
        measured: ``(canonical SMILES, method)`` pairs present in training.
        n_rows: Number of training rows.
        version: Package version that produced the model.
    """

    estimator: BaseEstimator
    model_name: str
    best_params: dict[str, Any]
    feature_names: list[str]
    method_features: pd.DataFrame
    descriptor_ranges: pd.DataFrame
    measured: set[tuple[str, str]] = field(default_factory=set)
    n_rows: int = 0
    version: str = __version__

    @property
    def methods(self) -> list[str]:
        """Names of the methods this model can predict for."""
        return list(self.method_features.index)

    def save(self, path: Path | str) -> Path:
        """Write the model to a single file.

        Args:
            path: Destination, conventionally ending in ``.joblib``.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> TrainedModel:
        """Read a model written by :meth:`save`.

        Loading runs Python's unpickler, which can execute code, so only load
        model files you created or trust.

        Args:
            path: A ``.joblib`` file.

        Returns:
            The :class:`TrainedModel`.

        Raises:
            TypeError: If the file holds something other than a TrainedModel.
        """
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain an hplc-rt model")
        if model.version != __version__:
            logger.warning(
                "model was trained with hplc-rt %s, this is %s; retrain if "
                "predictions look wrong", model.version, __version__,
            )
        return model

    def predict(
        self, smiles: Iterable[str], methods: Iterable[str] | None = None
    ) -> pd.DataFrame:
        """Predict retention times for molecules under training methods.

        Args:
            smiles: SMILES strings.
            methods: Method names from :attr:`methods`; defaults to all of them.

        Returns:
            One row per molecule and method, with columns ``SMILES``, ``Method``,
            ``Predicted RT (min)`` and ``Note``. Invalid SMILES get no prediction.
            The note also flags descriptors outside the training range and
            molecules that were measured under that method in training.

        Raises:
            KeyError: If a method is not in the model.
        """
        methods = self.methods if methods is None else list(methods)
        unknown = [m for m in methods if m not in self.method_features.index]
        if unknown:
            raise KeyError(
                f"unknown method(s) {unknown}; available: {', '.join(self.methods)}"
            )

        smiles = list(smiles)
        rows = []
        for s in smiles:
            desc = descriptors.descriptors_for_smiles(s)
            canonical = canonical_smiles(s)
            for method in methods:
                rows.append({
                    "SMILES": s,
                    "Method": method,
                    **desc.to_dict(),
                    **self.method_features.loc[method].to_dict(),
                    "Note": self._note(desc, canonical, method),
                })
        table = pd.DataFrame(rows)
        if table.empty:
            return pd.DataFrame(columns=["SMILES", "Method", PREDICTION_COLUMN, "Note"])

        valid = table[config.DESCRIPTOR_COLUMNS].notna().all(axis=1)
        table[PREDICTION_COLUMN] = float("nan")
        if valid.any():
            X = table.loc[valid, self.feature_names].astype(float)
            table.loc[valid, PREDICTION_COLUMN] = self.estimator.predict(X)
        return table[["SMILES", "Method", PREDICTION_COLUMN, "Note"]]

    def _note(self, desc: pd.Series, canonical: str | None, method: str) -> str:
        """Warnings for one prediction, joined into one string."""
        if canonical is None or desc.isna().any():
            return "invalid SMILES"
        notes = []
        low, high = self.descriptor_ranges.loc["min"], self.descriptor_ranges.loc["max"]
        outside = [c for c in config.DESCRIPTOR_COLUMNS if not low[c] <= desc[c] <= high[c]]
        if outside:
            notes.append(f"outside training range: {', '.join(outside)}")
        if (canonical, method) in self.measured:
            notes.append("measured in training data")
        return "; ".join(notes)


def train(
    data_dir: Path | str | None = None,
    model_name: str = "RandomForest",
    inner_folds: int = config.INNER_CV_FOLDS,
    n_jobs: int = -1,
) -> TrainedModel:
    """Tune and fit the pooled model on every row, for use in prediction.

    Unlike :func:`hplc_rt.pipeline.run` nothing is held out: the goal is the best
    model to use, not an estimate of its error. Use ``hplc-rt evaluate`` for the
    error estimate.

    Args:
        data_dir: Directory of ``.xlsx`` files.
        model_name: One of :data:`hplc_rt.models.MODEL_FACTORIES`.
        inner_folds: Cross-validation folds for hyperparameter selection.
        n_jobs: Parallelism for the grid search.

    Returns:
        The :class:`TrainedModel`.

    Raises:
        KeyError: If ``model_name`` is not a known model.
        ValueError: If a method feature varies within one method, which would
            mean it cannot be looked up by method name at prediction time.
    """
    if model_name not in models.MODEL_FACTORIES:
        raise KeyError(
            f"unknown model {model_name!r}; expected one of {list(models.MODEL_FACTORIES)}"
        )

    _, pooled = pipeline.build_tables(data_dir)
    feature_names = splits.pooled_feature_names(pooled)
    usable = pooled.dropna(subset=feature_names + [config.TARGET_COLUMN])

    method_columns = [c for c in feature_names if c not in config.DESCRIPTOR_COLUMNS]
    by_method = usable.groupby(config.METHOD_COLUMN)[method_columns]
    distinct = by_method.nunique()
    varying = list(distinct.columns[(distinct > 1).any()])
    if varying:
        raise ValueError(
            f"features vary within a method and cannot be looked up by name: {varying}"
        )

    estimator, param_grid = models.MODEL_FACTORIES[model_name]()
    grid = GridSearchCV(
        estimator, param_grid, cv=inner_folds, scoring=config.SELECTION_METRIC,
        n_jobs=n_jobs,
    )
    grid.fit(usable[feature_names], usable[config.TARGET_COLUMN])
    logger.info("%s trained on %d rows, params %s", model_name, len(usable), grid.best_params_)

    measured = {
        (canonical, method)
        for canonical, method in zip(
            usable[config.GROUP_COLUMN].map(canonical_smiles),
            usable[config.METHOD_COLUMN],
            strict=True,
        )
        if canonical is not None
    }

    return TrainedModel(
        estimator=grid.best_estimator_,
        model_name=model_name,
        best_params=grid.best_params_,
        feature_names=feature_names,
        method_features=by_method.first(),
        descriptor_ranges=usable[config.DESCRIPTOR_COLUMNS].agg(["min", "max"]),
        measured=measured,
        n_rows=len(usable),
    )
