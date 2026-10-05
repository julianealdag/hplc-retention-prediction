"""End-to-end orchestration: raw workbooks in, results table out.

This is the module that replaces "run all cells in order".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config, curation, descriptors, evaluate, features, loading, models, splits
from .evaluate import TestResult
from .models import CVResult
from .splits import Split

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """Everything a run produces.

    Attributes:
        splits: Train/test splits keyed by dataset name.
        cv_results: Nested-CV results, keyed by ``(dataset, model)``.
        test_results: Held-out test results, including the dummy baseline.
        cv_summary: Tabular view of ``cv_results``.
        test_summary: Tabular view of ``test_results``.
        fold_scores: Every outer CV fold of every model, one row each.
        predictions: Measured and predicted retention time for every test row.
    """

    splits: dict[str, Split] = field(default_factory=dict)
    cv_results: dict[tuple[str, str], CVResult] = field(default_factory=dict)
    test_results: list[TestResult] = field(default_factory=list)
    cv_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_tables(
    data_dir: Path | str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load, curate and featurise the raw workbooks, without splitting.

    Args:
        data_dir: Directory of ``.xlsx`` files; defaults to :data:`config.DATA_DIR`.

    Returns:
        ``(per_experiment, pooled)``: one frame per experiment, and the pooled
        frame with method features and a ``Dataset`` column. Both have the
        molecular descriptors attached.
    """
    experiments = loading.load_all(data_dir)
    curation.curate(experiments)
    features.build_features(experiments)

    pooled = features.pool_experiments(experiments)

    per_experiment = {
        name: descriptors.add_descriptors(exp.rt) for name, exp in experiments.items()
    }
    pooled = descriptors.add_descriptors(pooled)
    return per_experiment, pooled


def prepare_data(
    data_dir: Path | str | None = None,
    exclusions: list[str] | None = None,
    group_by: str | None = None,
) -> dict[str, Split]:
    """Load, curate, featurise and split the raw workbooks.

    Args:
        data_dir: Directory of ``.xlsx`` files; defaults to :data:`config.DATA_DIR`.
        exclusions: Columns kept out of the pooled feature matrix; defaults to
            :data:`config.NON_FEATURE_COLUMNS`.
        group_by: If set, hold out entire groups (typically SMILES) rather than
            splitting row-wise.

    Returns:
        Splits for every experiment plus the pooled dataset.
    """
    per_experiment, pooled = build_tables(data_dir)
    return splits.make_all_splits(per_experiment, pooled, exclusions, group_by=group_by)


def run(
    data_dir: Path | str | None = None,
    model_names: list[str] | None = None,
    datasets: list[str] | None = None,
    exclusions: list[str] | None = None,
    group_by: str | None = None,
) -> PipelineOutput:
    """Run the full pipeline.

    For each dataset and model: nested CV on the training half, then a final fit
    with the selected hyperparameters, then a single score against the held-out
    test half. The dummy baseline is scored once per dataset.

    Args:
        data_dir: Directory of ``.xlsx`` files.
        model_names: Models to run; defaults to all of
            :data:`hplc_rt.models.MODEL_FACTORIES`.
        datasets: Restrict to these dataset names. Useful for a quick check —
            ``datasets=["Dataset_all"]`` skips the 30 per-experiment models.
        exclusions: Columns kept out of the pooled feature matrix.
        group_by: If set, hold out entire groups (typically SMILES) rather than
            splitting row-wise.

    Returns:
        The :class:`PipelineOutput`.
    """
    model_names = model_names or list(models.MODEL_FACTORIES)
    all_splits = prepare_data(data_dir, exclusions, group_by=group_by)
    if datasets is not None:
        all_splits = {k: v for k, v in all_splits.items() if k in datasets}

    output = PipelineOutput(splits=all_splits)

    for name, split in all_splits.items():
        for model_name in model_names:
            logger.info("fitting %s on %s", model_name, name)
            cv_result = models.nested_cv(split, model_name)
            output.cv_results[(name, model_name)] = cv_result

            estimator = models.fit_final(split, model_name, cv_result.best_params)
            output.test_results.append(
                evaluate.evaluate(estimator, split, model_name, cv_result.best_params)
            )
        output.test_results.append(evaluate.dummy_baseline(split))

    output.cv_summary = pd.DataFrame([r.to_row() for r in output.cv_results.values()])
    output.test_summary = evaluate.summarize(output.test_results)
    folds = [r.fold_scores for r in output.cv_results.values()]
    if folds:
        output.fold_scores = pd.concat(folds, ignore_index=True)
    predictions = [r.predictions for r in output.test_results if r.predictions is not None]
    if predictions:
        output.predictions = pd.concat(predictions, ignore_index=True)
    return output


def importance_matrix(
    output: PipelineOutput, model_name: str, feature_subset: list[str] | None = None
) -> pd.DataFrame:
    """Assemble per-dataset feature importances into a features x datasets matrix.

    Args:
        output: A completed run.
        model_name: Which model's importances to collect.
        feature_subset: Restrict to these features; defaults to the nine molecular
            descriptors, which are the only features shared by every dataset.

    Returns:
        Features as rows, datasets as columns.
    """
    feature_subset = feature_subset or config.DESCRIPTOR_COLUMNS
    columns = {}
    for (dataset, model), result in output.cv_results.items():
        if model != model_name:
            continue
        series = result.importance_series()
        columns[dataset] = series.reindex(feature_subset)
    return pd.DataFrame(columns)
