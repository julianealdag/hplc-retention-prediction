"""Train/test splitting and feature-matrix selection.

Two modelling settings are supported, and they use different feature sets:

*Per-experiment* models
    Trained on one experiment at a time. The chromatographic conditions are
    constant within an experiment, so only the nine molecular descriptors vary and
    only they are used. The question these answer is "given this fixed method, how
    well can structure alone predict retention?"

*Pooled* model
    Trained on all 30 experiments together, using descriptors **and** conditions.
    The question here is "can one model transfer across methods?" — which is the
    actual point of the project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)


@dataclass
class Split:
    """An 80/20 train/test split of one dataset.

    Attributes:
        name: Dataset the split came from.
        X_train, X_test: Feature matrices.
        y_train, y_test: Retention times in minutes.
        feature_names: Columns of the feature matrices, in order.
    """

    name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Split({self.name!r}, train={self.X_train.shape}, "
            f"test={self.X_test.shape}, n_features={len(self.feature_names)})"
        )


def pooled_feature_names(pooled: pd.DataFrame) -> list[str]:
    """Feature columns for the pooled model.

    Derived from the *pooled* frame's own columns, so the ``Col_*`` one-hot
    indicators created during pooling are included. (The original notebook derived
    this list from a leftover per-experiment frame, which has no ``Col_*`` columns,
    so column identity silently never reached the model — see
    ``docs/corrections.md``.)

    Args:
        pooled: Output of :func:`hplc_rt.features.pool_experiments` with descriptors.

    Returns:
        Feature column names, excluding identifiers, the target, and the leaky
        columns listed in :data:`config.NON_FEATURE_COLUMNS`.
    """
    return [c for c in pooled.columns if c not in config.NON_FEATURE_COLUMNS]


def make_split(
    frame: pd.DataFrame,
    feature_names: list[str],
    name: str,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Split:
    """Split one frame into train and test sets.

    Note:
        The split is random over compound/experiment rows. Because the same
        compounds recur across experiments, a compound can appear in both train and
        test under different conditions — an optimistic setting. See the
        "Limitations" section of the README.

    Args:
        frame: Table with features and the target column.
        feature_names: Columns to use as features.
        name: Label for the resulting split.
        test_size: Fraction held out.
        random_state: Seed.

    Returns:
        The :class:`Split`.

    Raises:
        KeyError: If the target column is missing.
    """
    if config.TARGET_COLUMN not in frame.columns:
        raise KeyError(f"{config.TARGET_COLUMN!r} not found in {name}")

    usable = frame.dropna(subset=feature_names + [config.TARGET_COLUMN])
    n_dropped = len(frame) - len(usable)
    if n_dropped:
        logger.info("%s: dropped %d rows with missing values", name, n_dropped)

    X = usable[feature_names]
    y = usable[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return Split(name, X_train, X_test, y_train, y_test, list(feature_names))


def make_all_splits(
    per_experiment: dict[str, pd.DataFrame], pooled: pd.DataFrame
) -> dict[str, Split]:
    """Build splits for every experiment plus the pooled dataset.

    Args:
        per_experiment: Per-experiment frames with descriptors attached.
        pooled: The pooled frame with descriptors attached.

    Returns:
        Mapping from dataset name to :class:`Split`, including the pooled dataset
        under :data:`config.POOLED_DATASET_KEY`.
    """
    splits = {
        name: make_split(frame, config.DESCRIPTOR_COLUMNS, name)
        for name, frame in per_experiment.items()
    }
    splits[config.POOLED_DATASET_KEY] = make_split(
        pooled, pooled_feature_names(pooled), config.POOLED_DATASET_KEY
    )
    logger.info(
        "built %d splits (pooled model uses %d features)",
        len(splits),
        len(splits[config.POOLED_DATASET_KEY].feature_names),
    )
    return splits
