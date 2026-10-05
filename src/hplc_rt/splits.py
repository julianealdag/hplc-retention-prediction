"""Train/test splitting and feature-matrix selection.

Two modelling settings are supported, and they use different feature sets:

*Per-experiment* models
    Trained on one experiment at a time. The chromatographic conditions are
    constant within an experiment, so only the nine molecular descriptors vary and
    only they are used. The question these answer is "given this fixed method, how
    well can structure alone predict retention?"

*Pooled* model
    Trained on all 30 experiments together, using descriptors **and** conditions.
    Question: can one model transfer across methods?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
        groups_train, groups_test: Group labels (typically SMILES) when the
            split is compound-disjoint; ``None`` for a row-wise split.
        methods_test: Method of each test row, when the frame records it (the
            pooled frame does); used to break down errors by method.
    """

    name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    groups_train: pd.Series | None = None
    groups_test: pd.Series | None = None
    methods_test: pd.Series | None = None

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Split({self.name!r}, train={self.X_train.shape}, "
            f"test={self.X_test.shape}, n_features={len(self.feature_names)})"
        )


def pooled_feature_names(
    pooled: pd.DataFrame, exclusions: list[str] | None = None
) -> list[str]:
    """Feature columns for the pooled model.

    Derived from the *pooled* frame's own columns, so the ``Col_*`` one-hot
    indicators created during pooling are included. (The original coursework code derived
    this list from a leftover per-experiment frame, which has no ``Col_*`` columns,
    so column identity silently never reached the model — see
    ``docs/corrections.md``.)

    Args:
        pooled: Output of :func:`hplc_rt.features.pool_experiments` with descriptors.
        exclusions: Columns to leave out; defaults to
            :data:`config.NON_FEATURE_COLUMNS`. Override only to *measure* the
            effect of the leaking columns, never to model with them.

    Returns:
        Feature column names, excluding identifiers, the target, and the
        target-derived columns.
    """
    exclusions = config.NON_FEATURE_COLUMNS if exclusions is None else exclusions
    return [c for c in pooled.columns if c not in exclusions]


def make_split(
    frame: pd.DataFrame,
    feature_names: list[str],
    name: str,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    group_by: str | None = None,
) -> Split:
    """Split one frame into train and test sets.

    Note:
        The default split is random over compound/experiment rows. Because the
        same compounds recur across experiments, a compound can appear in both
        train and test under different conditions. Pass ``group_by``
        (typically :data:`config.GROUP_COLUMN`) to hold out entire molecules
        instead. See the Limitations section of the README.

    Args:
        frame: Table with features and the target column.
        feature_names: Columns to use as features.
        name: Label for the resulting split.
        test_size: Fraction held out.
        random_state: Seed.
        group_by: If set, split on this column so no group appears on both
            sides. ``None`` keeps the original row-wise split.

    Returns:
        The :class:`Split`.

    Raises:
        KeyError: If the target column, or ``group_by`` when requested, is missing.
    """
    if config.TARGET_COLUMN not in frame.columns:
        raise KeyError(f"{config.TARGET_COLUMN!r} not found in {name}")

    needed = feature_names + [config.TARGET_COLUMN]
    if group_by is not None:
        if group_by not in frame.columns:
            raise KeyError(f"{group_by!r} not found in {name}")
        needed = needed + [group_by]

    usable = frame.dropna(subset=needed)
    n_dropped = len(frame) - len(usable)
    if n_dropped:
        logger.info("%s: dropped %d rows with missing values", name, n_dropped)

    X = usable[feature_names]
    y = usable[config.TARGET_COLUMN]
    methods = usable.get(config.METHOD_COLUMN)

    if group_by is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return Split(
            name, X_train, X_test, y_train, y_test, list(feature_names),
            methods_test=None if methods is None else methods.loc[X_test.index],
        )

    groups = usable[group_by]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return Split(
        name,
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        list(feature_names),
        groups_train=groups.iloc[train_idx],
        groups_test=groups.iloc[test_idx],
        methods_test=None if methods is None else methods.iloc[test_idx],
    )


def make_all_splits(
    per_experiment: dict[str, pd.DataFrame],
    pooled: pd.DataFrame,
    exclusions: list[str] | None = None,
    group_by: str | None = None,
) -> dict[str, Split]:
    """Build splits for every experiment plus the pooled dataset.

    Args:
        per_experiment: Per-experiment frames with descriptors attached.
        pooled: The pooled frame with descriptors attached.
        exclusions: Passed through to :func:`pooled_feature_names`.
        group_by: Passed through to :func:`make_split`. With
            :data:`config.METHOD_COLUMN` only the pooled dataset is split, since
            each experiment holds a single method.

    Returns:
        Mapping from dataset name to :class:`Split`, including the pooled dataset
        under :data:`config.POOLED_DATASET_KEY`.
    """
    if group_by == config.METHOD_COLUMN:
        # Each experiment is a single method, so it cannot be split by method.
        logger.info("method split: per-experiment models skipped, pooled model only")
        splits = {}
    else:
        splits = {
            name: make_split(frame, config.DESCRIPTOR_COLUMNS, name, group_by=group_by)
            for name, frame in per_experiment.items()
        }
    splits[config.POOLED_DATASET_KEY] = make_split(
        pooled,
        pooled_feature_names(pooled, exclusions),
        config.POOLED_DATASET_KEY,
        group_by=group_by,
    )
    logger.info(
        "built %d splits (pooled model uses %d features)",
        len(splits),
        len(splits[config.POOLED_DATASET_KEY].feature_names),
    )
    return splits
