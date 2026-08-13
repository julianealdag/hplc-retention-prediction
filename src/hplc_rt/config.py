"""Central configuration: paths, column names, hyperparameter grids, seed.

Paths, column names, hyperparameter grids and the random seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

RANDOM_STATE: int = 42
"""Seed used for every train/test split, Ridge/Lasso solver and Random Forest."""

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
"""Directory holding the per-experiment ``.xlsx`` files (one file per LC setup)."""

RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------
# Excel schema
# --------------------------------------------------------------------------

SHEET_RT: str = "RT"
"""Sheet holding one row per compound: identifiers plus the measured retention time."""

SHEET_LC: str = "LC setups"
"""Sheet holding the instrument/column metadata and the gradient elution program."""

GRADIENT_HEADER: str = "Gradient elution program"
"""Marker row in ``SHEET_LC`` below which the gradient table starts."""

GRADIENT_COLUMNS: list[str] = ["Time (min)", "Flow rate (mL/min)", "B (%)"]

TARGET_COLUMN: str = "RT (min)"
"""Regression target: measured retention time in minutes."""

LC_KEYS_OF_INTEREST: list[str] = [
    "Instrument",
    "Instrument manufacturer",
    "Model type",
    "Analytical column",
    "Column dimensions",
    "Sample temperature (°C)",
    "Column temperature (°C)",
    "Dead time (min)",
    "Mobile phase A",
    "Mobile phase B",
]

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

DESCRIPTOR_COLUMNS: list[str] = [
    "MolWt",
    "LogP",
    "TPSA",
    "RotatableBonds",
    "HBDonors",
    "HBAcceptors",
    "AromaticRings",
    "MolRefractivity",
    "BranchingIndex",
]
"""The nine RDKit 2D descriptors computed per compound.

These alone are the feature set for the *per-dataset* models: within a single
experiment the chromatographic conditions are constant, so they carry no
information and only the molecule varies.
"""

GRADIENT_RESOLUTION: int = 100
"""Number of points each gradient program is resampled onto.

Gradient programs are specified as a handful of (time, flow rate, %B) breakpoints
and differ in length between experiments. Resampling every program onto a fixed
100-point grid turns a variable-length program into a fixed-length vector that a
regressor can consume. Interpolation is step-wise (``kind="previous"``) because an
HPLC gradient holds its setting until the next programmed change.
"""

GRADIENT_TIME_MAX: float = 100.0
"""Upper bound of the resampling grid in minutes; covers the longest program."""

NON_FEATURE_COLUMNS: list[str] = [
    "MCMRT\nNumber",
    "Compound \nName",
    "IUPAC \nName",
    "Formula",
    "Pubchem \nNumber",
    "Isomeric SMILES",
    "InChI",
    "Retention Factor (k)",
    "RT (min)",
    "RSD",
    "Dataset",
    "AnalyticalColumn",
]
"""Identifier, target and post-hoc columns excluded from the pooled feature matrix.

Three of these are excluded for leakage reasons rather than because they are
identifiers:

``RT (min)``
    The target itself.

``Retention Factor (k)``
    Defined as ``k = (RT - t_dead) / t_dead``. It is an algebraic rearrangement of
    the target, so a model given ``k`` and the dead time can invert it exactly.

``RSD``
    The relative standard deviation of the replicate retention-time measurements,
    i.e. ``std(RT) / mean(RT)``. Two problems: it is a deterministic function of
    the quantity being predicted, and it does not exist until the compound has
    actually been run on the instrument — so it could never be supplied at
    prediction time for a new molecule. See ``docs/corrections.md``; the original
    submission included this column in the pooled feature matrix.
"""

LEAKY_COLUMNS: list[str] = ["Retention Factor (k)", "RSD"]
"""The subset of :data:`NON_FEATURE_COLUMNS` excluded for leakage, not identity.

Used by ``scripts/quantify_leakage.py`` to restore the original feature
matrix. The normal pipeline does not include them.
"""

POOLED_DATASET_KEY: str = "Dataset_all"
"""Key under which the concatenation of all experiments is stored."""

GROUP_COLUMN: str = "Isomeric SMILES"
"""Column used to hold out entire molecules when splitting by compound.

Used by ``hplc-rt --split-by compound`` so that a molecule is not in both
train and test. The default split is still row-wise. See the README
limitations section.
"""

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------

TEST_SIZE: float = 0.2
INNER_CV_FOLDS: int = 5
OUTER_CV_FOLDS: int = 5

SCORING: dict[str, str] = {
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}
SELECTION_METRIC: str = "r2"
"""Metric the inner loop optimises when choosing hyperparameters."""

ALPHA_GRID: np.ndarray = np.logspace(-2, 2, 5)
"""Regularisation strengths tried for Ridge and Lasso: 0.01, 0.1, 1, 10, 100."""

PARAM_GRID_RIDGE: dict[str, np.ndarray] = {"ridge__alpha": ALPHA_GRID}
PARAM_GRID_LASSO: dict[str, np.ndarray] = {"lasso__alpha": ALPHA_GRID}
PARAM_GRID_RF: dict[str, list] = {
    "n_estimators": [50, 100],
    "max_features": [0.6, 1.0],
    "min_samples_leaf": [1, 5],
}

LINEAR_MAX_ITER: int = 10_000
ZERO_COEF_TOLERANCE: float = 1e-6
"""A Lasso coefficient below this magnitude counts as "eliminated"."""
