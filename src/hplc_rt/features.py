"""Turning experimental conditions into numeric features.

Three families of feature come out of this module:

1. **Mobile phase composition** — parsed out of free-text descriptions such as
   ``"water:methanol 90:10 + 0.1% formic acid"``.
2. **Gradient program** — resampled onto a fixed-length vector so that programs
   with different numbers of breakpoints become comparable.
3. **Instrument settings** — temperatures, dead time, and a one-hot encoding of
   the analytical column.

Together with the molecular descriptors from :mod:`hplc_rt.descriptors`, these
form the feature matrix for the pooled model.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from . import config
from .curation import normalize_text
from .loading import Experiment

logger = logging.getLogger(__name__)

RATIO_PATTERN = re.compile(r"(\d+):(\d+)")
FORMIC_ACID_PATTERN = re.compile(r"(\d*\.?\d+)%formicacid")
SALT_CONC_PATTERN = re.compile(r"(\d*\.?\d+)(mm|mmol)")

MOBILE_PHASE_FEATURES: tuple[str, ...] = (
    "Water",
    "Methanol",
    "ACN",
    "Ratio_Water",
    "Ratio_Methanol",
    "Ratio_ACN",
    "Acid_Formic",
    "Acid_Conc",
    "Salt_Formate",
    "Salt_Acetate",
    "Salt_Conc",
)


def parse_mobile_phase(description: str | None) -> dict[str, float]:
    """Extract numeric descriptors from a free-text mobile phase description.

    Handles the three solvents present in this dataset (water, methanol,
    acetonitrile), an optional volume ratio, formic acid as a modifier, and
    ammonium formate/acetate as buffer salts.

    Args:
        description: Free text, e.g. ``"Water:Methanol 90:10 + 0.1% formic acid"``.
            ``None`` and empty strings yield an all-zero descriptor.

    Returns:
        Mapping with the keys in :data:`MOBILE_PHASE_FEATURES`. Solvent presence is
        0/1; ratios are percentages; concentrations are in % (acid) and mM (salt).

    Example:
        >>> parse_mobile_phase("Water:Methanol 90:10 + 0.1% formic acid")["Ratio_Water"]
        90
    """
    text = normalize_text(description or "")
    if not isinstance(text, str):
        text = ""

    water = int("water" in text)
    methanol = int("methanol" in text)
    acn = int("acetonitrile" in text)

    ratio_water = ratio_methanol = ratio_acn = 0
    match = RATIO_PATTERN.search(text)
    if match:
        first, second = int(match.group(1)), int(match.group(2))
        # The ratio is written in the order the solvents are named.
        if water and methanol:
            ratio_water, ratio_methanol = first, second
        elif water and acn:
            ratio_water, ratio_acn = first, second
        elif methanol and acn:
            ratio_methanol, ratio_acn = first, second
    elif water:
        ratio_water = 100
    elif methanol:
        ratio_methanol = 100
    elif acn:
        ratio_acn = 100

    acid_conc = 0.0
    acid_match = FORMIC_ACID_PATTERN.search(text)
    if acid_match:
        acid_conc = float(acid_match.group(1))
    is_formic = int("formicacid" in text)

    salt_formate = salt_acetate = 0
    salt_conc = 0.0
    if "ammoniumformate" in text:
        salt_formate = 1
    elif "ammoniumacetate" in text:
        salt_acetate = 1
    if salt_formate or salt_acetate:
        salt_match = SALT_CONC_PATTERN.search(text)
        if salt_match:
            salt_conc = float(salt_match.group(1))

    return {
        "Water": water,
        "Methanol": methanol,
        "ACN": acn,
        "Ratio_Water": ratio_water,
        "Ratio_Methanol": ratio_methanol,
        "Ratio_ACN": ratio_acn,
        "Acid_Formic": is_formic,
        "Acid_Conc": acid_conc,
        "Salt_Formate": salt_formate,
        "Salt_Acetate": salt_acetate,
        "Salt_Conc": salt_conc,
    }


def gradient_column_names(resolution: int = config.GRADIENT_RESOLUTION) -> list[str]:
    """Names for the flattened gradient vector.

    Args:
        resolution: Number of resampling points.

    Returns:
        ``["HPLC_0_Rate", "HPLC_0_Comp", "HPLC_1_Rate", ...]`` — interleaved to
        match the flattening order in :func:`vectorize_gradient`.
    """
    return [f"HPLC_{i}_{feat}" for i in range(resolution) for feat in ("Rate", "Comp")]


def vectorize_gradient(
    gradient: pd.DataFrame,
    resolution: int = config.GRADIENT_RESOLUTION,
    time_max: float = config.GRADIENT_TIME_MAX,
) -> np.ndarray:
    """Resample a gradient program onto a fixed-length vector.

    A gradient is a list of programmed breakpoints — "at t=0 run 0.3 mL/min at 5%
    B, at t=12 ramp to 95% B" — and different experiments use different numbers of
    breakpoints. A regressor needs a fixed-width input, so each program is sampled
    at ``resolution`` evenly spaced times.

    Interpolation is step-wise (``kind="previous"``): the pump holds its last
    programmed setting until the next breakpoint, so a linear interpolation would
    invent ramps that the instrument never ran. After the last breakpoint the
    same hold applies — the final rate and %B continue to the end of the grid,
    they do not snap back to the initial values. (The original coursework code used a
    single ``fill_value`` for both sides of the range, which did exactly that;
    see ``docs/corrections.md``.)

    Args:
        gradient: Frame with :data:`config.GRADIENT_COLUMNS`.
        resolution: Number of sample points.
        time_max: Upper bound of the sampling grid, in minutes.

    Returns:
        Flat array of length ``2 * resolution``, interleaved rate/composition.

    Raises:
        ValueError: If the gradient table is empty.
    """
    if gradient.empty:
        raise ValueError("gradient program is empty")

    times = gradient["Time (min)"].tolist()
    rates = gradient["Flow rate (mL/min)"].tolist()
    comps = gradient["B (%)"].tolist()

    grid = np.linspace(0, time_max, resolution)
    # (below first breakpoint, after last breakpoint). After the last programmed
    # time the pump holds the final setting; a scalar fill_value would reuse the
    # *initial* setting for the tail of the grid.
    rate_interp = interp1d(
        times, rates, kind="previous", bounds_error=False,
        fill_value=(rates[0], rates[-1]),
    )(grid)
    comp_interp = interp1d(
        times, comps, kind="previous", bounds_error=False,
        fill_value=(comps[0], comps[-1]),
    )(grid)

    return np.stack([rate_interp, comp_interp], axis=1).flatten()


def attach_condition_features(experiment: Experiment) -> None:
    """Broadcast this experiment's conditions onto every compound row, in place.

    Within one experiment the chromatographic conditions are constant, so each
    feature is a single value repeated down the column. It only becomes
    informative once experiments are pooled.

    Args:
        experiment: Modified in place.
    """
    lc = experiment.lc

    for prefix, key in (("MPA", "Mobile phase A"), ("MPB", "Mobile phase B")):
        parsed = parse_mobile_phase(lc[key][0])
        for name, value in parsed.items():
            experiment.rt[f"{prefix}_{name}"] = value

    experiment.rt["AnalyticalColumn"] = lc["Analytical column"][0]
    experiment.rt["SampleTemperature"] = lc["Sample temperature (°C)"][0]
    experiment.rt["ColumnTemperature"] = lc["Column temperature (°C)"][0]
    experiment.rt["DeadTime"] = lc["Dead time (min)"][0]

    gradient_vector = vectorize_gradient(experiment.gradient)
    gradient_frame = pd.DataFrame(
        np.tile(gradient_vector, (len(experiment.rt), 1)),
        columns=gradient_column_names(),
        index=experiment.rt.index,
    )
    experiment.rt = pd.concat([experiment.rt, gradient_frame], axis=1)


def build_features(experiments: dict[str, Experiment]) -> dict[str, Experiment]:
    """Attach condition features to every experiment.

    Args:
        experiments: Mapping from name to experiment; modified in place.

    Returns:
        The same mapping, for chaining.
    """
    for experiment in experiments.values():
        attach_condition_features(experiment)
    logger.info("attached condition features to %d experiments", len(experiments))
    return experiments


def pool_experiments(experiments: dict[str, Experiment]) -> pd.DataFrame:
    """Concatenate every experiment into one frame and one-hot the column type.

    Args:
        experiments: Experiments that have already been through
            :func:`build_features`.

    Returns:
        The pooled frame, with a ``Dataset`` column recording provenance and the
        analytical column replaced by ``Col_*`` indicator columns.
    """
    pooled = pd.concat(
        [exp.rt.assign(Dataset=name) for name, exp in experiments.items()],
        ignore_index=True,
    )
    pooled = pd.get_dummies(pooled, columns=["AnalyticalColumn"], prefix="Col")

    bool_columns = pooled.select_dtypes(include="bool").columns
    pooled[bool_columns] = pooled[bool_columns].astype(int)

    logger.info("pooled frame: %d rows x %d columns", *pooled.shape)
    return pooled
