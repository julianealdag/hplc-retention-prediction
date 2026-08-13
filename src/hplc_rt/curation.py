"""Cleaning the raw tables before any feature is computed.

The workbooks were assembled by hand over time, so the same analytical column
appears as ``"XBridge C18"``, ``"xbridge c18"`` and ``"XBridge C18"`` (with a
non-breaking space). Left alone, one-hot encoding would treat those as three
different columns. Normalising text is therefore not cosmetic — it changes the
feature matrix.
"""

from __future__ import annotations

import logging

import pandas as pd

from .loading import Experiment

logger = logging.getLogger(__name__)

COMPOUND_NAME_CANDIDATES: tuple[str, ...] = ("Compound \nName", "Compound Name")
CAS_COLUMN: str = "CAS\nNumber"


def normalize_text(value: object) -> object:
    """Collapse a text value to a comparable form.

    Strips surrounding whitespace, lowercases, removes *all* internal spaces and
    replaces non-breaking spaces. Non-string and missing values pass through
    unchanged so the function is safe to apply across a whole frame.

    Args:
        value: Any cell value.

    Returns:
        The normalised string, or the input unchanged if it was null.

    Example:
        >>> normalize_text("  XBridge C18 ")
        'xbridgec18'
    """
    if pd.isnull(value):
        return value
    return str(value).strip().lower().replace(" ", "").replace(" ", "")


def strip_compound_names(experiment: Experiment) -> None:
    """Strip stray whitespace from the compound-name column, in place.

    Compound names are the key used to check which compounds appear in which
    experiment, so trailing spaces cause spurious mismatches.

    Args:
        experiment: Modified in place.
    """
    for candidate in COMPOUND_NAME_CANDIDATES:
        if candidate in experiment.rt.columns:
            experiment.rt[candidate] = experiment.rt[candidate].astype(str).str.strip()
            return
    logger.warning("%s: no compound-name column found", experiment.name)


def drop_cas_number(experiment: Experiment) -> None:
    """Remove the CAS registry number column, in place.

    A CAS number is an arbitrary registry identifier with no physical meaning; it
    would be noise at best and a leak-by-proxy at worst if compounds were
    registered in an order correlated with the target.

    Args:
        experiment: Modified in place.
    """
    if CAS_COLUMN in experiment.rt.columns:
        experiment.rt = experiment.rt.drop(columns=[CAS_COLUMN])


def normalize_lc_metadata(experiment: Experiment) -> None:
    """Normalise every cell of the LC metadata frame, in place.

    Args:
        experiment: Modified in place.
    """
    experiment.lc = experiment.lc.map(normalize_text)


def curate(experiments: dict[str, Experiment]) -> dict[str, Experiment]:
    """Apply every curation step to every experiment.

    Args:
        experiments: Mapping from name to experiment; modified in place.

    Returns:
        The same mapping, for chaining.
    """
    for experiment in experiments.values():
        strip_compound_names(experiment)
        drop_cas_number(experiment)
        normalize_lc_metadata(experiment)
    logger.info("curated %d experiments", len(experiments))
    return experiments
