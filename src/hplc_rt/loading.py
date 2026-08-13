"""Reading the raw Excel workbooks into memory.

Each experiment is one ``.xlsx`` file with two sheets:

``RT``
    One row per compound — identifiers (name, SMILES, InChI, PubChem number) plus
    the measured retention time.

``LC setups``
    A key/value block describing the instrument and column, followed by a marker
    row (``Gradient elution program``) and then the gradient table itself.

The awkward part is that the second sheet mixes two different shapes in one grid,
so it is read with ``header=None`` and split at the marker row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """One HPLC experiment: its compounds, its instrument setup, its gradient.

    Attributes:
        name: Base filename, e.g. ``Dataset_2010``.
        rt: Compound table from the ``RT`` sheet, one row per compound.
        lc: Single-row frame of instrument/column metadata.
        gradient: Gradient elution program with columns
            ``Time (min)``, ``Flow rate (mL/min)``, ``B (%)``.
    """

    name: str
    rt: pd.DataFrame
    lc: pd.DataFrame
    gradient: pd.DataFrame


def _split_lc_sheet(lc_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the ``LC setups`` sheet into its metadata block and gradient table.

    Args:
        lc_raw: The sheet read with ``header=None``, so column 0 holds the keys.

    Returns:
        ``(lc, gradient)`` — a one-row metadata frame and the gradient program.

    Raises:
        ValueError: If the ``Gradient elution program`` marker row is absent.
    """
    marker = lc_raw[lc_raw[0] == config.GRADIENT_HEADER].index
    if len(marker) == 0:
        raise ValueError(f"no {config.GRADIENT_HEADER!r} marker row found in LC sheet")
    marker_idx = marker[0]

    # Metadata: everything above the marker, as key -> value.
    lc_info = lc_raw.iloc[:marker_idx]
    lc = pd.DataFrame([dict(zip(lc_info[0], lc_info[1], strict=False))])

    # Gradient: everything below the marker. Drop the marker row itself, then the
    # header row beneath it, then the empty first column.
    gradient = lc_raw.iloc[marker_idx + 1 :].reset_index(drop=True)
    gradient = gradient.dropna(how="all")
    gradient = gradient.iloc[1:].copy().reset_index(drop=True)
    gradient = gradient.iloc[:, 1:]
    gradient.columns = config.GRADIENT_COLUMNS
    return lc, gradient


def load_experiment(path: Path) -> Experiment:
    """Read a single workbook.

    Args:
        path: Path to the ``.xlsx`` file.

    Returns:
        The parsed :class:`Experiment`.

    Raises:
        ValueError: If a required sheet or the gradient marker is missing.
    """
    xls = pd.ExcelFile(path)
    rt = xls.parse(config.SHEET_RT)
    lc_raw = xls.parse(config.SHEET_LC, header=None)
    lc, gradient = _split_lc_sheet(lc_raw)
    return Experiment(name=path.stem, rt=rt, lc=lc, gradient=gradient)


def load_all(data_dir: Path | str | None = None) -> dict[str, Experiment]:
    """Read every workbook in a directory.

    Files that fail to parse are logged and skipped rather than aborting the run —
    with 30 files, one malformed workbook should not cost you the other 29.

    Args:
        data_dir: Directory of ``.xlsx`` files. Defaults to :data:`config.DATA_DIR`.

    Returns:
        Mapping from experiment name to :class:`Experiment`, sorted by name so that
        runs are reproducible regardless of filesystem ordering.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no workbooks.
    """
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"data directory not found: {data_dir}\n"
            "See README section 'Getting the data'."
        )

    paths = sorted(data_dir.glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"no .xlsx files in {data_dir}")

    experiments: dict[str, Experiment] = {}
    for path in paths:
        try:
            experiments[path.stem] = load_experiment(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            logger.warning("skipping %s: %s", path.name, exc)

    logger.info("loaded %d of %d workbooks", len(experiments), len(paths))
    return experiments
