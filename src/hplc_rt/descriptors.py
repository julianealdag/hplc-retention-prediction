"""Molecular descriptors computed from SMILES with RDKit.

Nine cheap 2D descriptors, chosen to span the properties that govern reversed-phase
retention: size (molecular weight, molar refractivity), lipophilicity (LogP),
polarity (TPSA, hydrogen-bond donors and acceptors), flexibility (rotatable bonds)
and shape/complexity (aromatic rings, Bertz index).

LogP dominates every feature-importance ranking in this project, which is the
expected result: reversed-phase HPLC separates largely by hydrophobicity.
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from . import config

logger = logging.getLogger(__name__)

# RDKit prints parse failures straight to stderr; route them through logging instead.
RDLogger.DisableLog("rdApp.*")

SMILES_COLUMN: str = "Isomeric SMILES"


def descriptors_for_smiles(smiles: str | None) -> pd.Series:
    """Compute the nine descriptors for one molecule.

    Args:
        smiles: SMILES string. Missing, empty and unparseable values yield a series
            of ``None`` rather than raising, so one bad structure does not abort a
            30-file run.

    Returns:
        Series indexed by :data:`config.DESCRIPTOR_COLUMNS`.

    Example:
        >>> descriptors_for_smiles("CCO")["MolWt"]  # doctest: +ELLIPSIS
        46.0...
    """
    empty = pd.Series([None] * len(config.DESCRIPTOR_COLUMNS), index=config.DESCRIPTOR_COLUMNS)

    if pd.isna(smiles) or not str(smiles).strip():
        return empty

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        logger.warning("unparseable SMILES, descriptors set to NaN: %s", smiles)
        return empty

    return pd.Series(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            Crippen.MolMR(mol),
            Descriptors.BertzCT(mol),
        ],
        index=config.DESCRIPTOR_COLUMNS,
    )


def add_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    """Append descriptor columns to a compound table.

    Descriptors are computed once per unique SMILES and then joined back, rather
    than once per row. The same ~340 compounds appear in all 30 experiments, so
    this is roughly a 30x saving on RDKit calls.

    Args:
        frame: Table containing a ``Isomeric SMILES`` column.

    Returns:
        A new frame with :data:`config.DESCRIPTOR_COLUMNS` appended.

    Raises:
        KeyError: If the SMILES column is absent.
    """
    if SMILES_COLUMN not in frame.columns:
        raise KeyError(f"{SMILES_COLUMN!r} column not found")

    unique = frame[SMILES_COLUMN].dropna().unique()
    lookup = pd.DataFrame(
        [descriptors_for_smiles(s) for s in unique],
        index=pd.Index(unique, name=SMILES_COLUMN),
    )

    joined = frame.reset_index(drop=True).join(
        lookup, on=SMILES_COLUMN, rsuffix="_descriptor"
    )
    n_failed = joined[config.DESCRIPTOR_COLUMNS[0]].isna().sum()
    if n_failed:
        logger.warning("%d rows have no descriptors (bad or missing SMILES)", n_failed)
    return joined
