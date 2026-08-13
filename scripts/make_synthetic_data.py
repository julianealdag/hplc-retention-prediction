#!/usr/bin/env python3
"""Generate small synthetic workbooks matching the real Excel schema.

The real dataset is not redistributable here (see README, "Getting the data"), so
this script fabricates a handful of workbooks with the same two-sheet layout,
column names and quirks — including the newline-containing headers such as
``Compound \\nName``. It exists so the test suite can exercise the whole pipeline
without the real files.

The retention times are generated from a crude but monotone function of LogP so
that the models have a real signal to find; the numbers are not chemistry.

Usage:
    python scripts/make_synthetic_data.py --out data/synthetic --n-experiments 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COMPOUNDS: list[tuple[str, str]] = [
    ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("Diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("Benzene", "c1ccccc1"),
    ("Toluene", "Cc1ccccc1"),
    ("Phenol", "Oc1ccccc1"),
    ("Aniline", "Nc1ccccc1"),
    ("Naphthalene", "c1ccc2ccccc2c1"),
    ("Anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("Nicotine", "CN1CCCC1c1cccnc1"),
    ("Theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O"),
    ("Salicylic acid", "OC(=O)c1ccccc1O"),
    ("Estradiol", "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O"),
    ("Bisphenol A", "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"),
    ("Atrazine", "CCNc1nc(Cl)nc(NC(C)C)n1"),
    ("Carbamazepine", "NC(=O)N1c2ccccc2C=Cc2ccccc21"),
    ("Triclosan", "Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl"),
]

COLUMNS: list[tuple[str, str]] = [
    ("XBridge C18", "150 x 4.6 mm, 5 um"),
    ("Zorbax Eclipse XDB-C18", "100 x 2.1 mm, 1.8 um"),
    ("Kinetex Biphenyl", "50 x 3.0 mm, 2.6 um"),
]

MOBILE_PHASES_A: list[str] = [
    "Water + 0.1% formic acid",
    "Water + 10 mM ammonium formate",
    "Water:Methanol 90:10 + 0.1% formic acid",
]
MOBILE_PHASES_B: list[str] = [
    "Acetonitrile + 0.1% formic acid",
    "Methanol + 0.1% formic acid",
    "Acetonitrile",
]

RT_SHEET_COLUMNS: list[str] = [
    "MCMRT\nNumber", "Compound \nName", "IUPAC \nName", "Formula",
    "CAS\nNumber", "Pubchem \nNumber", "Isomeric SMILES", "InChI",
    "Retention Factor (k)", "RT (min)", "RSD",
]


def _rt_sheet(rng: np.random.Generator, n: int, dead_time: float) -> pd.DataFrame:
    """Build a synthetic RT sheet with the real column names."""
    from rdkit import Chem
    from rdkit.Chem import Crippen

    chosen = [COMPOUNDS[i] for i in rng.choice(len(COMPOUNDS), size=n, replace=False)]
    rows = []
    for idx, (name, smiles) in enumerate(chosen):
        logp = Crippen.MolLogP(Chem.MolFromSmiles(smiles))
        # Monotone in LogP, which is the real physical trend, plus noise.
        rt = float(2.0 + 1.8 * logp + rng.normal(0, 0.4))
        rt = max(rt, 0.3)
        rows.append({
            "MCMRT\nNumber": f"MCMRT{idx:04d}",
            "Compound \nName": f" {name} ",  # deliberate stray whitespace
            "IUPAC \nName": name.lower(),
            "Formula": "C0H0",
            "CAS\nNumber": f"{idx:05d}-00-0",
            "Pubchem \nNumber": 1000 + idx,
            "Isomeric SMILES": smiles,
            "InChI": f"InChI=1S/{name}",
            "Retention Factor (k)": (rt - dead_time) / dead_time,
            "RT (min)": rt,
            "RSD": float(abs(rng.normal(0, 0.05)) / max(rt, 0.3)),
        })
    return pd.DataFrame(rows, columns=RT_SHEET_COLUMNS)


def _lc_sheet(rng: np.random.Generator, index: int, dead_time: float) -> pd.DataFrame:
    """Build the LC setups sheet: metadata block, marker row, gradient table."""
    column, dimensions = COLUMNS[index % len(COLUMNS)]
    metadata = [
        ("Instrument", "Synthetic LC"),
        ("Instrument manufacturer", "Nobody"),
        ("Model type", "Model X"),
        ("Analytical column", column),
        ("Column dimensions", dimensions),
        ("Sample temperature (°C)", 10.0),
        ("Column temperature (°C)", float(25 + index % 3 * 5)),
        ("Dead time (min)", dead_time),
        ("Mobile phase A", MOBILE_PHASES_A[index % len(MOBILE_PHASES_A)]),
        ("Mobile phase B", MOBILE_PHASES_B[index % len(MOBILE_PHASES_B)]),
        ("Source", "synthetic"),
    ]
    rows: list[list] = [[key, value, None] for key, value in metadata]
    rows.append(["Gradient elution program", None, None])
    rows.append([None, "Time (min)", "Flow rate (mL/min)", "B (%)"])

    flow = round(float(rng.uniform(0.2, 0.6)), 2)
    breakpoints = [(0.0, flow, 5.0), (2.0, flow, 5.0), (12.0, flow, 95.0), (15.0, flow, 95.0)]
    for time, rate, pct_b in breakpoints:
        rows.append([None, time, rate, pct_b])

    width = max(len(r) for r in rows)
    padded = [r + [None] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded)


def make_workbook(path: Path, index: int, n_compounds: int, seed: int) -> None:
    """Write one synthetic ``.xlsx`` file."""
    rng = np.random.default_rng(seed + index)
    dead_time = round(float(rng.uniform(0.5, 1.5)), 2)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _rt_sheet(rng, n_compounds, dead_time).to_excel(
            writer, sheet_name="RT", index=False
        )
        _lc_sheet(rng, index, dead_time).to_excel(
            writer, sheet_name="LC setups", index=False, header=False
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--n-experiments", type=int, default=3)
    parser.add_argument("--n-compounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n_experiments):
        path = args.out / f"Dataset_{9000 + i}.xlsx"
        make_workbook(path, i, args.n_compounds, args.seed)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
