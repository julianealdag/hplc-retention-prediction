#!/usr/bin/env python3
"""Compare the pooled model with and without leaking columns (RSD, k).

Runs twice: corrected exclusions, then with the chosen leaking columns restored.
Prints the difference in R² and MAE. Needs the real files in ``data/raw/``.

The original coursework code leaked only ``RSD``; ``--columns RSD`` reproduces
that. The default restores both, which shows the effect of the retention factor.

Usage:
    python scripts/quantify_leakage.py --data-dir data/raw --columns RSD
    python scripts/quantify_leakage.py --data-dir data/raw --models RandomForest Ridge
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from hplc_rt import config, pipeline


def run_both(
    data_dir: Path, model_names: list[str], columns: list[str] | None = None
) -> pd.DataFrame:
    """Run the pooled model with and without leaking columns.

    Args:
        data_dir: Directory of ``.xlsx`` files.
        model_names: Models to compare.
        columns: Leaking columns to restore; defaults to all of
            :data:`config.LEAKY_COLUMNS`.

    Returns:
        One row per model, with corrected and original metrics side by side.
    """
    pooled_only = [config.POOLED_DATASET_KEY]

    logging.info("run 1/2: corrected — leaking columns excluded")
    corrected = pipeline.run(data_dir, model_names=model_names, datasets=pooled_only)

    columns = list(config.LEAKY_COLUMNS) if columns is None else columns
    with_leak = [c for c in config.NON_FEATURE_COLUMNS if c not in columns]
    logging.info("run 2/2: leaking columns restored (%s)", columns)
    original = pipeline.run(
        data_dir, model_names=model_names, datasets=pooled_only, exclusions=with_leak
    )

    def lookup(output, model: str) -> dict:
        rows = output.test_summary
        row = rows[(rows["Model"] == model)].iloc[0]
        return {"R2": row["R2_test"], "MAE": row["MAE_test"], "MSE": row["MSE_test"]}

    records = []
    for model in model_names:
        fixed, leaked = lookup(corrected, model), lookup(original, model)
        records.append(
            {
                "Model": model,
                "R2_corrected": fixed["R2"],
                "R2_with_leak": leaked["R2"],
                "R2_inflation": leaked["R2"] - fixed["R2"],
                "MAE_corrected": fixed["MAE"],
                "MAE_with_leak": leaked["MAE"],
                "MAE_understated_by": fixed["MAE"] - leaked["MAE"],
            }
        )

    n_corrected = len(corrected.splits[config.POOLED_DATASET_KEY].feature_names)
    n_original = len(original.splits[config.POOLED_DATASET_KEY].feature_names)
    logging.info("features: %d corrected vs %d with leak", n_corrected, n_original)
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--models", nargs="+", default=["RandomForest"])
    parser.add_argument(
        "--columns", nargs="+", choices=config.LEAKY_COLUMNS, default=None,
        help="leaking columns to restore (default: all; the original code leaked RSD)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    try:
        comparison = run_both(args.data_dir, args.models, args.columns)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    print("\n--- Effect of the leaking columns on the pooled model ---")
    print(comparison.to_string(index=False))
    print(
        "\nR2_inflation > 0 means the restored columns made the score optimistic."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output, index=False)
        print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
