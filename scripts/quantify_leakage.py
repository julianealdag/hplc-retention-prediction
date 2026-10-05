#!/usr/bin/env python3
"""Compare the pooled model with and without the leaking columns (RSD, k).

Runs twice: corrected exclusions, then the original coursework code feature set.
Prints the difference in R² and MAE. Needs the real files in ``data/raw/``.

Usage:
    python scripts/quantify_leakage.py --data-dir data/raw
    python scripts/quantify_leakage.py --data-dir data/raw --models RandomForest Ridge
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from hplc_rt import config, pipeline


def run_both(data_dir: Path, model_names: list[str]) -> pd.DataFrame:
    """Run the pooled model with and without the leaking columns.

    Args:
        data_dir: Directory of ``.xlsx`` files.
        model_names: Models to compare.

    Returns:
        One row per model, with corrected and original metrics side by side.
    """
    pooled_only = [config.POOLED_DATASET_KEY]

    logging.info("run 1/2: corrected — leaking columns excluded")
    corrected = pipeline.run(data_dir, model_names=model_names, datasets=pooled_only)

    # Drop leaky names from the exclusion list to match the original coursework code.
    with_leak = [c for c in config.NON_FEATURE_COLUMNS if c not in config.LEAKY_COLUMNS]
    logging.info("run 2/2: original — leaking columns restored (%s)", config.LEAKY_COLUMNS)
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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    try:
        comparison = run_both(args.data_dir, args.models)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    print("\n--- Effect of the leaking columns on the pooled model ---")
    print(comparison.to_string(index=False))
    print(
        "\nR2_inflation > 0 means the original result was optimistic by that much.\n"
        "Paste the corrected figures into README.md and docs/corrections.md."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output, index=False)
        print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
