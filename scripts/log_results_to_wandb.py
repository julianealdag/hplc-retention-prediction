#!/usr/bin/env python3
"""Upload results of an earlier ``hplc-rt evaluate`` run to Weights & Biases.

For runs made without ``--wandb``: reads ``cv_summary.csv``, ``test_summary.csv``
and ``figures/`` from an output directory and logs them as a new W&B run, so
long runs do not have to be repeated. Optionally uploads a trained model too.

Usage:
    python scripts/log_results_to_wandb.py results/pooled_row --split-by row
    python scripts/log_results_to_wandb.py --model results/model.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from hplc_rt import config, predictor, tracking


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, nargs="?", help="evaluate output directory")
    parser.add_argument("--split-by", choices=list(config.SPLIT_COLUMNS), default="row")
    parser.add_argument("--model", type=Path, help="model file from 'hplc-rt train'")
    parser.add_argument("--project", default=tracking.DEFAULT_PROJECT)
    args = parser.parse_args()
    if args.results is None and args.model is None:
        parser.error("give a results directory, --model, or both")

    if args.results is not None:
        cv = pd.read_csv(args.results / "cv_summary.csv")
        test = pd.read_csv(args.results / "test_summary.csv")
        pooled_only = set(test["Dataset"]) == {config.POOLED_DATASET_KEY}
        run = tracking.start_run(
            "evaluate",
            {
                "split_by": args.split_by,
                "models": sorted(set(test["Model"]) - {"Dummy"}),
                "datasets": [config.POOLED_DATASET_KEY] if pooled_only else "all",
                "source": str(args.results),
                "backfilled": True,
            },
            project=args.project,
            name=f"evaluate-{args.split_by}" + ("-pooled" if pooled_only else ""),
            tags=[args.split_by, "backfilled"],
        )
        optional = {
            name: pd.read_csv(args.results / f"{name}.csv")
            for name in ("fold_scores", "predictions")
            if (args.results / f"{name}.csv").exists()
        }
        tracking.log_results(run, cv, test, args.results / "figures", **optional)
        run.finish()

    if args.model is not None:
        model = predictor.TrainedModel.load(args.model)
        summary = {
            "model_type": model.model_name,
            "best_params": {k: str(v) for k, v in model.best_params.items()},
            "n_rows": model.n_rows,
            "n_methods": len(model.methods),
            "n_features": len(model.feature_names),
            "backfilled": True,
        }
        run = tracking.start_run(
            "train", summary, project=args.project, name=f"train-{model.model_name}",
            tags=["backfilled"],
        )
        tracking.log_model(run, args.model, summary)
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
