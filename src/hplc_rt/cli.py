"""Command-line entry point: ``hplc-rt``.

Three subcommands:

``evaluate``
    Nested cross-validation and held-out scores (the default, so
    ``hplc-rt --data-dir data/raw`` still works).
``train``
    Fit the pooled model on all data and save it to a file.
``predict``
    Predict retention times for SMILES under the methods in a saved model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import config, models, pipeline, plots, predictor, tracking

COMMANDS: tuple[str, ...] = ("evaluate", "train", "predict")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with its three subcommands."""
    parser = argparse.ArgumentParser(
        prog="hplc-rt",
        description="Predict HPLC retention time from structure and method conditions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser(
        "evaluate", help="cross-validate and score models (default)",
        description="Nested cross-validation and held-out test scores.",
    )
    _add_data_dir(evaluate)
    evaluate.add_argument(
        "--output", type=Path, default=config.RESULTS_DIR,
        help="where to write result tables and figures (default: %(default)s)",
    )
    evaluate.add_argument(
        "--models", nargs="+", default=list(models.MODEL_FACTORIES),
        choices=list(models.MODEL_FACTORIES),
        help="models to run (default: all)",
    )
    evaluate.add_argument(
        "--datasets", nargs="+", default=None,
        help="restrict to these datasets, e.g. Dataset_all",
    )
    evaluate.add_argument(
        "--pooled-only", action="store_true",
        help=f"shorthand for --datasets {config.POOLED_DATASET_KEY}",
    )
    evaluate.add_argument(
        "--split-by", choices=list(config.SPLIT_COLUMNS), default="row",
        help="row: random over measurements (optimistic). "
             "compound: hold out entire molecules. "
             "method: hold out entire methods; pooled model only "
             "(default: %(default)s)",
    )
    evaluate.add_argument(
        "--no-figures", action="store_true", help="skip figure generation",
    )
    _add_wandb(evaluate)
    _add_verbose(evaluate)

    train = sub.add_parser(
        "train", help="fit the pooled model on all data and save it",
        description="Tune and fit the pooled model on every row, then save it.",
    )
    _add_data_dir(train)
    train.add_argument(
        "--model-type", default="RandomForest", choices=list(models.MODEL_FACTORIES),
        help="regressor to train (default: %(default)s)",
    )
    train.add_argument(
        "--out", type=Path, default=Path("model.joblib"),
        help="model file to write (default: %(default)s)",
    )
    _add_wandb(train)
    _add_verbose(train)

    predict = sub.add_parser(
        "predict", help="predict retention times with a saved model",
        description="Predict retention times for molecules under the methods in a "
                    "saved model.",
    )
    predict.add_argument(
        "--model", type=Path, required=True, help="model file from 'hplc-rt train'",
    )
    source = predict.add_mutually_exclusive_group()
    source.add_argument("--smiles", nargs="+", help="one or more SMILES strings")
    source.add_argument("--input", type=Path, help="CSV file with a SMILES column")
    predict.add_argument(
        "--smiles-column", default="SMILES",
        help="column to read from --input (default: %(default)s)",
    )
    predict.add_argument(
        "--method", nargs="+", default=None,
        help="methods to predict for (default: every method in the model)",
    )
    predict.add_argument("--output", type=Path, help="write predictions to this CSV")
    predict.add_argument(
        "--list-methods", action="store_true", help="list the model's methods and exit",
    )
    _add_verbose(predict)
    return parser


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir", type=Path, default=config.DATA_DIR,
        help="directory of .xlsx experiment files (default: %(default)s)",
    )


def _add_wandb(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb", action="store_true",
        help="log this run to Weights & Biases (needs the wandb extra and a login)",
    )
    parser.add_argument(
        "--wandb-project", default=tracking.DEFAULT_PROJECT,
        help="W&B project name (default: %(default)s)",
    )


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="debug-level logging",
    )


def main(argv: list[str] | None = None) -> int:
    """Run a subcommand from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``. Without a subcommand,
            ``evaluate`` is assumed.

    Returns:
        Process exit code — 0 on success, 1 on a missing file or bad input.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in (*COMMANDS, "-h", "--help"):
        argv = ["evaluate", *argv]
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    handlers = {"evaluate": _evaluate, "train": _train, "predict": _predict}
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ImportError, KeyError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _evaluate(args: argparse.Namespace) -> int:
    datasets = args.datasets
    if args.pooled_only:
        datasets = [config.POOLED_DATASET_KEY]

    group_by = config.SPLIT_COLUMNS[args.split_by]
    run = None
    if args.wandb:
        run = tracking.start_run(
            "evaluate",
            {
                "split_by": args.split_by,
                "models": args.models,
                "datasets": datasets or "all",
                "data_dir": str(args.data_dir),
            },
            project=args.wandb_project,
            name=f"evaluate-{args.split_by}" + ("-pooled" if args.pooled_only else ""),
            tags=[args.split_by],
        )
    output = pipeline.run(
        data_dir=args.data_dir,
        model_names=args.models,
        datasets=datasets,
        group_by=group_by,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    output.cv_summary.to_csv(args.output / "cv_summary.csv", index=False)
    output.test_summary.to_csv(args.output / "test_summary.csv", index=False)
    output.fold_scores.to_csv(args.output / "fold_scores.csv", index=False)
    output.predictions.to_csv(args.output / "predictions.csv", index=False)
    print(output.test_summary.to_string(index=False))

    if not args.no_figures:
        figures_dir = args.output / "figures"
        plots.save(
            plots.model_comparison(output.test_summary, "R2_test", include_dummy=False),
            "r2_comparison", figures_dir,
        )
        plots.save(
            plots.model_comparison(output.test_summary, "MAE_test"),
            "mae_comparison", figures_dir,
        )
        _save_pooled_figures(output, figures_dir)
        print(f"\nfigures written to {figures_dir}")

    print(f"tables written to {args.output}")
    if run is not None:
        figures = None if args.no_figures else args.output / "figures"
        tracking.log_results(
            run, output.cv_summary, output.test_summary, figures,
            fold_scores=output.fold_scores, predictions=output.predictions,
        )
        run.finish()
    return 0


def _save_pooled_figures(output: pipeline.PipelineOutput, figures_dir: Path) -> None:
    """Predicted-vs-measured per model and per-fold scores for the pooled model."""
    pooled = config.POOLED_DATASET_KEY
    predictions = output.predictions
    if not predictions.empty:
        for model_name, rows in predictions[predictions["Dataset"] == pooled].groupby("Model"):
            plots.save(
                plots.predicted_vs_measured_by_method(
                    rows, title=f"{model_name}, pooled model, test set"
                ),
                f"pooled_{model_name}_predicted_vs_measured", figures_dir,
            )
    folds = output.fold_scores
    if not folds.empty and (folds["Dataset"] == pooled).any():
        plots.save(
            plots.fold_scores(folds[folds["Dataset"] == pooled]),
            "pooled_fold_scores_r2", figures_dir,
        )


def _train(args: argparse.Namespace) -> int:
    model = predictor.train(args.data_dir, model_name=args.model_type)
    path = model.save(args.out)
    print(
        f"{model.model_name} trained on {model.n_rows} rows from "
        f"{len(model.methods)} methods, saved to {path}"
    )
    if args.wandb:
        summary = {
            "model_type": model.model_name,
            "best_params": {k: str(v) for k, v in model.best_params.items()},
            "n_rows": model.n_rows,
            "n_methods": len(model.methods),
            "n_features": len(model.feature_names),
        }
        run = tracking.start_run(
            "train", summary, project=args.wandb_project,
            name=f"train-{model.model_name}",
        )
        tracking.log_model(run, path, summary)
        run.finish()
    return 0


def _predict(args: argparse.Namespace) -> int:
    model = predictor.TrainedModel.load(args.model)
    if args.list_methods:
        print("\n".join(model.methods))
        return 0

    if args.smiles:
        smiles = args.smiles
    elif args.input:
        table = pd.read_csv(args.input)
        if args.smiles_column not in table.columns:
            raise KeyError(f"column {args.smiles_column!r} not found in {args.input}")
        smiles = table[args.smiles_column].tolist()
    else:
        raise ValueError("give molecules with --smiles or --input")

    predictions = model.predict(smiles, args.method)
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"{len(predictions)} predictions written to {args.output}")
    else:
        print(predictions.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
