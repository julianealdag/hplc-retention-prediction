"""Command-line entry point: ``hplc-rt``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config, models, pipeline, plots


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hplc-rt",
        description="Predict HPLC retention time from structure and method conditions.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=config.DATA_DIR,
        help="directory of .xlsx experiment files (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=config.RESULTS_DIR,
        help="where to write result tables and figures (default: %(default)s)",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(models.MODEL_FACTORIES),
        choices=list(models.MODEL_FACTORIES),
        help="models to run (default: all)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="restrict to these datasets, e.g. Dataset_all",
    )
    parser.add_argument(
        "--pooled-only", action="store_true",
        help=f"shorthand for --datasets {config.POOLED_DATASET_KEY}",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="skip figure generation",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="debug-level logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code — 0 on success, 1 if the data directory is missing.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    datasets = args.datasets
    if args.pooled_only:
        datasets = [config.POOLED_DATASET_KEY]

    try:
        output = pipeline.run(
            data_dir=args.data_dir, model_names=args.models, datasets=datasets
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    output.cv_summary.to_csv(args.output / "cv_summary.csv", index=False)
    output.test_summary.to_csv(args.output / "test_summary.csv", index=False)
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
        print(f"\nfigures written to {figures_dir}")

    print(f"tables written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
