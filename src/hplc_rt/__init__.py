"""Predicting HPLC retention time from molecular structure and method conditions.

Example::

    from hplc_rt import pipeline
    output = pipeline.run(data_dir="data/raw")
    print(output.test_summary)

Or from the command line::

    hplc-rt evaluate --data-dir data/raw --output results/
    hplc-rt train --data-dir data/raw --out model.joblib
    hplc-rt predict --model model.joblib --smiles "CCO" --method <name>
"""

from __future__ import annotations

__version__ = "1.1.0"

from . import (
    config,
    curation,
    descriptors,
    evaluate,
    features,
    loading,
    models,
    pipeline,
    plots,
    predictor,
    splits,
)

__all__ = [
    "config",
    "curation",
    "descriptors",
    "evaluate",
    "features",
    "loading",
    "models",
    "pipeline",
    "plots",
    "predictor",
    "splits",
    "__version__",
]
