"""Predicting HPLC retention time from molecular structure and method conditions.

Typical use::

    from hplc_rt import pipeline
    output = pipeline.run(data_dir="data/raw")
    print(output.test_summary)

Or from the command line::

    hplc-rt --data-dir data/raw --output results/
"""

from __future__ import annotations

__version__ = "1.0.0"

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
    "splits",
    "__version__",
]
