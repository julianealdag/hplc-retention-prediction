# Predicting HPLC retention time across variable chromatographic conditions

[![Tests](https://github.com/julianealdag/hplc-retention-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/julianealdag/hplc-retention-prediction/actions/workflows/ci.yml)

Retention time in reversed-phase HPLC depends on the molecule and on the method
(solvent composition, gradient, column, temperature). A model trained on one
fixed method often fails when the conditions change. This project asks whether
one model can predict elution time across many methods.

The models are trained on 10,073 measured retention times from 30 reversed-phase
LC methods (343 compounds).

The pipeline is a rewrite of earlier coursework code. Five bugs found along
the way, including target leakage through the `RSD` column, are documented and
fixed in [`docs/corrections.md`](docs/corrections.md). Results from the
corrected pipeline are being regenerated; see [Results](#results).

## Data

[MCMRT](https://doi.org/10.1038/s41597-024-03780-5) (Zhang *et al.*,
*Scientific Data* 2024): 343 small molecules measured under 30 reversed-phase
LC methods.

| | |
|---|---|
| Retention time measurements | 10,073 |
| Distinct molecules | 343 |
| Chromatographic methods | 30 |
| Compounds per method | 330–343 |
| Features, per-method model | 9 |
| Features, pooled model | 240 |

Not every molecule appears in every method.

## Features

**Molecular (RDKit, 2D):** molecular weight, LogP, TPSA, rotatable bonds,
H-bond donors and acceptors, aromatic rings, molar refractivity, Bertz
complexity. These cover size, lipophilicity, polarity, flexibility and shape.

**Method** (used only in the pooled model):

- Mobile phase: parsed from text such as `"Water:Methanol 90:10 + 0.1% formic acid"`
  into solvent indicators, volume ratios, and modifier/buffer concentrations.
- Gradient: variable-length (time, flow, %B) breakpoints, resampled onto a
  fixed 100-point grid (200 values). Interpolation is step-wise
  (`kind="previous"`), matching how the pump holds a setting until the next
  breakpoint.
- Instrument: column and sample temperature, dead time, and a one-hot encoding
  of the analytical column.

## Method

**Per-method models** (30): only the nine molecular descriptors. Conditions are
constant within one method, so they add no information. Question: how well does
structure alone predict retention for a fixed method?

**Pooled model:** descriptors plus method features, all 10,073 rows. Question:
can one model transfer across methods?

Regressors: Ridge, Lasso, Random Forest. Ridge and Lasso are used with
`StandardScaler` because the penalties are scale-sensitive and so that
coefficients can be compared. The forest is not scaled.

Hyperparameters are chosen with nested cross-validation (inner 5-fold for
tuning, outer 5-fold for scoring). A dummy regressor that predicts the training
mean is the baseline.

## Results

Results from the corrected pipeline are being regenerated. Earlier numbers came
from code affected by the issues in [`docs/corrections.md`](docs/corrections.md)
(most importantly the `RSD` leak) and are not reported here.

The new results will compare three ways of splitting the data:

- **Row-wise:** random rows. The same molecule can be in training under one
  method and in test under another.
- **Compound-held-out** (`--split-by compound`): whole molecules are held out.
  This matches the identification use case: a new compound on a known method.
- **Method-held-out:** whole methods are held out. This tests the project's
  actual question, transfer to an unseen method (planned).

## Limitations

**Row-wise split.** The default split is random over rows, not over compounds,
so it overstates how well the model handles new molecules. Use
`--split-by compound` for a compound-disjoint split.

**Column chemistry** is only a one-hot identity, so the model cannot generalise
to columns that were not in the training data.

**Descriptors are 2D.** Shape and conformation are not represented.

**Thirty reversed-phase methods.** All are reversed-phase, so they may be more
similar than the count implies.

## Getting the data

The dataset is not included. Download it from:

> Zhang, Y., Liu, F., Li, X.Q., Gao, Y., Li, K.C., Zhang, Q.H. Retention time
> dataset for heterogeneous molecules in reversed-phase liquid chromatography.
> *Scientific Data* **11**, 946 (2024). https://doi.org/10.1038/s41597-024-03780-5

Put the 30 `.xlsx` files in `data/raw/`. Each file needs an `RT` sheet and an
`LC setups` sheet (metadata, then a `Gradient elution program` marker, then the
gradient table).

Without the real files:

```bash
python scripts/make_synthetic_data.py --out data/synthetic --n-experiments 3
hplc-rt --data-dir data/synthetic
```

Synthetic retention times are a function of LogP used to test the code, not
real measurements.

## Install and run

```bash
git clone https://github.com/julianealdag/hplc-retention-prediction.git
cd hplc-retention-prediction
pip install -e .
```

Python 3.10+.

```bash
# 30 per-method models plus the pooled model, 3 regressors each
hplc-rt --data-dir data/raw --output results/

# pooled model only
hplc-rt --data-dir data/raw --pooled-only

# one regressor
hplc-rt --data-dir data/raw --models RandomForest

# hold out entire molecules
hplc-rt --data-dir data/raw --split-by compound --pooled-only
```

```python
from hplc_rt import pipeline

output = pipeline.run(data_dir="data/raw")
print(output.test_summary)
```

A full run takes about an hour (mostly Random Forest grid search).
`--pooled-only` takes a few minutes.

```bash
pip install -e ".[dev]"
pytest          # 43 tests, about 1 min, synthetic data
```

Some tests check that the issues in
[`docs/corrections.md`](docs/corrections.md) do not return.

```bash
python scripts/quantify_leakage.py --data-dir data/raw --models RandomForest
```

## Layout

```
src/hplc_rt/          pipeline (load, features, models, plots, CLI)
scripts/              synthetic data and leakage comparison
tests/                pytest suite
docs/                 corrections to the original coursework code
data/                 download instructions (raw files are gitignored)
```

## About this repository

This project started as a group coursework project for **Digital Chemistry
(FS2025) at ETH Zürich** in June 2025. I have continued it on my own since then.
The code in this repository is that later work: the installable package, the
corrected pipeline, the tests, the command-line interface and
[`docs/corrections.md`](docs/corrections.md).

## Acknowledgements

The package, tests and documentation were developed with the help of [Claude Code](https://claude.com/claude-code) (Anthropic) as an agentic coding assistant.

## References

1. Zhang, Y., Liu, F., Li, X.Q., Gao, Y., Li, K.C. & Zhang, Q.H. Retention time
   dataset for heterogeneous molecules in reversed-phase liquid chromatography.
   *Scientific Data* **11**, 946 (2024).
   [doi:10.1038/s41597-024-03780-5](https://doi.org/10.1038/s41597-024-03780-5)
2. Landrum, G. *et al.* RDKit: Open-source cheminformatics toolkit.
   [rdkit.org](https://www.rdkit.org/)
3. Pedregosa, F. *et al.* Scikit-learn: Machine learning in Python. *JMLR* **12**,
   2825–2830 (2011).

## License

MIT, see [LICENSE](LICENSE). The MCMRT dataset has a separate licence from its
authors.
