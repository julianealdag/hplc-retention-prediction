# Predicting HPLC retention time across variable chromatographic conditions

[![Tests](https://github.com/julianealdag/hplc-retention-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/julianealdag/hplc-retention-prediction/actions/workflows/ci.yml)

Retention time in reversed-phase HPLC depends on the molecule and on the method
(solvent composition, gradient, column, temperature). A model trained on one
fixed method often fails when the conditions change. This project asks whether
one model can predict elution time across many methods.

The models are trained on 10,073 measured retention times from 30 reversed-phase
LC methods (343 compounds).

**Result (original submission):** Random Forest on the pooled data reached
**R² = 0.955, MAE = 1.09 min** on a held-out test set, compared with 7.25 min
MAE for a mean baseline and 3.16 min for Ridge/Lasso. Linear models do not
capture the nonlinear retention behaviour.

Five issues found after submission, including target leakage (`RSD`), are
documented in [`docs/corrections.md`](docs/corrections.md). The pipeline in
`src/` excludes the leaking column. The numbers above are from the original
notebook and are therefore optimistic.

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

Held-out test set: 20% of rows, not used in training or tuning.

**Pooled model** (all 30 methods):

| Model | R² | MAE (min) | MSE (min²) |
|---|---|---|---|
| **Random Forest** | **0.955** | **1.09** | **4.84** |
| Ridge | 0.792 | 3.16 | 22.31 |
| Lasso | 0.792 | 3.16 | 22.35 |
| Dummy (mean) | 0.000 | 7.25 | 107.40 |

**Mean over the 30 per-method models:**

| Model | R² | MAE (min) |
|---|---|---|
| **Random Forest** | **0.797** | **1.93** |
| Lasso | 0.701 | 2.48 |
| Ridge | 0.689 | 2.50 |

Ridge and Lasso are almost identical on the pooled data (within 0.001 R²), which
points to a linear model class hitting a nonlinear problem rather than to
overfitting.

Pooling helps the forest (R² 0.797 → 0.955) and barely helps the linear models
(0.69 → 0.79), while their MAE gets worse (2.50 → 3.16 min) because the pooled
retention range is wider.

LogP is the strongest feature in every model and both settings, which matches
reversed-phase HPLC (separation mainly by hydrophobicity).

![Model comparison on held-out test sets](docs/figures/test_comparison_r2.png)

*R² on held-out test data for each of the 30 methods and for the pooled dataset
(leftmost). Green: Random Forest, red: Lasso, blue: Ridge.*

All 23 figures from the submitted notebook are in
[`docs/figures/`](docs/figures/).

## Limitations

**Row-wise split.** The default split is random over rows, not over compounds.
The same molecule can appear in train under one method and in test under
another. `hplc-rt --split-by compound` holds out entire molecules. The numbers
above use the original row-wise split.

**`RSD` leakage.** See
[`docs/corrections.md`](docs/corrections.md#1-rsd-was-used-as-a-predictor--target-leakage).
The pooled results above are optimistic. `scripts/quantify_leakage.py` measures
the difference if the real data are available.

**Gradient tail.** After the last breakpoint the original interpolator reused
the initial flow/%B instead of holding the final setting. See
[`docs/corrections.md`](docs/corrections.md#5-gradient-vectors-snapped-back-to-the-initial-setting-after-the-last-breakpoint).
The current code holds the final setting.

**Column chemistry** is only a one-hot identity, so the model cannot generalise
to columns that were not in the training data.

**Descriptors are 2D.** Shape and conformation are not represented.

**Thirty reversed-phase methods.** LogP dominance suggests they are more similar
than the count implies.

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
notebooks/            original submission (June 2025)
docs/                 corrections, figures, course report
data/                 download instructions (raw files are gitignored)
```

## About this repository

This started as coursework for **Digital Chemistry (FS2025) at ETH Zürich**,
submitted June 2025 by **Juliane Aldag** and two fellow students. The results above are from that submission and are joint work.

In the team I was responsible for the train/test split, feature
standardisation, and nested cross-validation. Data curation and feature
engineering were shared. A teammate led model evaluation and feature-importance
analysis.

I have continued the project on my own since then. The installable package,
tests, command-line interface, and the notes in
[`docs/corrections.md`](docs/corrections.md) are that later work. The original
notebook is in `notebooks/`.

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
