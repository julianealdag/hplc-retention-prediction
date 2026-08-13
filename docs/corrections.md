# Corrections to the original submission

The code in `src/` is a refactor of the notebook submitted for assessment in June
2025 (`notebooks/original_submission.ipynb`, preserved unchanged). While
restructuring it, four defects came to light. They are listed here rather than
quietly fixed, so that the submitted notebook, the report, and this package can
each be read against a clear record of what differs.

Three of the four affect the pooled `Dataset_all` model. The 30 per-experiment
models use only the nine molecular descriptors and are unaffected by items 1–3.

---

## 1. `RSD` was used as a predictor — target leakage

**Severity: high. Affects the headline pooled result.**

`RSD` is one of the columns in the source dataset's `RT` sheet. The dataset paper
defines it as the relative standard deviation across three replicate analyses of
the same molecule under the same conditions — that is, `std(RT) / mean(RT)` over
replicates.

Two things follow. First, it is a deterministic function of the quantity being
predicted: the denominator *is* the target. Where the absolute replicate scatter
is roughly instrument-limited and similar across compounds, `RSD ≈ constant / RT`,
which hands the model something close to the inverse of the answer. Second — and
independent of the algebra — `RSD` does not exist until the compound has been run
on the instrument three times. A model whose stated purpose is to predict elution
time *before* running the experiment can never be given this value at prediction
time.

In the original code the exclusion list read:

```python
columns_to_remove = [
    'MCMRT\nNumber', 'Compound \nName', 'IUPAC \nName', 'Formula',
    'Pubchem \nNumber', 'Isomeric SMILES', 'InChI',
    'Retention Factor (k)', 'RT (min)', 'Dataset', 'AnalyticalColumn'
]
```

`Retention Factor (k)` was correctly excluded on exactly this reasoning — it is
`(RT − t_dead) / t_dead` — but `RSD` was not, so it entered the feature matrix.
It ranks third in the Random Forest importances of the submitted notebook, at
0.052, behind LogP (0.287) and one gradient-composition feature.

**Consequence.** The pooled test-set figures in the report and notebook — Random
Forest R² = 0.955, MAE = 1.09 min — are optimistic by an unquantified margin. The
importance of 0.052 suggests the effect is real but not the main driver of the
result; LogP remains dominant. Re-running with `RSD` excluded would settle it.

**Fixed in:** `config.NON_FEATURE_COLUMNS` now excludes `RSD`, with the reasoning
recorded in the docstring. Guarded by
`test_target_derived_columns_never_enter_the_feature_matrix`.

---

## 2. The one-hot column identity never reached the pooled model

**Severity: medium. A stated method feature was silently absent.**

The feature list for the pooled model was built like this:

```python
for dataset_name, data_parts in individual_datasets.items():
    df = data_parts['combined_df'].copy()      # loop variable
    ...

feature_names_dataset_all = [c for c in df.columns if c not in columns_to_remove]
df_all = all_data["Dataset_all"]['combined_df'].copy()
X_dataset_all = df_all[feature_names_dataset_all]
```

`df` on the third line is whatever the preceding loop left behind — the *last
individual experiment*, not the pooled frame. One-hot encoding was applied only
to the pooled frame, so individual experiments have no `Col_*` columns, so no
`Col_*` column ever entered `feature_names_dataset_all`.

**Consequence.** The report states that "the analytical column type was one-hot
encoded", and it was — but the resulting six indicator columns were then dropped
before modelling. The pooled model had 235 features where it should have had
240 — the 235 being 234 legitimate features plus the leaking `RSD` of item 1, and
the missing six being the `Col_*` indicators. It could not distinguish one
stationary phase from another except
indirectly through the mobile-phase and gradient features. This is visible in the
submitted notebook: the feature-importance listing for `Dataset_all` runs to 235
entries and contains no `Col_*` name.

**Fixed in:** `splits.pooled_feature_names()` derives the list from the pooled
frame itself. Guarded by `test_pooled_features_include_column_identity`.

---

## 3. The Ridge coefficient heatmap showed one dataset thirty times

**Severity: medium. One published figure is wrong.**

```python
start_index = 0
for dataset_name_temp, data_parts_temp in all_data.items():
    end_index = start_index + 5
    dataset_fold_coefficients = all_ridge_coefficients[start_index:end_index]
    ...
    ridge_abs_coef_per_dataset[dataset_name_temp] = mean_abs_for_dataset
```

`start_index` is initialised once and never advanced, so every iteration slices
the same first five entries — the five outer folds of the first dataset only.

**Consequence.** In "Mean Absolute Ridge Coefficients per Dataset (Features vs.
Datasets)" every column is identical (MolWt 1.08, LogP 8.12, TPSA 1.82, and so on
across all 31 columns). The figure appears to show that feature importance is
remarkably stable across chromatographic conditions; in fact it shows one dataset
repeated. The equivalent Random Forest heatmap was built differently and is
unaffected — it does vary across datasets, which is the honest version of the
same comparison.

**Fixed in:** `pipeline.importance_matrix()` keys importances by dataset name
rather than by positional slice, which removes the class of bug rather than the
instance.

---

## 4. Lasso zero-frequency percentages used the wrong denominator

**Severity: low. Percentages understated by ~3%.**

```python
num_datasets = len(all_data)          # 31 - includes 'Dataset_all'
total_outer_folds = num_datasets * 5  # 155
```

At that point `all_data` already contained the pooled `Dataset_all` entry, but the
loop that produced the counts ran over the 30 individual experiments only, giving
150 folds. Dividing 150 folds' worth of counts by 155 understates every percentage
by a factor of 150/155.

**Consequence.** Cosmetic — the ranking of features is unchanged, only the
absolute percentages. A feature eliminated in every fold reads as 96.8% rather
than 100%.

**Fixed in:** counts are derived from the results actually collected rather than
from a separately computed constant.

---

## Numbers that differ between the report and the notebook

Not defects, but worth recording. The report quotes pooled Random Forest
R² = 0.965 and MSE = 3.86 min²; the submitted notebook shows R² = 0.955 /
MSE = 4.84 on the held-out test set and R² = 0.957 / MSE = 4.24 from nested CV.
Report averages across the 30 individual datasets (RF R² = 0.802, Ridge
R² = 0.692) likewise differ slightly from the notebook (0.797 and 0.689). The
report appears to have been written from an earlier run.

The README quotes the notebook figures throughout, on the grounds that the
notebook is the artefact anyone can open and check.
