# Corrections to the original submission

`src/` is a refactor of the notebook submitted in June 2025
(`notebooks/original_submission.ipynb`, unchanged). Five bugs showed up while
restructuring the code. They are listed here so the notebook, the report, and
this package can be compared.

Items 1–3 and 5 affect the pooled `Dataset_all` model. The 30 per-experiment
models use only the nine molecular descriptors and are not affected by those
four.

---

## 1. `RSD` was used as a predictor (target leakage)

**Severity: high. Affects the headline pooled result.**

`RSD` is a column on the source `RT` sheet. The dataset paper defines it as the
relative standard deviation across three replicate analyses of the same molecule
under the same conditions: `std(RT) / mean(RT)`.

Two problems follow. First, it is a function of the target (the denominator is
the retention time). If replicate scatter is similar across compounds,
`RSD ≈ constant / RT`, which is close to the inverse of the answer. Second,
`RSD` is only known after the compound has been run three times, so it cannot
be used to predict elution time before the experiment.

The original exclusion list was:

```python
columns_to_remove = [
    'MCMRT\nNumber', 'Compound \nName', 'IUPAC \nName', 'Formula',
    'Pubchem \nNumber', 'Isomeric SMILES', 'InChI',
    'Retention Factor (k)', 'RT (min)', 'Dataset', 'AnalyticalColumn'
]
```

`Retention Factor (k) = (RT − t_dead) / t_dead` was excluded for the same
reason, but `RSD` was not. In the submitted notebook it ranks third in Random
Forest importance (0.052), after LogP (0.287) and one gradient-composition
feature.

**Effect.** The pooled test figures (Random Forest R² = 0.955, MAE = 1.09 min)
are optimistic by an amount that has not been measured on the real data. An
importance of 0.052 suggests a real but secondary effect; LogP is still
dominant.

**Fix.** `config.NON_FEATURE_COLUMNS` excludes `RSD`. Covered by
`test_target_derived_columns_never_enter_the_feature_matrix`.

With the real data in `data/raw/`:

```bash
python scripts/quantify_leakage.py --data-dir data/raw --models RandomForest
```

This runs the pooled model twice (corrected vs. leaking columns restored) and
prints the difference in R² and MAE.

On synthetic data, where `RSD` is generated as `|noise| / RT`, restoring the
leak increases Ridge R² by about 0.012. That checks the mechanism, not the
size of the effect on MCMRT.

---

## 2. The one-hot column identity never reached the pooled model

**Severity: medium. A method feature described in the report was dropped.**

The pooled feature list was built like this:

```python
for dataset_name, data_parts in individual_datasets.items():
    df = data_parts['combined_df'].copy()      # loop variable
    ...

feature_names_dataset_all = [c for c in df.columns if c not in columns_to_remove]
df_all = all_data["Dataset_all"]['combined_df'].copy()
X_dataset_all = df_all[feature_names_dataset_all]
```

`df` is the last individual experiment from the loop, not the pooled frame.
One-hot encoding was applied only to the pooled frame, so individual
experiments have no `Col_*` columns, and none entered
`feature_names_dataset_all`.

**Effect.** The report says the analytical column was one-hot encoded. The
indicators were created and then dropped. The pooled model had 235 features
instead of 240 (234 valid features plus leaking `RSD`, and none of the six
`Col_*` columns). The submitted feature-importance list for `Dataset_all` has
235 entries and no `Col_*` name.

**Fix.** `splits.pooled_feature_names()` uses the pooled frame. Covered by
`test_pooled_features_include_column_identity`.

---

## 3. The Ridge coefficient heatmap showed one dataset thirty times

**Severity: medium. One figure in the submission is wrong.**

```python
start_index = 0
for dataset_name_temp, data_parts_temp in all_data.items():
    end_index = start_index + 5
    dataset_fold_coefficients = all_ridge_coefficients[start_index:end_index]
    ...
    ridge_abs_coef_per_dataset[dataset_name_temp] = mean_abs_for_dataset
```

`start_index` is never increased, so every iteration takes the first five
entries (the five outer folds of the first dataset).

**Effect.** In "Mean Absolute Ridge Coefficients per Dataset (Features vs.
Datasets)" every column is the same (MolWt 1.08, LogP 8.12, TPSA 1.82, …). The
plot looks like stable importance across methods; it is one dataset repeated.
The Random Forest heatmap was built another way and does vary across datasets.

**Fix.** `pipeline.importance_matrix()` keys importances by dataset name.

---

## 4. Lasso zero-frequency percentages used the wrong denominator

**Severity: low. Percentages are about 3% too low.**

```python
num_datasets = len(all_data)          # 31 - includes 'Dataset_all'
total_outer_folds = num_datasets * 5  # 155
```

`all_data` already included `Dataset_all`, but the counting loop only ran over
the 30 individual experiments (150 folds). Dividing by 155 understates every
percentage by 150/155.

**Effect.** Rankings stay the same. A feature that is zero in every fold is
reported as 96.8% instead of 100%.

**Fix.** The denominator is the number of folds actually collected.

---

## 5. Gradient vectors reused the initial setting after the last breakpoint

**Severity: medium. The tail of every gradient feature vector was wrong.**

Gradient programs are short lists of breakpoints, usually finished well before
100 min. They are resampled onto a 100-point grid from 0 to 100 min.
Interpolation is step-wise (`kind="previous"`): the pump holds flow and %B
until the next breakpoint.

Original call:

```python
rate_interp = interp1d(
    times, rates, kind="previous", bounds_error=False, fill_value=rates[0]
)(target_times)
```

A single `fill_value` is used on both sides of the data range. Points before
the first breakpoint correctly get the initial setting. Points after the last
breakpoint (most of the grid if the program ends at 15–30 min) also get the
*initial* setting, instead of the final hold. The instrument does not jump
back to 5% B at the end of the run.

**Effect.** Later `HPLC_*_Rate` / `HPLC_*_Comp` features repeated the start of
the gradient. Methods still differed by program length, so the vector was not
empty of information, but it did not describe the pump. Per-method models do
not use these features.

**Fix.** `features.vectorize_gradient` fills with `(first_value, last_value)`.
Covered by `test_gradient_holds_final_setting_after_last_breakpoint`.

---

## Report vs. notebook numbers

Not bugs, but the numbers differ. The report quotes pooled Random Forest
R² = 0.965 and MSE = 3.86 min². The submitted notebook shows R² = 0.955 /
MSE = 4.84 on the test set and R² = 0.957 / MSE = 4.24 from nested CV. Report
averages over the 30 individual datasets (RF R² = 0.802, Ridge R² = 0.692)
also differ slightly from the notebook (0.797 and 0.689). The report was
probably written from an earlier run.

The README uses the notebook numbers, because that file can be opened and
checked.
