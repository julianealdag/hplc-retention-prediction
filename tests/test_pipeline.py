"""Tests against synthetic workbooks: loading, features, leakage exclusions, CV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hplc_rt import (
    config,
    curation,
    descriptors,
    evaluate,
    features,
    loading,
    models,
    pipeline,
    splits,
)

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_loads_every_workbook(data_dir: Path) -> None:
    experiments = loading.load_all(data_dir)
    assert len(experiments) == 3
    assert set(experiments) == {"Dataset_9000", "Dataset_9001", "Dataset_9002"}


def test_lc_sheet_splits_into_metadata_and_gradient(data_dir: Path) -> None:
    experiment = loading.load_all(data_dir)["Dataset_9000"]
    assert "Analytical column" in experiment.lc.columns
    assert list(experiment.gradient.columns) == config.GRADIENT_COLUMNS
    assert len(experiment.gradient) == 4


def test_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        loading.load_all("/nonexistent/path")


# --------------------------------------------------------------------------
# Curation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  XBridge C18 ", "xbridgec18"),
        ("XBridge C18", "xbridgec18"),
        ("xbridge c18", "xbridgec18"),
    ],
)
def test_normalize_text_collapses_variants(raw: str, expected: str) -> None:
    """The three spellings that would otherwise become three one-hot columns."""
    assert curation.normalize_text(raw) == expected


def test_normalize_text_passes_through_nulls() -> None:
    assert pd.isna(curation.normalize_text(np.nan))


def test_curation_strips_names_and_drops_cas(data_dir: Path) -> None:
    experiments = curation.curate(loading.load_all(data_dir))
    rt = experiments["Dataset_9000"].rt
    assert "CAS\nNumber" not in rt.columns
    assert not rt["Compound \nName"].str.startswith(" ").any()


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def test_parse_mobile_phase_reads_ratio_and_additive() -> None:
    parsed = features.parse_mobile_phase("Water:Methanol 90:10 + 0.1% formic acid")
    assert parsed["Water"] == 1
    assert parsed["Methanol"] == 1
    assert parsed["Ratio_Water"] == 90
    assert parsed["Ratio_Methanol"] == 10
    assert parsed["Acid_Formic"] == 1
    assert parsed["Acid_Conc"] == pytest.approx(0.1)


def test_parse_mobile_phase_reads_buffer_salt() -> None:
    parsed = features.parse_mobile_phase("Water + 10 mM ammonium formate")
    assert parsed["Salt_Formate"] == 1
    assert parsed["Salt_Conc"] == pytest.approx(10.0)
    assert parsed["Salt_Acetate"] == 0


def test_parse_mobile_phase_handles_empty() -> None:
    parsed = features.parse_mobile_phase(None)
    assert set(parsed) == set(features.MOBILE_PHASE_FEATURES)
    assert all(value == 0 for value in parsed.values())


def test_gradient_vector_is_fixed_length_and_stepwise() -> None:
    gradient = pd.DataFrame(
        {
            "Time (min)": [0.0, 5.0, 10.0],
            "Flow rate (mL/min)": [0.3, 0.3, 0.5],
            "B (%)": [5.0, 5.0, 95.0],
        }
    )
    vector = features.vectorize_gradient(gradient)
    assert vector.shape == (2 * config.GRADIENT_RESOLUTION,)
    # Step-wise: at t just below 10 the composition must still be the old value,
    # never an interpolated intermediate.
    comps = vector[1::2]
    assert set(np.unique(comps)) <= {5.0, 95.0}


def test_gradient_vector_length_is_independent_of_breakpoint_count() -> None:
    """The whole point of resampling: different programs, same feature width."""
    short = pd.DataFrame({
        "Time (min)": [0.0, 9.0],
        "Flow rate (mL/min)": [0.3, 0.3],
        "B (%)": [5.0, 90.0],
    })
    long = pd.DataFrame({
        "Time (min)": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        "Flow rate (mL/min)": [0.3] * 6,
        "B (%)": [5.0, 20.0, 40.0, 60.0, 80.0, 95.0],
    })
    assert features.vectorize_gradient(short).shape == features.vectorize_gradient(long).shape


def test_empty_gradient_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        features.vectorize_gradient(pd.DataFrame(columns=config.GRADIENT_COLUMNS))


def test_gradient_holds_final_setting_after_last_breakpoint() -> None:
    """Regression test: the tail of the grid must hold the last programmed values.

    The original coursework code passed a scalar fill_value to interp1d, so every point
    after the last breakpoint snapped back to the *initial* flow and %B.
    """
    gradient = pd.DataFrame(
        {
            "Time (min)": [0.0, 10.0],
            "Flow rate (mL/min)": [0.3, 0.5],
            "B (%)": [5.0, 95.0],
        }
    )
    vector = features.vectorize_gradient(gradient, resolution=100, time_max=100.0)
    rates = vector[0::2]
    comps = vector[1::2]
    assert rates[-1] == pytest.approx(0.5)
    assert comps[-1] == pytest.approx(95.0)
    # And the hold should start as soon as we pass the last breakpoint.
    assert rates[50] == pytest.approx(0.5)
    assert comps[50] == pytest.approx(95.0)


def test_pooling_one_hot_encodes_the_column(data_dir: Path) -> None:
    experiments = features.build_features(curation.curate(loading.load_all(data_dir)))
    pooled = features.pool_experiments(experiments)
    col_indicators = [c for c in pooled.columns if c.startswith("Col_")]
    assert len(col_indicators) == 3, "three synthetic experiments use three columns"
    assert pooled[col_indicators].isin([0, 1]).all().all()


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------

def test_descriptors_computed_for_known_molecule() -> None:
    series = descriptors.descriptors_for_smiles("CCO")
    assert series["MolWt"] == pytest.approx(46.07, abs=0.01)
    assert series["HBDonors"] == 1


def test_bad_smiles_yields_nan_not_exception() -> None:
    series = descriptors.descriptors_for_smiles("not_a_molecule")
    assert series.isna().all()


def test_descriptors_ordered_by_logp_as_expected() -> None:
    """Sanity check on the physics: benzene is more lipophilic than ethanol."""
    ethanol = descriptors.descriptors_for_smiles("CCO")["LogP"]
    benzene = descriptors.descriptors_for_smiles("c1ccccc1")["LogP"]
    assert benzene > ethanol


# --------------------------------------------------------------------------
# Splits and leakage
# --------------------------------------------------------------------------

def test_pooled_features_include_column_identity(data_dir: Path) -> None:
    """Regression test for the stale-variable bug in the original coursework code.

    The pooled feature list must be derived from the pooled frame, so the Col_*
    indicators created during pooling are present.
    """
    experiments = features.build_features(curation.curate(loading.load_all(data_dir)))
    pooled = descriptors.add_descriptors(features.pool_experiments(experiments))
    names = splits.pooled_feature_names(pooled)
    assert any(n.startswith("Col_") for n in names), "one-hot column identity missing"


@pytest.mark.parametrize("leaky", ["RT (min)", "Retention Factor (k)", "RSD"])
def test_target_derived_columns_never_enter_the_feature_matrix(
    data_dir: Path, leaky: str
) -> None:
    """Regression test: RSD and k are functions of the target and must be excluded."""
    experiments = features.build_features(curation.curate(loading.load_all(data_dir)))
    pooled = descriptors.add_descriptors(features.pool_experiments(experiments))
    assert leaky not in splits.pooled_feature_names(pooled)


def test_per_experiment_split_uses_descriptors_only(data_dir: Path) -> None:
    all_splits = pipeline.prepare_data(data_dir)
    per_experiment = all_splits["Dataset_9000"]
    assert per_experiment.feature_names == config.DESCRIPTOR_COLUMNS


def test_split_proportions(data_dir: Path) -> None:
    split = pipeline.prepare_data(data_dir)["Dataset_9000"]
    total = len(split.X_train) + len(split.X_test)
    assert len(split.X_test) / total == pytest.approx(config.TEST_SIZE, abs=0.1)


def test_compound_split_holds_out_entire_molecules(data_dir: Path) -> None:
    """No SMILES should appear on both sides of a compound-disjoint split."""
    all_splits = pipeline.prepare_data(data_dir, group_by=config.GROUP_COLUMN)
    pooled = all_splits[config.POOLED_DATASET_KEY]
    assert pooled.groups_train is not None
    assert pooled.groups_test is not None
    overlap = set(pooled.groups_train) & set(pooled.groups_test)
    assert overlap == set(), f"molecules leaked across the split: {overlap}"


def test_method_split_holds_out_entire_methods(many_methods_dir: Path) -> None:
    all_splits = pipeline.prepare_data(many_methods_dir, group_by=config.METHOD_COLUMN)
    assert list(all_splits) == [config.POOLED_DATASET_KEY], "one method cannot be split"
    pooled = all_splits[config.POOLED_DATASET_KEY]
    overlap = set(pooled.groups_train) & set(pooled.groups_test)
    assert overlap == set(), f"methods leaked across the split: {overlap}"
    assert pooled.groups_test.nunique() >= 1


@pytest.mark.parametrize("group_by", [config.GROUP_COLUMN, config.METHOD_COLUMN])
def test_outer_cv_folds_respect_the_grouping(many_methods_dir: Path, group_by: str) -> None:
    """A grouped split must not be scored with folds that share groups.

    Plain k-fold here would put the same molecule (or method) in both the
    training and validation part of a fold, making CV scores optimistic.
    """
    split = pipeline.prepare_data(many_methods_dir, group_by=group_by)[
        config.POOLED_DATASET_KEY
    ]
    splitter = models.outer_splitter(split, n_folds=2)
    for train_idx, val_idx in splitter.split(
        split.X_train, split.y_train, split.groups_train
    ):
        shared = set(split.groups_train.iloc[train_idx]) & set(
            split.groups_train.iloc[val_idx]
        )
        assert shared == set()


def test_grouped_cv_needs_enough_groups(many_methods_dir: Path) -> None:
    split = pipeline.prepare_data(many_methods_dir, group_by=config.METHOD_COLUMN)[
        config.POOLED_DATASET_KEY
    ]
    with pytest.raises(ValueError, match="groups in training"):
        models.outer_splitter(split, n_folds=split.groups_train.nunique() + 1)


def test_nested_cv_runs_on_a_method_split(many_methods_dir: Path) -> None:
    split = pipeline.prepare_data(many_methods_dir, group_by=config.METHOD_COLUMN)[
        config.POOLED_DATASET_KEY
    ]
    result = models.nested_cv(split, "Ridge", inner_folds=2, outer_folds=2)
    assert np.isfinite(result.mae)


def test_row_split_is_the_default_and_can_share_molecules(data_dir: Path) -> None:
    """The original (optimistic) split is still the default."""
    split = pipeline.prepare_data(data_dir)[config.POOLED_DATASET_KEY]
    assert split.groups_train is None
    assert split.groups_test is None


def test_compound_split_requires_the_group_column() -> None:
    frame = pd.DataFrame(
        {
            "MolWt": [1.0, 2.0, 3.0, 4.0],
            "RT (min)": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(KeyError, match="Isomeric SMILES"):
        splits.make_split(frame, ["MolWt"], "toy", group_by=config.GROUP_COLUMN)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["Ridge", "Lasso", "RandomForest"])
def test_nested_cv_runs_and_reports_spread(data_dir: Path, model_name: str) -> None:
    split = pipeline.prepare_data(data_dir)["Dataset_9000"]
    result = models.nested_cv(split, model_name, inner_folds=2, outer_folds=2)
    assert result.mae > 0, "MAE must be reported positive, not scikit-learn's negation"
    assert result.mse > 0
    assert result.r2_std >= 0
    assert len(result.fold_importances) == 2


def test_unknown_model_raises(data_dir: Path) -> None:
    split = pipeline.prepare_data(data_dir)["Dataset_9000"]
    with pytest.raises(KeyError, match="unknown model"):
        models.nested_cv(split, "SupportVectorEverything")


def test_importance_series_names_match_features(data_dir: Path) -> None:
    split = pipeline.prepare_data(data_dir)["Dataset_9000"]
    result = models.nested_cv(split, "Ridge", inner_folds=2, outer_folds=2)
    assert set(result.importance_series().index) == set(config.DESCRIPTOR_COLUMNS)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def test_dummy_baseline_scores_zero_on_its_own_training_data(data_dir: Path) -> None:
    split = pipeline.prepare_data(data_dir)["Dataset_9000"]
    assert evaluate.dummy_baseline(split, on="train").r2 == pytest.approx(0.0, abs=1e-9)


def test_dummy_baseline_rejects_bad_argument(data_dir: Path) -> None:
    split = pipeline.prepare_data(data_dir)["Dataset_9000"]
    with pytest.raises(ValueError, match="test.*train"):
        evaluate.dummy_baseline(split, on="validation")


def test_model_beats_dummy_on_synthetic_signal(data_dir: Path) -> None:
    """The synthetic RT is monotone in LogP, so any working model must beat the mean."""
    split = pipeline.prepare_data(data_dir)["Dataset_9000"]
    result = models.nested_cv(split, "Ridge", inner_folds=2, outer_folds=2)
    estimator = models.fit_final(split, "Ridge", result.best_params)
    fitted = evaluate.evaluate(estimator, split, "Ridge")
    assert fitted.mae < evaluate.dummy_baseline(split).mae


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_full_pipeline_produces_a_summary(data_dir: Path) -> None:
    output = pipeline.run(data_dir, model_names=["Ridge"])
    assert set(output.test_summary["Dataset"]) == {
        "Dataset_9000", "Dataset_9001", "Dataset_9002", config.POOLED_DATASET_KEY,
    }
    assert set(output.test_summary["Model"]) == {"Ridge", "Dummy"}
    assert output.test_summary["R2_test"].notna().all()


def test_pooled_model_sees_more_features_than_per_experiment(data_dir: Path) -> None:
    all_splits = pipeline.prepare_data(data_dir)
    pooled = all_splits[config.POOLED_DATASET_KEY]
    single = all_splits["Dataset_9000"]
    assert len(pooled.feature_names) > len(single.feature_names)


# --------------------------------------------------------------------------
# Regression tests for the remaining defects in docs/corrections.md
# --------------------------------------------------------------------------

def test_importance_matrix_columns_differ_between_datasets(data_dir: Path) -> None:
    """Regression test for the never-advanced slice index.

    The original coursework code built its per-dataset coefficient heatmap by slicing a
    flat list with a ``start_index`` initialised to 0 and never incremented, so
    all 31 columns showed the first dataset's five folds. The original figure had
    31 identical columns.

    Different experiments have different compounds and conditions, so their fitted
    coefficients cannot legitimately be identical to full float precision.
    """
    output = pipeline.run(
        data_dir, model_names=["Ridge"], datasets=["Dataset_9000", "Dataset_9001"]
    )
    matrix = pipeline.importance_matrix(output, "Ridge")

    assert list(matrix.columns) == ["Dataset_9000", "Dataset_9001"]
    assert not matrix["Dataset_9000"].equals(matrix["Dataset_9001"]), (
        "per-dataset importances are identical - the slicing bug is back"
    )


def test_importance_matrix_is_keyed_by_name_not_position(data_dir: Path) -> None:
    """Each column must carry its own dataset's numbers, whatever the run order."""
    output = pipeline.run(data_dir, model_names=["Ridge"], datasets=["Dataset_9001"])
    matrix = pipeline.importance_matrix(output, "Ridge")
    expected = output.cv_results[("Dataset_9001", "Ridge")].importance_series()
    pd.testing.assert_series_equal(
        matrix["Dataset_9001"].dropna().sort_index(),
        expected.reindex(config.DESCRIPTOR_COLUMNS).dropna().sort_index(),
        check_names=False,
    )


def test_zero_coefficient_denominator_matches_folds_actually_run(
    data_dir: Path,
) -> None:
    """Regression test for the 155-vs-150 denominator.

    The original computed it as ``len(all_data) * 5``, counting a dataset the
    counting loop never visited. Here it must equal the folds actually run.
    """
    outer_folds, datasets = 2, ["Dataset_9000", "Dataset_9001"]
    all_splits = pipeline.prepare_data(data_dir)
    results = [
        models.nested_cv(
            all_splits[name], "Lasso", inner_folds=2, outer_folds=outer_folds
        )
        for name in datasets
    ]
    frequency = models.zero_coefficient_frequency(results)

    assert (frequency["Total_Folds"] == len(datasets) * outer_folds).all()
    assert (frequency["Zero_Frequency (%)"] <= 100.0).all()


def test_zero_coefficient_frequency_empty_without_lasso(data_dir: Path) -> None:
    split = pipeline.prepare_data(data_dir)["Dataset_9000"]
    ridge = models.nested_cv(split, "Ridge", inner_folds=2, outer_folds=2)
    assert models.zero_coefficient_frequency([ridge]).empty


def test_leakage_comparison_path_changes_the_feature_count(data_dir: Path) -> None:
    """scripts/quantify_leakage.py relies on exclusions being overridable."""
    corrected = pipeline.prepare_data(data_dir)[config.POOLED_DATASET_KEY]
    with_leak = [
        c for c in config.NON_FEATURE_COLUMNS if c not in config.LEAKY_COLUMNS
    ]
    leaked = pipeline.prepare_data(data_dir, exclusions=with_leak)[
        config.POOLED_DATASET_KEY
    ]

    assert "RSD" not in corrected.feature_names
    assert "RSD" in leaked.feature_names
    assert len(leaked.feature_names) == len(corrected.feature_names) + len(
        config.LEAKY_COLUMNS
    )


# --------------------------------------------------------------------------
# Per-fold scores and test predictions
# --------------------------------------------------------------------------

def test_fold_scores_name_the_methods_each_fold_held_out(many_methods_dir: Path) -> None:
    split = pipeline.prepare_data(many_methods_dir, group_by=config.METHOD_COLUMN)[
        config.POOLED_DATASET_KEY
    ]
    result = models.nested_cv(split, "Ridge", inner_folds=2, outer_folds=2)

    assert len(result.fold_scores) == 2
    assert result.fold_scores["R2"].mean() == pytest.approx(result.r2)
    held_out = {m for cell in result.fold_scores["HeldOut"] for m in cell.split(", ")}
    assert held_out == set(split.groups_train)


def test_predictions_record_the_method_of_every_test_row(data_dir: Path) -> None:
    output = pipeline.run(
        data_dir, model_names=["Ridge"], datasets=[config.POOLED_DATASET_KEY]
    )
    split = output.splits[config.POOLED_DATASET_KEY]
    predictions = output.predictions

    assert len(predictions) == len(split.y_test)
    assert set(predictions["Method"]) <= {"Dataset_9000", "Dataset_9001", "Dataset_9002"}
    assert predictions["Measured"].tolist() == split.y_test.tolist()
    mae = (predictions["Measured"] - predictions["Predicted"]).abs().mean()
    ridge = output.test_summary[output.test_summary["Model"] == "Ridge"]
    assert mae == pytest.approx(ridge["MAE_test"].iloc[0])


def test_leakage_script_can_restore_rsd_alone(data_dir: Path) -> None:
    """The original code leaked RSD only; the script must be able to reproduce that."""
    from quantify_leakage import run_both

    rsd_only = run_both(data_dir, ["Ridge"], columns=["RSD"])
    both = run_both(data_dir, ["Ridge"])

    assert list(rsd_only["Model"]) == ["Ridge"]
    # k plus the dead time pins down RT, so restoring it too must fit better.
    assert both["R2_with_leak"].iloc[0] > rsd_only["R2_with_leak"].iloc[0]
