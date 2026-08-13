"""Tests exercising the pipeline against synthetic workbooks.

These do not verify chemistry — the synthetic retention times are made up. What
they verify is that the refactored package still runs end to end, that the schema
parsing handles the awkward two-shapes-in-one-sheet layout, and that the leakage
columns stay out of the feature matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_synthetic_data import make_workbook  # noqa: E402

from hplc_rt import (  # noqa: E402
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


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three synthetic experiments of 20 compounds each."""
    directory = tmp_path_factory.mktemp("synthetic")
    for i in range(3):
        make_workbook(directory / f"Dataset_{9000 + i}.xlsx", i, 20, seed=0)
    return directory


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

    The original notebook passed a scalar fill_value to interp1d, so every point
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
    """Regression test for the stale-variable bug in the original notebook.

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

    The original notebook built its per-dataset coefficient heatmap by slicing a
    flat list with a ``start_index`` initialised to 0 and never incremented, so
    all 31 columns showed the first dataset's five folds. The submitted figure has
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
