"""Tests for training a saved model and predicting with it, including the CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hplc_rt import cli, config, pipeline, predictor, splits


@pytest.fixture(scope="module")
def model(data_dir: Path) -> predictor.TrainedModel:
    return predictor.train(data_dir, model_name="Ridge", inner_folds=2)


@pytest.fixture(scope="module")
def pooled(data_dir: Path) -> pd.DataFrame:
    return pipeline.build_tables(data_dir)[1]


def test_model_knows_every_training_method(model: predictor.TrainedModel) -> None:
    assert model.methods == ["Dataset_9000", "Dataset_9001", "Dataset_9002"]


def test_prediction_features_match_training_features(
    model: predictor.TrainedModel, pooled: pd.DataFrame
) -> None:
    """A training row predicted from its SMILES alone must give the same value.

    This checks that predict() rebuilds the feature vector exactly as the
    pipeline did: same descriptors, same method features, same column order.
    """
    row = pooled.iloc[0]
    from_pipeline = model.estimator.predict(
        row[splits.pooled_feature_names(pooled)].to_frame().T.astype(float)
    )[0]
    from_smiles = model.predict([row[config.GROUP_COLUMN]], [row["Dataset"]])

    assert from_smiles[predictor.PREDICTION_COLUMN].iloc[0] == pytest.approx(from_pipeline)


def test_predicts_one_row_per_molecule_and_method(model: predictor.TrainedModel) -> None:
    result = model.predict(["CCO", "c1ccccc1"])
    assert len(result) == 2 * len(model.methods)
    assert result[predictor.PREDICTION_COLUMN].notna().all()


def test_lipophilic_molecule_elutes_later(model: predictor.TrainedModel) -> None:
    """Synthetic retention rises with LogP; the model should have learned that."""
    result = model.predict(["CCO", "c1ccc2ccccc2c1"], ["Dataset_9000"])
    ethanol, naphthalene = result[predictor.PREDICTION_COLUMN]
    assert naphthalene > ethanol


def test_invalid_smiles_gets_no_prediction(model: predictor.TrainedModel) -> None:
    result = model.predict(["not a molecule", "CCO"], ["Dataset_9000"])
    assert np.isnan(result[predictor.PREDICTION_COLUMN].iloc[0])
    assert result["Note"].iloc[0] == "invalid SMILES"
    assert not np.isnan(result[predictor.PREDICTION_COLUMN].iloc[1])


def test_flags_molecules_outside_the_training_range(
    model: predictor.TrainedModel,
) -> None:
    long_alkane = "C" * 40
    note = model.predict([long_alkane], ["Dataset_9000"])["Note"].iloc[0]
    assert "outside training range" in note
    assert "MolWt" in note


def test_flags_molecules_measured_in_training(
    model: predictor.TrainedModel, pooled: pd.DataFrame
) -> None:
    row = pooled.iloc[0]
    # A different spelling of the same molecule must still be recognised.
    respelled = predictor.Chem.MolToSmiles(
        predictor.Chem.MolFromSmiles(row[config.GROUP_COLUMN]), doRandom=True
    )
    note = model.predict([respelled], [row["Dataset"]])["Note"].iloc[0]
    assert "measured in training data" in note


def test_unknown_method_raises(model: predictor.TrainedModel) -> None:
    with pytest.raises(KeyError, match="available"):
        model.predict(["CCO"], ["Dataset_0000"])


def test_save_and_load_round_trip(model: predictor.TrainedModel, tmp_path: Path) -> None:
    path = model.save(tmp_path / "model.joblib")
    loaded = predictor.TrainedModel.load(path)
    pd.testing.assert_frame_equal(loaded.predict(["CCO"]), model.predict(["CCO"]))


def test_load_rejects_other_objects(tmp_path: Path) -> None:
    path = tmp_path / "other.joblib"
    predictor.joblib.dump({"not": "a model"}, path)
    with pytest.raises(TypeError):
        predictor.TrainedModel.load(path)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def test_cli_train_then_predict(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model.joblib"
    assert cli.main([
        "train", "--data-dir", str(data_dir), "--model-type", "Ridge",
        "--out", str(model_path),
    ]) == 0

    molecules = tmp_path / "molecules.csv"
    pd.DataFrame({"SMILES": ["CCO", "c1ccccc1"]}).to_csv(molecules, index=False)
    out = tmp_path / "predictions.csv"
    assert cli.main([
        "predict", "--model", str(model_path), "--input", str(molecules),
        "--method", "Dataset_9001", "--output", str(out),
    ]) == 0
    predictions = pd.read_csv(out)
    assert list(predictions["SMILES"]) == ["CCO", "c1ccccc1"]
    assert predictions[predictor.PREDICTION_COLUMN].notna().all()

    capsys.readouterr()
    assert cli.main(["predict", "--model", str(model_path), "--list-methods"]) == 0
    assert "Dataset_9002" in capsys.readouterr().out


def test_cli_predict_reports_unknown_method(
    model: predictor.TrainedModel, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = model.save(tmp_path / "model.joblib")
    code = cli.main(["predict", "--model", str(path), "--smiles", "CCO", "--method", "nope"])
    assert code == 1
    assert "unknown method" in capsys.readouterr().err


def test_cli_without_subcommand_still_evaluates(data_dir: Path, tmp_path: Path) -> None:
    """``hplc-rt --data-dir ...`` from before the subcommands must keep working."""
    code = cli.main([
        "--data-dir", str(data_dir), "--pooled-only", "--models", "Ridge",
        "--no-figures", "--output", str(tmp_path),
    ])
    assert code == 0
    assert (tmp_path / "test_summary.csv").exists()
    for name in ("fold_scores.csv", "predictions.csv"):
        assert (tmp_path / name).exists()


def test_cli_method_split_evaluates_the_pooled_model_only(
    many_methods_dir: Path, tmp_path: Path
) -> None:
    code = cli.main([
        "evaluate", "--data-dir", str(many_methods_dir), "--split-by", "method",
        "--models", "Ridge", "--no-figures", "--output", str(tmp_path),
    ])
    assert code == 0
    summary = pd.read_csv(tmp_path / "test_summary.csv")
    assert set(summary["Dataset"]) == {config.POOLED_DATASET_KEY}


def test_cli_saves_pooled_diagnostic_figures(data_dir: Path, tmp_path: Path) -> None:
    code = cli.main([
        "evaluate", "--data-dir", str(data_dir), "--pooled-only", "--models", "Ridge",
        "--output", str(tmp_path),
    ])
    assert code == 0
    figures = {p.stem for p in (tmp_path / "figures").glob("*.png")}
    assert "pooled_Ridge_predicted_vs_measured" in figures
    assert "pooled_fold_scores_r2" in figures
