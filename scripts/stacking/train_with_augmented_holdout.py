import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import dacite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm.engine import _CVBooster
from sklearn.model_selection import StratifiedKFold

import cuml
import optuna.integration.lightgbm as lgb
import somen
from alaska2.metrics import alaska_weighted_auc
from alaska2.validation import get_folds_as_filename


class CVBoosterStore(object):
    def __init__(self) -> None:
        self._cv_booster: Optional[_CVBooster] = None

    def __call__(self, env) -> None:
        self._cv_booster = env.model

    def get(self) -> _CVBooster:
        if self._cv_booster is None:
            raise RuntimeError
        return self._cv_booster


def alaska_weighted_auc_for_lightgbm(preds, train_data):
    labels = train_data.get_label()
    if len(preds) != len(labels):
        preds = preds.reshape(-1, len(labels))
        preds = -preds[0]
    return "weighted_auc", alaska_weighted_auc(labels, preds), True


@dataclass
class LightGBMParams:
    num_boost_round: int
    params: Optional[Mapping[str, Any]]  # If None, search params using LightGBMTunerCV


@dataclass
class SecondLayerModelConfig:
    model_name: str  # LightGBM  # TODO: LightGBM 以外も試したい
    params: Mapping[str, Any]  # parsed as LightGBMParams


@dataclass
class TSNEFeatureConfig:
    run_names: Sequence[str]
    n_components: int = 5
    perplexity: float = 30.0


@dataclass
class TrainingConfig:
    run_name: str
    fold_index: int
    n_folds: int
    second_layer_n_folds: int
    first_layer_runs: Sequence[str]
    second_layer_model: SecondLayerModelConfig
    task: str  # binary, 4class
    tsne: Optional[TSNEFeatureConfig]


def _load_train_df(config: TrainingConfig):
    folds = get_folds_as_filename(config.n_folds)
    valid_filenames = folds[config.fold_index][1]

    run_dfs = []
    for first_layer_run in config.first_layer_runs:
        pred_valid_path = Path(f"data/working/{first_layer_run}/{config.fold_index}/pred_valid.h5")
        assert pred_valid_path.exists()

        pred_valid = somen.file_io.load_array(pred_valid_path)
        assert pred_valid.ndim == 3
        assert pred_valid.shape == (len(valid_filenames) * 4, 4, 8)

        pred_valid = np.transpose(pred_valid, (0, 2, 1)).reshape(-1, 4)
        assert pred_valid.shape == (len(valid_filenames) * 4 * 8, 4)

        df = {}
        for j, folder in enumerate(["Cover", "JMiPOD", "JUNIWARD", "UERD"]):
            df[f"p_{folder}"] = pred_valid[:, j]
        df["filename"] = np.repeat(valid_filenames, 4 * 8)
        df["folder"] = np.repeat(np.tile(["Cover", "JMiPOD", "JUNIWARD", "UERD"], len(valid_filenames)), 8)
        df["aug_index"] = np.tile(np.arange(8), len(pred_valid) // 8)

        df = pd.DataFrame(df)

        assert (df["folder"].iloc[:8] == "Cover").all()
        assert (df["folder"].iloc[8:16] == "JMiPOD").all()

        run_dfs.append(df)

    df_train = run_dfs[0]
    for k, run_df in enumerate(run_dfs[1:]):
        if k == 0:
            suffixes = (config.first_layer_runs[k], config.first_layer_runs[k + 1])
        else:
            suffixes = ("", config.first_layer_runs[k + 1])
        df_train = df_train.merge(run_df, on=["filename", "folder", "aug_index"], suffixes=suffixes)

    payload_df = pd.read_csv("data/working/payload_stats_train.csv")
    quality_df = pd.read_csv("data/working/quality_train.csv")
    payload_df["filename"] = quality_df["filename"]
    payload_df["quality"] = quality_df["quality"]
    # other_stats = payload_df[
    #     [
    #         "filename",
    #         "quality",
    #         "nonzero",
    #         "JMiPOD_count_per_nonzero",
    #         "JUNIWARD_count_per_nonzero",
    #         "UERD_count_per_nonzero",
    #     ]
    # ]
    # other_stats = payload_df[["filename", "quality", "nonzero"]]
    other_stats = payload_df[["filename", "quality"]]
    df_train = df_train.merge(other_stats, on="filename", how="left")

    return df_train


def _load_test_df(config: TrainingConfig):
    test_filenames = [f"{i:0>4}.jpg" for i in range(1, 5001)]

    run_dfs = []
    for first_layer_run in config.first_layer_runs:
        pred_test_path = Path(f"data/working/{first_layer_run}/{config.fold_index}/pred_test.h5")
        assert pred_test_path.exists()

        pred_test = somen.file_io.load_array(pred_test_path)
        assert pred_test.ndim == 3
        assert pred_test.shape == (5000, 4, 8)

        pred_test = np.transpose(pred_test, (0, 2, 1)).reshape(-1, 4)
        assert pred_test.shape == (5000 * 8, 4)

        df = {}
        for j, folder in enumerate(["Cover", "JMiPOD", "JUNIWARD", "UERD"]):
            df[f"p_{folder}"] = pred_test[:, j]

        df["filename"] = np.repeat(test_filenames, 8)
        df["aug_index"] = np.tile(np.arange(8), len(pred_test) // 8)
        df = pd.DataFrame(df)

        run_dfs.append(df)

    df_test = run_dfs[0]
    for k, run_df in enumerate(run_dfs[1:]):
        if k == 0:
            suffixes = (config.first_layer_runs[k], config.first_layer_runs[k + 1])
        else:
            suffixes = ("", config.first_layer_runs[k + 1])
        df_test = df_test.merge(run_df, on=["filename", "aug_index"], suffixes=suffixes)

    quality_df = pd.read_csv("data/working/quality_test.csv")
    quality_df["nonzero"] = pd.read_csv("data/working/payload_stats_test.csv")["nonzero"]
    # df_test = df_test.merge(quality_df[["filename", "quality", "nonzero"]], on="filename")
    df_test = df_test.merge(quality_df[["filename", "quality"]], on="filename", how="left")

    return df_test


def _load_tsne_df(config: TrainingConfig):
    folds = get_folds_as_filename(config.n_folds)
    valid_filenames = folds[config.fold_index][1]
    test_filenames = [f"{i:0>4}.jpg" for i in range(1, 5001)]

    df_valid, df_test = [], []

    for first_layer_run in config.tsne.run_names:
        cache_path = Path(
            f"data/working/{first_layer_run}/{config.fold_index}/"
            f"tsne_{config.tsne.n_components}_{config.tsne.perplexity}.h5"
        )
        if cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            transformed = somen.file_io.load_array(cache_path)
        else:
            print(f"Loading data/working/{first_layer_run}/{config.fold_index}/feature_{{valid,test}}.h5")
            feature_valid = somen.file_io.load_array(
                Path(f"data/working/{first_layer_run}/{config.fold_index}/feature_valid.h5")
            )
            feature_test = somen.file_io.load_array(
                Path(f"data/working/{first_layer_run}/{config.fold_index}/feature_test.h5")
            )

            assert feature_valid.ndim == 3
            assert feature_valid.shape[0] == len(valid_filenames) * 4
            assert feature_valid.shape[2] == 8

            assert feature_test.ndim == 3
            assert feature_test.shape[0] == len(test_filenames) == 5000
            assert feature_test.shape[2] == 8

            feature_valid = np.transpose(feature_valid, (0, 2, 1))
            feature_valid = feature_valid.reshape(-1, feature_valid.shape[-1])

            feature_test = np.transpose(feature_test, (0, 2, 1))
            feature_test = feature_test.reshape(-1, feature_test.shape[-1])

            print(feature_valid.shape, feature_test.shape)
            print("Computing t-SNE")
            tsne = cuml.TSNE(n_components=config.tsne.n_components, perplexity=config.tsne.perplexity)
            transformed = tsne.fit_transform(np.concatenate([feature_valid, feature_test], axis=0))
            print("Done")

            somen.file_io.save_array(transformed, cache_path)

        n_valid = len(valid_filenames) * 4 * 8
        n_test = len(test_filenames) * 8
        tsne_valid = transformed[:n_valid]
        tsne_test = transformed[n_valid:]
        assert len(tsne_test) == n_test

        df_valid.append(
            pd.DataFrame(tsne_valid, columns=[f"tsne_{i}_{first_layer_run}" for i in range(config.tsne.n_components)])
        )
        df_test.append(
            pd.DataFrame(tsne_test, columns=[f"tsne_{i}_{first_layer_run}" for i in range(config.tsne.n_components)])
        )

    df_valid = pd.concat(df_valid, axis=1)
    df_valid["filename"] = np.repeat(valid_filenames, 4 * 8)
    df_valid["folder"] = np.repeat(np.tile(["Cover", "JMiPOD", "JUNIWARD", "UERD"], len(valid_filenames)), 8)
    df_valid["aug_index"] = np.tile(np.arange(8), len(df_valid) // 8)

    df_test = pd.concat(df_test, axis=1)
    df_test["filename"] = np.repeat(test_filenames, 8)
    df_test["aug_index"] = np.tile(np.arange(8), len(df_test) // 8)

    return df_valid, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("overrides", nargs="*", type=str)
    args = parser.parse_args()

    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, args.config, args.overrides)

    if config.second_layer_model.model_name != "LightGBM":
        raise NotImplementedError
    lightgbm_params = dacite.from_dict(LightGBMParams, config.second_layer_model.params)

    working_dir = Path(f"data/working/stacking-{config.run_name}/")
    somen.file_io.save_json(vars(args), working_dir / "args.json", indent=4)
    somen.file_io.save_yaml_from_dataclass(config, working_dir / "config.yaml")

    df_train = _load_train_df(config)
    df_test = _load_test_df(config)

    if config.tsne is not None:
        tsne_train, tsne_test = _load_tsne_df(config)
        df_train = df_train.merge(tsne_train, on=["filename", "folder", "aug_index"])
        df_test = df_test.merge(tsne_test, on=["filename", "aug_index"])

    train_filenames = get_folds_as_filename(config.n_folds)[config.fold_index][1]
    quality_df = pd.read_csv("data/working/quality_train.csv").set_index("filename")
    quality = quality_df.loc[train_filenames, "quality"]
    kf = StratifiedKFold(config.second_layer_n_folds, shuffle=True, random_state=0)
    folds = []
    folds_for_tuner = []
    for train_indices, valid_indices in kf.split(quality, quality):
        train_indices = np.array([i * 4 * 8 + 8 * j + k for i in train_indices for j in range(4) for k in range(8)])
        valid_indices = np.array([i * 4 * 8 + 8 * j + k for i in valid_indices for j in range(4) for k in range(8)])
        assert len(train_indices) + len(valid_indices) == len(df_train)
        assert len(set(train_indices) & set(valid_indices)) == 0
        folds.append((train_indices, valid_indices))

        split_for_tuner = int(len(train_indices) * 0.8)
        folds_for_tuner.append((train_indices[:split_for_tuner], train_indices[split_for_tuner:]))

    X = df_train.drop(["filename", "folder"], axis=1)
    X_test = df_test.drop("filename", axis=1)
    objective_params = {}

    if config.task == "binary":
        y = (df_train["folder"] != "Cover").to_numpy()
        objective_params["objective"] = "binary"
    elif config.task == "4class":
        y = df_train["folder"].map(lambda x: ["Cover", "JMiPOD", "JUNIWARD", "UERD"].index(x))
        objective_params["objective"] = "multiclass"
        objective_params["num_class"] = 4
    else:
        raise NotImplementedError

    if lightgbm_params.params is None:
        params = {"verbosity": -1, "boosting_type": "gbdt", **objective_params}
        tuner = lgb.LightGBMTunerCV(
            params,
            lgb.Dataset(X, label=y),
            num_boost_round=20000,
            verbose_eval=100,
            early_stopping_rounds=100,
            folds=folds_for_tuner,
        )
        tuner.run()
        somen.file_io.save_json(tuner.best_params, working_dir / "best_params.json")
        params = tuner.best_params
    else:
        params = {**lightgbm_params.params, **objective_params}

    cv_booster_store = CVBoosterStore()
    results = lgb.cv(
        params=params,
        train_set=lgb.Dataset(X, label=y),
        num_boost_round=lightgbm_params.num_boost_round,
        feval=alaska_weighted_auc_for_lightgbm,
        folds=folds,
        callbacks=[cv_booster_store],
        verbose_eval=True,
    )

    cv_booster = cv_booster_store.get()
    best_iteration = np.nanargmax(results["weighted_auc-mean"]) + 1
    print(best_iteration)

    for fold_index, booster in enumerate(cv_booster.boosters):
        for importance_type in ["split", "gain"]:
            lgb.plot_importance(booster, importance_type=importance_type)
            plt.savefig(working_dir / f"importance_{importance_type}_{fold_index}.png")
            plt.tight_layout()
            plt.close()

    if config.task == "binary":
        pred_valid = np.zeros(len(X), dtype=np.float64)
        pred_test = np.zeros(len(df_test), dtype=np.float64)
    elif config.task == "4class":
        pred_valid = np.zeros((len(X), 4), dtype=np.float64)
        pred_test = np.zeros((len(df_test), 4), dtype=np.float64)
    else:
        raise NotImplementedError

    for fold_index, booster in enumerate(cv_booster.boosters):
        valid_indices = folds[fold_index][1]
        pred_valid[valid_indices] = booster.predict(X.iloc[valid_indices], num_iteration=best_iteration)
        pred_test += booster.predict(X_test, num_iteration=best_iteration) / config.n_folds

    if config.task == "binary":
        print("validation score:", alaska_weighted_auc(y[::8], np.mean(pred_valid.reshape(-1, 8), axis=1)))
    else:
        print("calculation of the validation score is not implemented yet")

    somen.file_io.save_array(pred_valid, working_dir / "pred_valid.h5")
    somen.file_io.save_array(pred_test, working_dir / "pred_test.h5")

    submission = pd.DataFrame({"Id": df_test["filename"].iloc[::8]})
    if config.task == "binary":
        submission["Label"] = np.mean(pred_test.reshape(-1, 8), axis=1)
    elif config.task == "4class":
        submission["Label"] = np.mean((1 - pred_test[:, 0]).reshape(-1, 8), axis=1)
    else:
        raise NotImplementedError
    submission.to_csv(working_dir / "submission.csv", index=False)
