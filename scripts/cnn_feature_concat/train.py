import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import somen
from alaska2.metrics import alaska_weighted_auc_metric_fun, cross_entropy_loss, reduced_focal_loss
from somen.pytorch_utility.extensions.mlflow import MLflowReporter, mlflow_start_run
from somen.pytorch_utility.misc import arbitrary_length_all_gather


@dataclass
class ModelConfig:
    apply_softmax_to_each_input: bool
    layers: Sequence[str]


@dataclass
class OptimizationConfig:
    optimizer: str
    optimizer_params: Mapping[str, Any]
    batch_size: int
    nb_epoch: int
    lr_scheduler: Optional[str]
    lr_scheduler_params: Optional[Mapping[str, Any]]
    use_reduced_focal_loss: bool


@dataclass
class TrainingConfig:
    run_name: str
    fold_index: int
    n_folds: int
    model: ModelConfig
    optim: OptimizationConfig
    perform_augmentation: bool
    cnn_run_names: Sequence[str]
    use_bn: bool
    pseudo_label_run: Optional[str]
    soft_label: bool
    label_smooth: Optional[float]


class Model(torch.nn.Sequential):
    def __init__(self, layers: Sequence[str], n_feats: Optional[int] = None):
        super(Model, self).__init__()
        self.n_feats = n_feats
        if n_feats is not None:
            self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(1) for _ in range(n_feats)])
        self.head = torch.nn.Sequential(*[eval(layer) for layer in layers])

    def forward(self, *feats):
        if self.n_feats is not None:
            assert len(feats) == self.n_feats
            x = torch.cat([bn(feat.reshape(-1, 1)).reshape(feat.shape) for feat, bn in zip(feats, self.bns)], dim=1)
        else:
            assert len(feats) == 1
            x = feats[0]
        return self.head(x)


class FeatureConcatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        feature_file_paths,
        use_aug_indices,
        feature_stats=None,
        inference=False,
        soft_label: bool = False,
        label_smooth: Optional[float] = None,
    ):
        assert len(use_aug_indices) >= 1

        features = [somen.file_io.load_array(path) for path in feature_file_paths]
        assert all([feat.ndim == 3 for feat in features])
        assert all([feat.shape[dim] == features[0].shape[dim] for feat in features[1:] for dim in [0, 2]])

        if feature_stats is None:
            feature_stats = [(np.mean(feat), np.std(feat)) for feat in features]
        features = [(feat - mean) / std for feat, (mean, std) in zip(features, feature_stats)]
        features = np.concatenate(features, axis=1)

        self.feature_file_paths = feature_file_paths
        self.feature_stats = feature_stats
        self.inference = inference
        self.use_aug_indices = use_aug_indices
        self.features = features
        self.soft_label = soft_label
        self.label_smooth = label_smooth

    def update_use_aug_indices(self, use_aug_indices) -> None:
        self.use_aug_indices = use_aug_indices

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index, aug_index=None):
        if aug_index is None:
            aug_index = np.random.choice(self.use_aug_indices)
        feat = self.features[index, :, aug_index]

        if self.inference:
            return feat
        else:
            class_index = index % 4
            if self.soft_label:
                label = np.zeros(4, dtype=np.float32)
                label[class_index] = 1.0
                if self.label_smooth is not None:
                    label = np.full_like(label, self.label_smooth / len(label)) + (1 - self.label_smooth) * label
            else:
                label = class_index
            return feat, label


class FeatureTupleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        feature_file_paths,
        use_aug_indices,
        inference=False,
        soft_label: bool = False,
        label_smooth: Optional[float] = None,
        apply_softmax_to_each_input: bool = True,
    ):
        assert len(use_aug_indices) >= 1

        features = [somen.file_io.load_array(path) for path in feature_file_paths]
        assert all([feat.ndim == 3 for feat in features])
        assert all([feat.shape[dim] == features[0].shape[dim] for feat in features[1:] for dim in [0, 2]])

        # NOTE: prediction に対して softmax をかけていた bug の結果に合わせるため
        if apply_softmax_to_each_input:
            for feat in features:
                feat[:] = F.softmax(torch.from_numpy(feat), dim=1).numpy()

        self.feature_file_paths = feature_file_paths
        self.inference = inference
        self.use_aug_indices = use_aug_indices
        self.features = features
        self.soft_label = soft_label
        self.label_smooth = label_smooth

    def update_use_aug_indices(self, use_aug_indices) -> None:
        self.use_aug_indices = use_aug_indices

    def __len__(self) -> int:
        return len(self.features[0])

    def __getitem__(self, index, aug_index=None):
        if aug_index is None:
            aug_index = np.random.choice(self.use_aug_indices)
        feats = [feat[index, :, aug_index] for feat in self.features]

        if self.inference:
            return tuple(feats)
        else:
            class_index = index % 4
            if self.soft_label:
                label = np.zeros(4, dtype=np.float32)
                label[class_index] = 1.0
                if self.label_smooth is not None:
                    label = np.full_like(label, self.label_smooth / len(label)) + (1 - self.label_smooth) * label
            else:
                label = class_index
            return tuple(feats) + (label,)


class PseudoLabelingDataset(torch.utils.data.Dataset):
    def __init__(self, prediction, dataset, soft_label: bool, label_smooth: Optional[float] = None):
        assert len(prediction) == len(dataset)
        self.prediction = prediction
        self.dataset = dataset
        self.soft_label = soft_label
        self.label_smooth = label_smooth

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        aug_index = np.random.randint(8)
        feats = self.dataset.__getitem__(index, aug_index=aug_index)
        if not isinstance(feats, tuple):
            feats = (feats,)
        if not self.dataset.inference:
            feats = feats[:-1]
        if self.soft_label:
            label = self.prediction[index, :, aug_index].astype(np.float32)
            if self.label_smooth is not None:
                label = np.full_like(label, self.label_smooth / len(label)) + (1 - self.label_smooth) * label
        else:
            label = np.random.choice(self.prediction.shape[1], p=self.prediction[index, :, aug_index])
        return tuple(feats) + (label,)


@torch.no_grad()
def get_pred(
    model,
    use_bn: bool,
    feature_stats,
    feature_file_paths: Sequence[Path],
    batch_size: int,
    num_workers: int,
    perform_augmentation: bool,
    apply_softmax_to_each_input: bool,
    local_rank: Optional[int] = None,
):
    if local_rank is not None:
        distributed = True
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        device = f"cuda:{local_rank}"
    else:
        distributed = False
        device = "cuda"

    preds = []

    if perform_augmentation:
        use_aug_indices = list(range(8))
    else:
        use_aug_indices = [-1]

    if use_bn:
        dataset = FeatureTupleDataset(
            feature_file_paths=feature_file_paths,
            use_aug_indices=["PlaceHolder"],
            inference=True,
            apply_softmax_to_each_input=apply_softmax_to_each_input,
        )
    else:
        dataset = FeatureConcatDataset(
            feature_file_paths=feature_file_paths,
            use_aug_indices=["PlaceHolder"],
            inference=True,
            feature_stats=feature_stats,
        )

    for use_aug_index in use_aug_indices:
        dataset.update_use_aug_indices([use_aug_index])

        if distributed:
            example_per_process = (len(dataset) + world_size - 1) // world_size
            sections = [example_per_process * r for r in range(world_size)] + [len(dataset)]
            dataset = torch.utils.data.Subset(dataset, range(sections[global_rank], sections[global_rank + 1]))

        pred = somen.pytorch_utility.predict(model, dataset, batch_size, num_workers, device=device)[0]
        pred = torch.nn.functional.softmax(torch.from_numpy(pred), dim=1).detach().numpy()

        if distributed:
            tensor_list = arbitrary_length_all_gather(torch.from_numpy(pred).to(device), axis=0, device=device)
            pred = torch.cat(tensor_list, dim=0).detach().cpu().numpy()

        preds.append(pred)

    preds = np.stack(preds, axis=-1)
    return preds


def make_prediction_h5(
    model,
    use_bn: bool,
    feature_stats,
    cnn_run_names: Sequence[str],
    fold_index: int,
    working_dir: Path,
    batch_size: int,
    num_workers: int,
    perform_augmentation: bool,
    apply_softmax_to_each_input: bool,
    prefix: str = "pred",
    local_rank: Optional[int] = None,
):
    train_feature_file_paths, valid_feature_file_paths, test_feature_file_paths = [], [], []
    for run_name in cnn_run_names:
        run_dir = Path(f"data/working/{run_name}/{fold_index}/")
        train_feature_file_paths.append(run_dir / "feature_train.h5")
        valid_feature_file_paths.append(run_dir / "feature_valid.h5")
        test_feature_file_paths.append(run_dir / "feature_test.h5")

    is_main_node = local_rank is None or torch.distributed.get_rank() == 0

    pred_train = get_pred(
        model=model,
        use_bn=use_bn,
        feature_stats=feature_stats,
        feature_file_paths=train_feature_file_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        perform_augmentation=perform_augmentation,
        apply_softmax_to_each_input=apply_softmax_to_each_input,
        local_rank=local_rank,
    )
    if is_main_node:
        somen.file_io.save_array(pred_train, working_dir / f"{prefix}_train.h5")
    del pred_train

    pred_valid = get_pred(
        model=model,
        use_bn=use_bn,
        feature_stats=feature_stats,
        feature_file_paths=valid_feature_file_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        perform_augmentation=perform_augmentation,
        apply_softmax_to_each_input=apply_softmax_to_each_input,
        local_rank=local_rank,
    )
    if is_main_node:
        somen.file_io.save_array(pred_valid, working_dir / f"{prefix}_valid.h5")
    del pred_valid

    pred_test = get_pred(
        model=model,
        use_bn=use_bn,
        feature_stats=feature_stats,
        feature_file_paths=test_feature_file_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        perform_augmentation=perform_augmentation,
        apply_softmax_to_each_input=apply_softmax_to_each_input,
        local_rank=local_rank,
    )
    if is_main_node:
        somen.file_io.save_array(pred_test, working_dir / f"{prefix}_test.h5")
    del pred_test


def setup_model(config):
    if config.use_bn:
        model = Model(config.model.layers, len(config.cnn_run_names))
    else:
        model = Model(config.model.layers)
    return model


def train(args):
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, args.config, args.overrides)
    working_dir = Path(f"data/working/{config.run_name}/{config.fold_index}/")

    distributed = args.local_rank is not None
    if distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    is_main_node = (not distributed) or (torch.distributed.get_rank() == 0)
    if is_main_node:
        somen.file_io.save_json(vars(args), working_dir / "args.json", indent=4)
        somen.file_io.save_yaml_from_dataclass(config, working_dir / "config.yaml")

    model = setup_model(config)

    get_update_fn = None
    if config.optim.use_reduced_focal_loss:
        loss_fn = reduced_focal_loss
    else:
        loss_fn = cross_entropy_loss

    train_feature_file_paths, valid_feature_file_paths = [], []
    for run_name in config.cnn_run_names:
        run_dir = Path(f"data/working/{run_name}/{config.fold_index}/")
        train_feature_file_paths.append(run_dir / "feature_train.h5")
        valid_feature_file_paths.append(run_dir / "feature_valid.h5")
    assert all([p.exists() for p in train_feature_file_paths])
    assert all([p.exists() for p in valid_feature_file_paths])

    train_use_aug_indices = list(range(8))
    if config.use_bn:
        train_dataset = FeatureTupleDataset(
            feature_file_paths=train_feature_file_paths,
            use_aug_indices=train_use_aug_indices,
            soft_label=config.soft_label,
            label_smooth=config.label_smooth,
            apply_softmax_to_each_input=config.model.apply_softmax_to_each_input,
        )
    else:
        train_dataset = FeatureConcatDataset(
            feature_file_paths=train_feature_file_paths,
            use_aug_indices=train_use_aug_indices,
            soft_label=config.soft_label,
            label_smooth=config.label_smooth,
        )
        feature_stats = train_dataset.feature_stats
        somen.file_io.save_pickle(feature_stats, working_dir / "feature_stats.pkl")

    valid_use_aug_indices = [-1]  # NOTE: itertools の product で全部 False が最後になっているはず
    if config.use_bn:
        valid_dataset = FeatureTupleDataset(
            feature_file_paths=valid_feature_file_paths,
            use_aug_indices=valid_use_aug_indices,
            apply_softmax_to_each_input=config.model.apply_softmax_to_each_input,
        )
    else:
        valid_dataset = FeatureConcatDataset(
            feature_file_paths=valid_feature_file_paths,
            use_aug_indices=valid_use_aug_indices,
            feature_stats=feature_stats,
        )

    if config.pseudo_label_run is not None:
        test_feature_file_paths = []
        for run_name in config.cnn_run_names:
            run_dir = Path(f"data/working/{run_name}/{config.fold_index}/")
            test_feature_file_paths.append(run_dir / "feature_test.h5")

        prediction_valid = somen.file_io.load_array(
            f"data/working/{config.pseudo_label_run}/{config.fold_index}/pred_valid.h5"
        )
        prediction_test = somen.file_io.load_array(
            f"data/working/{config.pseudo_label_run}/{config.fold_index}/pred_test.h5"
        )
        if config.use_bn:
            test_dataset = FeatureTupleDataset(
                feature_file_paths=test_feature_file_paths,
                use_aug_indices=[-1],
                apply_softmax_to_each_input=config.model.apply_softmax_to_each_input,
            )
        else:
            test_dataset = FeatureConcatDataset(
                feature_file_paths=test_feature_file_paths, use_aug_indices=[-1], feature_stats=feature_stats,
            )
        train_dataset = torch.utils.data.ConcatDataset(
            [
                train_dataset,
                PseudoLabelingDataset(
                    prediction=prediction_valid,
                    dataset=valid_dataset,
                    soft_label=config.soft_label,
                    label_smooth=config.label_smooth,
                ),
                PseudoLabelingDataset(
                    prediction=prediction_test,
                    dataset=test_dataset,
                    soft_label=config.soft_label,
                    label_smooth=config.label_smooth,
                ),
            ]
        )

    params = {
        "objective": loss_fn,
        "get_update_fn": get_update_fn,
        "optimizer": config.optim.optimizer,
        "optimizer_params": config.optim.optimizer_params,
        "nb_epoch": config.optim.nb_epoch,
        "batch_size": config.optim.batch_size,
        "device": args.device,
        "num_workers": args.num_workers,
        "resume": args.resume,
        "benchmark_mode": args.benchmark,
        "enable_cprofile": args.cprofile,
        "trainer_snapshot_n_saved": 1,
        "metric": [("weighted_auc", alaska_weighted_auc_metric_fun), ("loss", F.cross_entropy)],
        "batch_eval": True,
        "maximize": True,
        "lr_scheduler": config.optim.lr_scheduler,
        "lr_scheduler_params": config.optim.lr_scheduler_params,
        "local_rank": args.local_rank,
    }
    ext_extensions = []

    if args.mlflow:
        exp_name = config.dataset.name if args.mlflow_exp_name is None else args.mlflow_exp_name
        run_name = config.name if args.mlflow_run_name is None else args.mlflow_run_name
        mlflow_start_run(exp_name, working_dir, args.resume, run_name)
        ext_extensions.append(MLflowReporter())

    # if distributed:
    #     from somen.pytorch_utility.extensions.ddp_param_sync_checker import DDPParamSyncChecker
    #     ext_extensions.append(DDPParamSyncChecker(model))

    try:
        somen.pytorch_utility.train(
            model=model,
            params=params,
            train_set=train_dataset,
            valid_sets=[valid_dataset],
            working_dir=working_dir,
            load_best=False,
            ext_extensions=ext_extensions,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if is_main_node:
            torch.save(model.state_dict(), working_dir / "final.pth")


def predict(args):
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, args.config, args.overrides)
    working_dir = Path(f"data/working/{config.run_name}/{config.fold_index}/")

    model = setup_model(config)
    model.load_state_dict(torch.load(working_dir / "final.pth", map_location="cpu"))

    if config.use_bn:
        feature_stats = None
    else:
        feature_stats = somen.file_io.load_pickle(working_dir / "feature_stats.pkl")

    make_prediction_h5(
        model=model,
        use_bn=config.use_bn,
        feature_stats=feature_stats,
        cnn_run_names=config.cnn_run_names,
        fold_index=config.fold_index,
        working_dir=working_dir,
        batch_size=config.optim.batch_size * 2,
        num_workers=args.num_workers,
        perform_augmentation=config.perform_augmentation,
        apply_softmax_to_each_input=config.model.apply_softmax_to_each_input,
        prefix="pred",
        local_rank=args.local_rank,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--local_rank", type=int, default=None)

    parser.add_argument("--task", type=str, choices=["train", "predict"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", type=str)
    args = parser.parse_args()

    if args.task == "train":
        train(args)
    elif args.task == "predict":
        predict(args)
