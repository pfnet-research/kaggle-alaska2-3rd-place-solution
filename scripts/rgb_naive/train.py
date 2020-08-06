import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import albumentations as A
import dacite
import numpy as np
import torch
import torch.nn.functional as F
from ignite.engine.engine import Engine
from ignite.utils import convert_tensor

import somen
import somen.albumentations_utility as As
from alaska2.datasets import BasicDataset, DistributedCoverStegoPairSampler, PseudoLabelingDataset
from alaska2.loading import get_in_channels
from alaska2.metrics import alaska_weighted_auc_metric_fun, cross_entropy_loss, reduced_focal_loss
from alaska2.models import Model, patch_first_conv
from alaska2.validation import get_folds_as_filename
from somen.pytorch_utility.extensions.mlflow import MLflowReporter, mlflow_start_run

# from somen.pytorch_utility.misc import arbitrary_length_all_gather


@dataclass
class ModelConfig:
    backbone: str
    pretrained: Optional[str]


@dataclass
class OptimizationConfig:
    optimizer: str
    optimizer_params: Mapping[str, Any]
    batch_size: int
    nb_epoch: int
    lr_scheduler: Optional[str]
    lr_scheduler_params: Optional[Mapping[str, Any]]
    cover_stego_pair: bool
    use_reduced_focal_loss: bool


@dataclass
class MixedSampleDataAugmentationConfig:
    method: str
    params: Mapping[str, Any]


@dataclass
class CutMixParams:
    beta: float
    cutmix_prob: float


@dataclass
class TrainingConfig:
    run_name: str
    fold_index: int
    n_folds: int
    read_method: str
    subsample: float
    model: ModelConfig
    optim: OptimizationConfig
    perform_augmentation: bool
    msda: Optional[MixedSampleDataAugmentationConfig]
    pseudo_label_run: Optional[str]
    soft_label: bool
    label_smooth: Optional[float]


@torch.no_grad()
def get_pred(
    model,
    filenames: Sequence[str],
    folders: Tuple[str],
    zip_path: Path,
    read_method: str,
    batch_size: int,
    num_workers: int,
    perform_augmentation: bool,
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
        # TTA
        aug_candidates = [A.VerticalFlip(p=1.0), A.HorizontalFlip(p=1.0), As.FixedFactorRandomRotate90(p=1.0, factor=1)]
        use_augs_gen = itertools.product([True, False], repeat=len(aug_candidates))
    else:
        use_augs_gen = [(False, False)]

    for use_augs in use_augs_gen:
        aug_list = [aug for use, aug in zip(use_augs, aug_candidates) if use]

        dataset = BasicDataset(zip_path, filenames, read_method, aug_list, inference=True, folders=folders)

        if distributed:
            example_per_process = (len(dataset) + world_size - 1) // world_size
            sections = [example_per_process * r for r in range(world_size)] + [len(dataset)]
            dataset = torch.utils.data.Subset(dataset, range(sections[global_rank], sections[global_rank + 1]))

        pred = somen.pytorch_utility.predict(model, dataset, batch_size, num_workers, device=device)[0]

        # if distributed:
        #     tensor_list = arbitrary_length_all_gather(torch.from_numpy(pred).to(device), axis=0, device=device)
        #     pred = torch.cat(tensor_list, dim=0).detach().cpu().numpy()

        preds.append(pred)

    preds = np.stack(preds, axis=-1)
    return preds


def make_prediction_h5(
    model,
    fold_index: int,
    n_folds: int,
    working_dir: Path,
    zip_path: Path,
    read_method: str,
    batch_size: int,
    num_workers: int,
    perform_augmentation: bool,
    prefix: str = "pred",
    local_rank: Optional[int] = None,
):
    folds = get_folds_as_filename(n_folds)
    train_filenames = folds[fold_index][0]
    valid_filenames = folds[fold_index][1]
    test_filenames = [f"{i:0>4}.jpg" for i in range(1, 5001)]

    distributed = local_rank is not None
    is_main_node = (not distributed) or torch.distributed.get_rank() == 0

    pred_train = get_pred(
        model,
        train_filenames,
        ("Cover", "JMiPOD", "JUNIWARD", "UERD"),
        zip_path,
        read_method,
        batch_size,
        num_workers,
        perform_augmentation,
        local_rank,
    )
    # if is_main_node:
    #     somen.file_io.save_array(pred_valid, working_dir / f"{prefix}_train.h5")
    if distributed:
        somen.file_io.save_array(pred_train, working_dir / f"{prefix}_train_{torch.distributed.get_rank()}.h5")
    else:
        somen.file_io.save_array(pred_train, working_dir / f"{prefix}_train.h5")
    del pred_train

    pred_valid = get_pred(
        model,
        valid_filenames,
        ("Cover", "JMiPOD", "JUNIWARD", "UERD"),
        zip_path,
        read_method,
        batch_size,
        num_workers,
        perform_augmentation,
        local_rank,
    )
    # if is_main_node:
    #     somen.file_io.save_array(pred_valid, working_dir / f"{prefix}_valid.h5")
    if distributed:
        somen.file_io.save_array(pred_valid, working_dir / f"{prefix}_valid_{torch.distributed.get_rank()}.h5")
    else:
        somen.file_io.save_array(pred_valid, working_dir / f"{prefix}_valid.h5")
    del pred_valid

    pred_test = get_pred(
        model,
        test_filenames,
        ("Test",),
        zip_path,
        read_method,
        batch_size,
        num_workers,
        perform_augmentation,
        local_rank,
    )
    # if is_main_node:
    #     somen.file_io.save_array(pred_test, working_dir / f"{prefix}_test.h5")
    if distributed:
        somen.file_io.save_array(pred_test, working_dir / f"{prefix}_test_{torch.distributed.get_rank()}.h5")
    else:
        somen.file_io.save_array(pred_test, working_dir / f"{prefix}_test.h5")
    del pred_test

    if distributed:
        torch.distributed.barrier()
        if is_main_node:
            world_size = torch.distributed.get_world_size()
            for target in ["train", "valid", "test"]:
                preds = []
                for rank in range(world_size):
                    path = working_dir / f"{prefix}_{target}_{rank}.h5"
                    preds.append(somen.file_io.load_array(path))
                preds = np.concatenate(preds, axis=0)
                somen.file_io.save_array(preds, working_dir / f"{prefix}_{target}.h5")

            for target in ["train", "valid", "test"]:
                for rank in range(world_size):
                    path = working_dir / f"{prefix}_{target}_{rank}.h5"
                    path.unlink()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def _dot_call_me(*args, **kwargs):
    assert False


def mixed_sample_data_augmentation(config: MixedSampleDataAugmentationConfig):
    if config.method == "CutMix":
        params = dacite.from_dict(CutMixParams, config.params)
        print(f"CutMix is selected: {params}")

        def get_update_fn(
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: Union[Callable, torch.nn.Module],
            device: Optional[Union[str, torch.device]] = None,
            label_indices: Sequence[int] = [-1],
            non_blocking: bool = False,
        ) -> Callable:
            def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
                model.train()
                optimizer.zero_grad()

                assert len(batch) == 2
                x = convert_tensor(batch[0], device=device, non_blocking=non_blocking)
                labels = convert_tensor(batch[1], device=device, non_blocking=non_blocking)

                # https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L227-L244
                r = np.random.rand()
                if params.beta > 0 and r < params.cutmix_prob:
                    # generate mixed sample
                    lam = np.random.beta(params.beta, params.beta)
                    rand_index = torch.randperm(x.size()[0]).cuda()
                    labels_a = labels
                    labels_b = labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                    # compute output
                    output = model(x)
                    loss = loss_fn(output, labels_a) * lam + loss_fn(output, labels_b) * (1.0 - lam)
                else:
                    # compute output
                    output = model(x)
                    loss = loss_fn(output, labels)

                loss.backward()
                optimizer.step()

                return loss.item()

            return _update

        return get_update_fn

    if config.method == "SoftLabelCutMix":
        params = dacite.from_dict(CutMixParams, config.params)
        print(f"SoftLabelCutMix is selected: {params}")

        def get_update_fn(
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: Union[Callable, torch.nn.Module],
            device: Optional[Union[str, torch.device]] = None,
            label_indices: Sequence[int] = [-1],
            non_blocking: bool = False,
        ) -> Callable:
            def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
                model.train()
                optimizer.zero_grad()

                assert len(batch) == 2
                x = convert_tensor(batch[0], device=device, non_blocking=non_blocking)
                labels = convert_tensor(batch[1], device=device, non_blocking=non_blocking)

                assert labels.ndim == 2  # must be soft labels

                # https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L227-L244
                r = np.random.rand()
                if params.beta > 0 and r < params.cutmix_prob:
                    # generate mixed sample
                    lam = np.random.beta(params.beta, params.beta)
                    rand_index = torch.randperm(x.size()[0]).cuda()
                    labels_a = labels
                    labels_b = labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                    labels = labels_a * lam + labels_b * (1.0 - lam)

                output = model(x)
                loss = loss_fn(output, labels)

                loss.backward()
                optimizer.step()

                return loss.item()

            return _update

        return get_update_fn

    raise ValueError


def setup_model(config):
    model = Model(4, config.model.backbone)

    in_channels = get_in_channels(config.read_method)
    if in_channels != 3:
        patch_first_conv(model, in_channels, reuse=config.read_method.startswith("RGB"))

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
    if config.model.pretrained is not None:
        pretrained = config.model.pretrained.format(fold_index=config.fold_index)
        print(f"Load pretrained model from: {pretrained}")
        model.load_state_dict(torch.load(pretrained, map_location="cpu"))

    folds = get_folds_as_filename(config.n_folds)
    train_filenames = folds[config.fold_index][0]
    valid_filenames = folds[config.fold_index][1]

    if config.subsample < 1.0:
        rng = np.random.RandomState(0)
        train_permutation = rng.permutation(len(train_filenames))[: int(len(train_filenames) * config.subsample)]
        train_filenames = train_filenames[train_permutation]
        valid_permutation = rng.permutation(len(valid_filenames))[: int(len(valid_filenames) * config.subsample)]
        valid_filenames = valid_filenames[valid_permutation]

    if config.perform_augmentation:
        train_augs = [A.VerticalFlip(), A.HorizontalFlip(), As.FixedFactorRandomRotate90(factor=1)]
    else:
        train_augs = []

    if config.msda is not None:
        assert config.perform_augmentation
        get_update_fn = mixed_sample_data_augmentation(config.msda)
    else:
        get_update_fn = None

    train_dataset = BasicDataset(
        args.zip_path,
        train_filenames,
        config.read_method,
        train_augs,
        soft_label=config.soft_label,
        label_smooth=config.label_smooth,
    )
    if config.pseudo_label_run is not None:
        test_filenames = [f"{i:0>4}.jpg" for i in range(1, 5001)]
        prediction_valid = somen.file_io.load_array(
            f"data/working/{config.pseudo_label_run}/{config.fold_index}/pred_valid.h5"
        )
        prediction_test = somen.file_io.load_array(
            f"data/working/{config.pseudo_label_run}/{config.fold_index}/pred_test.h5"
        )
        if config.subsample < 1.0:
            shape1 = prediction_valid.shape[1:]
            prediction_valid = prediction_valid.reshape(-1, 4, *shape1)
            prediction_valid = prediction_valid[valid_permutation]
            prediction_valid = prediction_valid.reshape(-1, *shape1)

        train_dataset = torch.utils.data.ConcatDataset(
            [
                train_dataset,
                PseudoLabelingDataset(
                    zip_path=args.zip_path,
                    filenames=valid_filenames,
                    read_method=config.read_method,
                    prediction=prediction_valid,
                    soft_label=config.soft_label,
                    label_smooth=config.label_smooth,
                ),
                PseudoLabelingDataset(
                    zip_path=args.zip_path,
                    filenames=test_filenames,
                    read_method=config.read_method,
                    prediction=prediction_test,
                    folders=("Test",),
                    soft_label=config.soft_label,
                    label_smooth=config.label_smooth,
                ),
            ]
        )

    if config.optim.use_reduced_focal_loss:
        loss_fn = reduced_focal_loss
    else:
        loss_fn = cross_entropy_loss

    valid_augs = []
    valid_dataset = BasicDataset(args.zip_path, valid_filenames, config.read_method, valid_augs)

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
        "train_sampler": (DistributedCoverStegoPairSampler(train_dataset) if config.optim.cover_stego_pair else None),
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

    make_prediction_h5(
        model=torch.nn.Sequential(model, torch.nn.Softmax(dim=1)),
        fold_index=config.fold_index,
        n_folds=config.n_folds,
        working_dir=working_dir,
        zip_path=args.zip_path,
        read_method=config.read_method,
        batch_size=config.optim.batch_size * 2,
        num_workers=args.num_workers,
        perform_augmentation=config.perform_augmentation,
        local_rank=args.local_rank,
    )


def predict_feature(args):
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, args.config, args.overrides)
    working_dir = Path(f"data/working/{config.run_name}/{config.fold_index}/")

    model = setup_model(config)
    model.load_state_dict(torch.load(working_dir / "final.pth", map_location="cpu"))

    class OutputFeature(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.flatten = torch.nn.Flatten()

        def forward(self, x):
            feature = self.model.get_feature(x)
            feature = self.pool(feature)
            feature = self.flatten(feature)
            return feature

    make_prediction_h5(
        model=OutputFeature(),
        fold_index=config.fold_index,
        n_folds=config.n_folds,
        working_dir=working_dir,
        zip_path=args.zip_path,
        read_method=config.read_method,
        batch_size=config.optim.batch_size * 2,
        num_workers=args.num_workers,
        perform_augmentation=config.perform_augmentation,
        prefix="feature",
        local_rank=args.local_rank,
    )


def download_model(args):
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, args.config, args.overrides)
    setup_model(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--zip_path", type=str, default="data/input/alaska2-image-steganalysis.zip")
    parser.add_argument("--local_rank", type=int, default=None)

    parser.add_argument(
        "--task", type=str, choices=["train", "predict", "predict_feature", "download_model"], required=True
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", type=str)
    args = parser.parse_args()

    if args.task == "train":
        train(args)
    elif args.task == "predict":
        predict(args)
    elif args.task == "predict_feature":
        predict_feature(args)
    elif args.task == "download_model":
        download_model(args)
