import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import pfio
import torch
import torch.nn.functional as F
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

import somen
import somen.albumentations_utility as As
from alaska2.loading import read_image_from_container
from alaska2.metrics import alaska_weighted_auc_metric_fun
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


@dataclass
class TrainingConfig:
    run_name: str
    fold_index: int
    n_folds: int
    subsample: float
    model: ModelConfig
    optim: OptimizationConfig
    perform_augmentation: bool
    use_dilation_8: bool
    strides: Sequence[int]
    L: int
    include_Q_matrix: bool
    include_cos_matrix: bool
    parallel_by_pos: bool = False
    include_lsb: bool = False
    crop_quarter: bool = False
    patch_dilation_first_stride: bool = False
    replace_second_stride_by_max_pool: bool = False
    crop_hint_run: Optional[str] = None
    crop_softmax_s: float = 5.0


QUANT_TABLES = [
    [
        np.array(
            [
                [8, 6, 5, 8, 12, 20, 26, 31],
                [6, 6, 7, 10, 13, 29, 30, 28],
                [7, 7, 8, 12, 20, 29, 35, 28],
                [7, 9, 11, 15, 26, 44, 40, 31],
                [9, 11, 19, 28, 34, 55, 52, 39],
                [12, 18, 28, 32, 41, 52, 57, 46],
                [25, 32, 39, 44, 52, 61, 60, 51],
                [36, 46, 48, 49, 56, 50, 52, 50],
            ],
            dtype=np.int32,
        ),
        np.array(
            [
                [9, 9, 12, 24, 50, 50, 50, 50],
                [9, 11, 13, 33, 50, 50, 50, 50],
                [12, 13, 28, 50, 50, 50, 50, 50],
                [24, 33, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
            ],
            dtype=np.int32,
        ),
    ],
    [
        np.array(
            [
                [3, 2, 2, 3, 5, 8, 10, 12],
                [2, 2, 3, 4, 5, 12, 12, 11],
                [3, 3, 3, 5, 8, 11, 14, 11],
                [3, 3, 4, 6, 10, 17, 16, 12],
                [4, 4, 7, 11, 14, 22, 21, 15],
                [5, 7, 11, 13, 16, 21, 23, 18],
                [10, 13, 16, 17, 21, 24, 24, 20],
                [14, 18, 19, 20, 22, 20, 21, 20],
            ],
            dtype=np.int32,
        ),
        np.array(
            [
                [3, 4, 5, 9, 20, 20, 20, 20],
                [4, 4, 5, 13, 20, 20, 20, 20],
                [5, 5, 11, 20, 20, 20, 20, 20],
                [9, 13, 20, 20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20, 20, 20, 20],
            ],
            dtype=np.int32,
        ),
    ],
    [
        np.array(
            [
                [2, 1, 1, 2, 2, 4, 5, 6],
                [1, 1, 1, 2, 3, 6, 6, 6],
                [1, 1, 2, 2, 4, 6, 7, 6],
                [1, 2, 2, 3, 5, 9, 8, 6],
                [2, 2, 4, 6, 7, 11, 10, 8],
                [2, 4, 6, 6, 8, 10, 11, 9],
                [5, 6, 8, 9, 10, 12, 12, 10],
                [7, 9, 10, 10, 11, 10, 10, 10],
            ],
            dtype=np.int32,
        ),
        np.array(
            [
                [2, 2, 2, 5, 10, 10, 10, 10],
                [2, 2, 3, 7, 10, 10, 10, 10],
                [2, 3, 6, 10, 10, 10, 10, 10],
                [5, 7, 10, 10, 10, 10, 10, 10],
                [10, 10, 10, 10, 10, 10, 10, 10],
                [10, 10, 10, 10, 10, 10, 10, 10],
                [10, 10, 10, 10, 10, 10, 10, 10],
                [10, 10, 10, 10, 10, 10, 10, 10],
            ],
            dtype=np.int32,
        ),
    ],
]

QUANT_TABLES_AS_FEATURE = [[np.tile(t.astype(np.float32) / 50, (64, 64)) for t in ts] for ts in QUANT_TABLES]


class DCTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zip_path,
        quality_indices,
        filenames,
        deterministic_aug_list,
        include_Q_matrix: bool,
        include_cos_matrix: bool,
        include_lsb: bool,
        inference=False,
        folders=("Cover", "JMiPOD", "JUNIWARD", "UERD"),
        crop_prob: Optional[np.ndarray] = None,
    ):
        if crop_prob is not None:
            assert crop_prob.ndim == 3  # (example_index, pos_index, aug_index)
            assert crop_prob.shape == (len(filenames) * len(folders), 9, 8)

        self.quality_indices = quality_indices
        self.filenames = filenames
        self.folders = folders
        self.container = pfio.open_as_container(zip_path)
        self.deterministic_aug_list = deterministic_aug_list
        self.include_Q_matrix = include_Q_matrix
        self.include_cos_matrix = include_cos_matrix
        self.include_lsb = include_lsb
        self.inference = inference
        self.crop_prob = crop_prob

        cos_matrix = np.ones((64, 8, 64, 8), dtype=np.float32)
        cos_matrix *= np.cos(np.pi * np.arange(8) / 16).reshape(1, 8, 1, 1)
        cos_matrix *= np.cos(np.pi * np.arange(8) / 16).reshape(1, 1, 1, 8)
        cos_matrix = cos_matrix.reshape(512, 512)
        self._cos_matrix = cos_matrix

    def __len__(self) -> int:
        return len(self.filenames) * len(self.folders)

    def __getitem__(self, index):
        class_index = index % len(self.folders)
        filename = self.filenames[index // len(self.folders)]
        quality_index = self.quality_indices[index // len(self.folders)]

        dct = read_image_from_container(self.container, f"{self.folders[class_index]}/{filename}", "DCT")
        example = {"image": dct}

        mask = []

        if self.include_Q_matrix:
            mask.append(QUANT_TABLES_AS_FEATURE[quality_index][0].copy())
            mask.append(QUANT_TABLES_AS_FEATURE[quality_index][1].copy())

        if self.include_cos_matrix:
            mask.append(self._cos_matrix.copy())

        if self.include_lsb:
            lsb = (dct & 1).astype(np.float32)
            mask.append(lsb[..., 0])
            mask.append(lsb[..., 1])
            mask.append(lsb[..., 2])

        if len(mask) != 0:
            example["mask"] = np.stack(mask, axis=-1)

        aug_index = np.random.randint(len(self.deterministic_aug_list))
        example = self.deterministic_aug_list[aug_index](**example)

        img = example["image"]
        img = np.moveaxis(img, 2, 0)
        assert img.dtype == np.int32

        mask = example.get("mask", None)
        if mask is not None:
            mask = np.moveaxis(mask, 2, 0)

        if self.crop_prob is not None:
            crop_index = np.random.choice(9, p=self.crop_prob[index, :, aug_index])
            crop_row = crop_index // 3
            crop_col = crop_index % 3
            crop_slice = (
                slice(None),
                slice(crop_row * 128, crop_row * 128 + 256),
                slice(crop_col * 128, crop_col * 128 + 256),
            )
            img = img[crop_slice]
            if mask is not None:
                mask = mask[crop_slice]

        ret = (img,)
        if mask is not None:
            ret += (mask,)
        if not self.inference:
            ret += (class_index,)

        return ret


class DCTD8Model(Model):
    def __init__(
        self,
        num_classes: int,
        backbone: str,
        use_dilation_8: bool,
        strides: Sequence[int],
        L: int,
        include_Q_matrix: bool,
        include_cos_matrix: bool,
        parallel_by_pos: bool,
        include_lsb: bool,
        crop_quarter: bool,
        patch_dilation_first_stride: bool,
        replace_second_stride_by_max_pool: bool,
    ):
        super(DCTD8Model, self).__init__(num_classes, backbone)

        assert L % 2 == 0
        assert len(strides) == 2

        in_channels = 4 * L
        if include_Q_matrix:
            in_channels += 2
        if include_cos_matrix:
            in_channels += 1
        if include_lsb:
            in_channels += 1

        self._in_channels = in_channels
        self.use_dilation_8 = use_dilation_8
        self.strides = strides
        self.L = L
        self.include_Q_matrix = include_Q_matrix
        self.include_cos_matrix = include_cos_matrix
        self.parallel_by_pos = parallel_by_pos
        self.include_lsb = include_lsb
        self.crop_quarter = crop_quarter
        self.patch_dilation_first_stride = patch_dilation_first_stride
        self.replace_second_stride_by_max_pool = replace_second_stride_by_max_pool

        self._patch_backbone()

    def _patch_dilation(self, module, ignore=False):
        assert tuple(module.stride) == (1, 1)
        assert tuple(module.dilation) == (1, 1)
        module.dilation = (8, 8)

        kh, kw = module.weight.size()[-2:]
        assert kh == kw
        assert kh % 2 == 1
        pad_before = kh // 2
        pad = kh // 2 * 8

        if isinstance(module, Conv2dStaticSamePadding):
            assert (
                ignore
                or type(module.static_padding) is not torch.nn.ZeroPad2d
                or module.static_padding.padding == (pad_before, pad_before, pad_before, pad_before,)
            )
            assert ignore or module.padding == (0, 0)
            module.static_padding = torch.nn.Identity()
            module.padding = (pad, pad)
        else:
            assert ignore or type(module) is torch.nn.Conv2d
            assert ignore or module.padding == (pad_before, pad_before)
            module.padding = (pad, pad)

    def _patch_backbone(self):
        patch_first_conv(self.model, self._in_channels, reuse=False)

        # find the first conv
        modules_iter = iter(self.model.named_modules())
        for _, module in modules_iter:
            if isinstance(module, torch.nn.Conv2d) and tuple(module.stride) == (2, 2):
                break

        # patch first conv's stride
        module.stride = (self.strides[0], self.strides[0])
        if self.patch_dilation_first_stride:
            self._patch_dilation(module, True)

        if self.use_dilation_8:
            for name, module in modules_iter:
                if isinstance(module, torch.nn.Conv2d) and tuple(module.stride) == (2, 2):
                    if self.replace_second_stride_by_max_pool:
                        replacer = torch.nn.MaxPool2d(self.strides[1], self.strides[1])
                        keys = name.split(".")
                        parent = self.model
                        for key in keys[:-1]:
                            if key.isdigit():
                                parent = parent[int(key)]
                            else:
                                parent = getattr(parent, key)
                        key = keys[-1]
                        if key.isdigit():
                            parent[int(key)] = replacer
                        else:
                            setattr(parent, key, replacer)
                    else:
                        module.stride = (self.strides[1], self.strides[1])
                    break
                if isinstance(module, torch.nn.Conv2d):
                    self._patch_dilation(module)

    @torch.no_grad()
    def _get_x(self, dct, mask=None):
        x = torch.zeros((dct.shape[0], self._in_channels,) + dct.shape[2:], device=dct.device).float()

        arange = torch.arange(1, self.L)[np.newaxis, :, np.newaxis, np.newaxis].to(dct.device)

        # Luminance
        start, length = 0, self.L - 1
        x[:, start : start + length, :, :] = dct[:, 0:1, :, :] == arange
        start += length

        x[:, start, :, :] = dct[:, 0, :, :] >= self.L
        start += 1

        length = self.L - 1
        x[:, start : start + length, :, :] = dct[:, 0:1, :, :] == -arange
        start += length

        x[:, start, :, :] = dct[:, 0, :, :] <= -self.L
        start += 1

        # Chrominance-1
        L_per_2 = self.L // 2
        arange = arange[:, : L_per_2 - 1]

        length = L_per_2 - 1
        x[:, start : start + length, :, :] = dct[:, 1:2, :, :] == arange
        start += length

        x[:, start, :, :] = dct[:, 1, :, :] >= L_per_2
        start += 1

        length = L_per_2 - 1
        x[:, start : start + length, :, :] = dct[:, 1:2, :, :] == -arange
        start += length

        x[:, start, :, :] = dct[:, 1, :, :] <= -L_per_2
        start += 1

        # Chrominance-2
        length = L_per_2 - 1
        x[:, start : start + length, :, :] = dct[:, 2:3, :, :] == arange
        start += length

        x[:, start, :, :] = dct[:, 2, :, :] >= L_per_2
        start += 1

        length = L_per_2 - 1
        x[:, start : start + length, :, :] = dct[:, 2:3, :, :] == -arange
        start += length

        x[:, start, :, :] = dct[:, 2, :, :] <= -L_per_2
        start += 1

        if mask is not None:
            x[:, -mask.shape[1] :, :, :] = mask

        return x

    def get_feature_from_dct(self, dct, mask=None):
        assert not self.parallel_by_pos
        assert dct.dtype == torch.int32
        x = self._get_x(dct, mask)

        if self.crop_quarter:
            x = x[..., : x.shape[-2] // 2, : x.shape[-1] // 2]

        return self.get_feature(x)

    def forward(self, dct, mask=None):
        x = self._get_x(dct, mask)

        if self.crop_quarter:
            x = x[..., : x.shape[-2] // 2, : x.shape[-1] // 2]

        if self.parallel_by_pos:
            x = x.reshape(x.shape[0], x.shape[1], 64, 8, 64, 8)
            x = x.permute(0, 3, 5, 1, 2, 4)
            x = x.reshape(-1, *x.shape[-3:])

        y = super(DCTD8Model, self).forward(x)

        if self.parallel_by_pos:
            y = y.reshape(dct.shape[0], 64, y.shape[-1])
            y = torch.mean(y, dim=1)

        return y


@torch.no_grad()
def get_pred(
    model,
    quality_indices: Sequence[int],
    filenames: Sequence[str],
    folders: Tuple[str],
    zip_path: Path,
    batch_size: int,
    num_workers: int,
    perform_augmentation: bool,
    include_Q_matrix: bool,
    include_cos_matrix: bool,
    include_lsb: bool,
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
        aug = A.Compose([aug for use, aug in zip(use_augs, aug_candidates) if use], p=1.0)

        dataset = DCTDataset(
            zip_path=zip_path,
            quality_indices=quality_indices,
            filenames=filenames,
            deterministic_aug_list=[aug],
            include_Q_matrix=include_Q_matrix,
            include_cos_matrix=include_cos_matrix,
            include_lsb=include_lsb,
            inference=True,
            folders=folders,
        )

        if distributed:
            example_per_process = (len(dataset) + world_size - 1) // world_size
            sections = [example_per_process * r for r in range(world_size)] + [len(dataset)]
            dataset = torch.utils.data.Subset(dataset, range(sections[global_rank], sections[global_rank + 1]))

        pred = somen.pytorch_utility.predict(model, dataset, batch_size, num_workers, device=device)[0]

        # if distributed:
        #     tensor_list = arbitrary_length_all_gather(torch.from_numpy(pred).to(device))
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
    batch_size: int,
    num_workers: int,
    perform_augmentation: bool,
    include_Q_matrix: bool,
    include_cos_matrix: bool,
    include_lsb: bool,
    prefix: str = "pred",
    local_rank: Optional[int] = None,
):
    folds = get_folds_as_filename(n_folds)
    train_filenames = folds[fold_index][0]
    valid_filenames = folds[fold_index][1]
    test_filenames = [f"{i:0>4}.jpg" for i in range(1, 5001)]

    distributed = local_rank is not None
    is_main_node = (not distributed) or torch.distributed.get_rank() == 0

    quality_df = pd.read_csv("data/working/quality_train.csv")
    quality_df["quality_index"] = quality_df["quality"].map({75: 0, 90: 1, 95: 2})
    quality_df = quality_df.set_index("filename")

    train_q_indices = quality_df.loc[train_filenames, "quality_index"]
    pred_train = get_pred(
        model=model,
        quality_indices=train_q_indices,
        filenames=train_filenames,
        folders=("Cover", "JMiPOD", "JUNIWARD", "UERD"),
        zip_path=zip_path,
        batch_size=batch_size,
        num_workers=num_workers,
        perform_augmentation=perform_augmentation,
        include_Q_matrix=include_Q_matrix,
        include_cos_matrix=include_cos_matrix,
        include_lsb=include_lsb,
        local_rank=local_rank,
    )
    # if is_main_node:
    #     somen.file_io.save_array(pred_valid, working_dir / f"{prefix}_train.h5")
    if distributed:
        somen.file_io.save_array(pred_train, working_dir / f"{prefix}_train_{torch.distributed.get_rank()}.h5")
    else:
        somen.file_io.save_array(pred_train, working_dir / f"{prefix}_train.h5")
    del pred_train

    valid_q_indices = quality_df.loc[valid_filenames, "quality_index"]
    pred_valid = get_pred(
        model=model,
        quality_indices=valid_q_indices,
        filenames=valid_filenames,
        folders=("Cover", "JMiPOD", "JUNIWARD", "UERD"),
        zip_path=zip_path,
        batch_size=batch_size,
        num_workers=num_workers,
        perform_augmentation=perform_augmentation,
        include_Q_matrix=include_Q_matrix,
        include_cos_matrix=include_cos_matrix,
        include_lsb=include_lsb,
        local_rank=local_rank,
    )
    # if is_main_node:
    #     somen.file_io.save_array(pred_valid, working_dir / f"{prefix}_valid.h5")
    if distributed:
        somen.file_io.save_array(pred_valid, working_dir / f"{prefix}_valid_{torch.distributed.get_rank()}.h5")
    else:
        somen.file_io.save_array(pred_valid, working_dir / f"{prefix}_valid.h5")
    del pred_valid

    quality_df = pd.read_csv("data/working/quality_test.csv")
    quality_df["quality_index"] = quality_df["quality"].map({75: 0, 90: 1, 95: 2})
    quality_df = quality_df.set_index("filename")
    test_q_indices = quality_df.loc[test_filenames, "quality_index"]

    pred_test = get_pred(
        model=model,
        quality_indices=test_q_indices,
        filenames=test_filenames,
        folders=("Test",),
        zip_path=zip_path,
        batch_size=batch_size,
        num_workers=num_workers,
        perform_augmentation=perform_augmentation,
        include_Q_matrix=include_Q_matrix,
        include_cos_matrix=include_cos_matrix,
        include_lsb=include_lsb,
        local_rank=local_rank,
    )
    # if is_main_node:
    #     somen.file_io.save_array(pred_test, working_dir / f"{prefix}_test.h5")
    if distributed:
        somen.file_io.save_array(pred_test, working_dir / f"{prefix}_test_{torch.distributed.get_rank()}.h5")
    else:
        somen.file_io.save_array(pred_test, working_dir / f"{prefix}_test.h5")
    del pred_test

    if local_rank is not None:
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


def get_crop_prob(run_name: str, fold_index: int, target: str, crop_softmax_s: float = 5.0):
    hint_full = somen.file_io.load_array(f"data/working/{run_name}/{fold_index}/pred_{target}.h5")
    hint_crop = somen.file_io.load_array(f"data/working/{run_name}/{fold_index}/crop_pred_{target}.h5")

    assert hint_crop.shape[1:] == (3, 3, 4, 8)
    hint_crop = torch.from_numpy(hint_crop)
    hint_crop = F.log_softmax(hint_crop, dim=3)

    assert hint_full.shape[1:] == (4, 8)
    hint_full = torch.from_numpy(hint_full)
    hint_full = hint_full[:, np.newaxis, np.newaxis, :, :]

    crop_prob = torch.sum(hint_full * hint_crop, dim=3)
    crop_prob = crop_prob.reshape(-1, 9, 8)
    crop_prob = F.softmax(crop_softmax_s * crop_prob, dim=1)
    crop_prob = crop_prob.numpy()

    return crop_prob


def setup_model(config: TrainingConfig):
    model = DCTD8Model(
        num_classes=4,
        backbone=config.model.backbone,
        use_dilation_8=config.use_dilation_8,
        strides=config.strides,
        L=config.L,
        include_Q_matrix=config.include_Q_matrix,
        include_cos_matrix=config.include_cos_matrix,
        parallel_by_pos=config.parallel_by_pos,
        include_lsb=config.include_lsb,
        crop_quarter=config.crop_quarter,
        patch_dilation_first_stride=config.patch_dilation_first_stride,
        replace_second_stride_by_max_pool=config.replace_second_stride_by_max_pool,
    )
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

    quality_df = pd.read_csv("data/working/quality_train.csv")
    quality_df["quality_index"] = quality_df["quality"].map({75: 0, 90: 1, 95: 2})
    quality_df = quality_df.set_index("filename")

    if config.perform_augmentation:
        aug_candidates = [A.VerticalFlip(p=1.0), A.HorizontalFlip(p=1.0), As.FixedFactorRandomRotate90(p=1.0, factor=1)]
        train_aug_list = []
        for use_augs in itertools.product([True, False], repeat=len(aug_candidates)):
            aug = A.Compose([aug for use, aug in zip(use_augs, aug_candidates) if use], p=1.0)
            train_aug_list.append(aug)
    else:
        train_aug_list = [A.Compose([])]
    train_q_indices = quality_df.loc[train_filenames, "quality_index"].tolist()

    if config.crop_hint_run is not None:
        crop_prob = get_crop_prob(config.crop_hint_run, config.fold_index, "train", config.crop_softmax_s)
        if config.subsample < 1.0:
            shape1 = crop_prob.shape[1:]
            crop_prob = crop_prob.reshape(-1, 4, *shape1)
            crop_prob = crop_prob[train_permutation]
            crop_prob = crop_prob.reshape(-1, *shape1)
    else:
        crop_prob = None

    train_dataset = DCTDataset(
        zip_path=args.zip_path,
        quality_indices=train_q_indices,
        filenames=train_filenames,
        deterministic_aug_list=train_aug_list,
        include_Q_matrix=config.include_Q_matrix,
        include_cos_matrix=config.include_cos_matrix,
        include_lsb=config.include_lsb,
        crop_prob=crop_prob,
    )

    valid_q_indices = quality_df.loc[valid_filenames, "quality_index"].tolist()
    valid_aug_list = [A.Compose([])]

    if config.crop_hint_run is not None:
        max_pos = np.argmax(
            get_crop_prob(config.crop_hint_run, config.fold_index, "valid", config.crop_softmax_s), axis=1
        )
        crop_prob = np.zeros((max_pos.shape[0], 9, max_pos.shape[1]), dtype=np.float32)
        for i in range(max_pos.shape[0]):
            for j in range(max_pos.shape[1]):
                crop_prob[i, max_pos[i, j], j] = 1.0
        if config.subsample < 1.0:
            shape1 = crop_prob.shape[1:]
            crop_prob = crop_prob.reshape(-1, 4, *shape1)
            crop_prob = crop_prob[valid_permutation]
            crop_prob = crop_prob.reshape(-1, *shape1)
    else:
        crop_prob = None

    valid_dataset = DCTDataset(
        zip_path=args.zip_path,
        quality_indices=valid_q_indices,
        filenames=valid_filenames,
        deterministic_aug_list=valid_aug_list,
        include_Q_matrix=config.include_Q_matrix,
        include_cos_matrix=config.include_cos_matrix,
        include_lsb=config.include_lsb,
        crop_prob=crop_prob,
    )

    params = {
        "objective": torch.nn.functional.cross_entropy,
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
        "metric": [("weighted_auc", alaska_weighted_auc_metric_fun), "loss"],
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
        batch_size=config.optim.batch_size * 2,
        num_workers=args.num_workers,
        perform_augmentation=config.perform_augmentation,
        include_Q_matrix=config.include_Q_matrix,
        include_cos_matrix=config.include_cos_matrix,
        include_lsb=config.include_lsb,
        local_rank=args.local_rank,
    )


def predict_feature(args):
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, args.config, args.overrides)
    working_dir = Path(f"data/working/{config.run_name}/{config.fold_index}/")

    model = setup_model(config)
    model.load_state_dict(torch.load(working_dir / "final.pth", map_location="cpu"))

    distributed = args.local_rank is not None
    if distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    class OutputFeature(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.flatten = torch.nn.Flatten()

        def forward(self, dct, mask=None):
            feature = self.model.get_feature_from_dct(dct, mask)
            feature = self.pool(feature)
            feature = self.flatten(feature)
            return feature

    make_prediction_h5(
        model=OutputFeature(),
        fold_index=config.fold_index,
        n_folds=config.n_folds,
        working_dir=working_dir,
        zip_path=args.zip_path,
        batch_size=config.optim.batch_size * 2,
        num_workers=args.num_workers,
        perform_augmentation=config.perform_augmentation,
        include_Q_matrix=config.include_Q_matrix,
        include_cos_matrix=config.include_cos_matrix,
        include_lsb=config.include_lsb,
        prefix="feature",
        local_rank=args.local_rank,
    )


def predict_crop(args):
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, args.config, args.overrides)
    working_dir = Path(f"data/working/{config.run_name}/{config.fold_index}/")

    model = setup_model(config)
    model.load_state_dict(torch.load(working_dir / "final.pth", map_location="cpu"))

    distributed = args.local_rank is not None
    if distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    class OutputCropPrediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, dct, mask=None):
            def crop(x):
                assert x.shape[-2:] == (512, 512)
                y = x.unfold(-2, 256, 128).unfold(-2, 256, 128)
                assert y.shape[-4:] == (3, 3, 256, 256)
                y = y.permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0] * 9, x.shape[1], 256, 256)
                return y

            dct = crop(dct)
            if mask is not None:
                mask = crop(mask)

            crop_pred = self.model(dct, mask)
            crop_pred = crop_pred.reshape(-1, 3, 3, crop_pred.shape[-1])
            return crop_pred

    make_prediction_h5(
        model=OutputCropPrediction(),
        fold_index=config.fold_index,
        n_folds=config.n_folds,
        working_dir=working_dir,
        zip_path=args.zip_path,
        batch_size=config.optim.batch_size * 2,
        num_workers=args.num_workers,
        perform_augmentation=config.perform_augmentation,
        include_Q_matrix=config.include_Q_matrix,
        include_cos_matrix=config.include_cos_matrix,
        include_lsb=config.include_lsb,
        prefix="crop_pred",
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
        "--task",
        type=str,
        choices=["train", "predict", "predict_feature", "predict_crop", "download_model"],
        required=True,
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
    elif args.task == "predict_crop":
        predict_crop(args)
    elif args.task == "download_model":
        download_model(args)
