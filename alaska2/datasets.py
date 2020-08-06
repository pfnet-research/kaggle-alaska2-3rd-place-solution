import itertools
import math
from typing import Optional, Sequence

import albumentations as A
import numpy as np
import pfio
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

import somen.albumentations_utility as As
from alaska2.loading import get_in_channels, read_image_from_container

normalize_info = {
    "YUV": {"mean": (0.44407194, 0.52137992, 0.46948314), "std": (0.14583727, 0.12018817, 0.17555958)},
    "XYZ": {"mean": (0.41790948, 0.44706437, 0.52127571), "std": (0.12537038, 0.16354785, 0.21286224)},
    "YCrCb": {"mean": (0.44407194, 0.47552349, 0.52422309), "std": (0.14583727, 0.14297034, 0.13778356)},
    "HSV": {"mean": (0.36237398, 0.58788817, 0.6395621), "std": (0.18495916, 0.2408899, 0.16586202)},
    "HLS": {"mean": (0.36294025, 0.44981734, 0.51339546), "std": (0.18493012, 0.13014653, 0.26954808)},
    "Lab": {"mean": (0.50689092, 0.51218311, 0.49064476), "std": (0.15830732, 0.13684685, 0.14181956)},
    "Luv": {"mean": (0.50472141, 0.37165887, 0.50752101), "std": (0.1582691, 0.10318156, 0.1874939)},
    "HED": {"mean": (-0.00984648, 0.0027415, -0.00632396), "std": (0.00270167, 0.00223852, 0.00270556)},
}


class BasicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zip_path: str,
        filenames: Sequence[str],
        read_method: str,
        aug_list: Sequence,
        inference=False,
        folders=("Cover", "JMiPOD", "JUNIWARD", "UERD"),
        soft_label: bool = False,
        label_smooth: Optional[float] = None,
    ):
        if label_smooth is not None and not soft_label:
            raise ValueError("If label_smooth is set, soft_label must be True.")

        self.filenames = filenames
        self.folders = folders
        self.read_method = read_method
        self.container = pfio.open_as_container(zip_path)

        if read_method == "RGB":
            normalize = A.Normalize()
        elif read_method == "cv2_YUV":
            normalize = A.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        elif read_method == "DCT":
            normalize = A.Normalize((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
        elif read_method == "DCT_PE_L10":
            normalize = None
        elif read_method.startswith("DCT_BIN"):
            normalize = None
        elif read_method.startswith("DCT_TRI"):
            normalize = None
        elif read_method.startswith("DCT_LSB"):
            normalize = None
        elif read_method.startswith("RGB_DCT"):
            in_channel = get_in_channels(read_method)
            mean = (0.485, 0.456, 0.406) + (0,) * (in_channel - 3)
            std = (0.229, 0.224, 0.225) + (1,) * (in_channel - 3)
            normalize = A.Normalize(mean=mean, std=std)
        elif read_method in normalize_info.keys():
            info = normalize_info[read_method]
            normalize = A.Normalize(info["mean"], info["std"])

        self.augs = A.Compose([normalize] + aug_list, p=1.0)
        self.inference = inference
        self.soft_label = soft_label
        self.label_smooth = label_smooth

    def __len__(self) -> int:
        return len(self.filenames) * len(self.folders)

    def __getitem__(self, index):
        class_index = index % len(self.folders)
        filename = self.filenames[index // len(self.folders)]

        img = read_image_from_container(self.container, f"{self.folders[class_index]}/{filename}", self.read_method)

        if self.read_method == "DCT":
            scale = 5.9569343623802355  # 128 / np.log1p(np.iinfo(np.int32).max)
            img = np.sign(img) * np.log1p(np.abs(img)) * scale

        img = self.augs(image=img)["image"]
        img = np.moveaxis(img, 2, 0).astype(np.float32)

        if self.inference:
            return img
        else:
            if self.soft_label:
                label = np.zeros(4, dtype=np.float32)
                label[class_index] = 1.0
                if self.label_smooth is not None:
                    label = np.full_like(label, self.label_smooth / len(label)) + (1 - self.label_smooth) * label
            else:
                label = class_index
            return img, label


class PseudoLabelingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zip_path,
        filenames,
        read_method,
        prediction,
        folders=("Cover", "JMiPOD", "JUNIWARD", "UERD"),
        soft_label: bool = False,
        label_smooth: Optional[float] = None,
    ):
        if label_smooth is not None and not soft_label:
            raise ValueError("If label_smooth is set, soft_label must be True.")

        assert prediction.ndim == 3  # (example, class, 8 flip/rotate)
        assert prediction.shape[-1] == 8
        assert prediction.shape[0] == len(filenames) * len(folders)

        aug_candidates = [A.VerticalFlip(p=1.0), A.HorizontalFlip(p=1.0), As.FixedFactorRandomRotate90(p=1.0, factor=1)]
        datasets = []

        for use_augs in itertools.product([True, False], repeat=len(aug_candidates)):
            aug_list = [aug for use, aug in zip(use_augs, aug_candidates) if use]
            datasets.append(
                BasicDataset(
                    zip_path=zip_path,
                    filenames=filenames,
                    read_method=read_method,
                    aug_list=aug_list,
                    inference=True,
                    folders=folders,
                )
            )
            assert len(prediction) == len(datasets[-1])

        self.prediction = prediction
        self._datasets = datasets
        self.soft_label = soft_label
        self.label_smooth = label_smooth

    def __len__(self) -> int:
        return len(self.prediction)

    def __getitem__(self, index):
        aug_index = np.random.randint(8)
        img = self._datasets[aug_index][index]
        if self.soft_label:
            label = self.prediction[index, :, aug_index].astype(np.float32)
            if self.label_smooth is not None:
                label = np.full_like(label, self.label_smooth / len(label)) + (1 - self.label_smooth) * label
        else:
            label = np.random.choice(self.prediction.shape[1], p=self.prediction[index, :, aug_index])
        return img, label


class DistributedCoverStegoPairSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
        assert len(dataset) % 4 == 0
        self.dataset = dataset
        self.num_filenames = len(dataset) // 4
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(self.num_filenames * 6.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

        assert self.num_samples % 2 == 0
        assert self.total_size % 2 == 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.num_filenames * 3, generator=g).tolist()
        else:
            indices = list(range(self.num_filenames * 3))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size // 2 - len(indices))]
        assert len(indices) == self.total_size // 2

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples // 2

        actual_indices = []
        for i in indices:
            filename_index = i // 3
            class_index = i % 3 + 1
            actual_indices.append(filename_index * 4)  # Cover
            actual_indices.append(filename_index * 4 + class_index)  # Stego

        assert len(self) == len(actual_indices)
        return iter(actual_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
