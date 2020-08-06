from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ignite.engine import Events
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, TensorDataset


def extract_label(
    label_key: Union[int], args: List[Any], kwargs: Dict[str, Any]
) -> Tuple[Any, List[Any], Dict[str, Any]]:

    if isinstance(label_key, int):
        if not (-len(args) <= label_key < len(args)):
            msg = f"Label key {label_key} is out of bounds"
            raise ValueError(msg)

        label = args[label_key]
        if label_key == -1:
            args = args[:-1]
        else:
            args = args[:label_key] + args[label_key + 1 :]

    elif isinstance(label_key, str):
        if label_key not in kwargs:
            msg = f'Label key "{label_key}" is not found'
            raise ValueError(msg)
        kwargs = dict(kwargs)  # sallow copy
        label = kwargs[label_key]
        del kwargs[label_key]

    return label, args, kwargs


def str_to_event(event: Union[Events, str]) -> Events:

    if isinstance(event, str):
        interval, event_type = event.split(" ")
        if event_type == "epoch":
            e = Events.EPOCH_COMPLETED(every=int(interval))
        elif event_type == "iteration":
            e = Events.ITERATION_COMPLETED(every=int(interval))
        else:
            raise ValueError(f"Invalid event type is provided: `{event_type}`")

    return e


def to_dataloader(
    dataset: Union[Tuple[np.ndarray, ...], Dataset, DataLoader],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
) -> DataLoader:
    if isinstance(dataset, tuple):
        dataset = TensorDataset(*dataset)

    if isinstance(dataset, DataLoader):
        dataloader = dataset
    else:
        assert isinstance(dataset, Dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    return dataloader


def get_metric_func(metric_name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

    if metric_name == "rmse":

        def rmse(y_pred: torch.Tensor, y_true: torch.Tensor):
            return np.sqrt(np.mean((y_true.detach().numpy() - y_pred.detach().numpy()) ** 2))

        return rmse

    if hasattr(metrics, metric_name):
        # sklearn metrics
        sklearn_metric_fn = getattr(metrics, metric_name)

        def metric_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
            return sklearn_metric_fn(y_true.detach().numpy(), y_pred.detach().numpy())

        return metric_fn

    raise ValueError(f"metric_name '{metric_name}' is not supported")


def arbitrary_length_all_gather(tensor: torch.Tensor, axis=0, device="cuda") -> Sequence[torch.Tensor]:
    if axis != 0:
        raise NotImplementedError  # TODO

    world_size = torch.distributed.get_world_size()

    length_tensor = torch.from_numpy(np.array(tensor.shape[axis])).to(device)
    length_tensor_list = [torch.ones_like(length_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(length_tensor_list, length_tensor)
    lengths = [int(t.detach().cpu().numpy()) for t in length_tensor_list]
    max_length = max(lengths)

    # TODO: current implementation works only with axis=0
    tensor_pad = torch.ones((max_length,) + tensor.shape[1:], dtype=tensor.dtype).to(tensor.device)
    tensor_pad[: len(tensor)] = tensor
    tensor_list = [torch.ones_like(tensor_pad) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor_pad)
    tensor_list = [tensor[:length].detach() for tensor, length in zip(tensor_list, lengths)]

    return tensor_list
