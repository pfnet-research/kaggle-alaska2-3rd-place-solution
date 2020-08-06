from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from ignite.utils import convert_tensor

from somen.pytorch_utility import misc


def predict(
    model: Union[torch.nn.Module, Callable],
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 0,
    device: str = "cuda",
    non_blocking: bool = False,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
) -> Sequence[np.ndarray]:

    if isinstance(model, torch.nn.Module):
        model.to(device)
        model.eval()

    data_loader = misc.to_dataloader(dataset, batch_size, False, num_workers, collate_fn, pin_memory)

    with torch.no_grad():

        def _get_non_ref_array(x):
            return x.detach().cpu().numpy().copy()

        y_preds = None
        for batch in data_loader:
            if isinstance(batch, torch.Tensor):
                batch = [batch]

            inputs = [convert_tensor(v, device=device, non_blocking=non_blocking) for v in batch]
            y_pred = model(*inputs)

            if isinstance(y_pred, tuple):
                y_pred = [_get_non_ref_array(e) for e in y_pred]
            else:
                y_pred = [_get_non_ref_array(y_pred)]

            if y_preds is None:
                y_preds = [[] for _ in range(len(y_pred))]
            for k in range(len(y_pred)):
                y_preds[k].append(y_pred[k])

        y_preds = [np.concatenate(e, axis=0) for e in y_preds]
        return y_preds
