from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from somen.pytorch_utility.extensions.extension import PRIORITY_WRITER, Extension


class BatchEvaluator(Extension):

    priority = PRIORITY_WRITER
    main_process_only = True  # If you want to distribute the procedure, use DistributedBatchEvaluator

    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        prefix: str,
        metric_functions: Mapping[str, Callable],
        label_indices: Sequence[int],
        device: str,
        non_blocking: bool,
        micro_average: bool = True,
        call_event: Events = Events.EPOCH_COMPLETED,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.prefix = prefix
        self.metric_functions = metric_functions
        self.label_indices = label_indices
        self.device = device
        self.non_blocking = non_blocking
        self.micro_average = micro_average
        self.call_event = call_event

    @torch.no_grad()
    def __call__(self, engine: Engine) -> None:
        self.model.eval()

        def _get_non_ref_array(x):
            return x.detach().cpu().numpy().copy()

        y_trues, y_preds = None, None
        for batch in self.data_loader:
            inputs = [
                v
                for i, v in enumerate(batch)
                if i not in self.label_indices and i - len(batch) not in self.label_indices
            ]
            inputs = [convert_tensor(v, device=self.device, non_blocking=self.non_blocking) for v in inputs]
            y_pred = self.model(*inputs)

            # Store labels
            if y_trues is None:
                y_trues = [[] for _ in self.label_indices]
            for k in range(len(self.label_indices)):
                y_trues[k].append(_get_non_ref_array(batch[self.label_indices[k]]))

            if isinstance(y_pred, tuple):
                y_pred = [_get_non_ref_array(e) for e in y_pred]
            else:
                y_pred = [_get_non_ref_array(y_pred)]

            if y_preds is None:
                y_preds = [[] for _ in range(len(y_pred))]
            for k in range(len(y_pred)):
                y_preds[k].append(y_pred[k])

        y_trues = [torch.from_numpy(np.concatenate(e, axis=0)) for e in y_trues]
        y_preds = [torch.from_numpy(np.concatenate(e, axis=0)) for e in y_preds]

        for key, metric_fn in self.metric_functions.items():
            value = metric_fn(*y_preds, *y_trues)
            if isinstance(value, torch.Tensor):
                value = value.item()
            engine.state.metrics["observation"][self.prefix + key] = value
