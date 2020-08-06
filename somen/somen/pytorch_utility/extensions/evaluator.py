from collections import defaultdict
from typing import Callable, Dict, Mapping, Sequence

import torch
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from somen.pytorch_utility.extensions.extension import PRIORITY_WRITER, Extension


class Evaluator(Extension):

    priority = PRIORITY_WRITER
    main_process_only = True  # If you want to distribute the procedure, use DistributedEvaluator

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
        # torch.cuda.empty_cache()
        total: Dict[str, float] = defaultdict(float)
        divisor: Dict[str, float] = defaultdict(float)
        for batch in self.data_loader:
            inputs = [
                v
                for i, v in enumerate(batch)
                if i not in self.label_indices and i - len(batch) not in self.label_indices
            ]
            inputs = [convert_tensor(v, device=self.device, non_blocking=self.non_blocking) for v in inputs]
            labels = [batch[i] for i in self.label_indices]
            labels = [convert_tensor(v, device=self.device, non_blocking=self.non_blocking) for v in labels]

            y_pred = self.model(*inputs)

            for key, metric_fn in self.metric_functions.items():
                if isinstance(y_pred, tuple):
                    loss = metric_fn(*y_pred, *labels)
                else:
                    loss = metric_fn(y_pred, *labels)
                loss_scalar = loss.item() if isinstance(loss, torch.Tensor) else loss

                if self.micro_average:
                    total[key] += len(inputs[0]) * loss_scalar
                    divisor[key] += len(inputs[0])
                else:
                    total[key] += loss_scalar
                    divisor[key] += 1.0

        for key in total.keys():
            engine.state.metrics["observation"][self.prefix + key] = total[key] / divisor[key]
