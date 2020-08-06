from typing import Any, Mapping, Optional

import numpy as np
from ignite.engine import Engine, Events

from somen.pytorch_utility.extensions.extension import PRIORITY_WRITER, Extension


class LRScheduler(Extension):

    priority = PRIORITY_WRITER
    main_process_only = False

    def __init__(
        self,
        optimizer,
        lr,
        nb_epoch,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[Mapping[str, Any]] = None,
        call_event: Events = Events.EPOCH_STARTED,
    ) -> None:
        self.optimizer = optimizer
        self.lr = lr
        self.nb_epoch = nb_epoch + lr_scheduler_params.get("epoch_offset", 0)
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.call_event = call_event
        self._current_lr = lr

    def _set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def _get_lr(self, cur_epoch):
        if self.lr_scheduler is None:
            return self.lr

        if self.lr_scheduler == "cos":
            return 0.5 * self.lr * (1.0 + np.cos(np.pi * cur_epoch / self.nb_epoch))

        if self.lr_scheduler == "exp":
            return self.lr * self.lr_scheduler_params.get("gamma", 0) ** cur_epoch

    def iteration_completed(self, engine: Engine):
        # FIXME: あとでもっと良い方法を考える
        engine.state.metrics["observation"]["lr"] = self._current_lr

    def __call__(self, engine: Engine) -> None:
        # `engine.state.epoch` is incremented just before EPOCH_STARTED event
        cur_epoch = engine.state.epoch - 1 + self.lr_scheduler_params.get("epoch_offset", 0)
        new_lr = self._get_lr(cur_epoch)

        warmup_epochs = self.lr_scheduler_params.get("warmup_epochs", 0)
        if cur_epoch < warmup_epochs:
            alpha = cur_epoch / warmup_epochs
            warmup_factor = self.lr_scheduler_params.get("warmup_factor", 0.1) * (1.0 - alpha) + alpha
            new_lr *= warmup_factor

        self._set_lr(new_lr)

        # FIXME: あとで直す
        self._current_lr = new_lr
