import torch
from ignite.engine import Engine, Events

from somen.pytorch_utility.extensions.extension import PRIORITY_READER, Extension


class DDPParamSyncChecker(Extension):
    priority = PRIORITY_READER
    main_process_only = False

    def __init__(self, model: torch.nn.Module, call_event: Events = Events.ITERATION_COMPLETED,) -> None:
        self.model = model
        self.call_event = call_event
        self._world_size = torch.distributed.get_world_size()
        self._global_rank = torch.distributed.get_rank()

    @torch.no_grad()
    def __call__(self, engine: Engine) -> None:
        for param in self.model.parameters():
            param_list = [torch.zeros_like(param) for _ in range(self._world_size)]
            torch.distributed.all_gather(param_list, param)
            assert all([(p == param).all() for p in param_list])
        print(f"DDPParamSyncChecker: pass at rank {self._global_rank}")
