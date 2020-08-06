import operator
from pathlib import Path
from typing import Dict, Mapping

import torch
from ignite.engine import Engine, Events

from somen.pytorch_utility.extensions.extension import PRIORITY_READER, Extension


class BestValueSnapshot(Extension):

    priority = PRIORITY_READER
    main_process_only = True

    def __init__(self, obj, path: Path, key: str, maximize: bool, call_event: Events = Events.EPOCH_COMPLETED):
        self.obj = obj
        self.path = Path(path)
        self.key = key
        self.maximize = maximize
        self.call_event = call_event

        self._best_value = -float("inf") if maximize else float("inf")
        self._compare_op = operator.gt if maximize else operator.lt

    def __call__(self, engine: Engine) -> None:
        observation: Dict[str, float] = engine.state.metrics["observation"]
        if self.key in observation:
            value = observation[self.key]
            if self._compare_op(value, self._best_value):
                self._best_value = value
                torch.save(self.obj.state_dict(), self.path)

    def state_dict(self) -> Dict:
        return {"_best_value": self._best_value}

    def load_state_dict(self, to_load: Mapping) -> None:
        self._best_value = to_load["_best_value"]
