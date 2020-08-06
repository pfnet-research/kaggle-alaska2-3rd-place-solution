import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from ignite.engine import Engine, Events

from somen.pytorch_utility.extensions.extension import PRIORITY_READER, Extension
from somen.pytorch_utility.extensions.summary import DictSummary


class LogReport(Extension):

    # LogReport is a special extension. It don't write or edit observation
    # but it's `logs` field can be referred from other READER extensions.
    priority = PRIORITY_READER + 50
    main_process_only = True

    def __init__(
        self, log_file_path: Path, nb_epoch: Optional[int] = None, call_event: Events = Events.EPOCH_COMPLETED
    ) -> None:
        self.log_file_path = Path(log_file_path)
        self.nb_epoch = nb_epoch
        self.call_event = call_event

        self.logs: List[Mapping] = []
        self._snapshot_elapsed_time = 0.0
        self._start_at: Optional[float] = None
        self._init_summary()

    def _init_summary(self) -> None:
        self._summary = DictSummary()

    def _get_elapsed_time(self) -> float:
        if self._start_at is None:
            raise ValueError("Training is not started.")
        return time.time() - self._start_at + self._snapshot_elapsed_time

    def started(self, engine: Engine) -> None:
        self._start_at = time.time()

    def iteration_completed(self, engine: Engine) -> None:
        observation: Mapping[str, float] = engine.state.metrics["observation"]
        self._summary.add(observation)

    def __call__(self, engine: Engine) -> None:
        log = self._summary.pop_as_mean()
        log["epoch"] = engine.state.epoch
        log["iteration"] = engine.state.iteration
        log["elapsed_time"] = self._get_elapsed_time()

        if self.nb_epoch is not None:
            log["remain_time"] = log["elapsed_time"] / engine.state.epoch * (self.nb_epoch - engine.state.epoch)

        # Add entries that are added between (iteration end, this function call)
        observation: Mapping[str, float] = engine.state.metrics["observation"]
        for key, value in observation.items():
            if key not in log:
                log[key] = value

        self.logs.append(log)
        with self.log_file_path.open("w") as fp:
            json.dump(self.logs, fp, indent=4)

    def state_dict(self) -> Dict:
        return {
            "logs": self.logs,
            "_snapshot_elapsed_time": self._get_elapsed_time(),
            "_start_at": self._start_at,
            "_summary": self._summary.state_dict(),
        }

    def load_state_dict(self, to_load: Mapping) -> None:
        self.logs = to_load["logs"]
        self._snapshot_elapsed_time = to_load["_snapshot_elapsed_time"]
        self._start_at = to_load["_start_at"]
        self._init_summary()
        self._summary.load_state_dict(to_load["_summary"])
