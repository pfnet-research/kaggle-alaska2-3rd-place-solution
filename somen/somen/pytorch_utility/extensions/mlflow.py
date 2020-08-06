from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import mlflow
from ignite.engine import Engine, Events
from mlflow.tracking.fluent import ActiveRun

from somen.pytorch_utility.extensions.extension import PRIORITY_READER, Extension
from somen.pytorch_utility.extensions.get_attached_extension import get_attached_extension
from somen.pytorch_utility.extensions.log_report import LogReport
from somen.types import PathLike


def mlflow_start_run(exp_name: str, out: PathLike, resume: bool = False, run_name=None) -> ActiveRun:
    mlflow.set_experiment(exp_name)

    run_id_path = Path(out) / "run_id"
    if resume and run_id_path.exists():
        # resume
        with run_id_path.open("r") as fp:
            run_id = fp.read().strip()
        active_run = mlflow.start_run(run_id=run_id)
    else:
        # new run
        active_run = mlflow.start_run(run_name=run_name)
        run_id = active_run.info.run_id
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        with run_id_path.open("w") as fp:
            fp.write(run_id)
    return active_run


class MLflowReporter(Extension):

    priority = PRIORITY_READER

    def __init__(self, entries: Optional[Sequence[str]] = None, log_report: Union[str, LogReport] = "LogReport"):
        self.entries = entries
        self.log_report = log_report
        self.call_event = Events.ITERATION_COMPLETED
        self._step = 0

    def started(self, engine: Engine) -> None:
        if isinstance(self.log_report, str):
            ext = get_attached_extension(engine, self.log_report)
            if not isinstance(ext, LogReport):
                raise ValueError("Referenced extension must be an instance of `LogReport`.")
            self.log_report = ext

        assert isinstance(self.log_report, LogReport)

    def __call__(self, engine: Engine) -> None:
        if not isinstance(self.log_report, LogReport):
            raise ValueError("Please call started() before calling")

        for log in self.log_report.logs[self._step :]:
            keys = log.keys() if self.entries is None else self.entries
            mlflow.log_metrics({key: log[key] for key in keys if key in log}, step=self._step)
            self._step += 1

    def state_dict(self) -> Dict:
        return {"_step": self._step}

    def load_state_dict(self, to_load: Mapping) -> None:
        self._step = to_load["_step"]
