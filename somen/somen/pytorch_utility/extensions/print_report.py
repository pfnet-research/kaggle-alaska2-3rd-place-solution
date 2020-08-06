from typing import Optional, Sequence, Union

from ignite.engine import Engine, Events

from somen.pytorch_utility.extensions.extension import PRIORITY_READER, Extension
from somen.pytorch_utility.extensions.get_attached_extension import get_attached_extension
from somen.pytorch_utility.extensions.log_report import LogReport


class PrintReport(Extension):

    priority = PRIORITY_READER
    main_process_only = True

    def __init__(
        self,
        entries: Optional[Sequence[str]] = None,
        log_report: Union[str, LogReport] = "LogReport",
        call_event: Events = Events.EPOCH_COMPLETED,
    ):
        self.entries = entries
        self.log_report = log_report
        self.call_event = call_event
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

        logs = self.log_report.logs

        if self._step < len(logs):
            entries = list(logs[0].keys()) if self.entries is None else self.entries
            entry_widths = [max(10, len(s)) for s in entries]

            if self._step == 0:
                header = "  ".join([f"{{:{w}}}".format(e) for e, w in zip(entries, entry_widths)])
                print(header)

            for log in logs[self._step :]:
                for e, w in zip(entries, entry_widths):
                    if e in log:
                        print(f"{{:<{w}g}}  ".format(log[e]), end="")
                    else:
                        print(" " * (w + 2), end="")
                print()
                self._step += 1
