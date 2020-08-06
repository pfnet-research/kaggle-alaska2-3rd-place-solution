from typing import Dict, Mapping

from ignite.engine import Engine, Events

PRIORITY_WRITER = 300
PRIORITY_EDITOR = 200
PRIORITY_READER = 100


class Extension:

    priority: int
    call_event: Events
    main_process_only: bool

    def started(self, engine: Engine) -> None:
        pass

    def iteration_completed(self, observation: Dict[str, float]) -> None:
        pass

    def __call__(self, engine: Engine) -> None:
        pass

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, to_load: Mapping) -> None:
        pass
