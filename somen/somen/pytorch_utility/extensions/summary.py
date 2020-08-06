from collections import defaultdict
from typing import Dict, Mapping


class DictSummary:
    def __init__(self) -> None:
        self._x: Dict[str, float] = defaultdict(float)
        self._n: Dict[str, float] = defaultdict(float)

    def add(self, d: Mapping[str, float]) -> None:
        for key, value in d.items():
            self._x[key] += value
            self._n[key] += 1.0

    def pop_as_mean(self) -> Dict[str, float]:
        ret = {key: value / self._n[key] for key, value in self._x.items()}
        self._x = defaultdict(float)
        self._n = defaultdict(float)
        return ret

    def state_dict(self) -> Dict:
        return {"_x": self._x, "_n": self._n}

    def load_state_dict(self, state_dict: Mapping) -> None:
        self._x = state_dict["_x"]
        self._n = state_dict["_n"]
