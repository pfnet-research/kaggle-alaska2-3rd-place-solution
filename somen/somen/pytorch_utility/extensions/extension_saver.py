from typing import Dict, Mapping, Sequence

from somen.pytorch_utility.extensions.extension import Extension


class ExtensionSaver:
    def __init__(self, extensions: Sequence[Extension]) -> None:
        self.extensions = extensions
        self._extension_dict: Dict[str, Extension] = {}

        for e in extensions:
            name = type(e).__name__
            if name in self._extension_dict:
                ordinal = 1
                while f"{name}_{ordinal}" not in self._extension_dict:
                    ordinal += 1
                name = f"{name}_{ordinal}"
            self._extension_dict[name] = e

    def state_dict(self) -> Dict:
        ret = {}
        for name, e in self._extension_dict.items():
            ret[name] = e.state_dict()
        return ret

    def load_state_dict(self, to_load: Mapping) -> None:
        for name, to_load_for_e in to_load.items():
            e = self._extension_dict[name]
            e.load_state_dict(to_load_for_e)
