from ignite.engine import Engine

from somen.pytorch_utility.extensions.extension import Extension


def get_attached_extension(engine: Engine, class_name: str) -> Extension:
    # TODO: Use other ways of not having to touch the private field `_event_handlers`.
    for handlers in engine._event_handlers.values():
        for handler, _, _ in handlers:
            if hasattr(handler, "_parent"):
                handler = handler._parent()
            if isinstance(handler, Extension) and type(handler).__name__ == class_name:
                return handler
    raise ValueError(f"`{class_name}` is not found in attached extensions")
