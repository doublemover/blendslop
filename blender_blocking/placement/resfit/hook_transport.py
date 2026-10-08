"""Rebuild local objective closures from process-safe source data."""
from dataclasses import dataclass, field
import importlib


@dataclass
class RebuildableHook:
    module: str
    factory: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    _callable: object = field(default=None, init=False, repr=False)

    def __call__(self, primitives):
        if self._callable is None:
            factory = getattr(importlib.import_module(self.module), self.factory)
            self._callable = factory(*self.args, **self.kwargs)
        return self._callable(primitives)

    def __getstate__(self):
        return {"module": self.module, "factory": self.factory,
                "args": self.args, "kwargs": self.kwargs, "_callable": None}
