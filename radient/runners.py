from abc import ABC, abstractmethod

from typing import Any


class Runner(ABC):

    @abstractmethod
    def __init__(self, function: Any, *args, **kwargs):
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._result = None

    @property
    def result(self):
        return self._result

    def __call__(self, *args, **kwargs):
        return self.result(*args, **kwargs)


class LocalRunner(Runner):
    """Evaluate a function or instance locally.
    """

    def __init__(self, function: Any, *args, **kwargs):
        super().__init__(function, **kwargs)
        self._result = self._function(*args, **kwargs)


class LazyLocalRunner(Runner):
    """Lazily (on-demand) evaluate a function or instance.
    """

    def __init__(self, function: Any, *args, **kwargs):
        super().__init__(function, *args, **kwargs)

    @property
    def result(self):
        if not self._result:
            self._result = self._function(*self._args, **self._kwargs)
        return self._result
