from abc import ABC, abstractmethod
import random
from typing import Dict, Tuple, Type, Optional

from radient.utils.flatten_inputs import flattened


class Runner(ABC):

    @abstractmethod
    def __init__(
        self,
        task: Type,
        task_args: Optional[Tuple] = None,
        task_kwargs: Optional[Dict] = None,
        name: Optional[str] = None,
        flatten_inputs: Optional[str] = False
    ):
        self._task = task
        self._task_args = task_args or ()
        self._task_kwargs = task_kwargs or {}
        self._flatten_inputs = flatten_inputs
        self._name = name or f"{task.__name__}-{random.randint(0, 65536)}"
        self._result = None

    def _evaluate(self):
        self._result = self._task(
            *self._task_args,
            **self._task_kwargs
        )

    @property
    def name(self):
        return self._name

    @property
    def result(self):
        return self._result

    def __call__(self, *args, **kwargs):
        if self._flatten_inputs:
            outputs = []
            for flat_args, flat_kwargs in flattened(*args, **kwargs):
                outputs.append(self.result(*flat_args, **flat_kwargs))
            return outputs
        return self.result(*args, **kwargs)


class LocalRunner(Runner):
    """Evaluate a function or instance locally.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evaluate()


class LazyLocalRunner(Runner):
    """Lazily (on-demand) evaluate a function or instance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def result(self):
        if not self._result:
            self._evaluate()
        return self._result
