from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from itertools import cycle, islice
import random
from typing import Any, Dict, List, Tuple, Type, Optional, Union
import uuid

import numpy as np


def _traverse(
    data: Union[Any, List[Any], Dict[str, Union[Any, List[Any]]]]
) -> Iterator:
    """Traverse a nested data structure and yield its elements.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            yield from _traverse(v)
    elif isinstance(data, list):
        for d in data:
            yield from _traverse(d)
    else:
        yield data


def _datalen(
    data: Union[Any, List[Any], Dict[str, Union[Any, List[Any]]]]
) -> Iterator:
    """
    """
    if isinstance(data, dict):
        return sum([_datalen(v) for v in data.values()])
    elif isinstance(data, list):
        return len(data)
    else:
        return 1 if data else 0


def flattened(*args, **kwargs) -> Iterator:
    """For use when `flatten_inputs` evaluates to True. Parses out `dict` and
    `list` inputs so that each individual element is passed as an argument into
    the downstream function. 
    """

    kwargs = OrderedDict(kwargs)

    # Combine `args` and `kwargs` datas into a single list, then cycle through
    # all of them until the maximum length is reached.
    datas = list(kwargs.values()) + list(args)
    maxlen = max([_datalen(d) for d in datas])
    generator = zip(*[islice(cycle(_traverse(d)), maxlen) for d in datas])

    # Recombine the flattened inputs into the original form.
    for inputs in generator:
        flat_kwargs = dict(zip(kwargs.keys(), *inputs))
        flat_args = inputs[len(kwargs):]
        yield flat_args, flat_kwargs


class Runner(ABC):

    @abstractmethod
    def __init__(
        self,
        task: Type,
        task_args: Optional[Tuple] = None,
        task_kwargs: Optional[Dict] = None,
        name: Optional[str] = None,
        flatten_inputs: Optional[str] = None
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
            # TODO: outputs should be dict/df for dict/df input(s)
            outputs = []
            for flat_args, flat_kwargs in flattened(*args, **kwargs):
                print(flat_args, flat_kwargs)
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
