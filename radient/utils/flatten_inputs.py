from collections import OrderedDict
from collections.abc import Iterator
from itertools import cycle, islice

from typing import Any, Dict, List, Union


def _traverse(data: Union[Any, List[Any]]) -> Iterator:
    """Traverse a nested data structure and yield its elements.
    """
    #if isinstance(data, dict):
    #    for k, v in data.items():
    #        yield from _traverse(v)
    if isinstance(data, list):
        for d in data:
            yield from _traverse(d)
    else:
        yield data


def _datalen(data: Union[Any, List[Any]]) -> int:
    """Returns the length of the input data when used with `_traverse`.
    """
    #if isinstance(data, dict):
    #    return sum([_datalen(v) for v in data.values()])
    if isinstance(data, list):
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