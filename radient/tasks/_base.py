from abc import ABC, abstractmethod
from typing import Any, List


class Task(ABC):
    """Tasks are operators that can include transforms, vectorizers, and sinks.
    Data sources will be supported soon^{TM}.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> List[Any]:
        pass