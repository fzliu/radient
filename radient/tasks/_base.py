from abc import ABC, abstractmethod


class Task(ABC):
    """Tasks are operators that can include transforms, vectorizers, and sinks.
    Data sources will be supported soon^{TM}.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass