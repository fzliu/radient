from abc import abstractmethod
from typing import Any

from radient.tasks._base import Task


class Source(Task):
    """Sources in Radient are task objects that yield data. Depending on the
    downstream transform, this can be raw bytes, or it can be filenames/URIs.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.read(*args, **kwargs)

    @abstractmethod
    def read(self, **kwargs) -> Any:
        pass
