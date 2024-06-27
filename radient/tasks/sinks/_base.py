from abc import abstractmethod
from typing import List, Union

from radient.tasks._base import Task
from radient.vector import Vector


class Sink(Task):
    """Sinks in Radient are destinations for vector data. The penultimate
    operation prior to sinks is usually the result of some data merging
    function and, in some cases, can be the direct output of a vectorizer or
    set of vectorizers.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.transact(*args, **kwargs)

    @abstractmethod
    def transact(
        self,
        vectors: Union[Vector, List[Vector]],
        **kwargs
    ) -> bool:
        pass

