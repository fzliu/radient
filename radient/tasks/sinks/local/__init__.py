from typing import Union

from radient.tasks.sinks._base import Sink
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector


class LocalVectorSink(Sink):

    def __init__(
        self
    ):
        super().__init__()
        raise NotImplementedError

    def transact(
        self,
        vectors: Union[Vector, list[Vector]],
        **kwargs
    ) -> dict[str, Union[int, list[int]]]:
        raise NotImplementedError
