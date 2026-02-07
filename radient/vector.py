from collections import OrderedDict
from typing import Any, Sequence, Optional, Union

import numpy as np


class Vector(np.ndarray):
    """Wrapper around `numpy.ndarray` specifically for working with embeddings.
    We try to use Numpy naming conventions here where possible, such as concise
    function names and 
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __array_finalize__(self, obj):
        """Attach metadata to be associated with this vector.
        """
        self._metadata = OrderedDict()

    @property
    def metadata(self) -> OrderedDict:
        return self._metadata

    @metadata.setter
    def metadata(self, data: dict):
        self._metadata = OrderedDict(data)

    def putmeta(
        self,
        key: str,
        value: Union[dict[str, Union[str, float, int]], str, float, int]
    ) -> "Vector":
        self._metadata[key] = value
        # Enable chaining function calls together.
        return self

    def popmeta(
        self,
        key: str
    ) -> Union[dict[str, Union[str, float, int]], str, float, int]:
        return self._metadata.pop(key)

    def todict(
        self,
        vector_field: str = "vector"
    ) -> dict[str, Union["Vector", str, float, int]]:
        return dict(self._metadata, **{vector_field: self.tolist()})
