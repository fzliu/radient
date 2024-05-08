from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Optional, Union
import warnings

import numpy as np

from radient.vector import Vector
from radient.util.lazy_import import fully_qualified_name


def normalize_vector(vector: Vector, inplace: bool = True):
    if not np.issubdtype(vector.dtype, np.floating):
        warnings.warn("non-float vectors are not normalized")
    else:
        if inplace:
            vector /= np.linalg.norm(vector)
        else:
            vector = vector / np.linalg.norm(vector)
    return vector


class Vectorizer(ABC):

    @abstractmethod
    def __init__(self):
        self._model_name = None
        self._model = None

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def model(self) -> Optional[Any]:
        return self._model

    @property
    def modality(self) -> str:
        return fully_qualified_name(self).split(".")[-2]

    @classmethod
    def _preprocess(cls, item: Any) -> Any:
        return item

    @abstractmethod
    def _vectorize(self, data: List[Any]) -> List[Vector]:
        raise NotImplementedError

    @classmethod
    def _postprocess(
        cls,
        vector: Union[Vector, List[Vector]],
        normalize: bool = True
    ) -> Union[Vector, List[Vector]]:
        if normalize:
            # Some vectorizers return a _sequence_ of vectors for a single
            # piece of data (most commonly data that varies with time such as
            # videos and audio). Automatically check for these here and
            # normalize them if this is the case.
            if isinstance(vector, list):
                for v in vector:
                    normalize_vector(v)
            else:
                normalize_vector(vector)
        return vector

    def vectorize(
        self,
        data: Union[Any, List[Any]],
        normalize: bool = True
    ) -> List[Vector]:
        single_input = False
        if not isinstance(data, list):
            single_input = True
            data = [data]
        data_ = [self._preprocess(d) for d in data]
        vectors = [self._vectorize(d_) for d_ in data_]
        vectors_ = [self._postprocess(v) for v in vectors]
        if single_input:
            return vectors_[0]
        return vectors_

    def accelerate(self, **kwargs):
        warnings.warn("this vectorizer does not support acceleration")