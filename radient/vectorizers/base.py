from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Optional
import warnings

import numpy as np

from radient.vector import Vector
from radient.util.lazy_import import fully_qualified_name


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
    def _postprocess(cls, vector: Vector, normalize: bool = True) -> Vector:
        return vector
        #return vector / np.linalg.norm(vector)

    def vectorize(self, data: List[Any], normalize: bool = True) -> List[Vector]:
        data_ = [self._preprocess(d) for d in data]
        vectors = self._vectorize(data_)
        return [self._postprocess(v) for v in vectors]

    def accelerate(self, **kwargs):
        warnings.warn("this vectorizer does not support acceleration")