from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from radient.util import LazyImport

milvus = LazyImport("milvus", package="milvus")  # embedded Milvus server
pymilvus = LazyImport("pymilvus", package="pymilvus")  # Milvus Python library


class MilvusInterface(object):
    _collections = []

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def get_collection(cls, coll: str):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            # Put any initialization here.
        return cls._instance


class Vector(np.ndarray):
    """Wrapper around `numpy.ndarray` specifically for working with embeddings.
    """

    def store(self, coll: str = "default"):
        """Stores this vector in a collection.
        """
        pass


    def query(self, coll: str = "default", topk: int = 10) -> List[List[np.ndarray]]:
        """Queries the collection for nearest neighbors.
        """
        pass


class Vectorizer(ABC):

    @abstractmethod
    def __init__(self):
        self._model = None

    @property
    def model(self) -> Optional[Any]:
        return self._model

    @property
    def modality(self) -> str:
        return fully_qualified_name(self).split(".")[-2]

    @abstractmethod
    def vectorize(self, data: List[Any]) -> List[Vector]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def standardize_input(cls, item: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def standardize_inputs(cls, items: List[Any]) -> List[Any]:
        return [cls.standardize_input(item) for item in items]

    def accelerate(self, **kwargs):
        warnings.warn("")