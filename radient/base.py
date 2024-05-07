from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from radient.util import fully_qualified_name, LazyImport

milvus = LazyImport("milvus", package="milvus")  # embedded Milvus server
MilvusClient = LazyImport("pymilvus", package="pymilvus", attribute="MilvusClient")  # Milvus Python library


class MilvusInterface(object):
    _collections = {}

    @classmethod
    def get_client(cls, uri: str, coll: str = "default"):
        if (uri, coll) not in _collections:
            _collections[(uri, coll)] = MilvusClient(
                collection_name=coll,
                uri=uri,
                vector_field="vector",
                overwrite=True,
            )
        return _collections[(uri, coll)]


class Vector(np.ndarray):
    """Wrapper around `numpy.ndarray` specifically for working with embeddings.
    """

    def store(self, uri: str = "http://localhost:19530", coll: str = "_def"):
        """Stores this vector in a collection.
        """
        data = [{"vector": self.tolist()}]
        client.insert_data(data)


    def query(self, uri: str = "http://localhost:19530", coll: str = "default", topk: int = 10) -> List[List[np.ndarray]]:
        """Queries the collection for nearest neighbors.
        """
        pass


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

    @abstractmethod
    def vectorize(self, data: List[Any]) -> List[Vector]:
        #TODO(fzliu): add batch size
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