from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Optional, Union

import numpy as np

from radient.sinks.milvus import MilvusInterface


class Vector(np.ndarray):
    """Wrapper around `numpy.ndarray` specifically for working with embeddings.
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
    def metadata(self, data: Dict):
        self._metadata = OrderedDict(data)

    def add_key_value(self, key: str, value: Union[Dict, str, float, int]):
        self._metadata[key] = value

    def remove_key(self, key: str) -> Any:
        return self._metadata.pop(key)

    def store(
        self,
        sink_type: Union[Sequence[str], str] = "vectordb",
        **kwargs      
    ):
        """Stores this vector in the specified sink.
        """
        if isinstance(sink_type, str):
            sink_type = [sink_type]
        for sink in sink_type:
            if sink == "vectordb":
                return self._store_vectordb(**kwargs)

    def _store_vectordb(
        self,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "_radient",
        field_name: Optional[str] = None
    ) -> Dict[str, Union[str, List]]:
        """Stores this vector in the specified collection.
        """
        client, info = MilvusInterface(milvus_uri, collection_name, dim=self.size)
        field = info["dense"]
        # We can get away with using the dict constructor because all schema
        # field names are strings.
        data = dict(self._metadata, **{field: self.tolist()})
        return client.insert(
            collection_name=collection_name,
            data=data
        )

    def _query_vectordb(
        self,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "_radient",
        metric_type: Optional[str] = "COSINE",
        topk: int = 10
    ) -> List[List[np.ndarray]]:
        """Queries the specified collection for nearest neighbors.
        """
        client, info = MilvusInterface(milvus_uri, collection_name)
        return client.search(
            collection_name="test_collection",
            data=[self.tolist()],
            limit=topk,
            search_params={"metric_type": metric_type, "params": {}}
        )

    def _store_lakehouse(
        self
    ) -> None:
        """Stores this vector in a Lakehouse.
        """
        raise NotImplementedError
