from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Optional, Union

import numpy as np

from radient._milvus import _MilvusInterface


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
    def metadata(self, data: Dict):
        self._metadata = OrderedDict(data)

    def putmeta(
        self,
        key: str,
        value: Union[Dict[str, Union[str, float, int]], str, float, int]
    ) -> "Vector":
        self._metadata[key] = value
        # Enable chaining function calls together.
        return self

    def popmeta(
        self,
        key: str
    ) -> Union[Dict[str, Union[str, float, int]], str, float, int]:
        return self._metadata.pop(key)

    def todict(
        self,
        vector_field: str = "vector"
    ) -> Dict[str, Union["Vector", str, float, int]]:
        return dict(self._metadata, **{vector_field: self.tolist()})

    def store(
        self,
        sink_type: Union[Sequence[str], str] = "vectordb",
        **kwargs
    ):
        """Stores this vector in the specified sink. This function is for
        convenience only.
        """
        if isinstance(sink_type, str):
            sink_type = [sink_type]
        for sink in sink_type:
            if sink == "vectordb":
                return self._store_vectordb(**kwargs)

    def _store_vectordb(
        self,
        milvus_uri: str,
        collection_name: str,
        field_name: Optional[str] = None
    ) -> Dict[str, Union[str, List]]:
        """Stores this vector in the specified collection.
        """
        client, info = _MilvusInterface(milvus_uri, collection_name, dim=self.size)
        field = info["dense"]
        # We can get away with using the dict constructor because all schema
        # field names are strings.
        data = dict(self._metadata, **{field: self.tolist()})
        return client.insert(
            collection_name=collection_name,
            data=data
        )

    def _search_vectordb(
        self,
        milvus_uri: str,
        collection_name: str,
        metric_type: Optional[str] = "COSINE",
        topk: int = 10
    ) -> List[List[np.ndarray]]:
        """Queries the specified collection for nearest neighbors.
        """
        client, info = _MilvusInterface(milvus_uri, collection_name)
        return client.search(
            collection_name="test_collection",
            data=[self.tolist()],
            limit=topk,
            search_params={"metric_type": metric_type, "params": {}}
        )

