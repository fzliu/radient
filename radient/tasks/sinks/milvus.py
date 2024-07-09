from pathlib import Path

from typing import Dict, List, Optional, Tuple, Union

from radient._milvus import _MilvusInterface
from radient.tasks.sinks._base import Sink
from radient.utils import fully_qualified_name
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector


DEFAULT_MILVUS_URI = str(Path.home() / ".radient" / "default.db")
#DEFAULT_MILVUS_URI = "http://127.0.0.1:19530"
DEFAULT_COLLECTION_NAME = "radient"


class MilvusSink(Sink):

    def __init__(
        self,
        operation: str,
        milvus_uri: str = DEFAULT_MILVUS_URI,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_field: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self._operation = operation
        self._milvus_uri = milvus_uri
        self._collection_name = collection_name
        self._vector_field = vector_field

    def transact(
        self,
        vectors: Union[Vector, List[Vector]],
        **kwargs
    ) -> Dict[str, Union[int, List[int]]]:
        if not isinstance(vectors, list):
            vectors = [vectors]
        client, info = _MilvusInterface._get_client(
            milvus_uri=self._milvus_uri,
            collection_name=self._collection_name,
            dimension=vectors[0].size
        )
        # If `field_name` is None, attempt to automatically acquire the field
        # name from the collection info.
        vector_field = self._vector_field or info["dense"]

        if self._operation == "insert":
            return client.insert(
                collection_name=self._collection_name,
                data=[v.todict(vector_field=vector_field) for v in vectors],
                **kwargs
            )
        elif self._operation == "search":
            return client.search(
                collection_name=self._collection_name,
                data=[v.tolist() for v in vectors],
                **kwargs
            )
        else:
            raise TypeError("invalid Milvus operation")
