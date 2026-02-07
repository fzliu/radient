from pathlib import Path

from typing import Optional, Union, TYPE_CHECKING

from radient.tasks.sinks._base import Sink
from radient.utils import fully_qualified_name
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

if TYPE_CHECKING:
    from pymilvus import MilvusClient
    import pymilvus
else:
    MilvusClient = LazyImport("pymilvus", attribute="MilvusClient", min_version="2.4.2")
    pymilvus = LazyImport("pymilvus", min_version="2.4.2")  # Milvus Python SDK


DEFAULT_MILVUS_URI = str(Path.home() / ".radient" / "default.db")
#DEFAULT_MILVUS_URI = "http://127.0.0.1:19530"
DEFAULT_COLLECTION_NAME = "radient"


class _MilvusInterface(object):
    """Interface to the Milvus vector database.

    This interface also works with Zilliz Cloud (https://zilliz.com/cloud).
    """

    _clients: dict[str, MilvusClient] = {}
    _collection_fields: dict[tuple[str, str], dict[str, str]] = {}

    def __new__(cls, *args, **kwargs):
        return cls._get_client(*args, **kwargs)

    @classmethod
    def _get_client(
        cls,
        milvus_uri: str,
        collection_name: str,
        dimension: Optional[int] = None
    ) -> tuple[MilvusClient, dict[str, str]]:

        milvus_uri = milvus_uri.replace("localhost", "127.0.0.1")

        # If a local Milvus installation was specified, check to see if it's up
        # and running first. If not, prompt the user and start an embedded
        # Milvus instance.
        if milvus_uri not in cls._clients:
            pymilvus.connections.connect(uri=milvus_uri)
            cls._clients[milvus_uri] = MilvusClient(uri=milvus_uri)
        client = cls._clients[milvus_uri]

        # Grab the collection information. If it doesn't exist yet, create it
        # with some default settings. With the collection information, we then
        # store the vector field names inside the `_collection_fields` global
        # object.
        uri_and_coll = (milvus_uri, collection_name)
        if uri_and_coll not in cls._collection_fields:
            if not client.has_collection(collection_name=collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    dimension=dimension,
                    auto_id=True,
                    enable_dynamic_field=True
                )
            info = client.describe_collection(collection_name=collection_name)
            fields = {}
            # TODO(fzliu): support multiple vector fields of the same type.
            for f in info["fields"]:
                if f["type"] == pymilvus.DataType.BINARY_VECTOR:
                    fields["binary"] = f["name"]
                elif f["type"] == pymilvus.DataType.FLOAT_VECTOR:
                    fields["dense"] = f["name"]
                elif (pymilvus.__version__ >= "2.4.0" and
                      f["type"] == pymilvus.DataType.SPARSE_FLOAT_VECTOR):
                    fields["sparse"] = f["name"]
            cls._collection_fields[uri_and_coll] = fields
        info = cls._collection_fields[uri_and_coll]

        return (client, info)


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
        vectors: Union[Vector, list[Vector]],
        **kwargs
    ) -> dict[str, Union[int, list[int]]]:
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

        if self._operation == "search":
            return client.search(
                collection_name=self._collection_name,
                data=[v.tolist() for v in vectors],
                **kwargs
            )

        raise TypeError("invalid Milvus operation")
