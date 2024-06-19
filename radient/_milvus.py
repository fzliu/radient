from typing import Any, Dict, List, Tuple, Optional

from radient.utils.lazy_import import LazyImport

MilvusClient = LazyImport("pymilvus", attribute="MilvusClient", min_version="2.4.2")
pymilvus = LazyImport("pymilvus", min_version="2.4.2")  # Milvus Python SDK


class _MilvusInterface(object):
    """Interface to the Milvus vector database.

    This interface also works with Zilliz Cloud (https://zilliz.com/cloud).
    """

    _clients = {}
    _collection_fields = {}

    def __new__(cls, *args, **kwargs):
        return cls._get_client(*args, **kwargs)

    @classmethod
    def _get_client(
        cls,
        milvus_uri: str,
        collection_name: str,
        dimension: Optional[int] = None
    ) -> Tuple["MilvusClient", Dict[str, str]]:

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
