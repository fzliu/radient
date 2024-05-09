from typing import Optional

from radient.util.lazy_import import fully_qualified_name, LazyImport

milvus = LazyImport("milvus")  # embedded Milvus server
MilvusClient = LazyImport("pymilvus", attribute="MilvusClient")
pymilvus = LazyImport("pymilvus")  # Milvus Python SDK
validators = LazyImport("validators")


class MilvusInterface(object):
    """Interface to the Milvus vector database. Because there are 

    This interface also works with Zilliz Cloud (https://zilliz.com/cloud).
    """

    _clients = {}
    _collection_fields = {}

    def __new__(cls, *args, **kwargs):
        return cls._get_client(*args, **kwargs)

    @classmethod
    def _get_client(cls, uri: str, coll: str, dim: Optional[int] = None):

        uri = uri.replace("localhost", "127.0.0.1")

        # If a local Milvus installation was specified, check to see if it's up
        # and running first. If not, prompt the user and start an embedded
        # Milvus instance.
        if uri not in cls._clients:
            try:
                pymilvus.connections.connect(uri=uri, alias="_test")
            except pymilvus.exceptions.MilvusException:
                if "127.0.0.1" in uri:
                    print("Local Milvus instance not detected.")
                    if input(f"Start local Milvus? [Y/n]\n") == "Y":
                        milvus.default_server.start()
            pymilvus.connections.disconnect(alias="_test")
            cls._clients[uri] = MilvusClient(uri=uri)
        client = cls._clients[uri]

        # Grab the collection information. If it doesn't exist yet, create it
        # with some default settings. With the collection information, we then
        # store the vector field names inside the `_collection_fields` global
        # object.
        if (uri, coll) not in cls._collection_fields:
            if not client.has_collection(coll):
                client.create_collection(
                    collection_name=coll,
                    dimension=dim,
                    auto_id=True,
                    enable_dynamic_field=True)
            info = client.describe_collection(coll)
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
            cls._collection_fields[(uri, coll)] = fields
        info = cls._collection_fields[(uri, coll)]

        return (client, info)