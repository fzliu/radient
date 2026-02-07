from __future__ import annotations

from typing import Union, TYPE_CHECKING

from radient.tasks.sinks._base import Sink
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.mongo_client import MongoClient
else:
    pymongo = LazyImport("pymongo", package_name="pymongo", min_version="4.5.0")
    MongoClient = LazyImport(
        "pymongo",
        attribute="MongoClient",
        package_name="pymongo",
        min_version="4.5.0"
    )


DEFAULT_MONGODB_URI = "mongodb://127.0.0.1:27017"
DEFAULT_DATABASE_NAME = "radient"
DEFAULT_COLLECTION_NAME = "radient"
DEFAULT_VECTOR_FIELD = "vector"
DEFAULT_INDEX_NAME = "vector_index"


class _MongoDBInterface:
    """Interface to MongoDB configured for vector storage and search."""

    _clients: dict[str, "MongoClient"] = {}
    _collections: dict[tuple[str, str, str], "Collection"] = {}

    @classmethod
    def _get_collection(
        cls,
        mongodb_uri: str,
        database_name: str,
        collection_name: str
    ) -> "Collection":
        if mongodb_uri not in cls._clients:
            cls._clients[mongodb_uri] = MongoClient(mongodb_uri)
        key = (mongodb_uri, database_name, collection_name)
        if key not in cls._collections:
            client = cls._clients[mongodb_uri]
            cls._collections[key] = client[database_name][collection_name]
        return cls._collections[key]


class MongoDBSink(Sink):

    def __init__(
        self,
        operation: str,
        mongodb_uri: str = DEFAULT_MONGODB_URI,
        database_name: str = DEFAULT_DATABASE_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_field: str = DEFAULT_VECTOR_FIELD,
        index_name: str = DEFAULT_INDEX_NAME,
        **kwargs
    ):
        super().__init__()
        self._operation = operation
        self._mongodb_uri = mongodb_uri
        self._database_name = database_name
        self._collection_name = collection_name
        self._vector_field = vector_field
        self._index_name = index_name

    def transact(
        self,
        vectors: Union[Vector, list[Vector]],
        **kwargs
    ) -> Union[dict, list]:
        if not isinstance(vectors, list):
            vectors = [vectors]

        collection = _MongoDBInterface._get_collection(
            mongodb_uri=self._mongodb_uri,
            database_name=self._database_name,
            collection_name=self._collection_name
        )

        if self._operation == "insert":
            docs = [v.todict(vector_field=self._vector_field) for v in vectors]
            result = collection.insert_many(docs)
            return {
                "inserted_count": len(result.inserted_ids),
                "inserted_ids": list(result.inserted_ids)
            }

        if self._operation == "search":
            num_candidates = kwargs.pop("num_candidates", 150)
            limit = kwargs.pop("limit", 10)

            results = []
            for v in vectors:
                pipeline = [{
                    "$vectorSearch": {
                        "index": self._index_name,
                        "path": self._vector_field,
                        "queryVector": v.tolist(),
                        "numCandidates": num_candidates,
                        "limit": limit
                    }
                }, {
                    "$project": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }]
                cursor = collection.aggregate(pipeline)
                results.append(list(cursor))

            return results

        raise TypeError("invalid MongoDB operation")
