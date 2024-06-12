from radient.tasks.sinks._base import Sink
from radient.tasks.sinks.milvus import MilvusSink


def sink(datastore: str = "milvus", **kwargs) -> Sink:

    if datastore == "milvus":
        return MilvusSink(**kwargs)
    else:
        raise ValueError(f"unknown data store: {method}")


