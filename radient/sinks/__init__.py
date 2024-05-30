from radient.sinks.milvus import MilvusSink


def make_sink(datastore: str = "milvus", **kwargs):

    if datastore == "milvus":
        return MilvusSink(**kwargs)
    else:
        raise ValueError(f"unknown data store: {method}")


