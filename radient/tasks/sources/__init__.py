from radient.tasks.sources._base import Source
from radient.tasks.sources.local import LocalSource


def source(datasource: str = "local", **kwargs) -> Source:

    if datasource == "local":
        return LocalSource(**kwargs)
    else:
        raise ValueError(f"unknown data store: {method}")


