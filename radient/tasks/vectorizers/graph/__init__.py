__all__ = [
    "FastRPGraphVectorizer",
    "graph_vectorizer"
]

from typing import Optional

from radient.tasks.vectorizers.graph._base import GraphVectorizer
from radient.tasks.vectorizers.graph.fastrp import FastRPGraphVectorizer


def graph_vectorizer(method: str = "fastrp", **kwargs) -> GraphVectorizer:
    """Creates an image vectorizer specified by `method`.
    """

    # Return a reasonable default vectorizer in the event that the user does
    # not specify one.
    if method in ("fastrp"):
        return FastRPGraphVectorizer(**kwargs)
    else:
        raise NotImplementedError
