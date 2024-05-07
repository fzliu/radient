__all__ = [
    "FastRPGraphVectorizer",
    "graph_vectorizer"
]

from typing import Optional

from radient.vectorizers.graph.base import GraphVectorizer
from radient.vectorizers.graph.fastrp import FastRPGraphVectorizer


def graph_vectorizer(method: Optional[str] = None, **kwargs) -> GraphVectorizer:
    """Creates an image vectorizer specified by `method`.
    """

    # Return a reasonable default vectorizer in the event that the user does
    # not specify one.
    if method in (None, "fastrp"):
        return FastRPGraphVectorizer(**kwargs)
    else:
        raise NotImplementedError
