__all__ = [
    "GraphVectorizer"
]

from abc import abstractmethod
from typing import Any

import numpy as np

from radient.util.lazy_import import fully_qualified_name, LazyImport
from radient.vectorizers.base import Vector, Vectorizer

nx = LazyImport("networkx")
sp = LazyImport("scipy")


class GraphVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def _preprocess(cls, graph: Any) -> str:
        # Turn input graphs into adjacency matrices.
        full_name = fully_qualified_name(graph)
        if isinstance(graph, sp.sparse.sparray):
            return graph
        elif isinstance(graph, sp.sparse.spmatrix):
            # TODO(fzliu): determine if support for this is necessary
            raise NotImplementedError
        elif full_name == "networkx.classes.graph.Graph":
            return nx.to_scipy_sparse_array(graph)
        else:
            raise TypeError
