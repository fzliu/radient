__all__ = [
    "GraphVectorizer"
]

from abc import abstractmethod
from typing import Any

import numpy as np

from radient.tasks.vectorizers._base import Vectorizer
from radient.utils import fully_qualified_name
from radient.utils.lazy_import import LazyImport

nx = LazyImport("networkx")
sp = LazyImport("scipy")


class GraphVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def _preprocess(cls, graph: Any, **kwargs) -> str:
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
