__all__ = [
    "FastRPGraphVectorizer"
]

from typing import Any, List, Sequence

import numpy as np

from radient.tasks.vectorizers.graph._base import GraphVectorizer
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

sp = LazyImport("scipy")
SparseRandomProjection = LazyImport("sklearn.random_projection", attribute="SparseRandomProjection", package_name="scikit-learn")


class FastRPGraphVectorizer(GraphVectorizer):
    """Computes node (not graph) embeddings using the FastRP algorithm.
    """

    def __init__(
        self,
        dimension: int = 64,
        weights: Sequence = (0.1, 0.2, 1.0, 3.0),
        beta: float = 0.0
    ):
        super().__init__()
        self._projection = SparseRandomProjection(n_components=dimension)
        self._weights = weights
        self._beta = beta

    def _vectorize(self, graph: "sp.sparse.sparray", **kwargs) -> Vector:
        """Radient-specific implementation of the FastRP vectorization
        algorithm.
        """
        projector = self._projection.fit(graph)
        R = projector.components_.T

        # Compute \mathbf{D} as per Chen et al. equation 0:
        # D = sum(Sip, axis=p) if i = j else 0
        L = graph.sum(axis=1)
        L = 0.5 * L**self._beta / graph.shape[0]
        L = sp.sparse.diags_array(L)
        N_k = graph @ L @ R

        # Compute a weighted combination of the powers of the projections.
        result = self._weights[0] * N_k
        for k in range(1, len(self._weights)):
            N_k = graph @ N_k
            result += self._weights[k] * N_k

        result = result.todense().view(Vector)
        return [item for item in result]
