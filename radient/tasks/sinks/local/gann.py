from multiprocessing import Pool
import time

import numpy as np

from radient.tasks.sinks.local._gkmeans import GKMeans


MAX_LEAF_SIZE = 200


class GANNNode():

    def __init__(
        self,
        dataset: np.ndarray,
        indexes: np.ndarray | None = None,
        centroid: np.ndarray | None = None,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        self._dataset = dataset
        self._indexes = indexes
        self._centroid = centroid
        self._verbose = verbose
        self._children = None
    
    @property
    def centroid(self):
        return self._centroid

    @property
    def children(self):
        return self._children

    @property
    def vectors(self):
        if self._indexes is None:
            return self._dataset
        return self._dataset[self._indexes]

    def split(self, recursive: bool = True, spill: float = 0.0):
        """Splits the index.
        """
        if self._indexes is not None and len(self._indexes) <= MAX_LEAF_SIZE:
            self._indexes = set(self._indexes)
            return

        vectors = self.vectors
        #print("Splitting index with", len(vectors), "vectors.")

        # Get indexes for each cluster
        gkmeans = GKMeans(n_clusters=2, verbose=self._verbose)
        a = gkmeans.fit_predict(vectors)
        C = list(gkmeans.cluster_centers_)
        idxs_C = [np.where(a==n)[0] for n in range(2)]

        if spill > 0:
            # Determine the hyperplane that separates the two clusters
            # (i.e. the two centroids)
            w = C[1] - C[0]
            b = -(C[1] + C[0]).dot(w) / 2.0

            # Compute each point's distance to the hyperplane
            d = (vectors.dot(w) + b) / np.linalg.norm(w)

            # For each of the two clusters, add `spill` of the points that are
            # in the other cluster but are very close to the separating
            # hyperplane
            n_add = int(len(vectors) * spill / 2.0)
            idxs_C_add = [
                np.where(d > 0, d, np.inf).argsort()[:n_add],
                np.where(d <= 0, d, -np.inf).argsort()[-n_add:]
            ]
            idxs_C = [np.concatenate([idxs_C[n], idxs_C_add[n]]) for n in range(2)]

        if self._indexes is not None:
            idxs_C = [self._indexes[idxs_C[n]] for n in range(2)]

        # Create child nodes
        self._children = [
            GANNNode(
                dataset=self._dataset,
                indexes=idxs_C[n],
                centroid=C[n],
                verbose=self._verbose
            ) for n in range(2)
        ]

        # Build indexes for each child node
        if recursive:
            for child in self._children:
                child.split(recursive=recursive, spill=spill)

    def get_candidates(self, vector: np.ndarray, top_k: int = 10):
        """Searches for the nearest vector in the index.
        """
        if self._children is None:
            return self._indexes
    
        # Determine which child node to search
        d = [np.linalg.norm(node.centroid - vector) for node in self._children]
        return self._children[np.argmin(d)].get_candidates(vector, top_k=top_k)


class GANN():

    def __init__(
        self,
        n_trees: int = 4,
        spill: float = 0.0,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        self._n_trees = n_trees
        self._spill = spill
        self._verbose = verbose
        self._dataset = []

    def _build_tree(self, n: int) -> GANNNode:
        np.random.seed(None)
        node = GANNNode(dataset=self._dataset, verbose=self._verbose)
        node.split(recursive=True, spill=self._spill)
        return node

    @property
    def n_trees(self):
        return self._n_trees

    @property
    def sealed(self):
        return hasattr(self, "_trees")

    def insert(self, vector: np.ndarray):
        """Inserts a vector into the index.
        """
        if self.sealed:
            raise ValueError("Cannot insert into a sealed index.")
        self._dataset.append(vector)

    def build(self, n_proc: int = 1):
        if self.sealed:
            raise ValueError("Index is already built.")

        self._dataset = np.array(self._dataset, dtype=np.float32)

        self._trees = []
        with Pool(n_proc) as pool:
            self._trees = pool.map(self._build_tree, range(self._n_trees))
    
    def search(self, vector: np.ndarray, top_k: int = 10) -> set[int]:
        if not self.sealed:
            raise ValueError("Build the index before searching.")

        candidates = set()
        for tree in self._trees:
            candidates.update(tree.get_candidates(vector, top_k=top_k))
        candidates_ = np.array(list(candidates))
        distances = np.linalg.norm(self._dataset[candidates_] - vector, axis=1)
        return candidates_[distances.argsort()[:top_k]]
