from multiprocessing import Pool
import time

import numpy as np

from radient.tasks.sinks.local._gkmeans import GKMeans


MAX_LEAF_SIZE = 200


class _GANNTree():

    def __init__(
        self,
        dataset: np.ndarray,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        self._dataset = dataset
        self._verbose = verbose
        self._centers = []
        self._leaves = [np.arange(dataset.shape[0])]

    def build(self, spill: float = 0.0):
        """Builds the tree.
        """

        gkmeans = GKMeans(n_clusters=2, verbose=self._verbose)

        while True:

            # Get indexes for each cluster
            gkmeans.fit(self._dataset, groups=self._leaves)
            C = gkmeans.cluster_centers_
            #idxs_C = [np.where(a==n)[0] for n in range(len(C))]

            new_leaves = []
            for (n, leaf) in enumerate(self._leaves):
                # Determine the hyperplane that separates the two clusters
                # (i.e. the hyperplane orthogonal to the line between centroids)
                w = C[n,1,:] - C[n,0,:]
                b = -(C[n,1,:] + C[n,0,:]).dot(w) / 2.0

                # Compute each point's distance to the hyperplane
                vectors = self._dataset[leaf,:]
                d = (vectors.dot(w) + b) / np.linalg.norm(w)

                # For each of the two clusters, add `spill` of the points
                # that are in the other cluster but are very close to the
                # separating hyperplane
                n_add = int(vectors.shape[0] * spill / 2.0)
                cutoffs = (
                    d[np.where(d > 0, d, np.inf).argsort()[n_add]],
                    d[np.where(d <= 0, d, -np.inf).argsort()[-n_add-1]]
                )
                new_leaves.append(leaf[np.where(d < cutoffs[0])])
                new_leaves.append(leaf[np.where(d > cutoffs[1])])

            self._centers.append(C)
            self._leaves = new_leaves

            leaf_sizes = [len(leaf) for leaf in self._leaves]
            if self._verbose:
                print(f"Num leaves: {len(self._leaves)}")
                print(f"Leaf sizes: {leaf_sizes}")

            # Continue until the average leaf size is below the threshold
            if np.mean(leaf_sizes) < MAX_LEAF_SIZE:
                if self._verbose:
                    print(f"Done, avg leaf size {np.mean(leaf_sizes)}")
                    print()
                break

    def get_candidates(self, query: np.ndarray):
        """Returns nearest neighbor candidates for a query vector.
        """
        idx = 0
        for center in self._centers:
            idx = 2 * idx + np.linalg.norm(center[idx] - query, axis=1).argmin()
        return self._leaves[idx]


class GANN():

    def __init__(
        self,
        n_trees: int = 1,
        spill: float = 0.0,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        self._n_trees = n_trees
        self._spill = spill
        self._verbose = verbose

        self._dataset = []

    def _build_tree(self, n: int) -> _GANNTree:
        np.random.seed(None)
        tree = _GANNTree(dataset=self._dataset, verbose=self._verbose)
        tree.build(spill=self._spill)
        return tree

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
        with Pool(n_proc) as pool:
            self._trees = pool.map(self._build_tree, range(self._n_trees))
    
    def search(self, query: np.ndarray, top_k: int = 10) -> list[int]:
        if not self.sealed:
            raise ValueError("Build the index before searching.")

        candidates = set()
        candidates.update(*[t.get_candidates(query) for t in self._trees])
        candidates = np.array(list(candidates))
        vectors = self._dataset[candidates,:]
        best = np.linalg.norm(vectors - query, axis=1).argsort()
        return candidates[best[:top_k]]
