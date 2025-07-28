from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

from radient.tasks.sinks.local._gkmeans import GKMeans
from radient.tasks.sinks.local._gkmeans import torch_auto_device


MAX_LEAF_SIZE = 200


class _GANNNode:
    def __init__(self, indices):
        self.indices = indices    # indices of dataset points in this node
        self.left = None          # left child node
        self.right = None         # right child node
        self.left_center = None   # centroid for left child
        self.right_center = None  # centroid for right child

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


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
        self._leaves = np.arange(dataset.shape[0])[np.newaxis,:]

    @property
    def centers(self):
        return self._centers

    def build(self, spill: float = 0.0):
        """Builds the tree.
        """

        gkmeans = GKMeans(
            n_clusters=2,
            device=torch_auto_device(),
            verbose=self._verbose
        )

        # Initialize root node
        root = _GANNNode(indices=np.arange(self._dataset.shape[0]))
        nodes = [root]

        while True:
            groups = np.array([node.indices for node in nodes])
            gkmeans.fit(self._dataset, groups=groups)
            C = gkmeans.cluster_centers_

            new_nodes = []
            for n, node in enumerate(nodes):
                vectors = self._dataset[node.indices,:]

                # Compute distances to the hyperplane which separates the two cluster centroids
                w = C[n,1,:] - C[n,0,:]
                b = -(C[n,1,:] + C[n,0,:]).dot(w) / 2.0
                d = (vectors.dot(w) + b) / np.linalg.norm(w)

                child_size = int(vectors.shape[0] * (0.5 + spill))
                idxs_by_dist = np.argsort(d)
                left_indices = node.indices[idxs_by_dist[:child_size]]
                right_indices = node.indices[idxs_by_dist[-child_size:]]

                # Create child nodes and attach to parent
                node.left = _GANNNode(left_indices)
                node.right = _GANNNode(right_indices)
                node.left_center = C[n,0,:]
                node.right_center = C[n,1,:]
                node.indices = None

                new_nodes.append(node.left)
                new_nodes.append(node.right)

            nodes = new_nodes

            if self._verbose:
                print(f"Num leaves: {len(nodes)}")

            mean_leaf_size = np.mean([len(node.indices) for node in nodes])
            if mean_leaf_size < MAX_LEAF_SIZE:
                if self._verbose:
                    print(f"Done, avg leaf size {mean_leaf_size}")
                    print()
                break

        self._root = root

    def get_candidates(self, query: np.ndarray):
        """Returns nearest neighbor candidates for a query vector.
        """
        node = self._root
        while not node.is_leaf:
            d_left = np.linalg.norm(node.left_center - query)
            d_right = np.linalg.norm(node.right_center - query)
            node = node.left if d_left <= d_right else node.right
        return node.indices


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

    def index(self, n_proc: int = 1):
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
