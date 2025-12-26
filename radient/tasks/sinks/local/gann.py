from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, asdict, field

from numba import jit
import numpy as np

from radient.tasks.sinks.local._gkmeans import GKMeans
from radient.tasks.sinks.local._gkmeans import torch_auto_device
from radient.utils.json_tools import ExtendedJSONEncoder


MAX_LEAF_SIZE = 200


def _maybe_apply(func: Callable, value: Any) -> Any:
    if value is not None:
        return func(value)
    return None


@jit(nopython=True, cache=True, fastmath=True)
def _hyperplane_distance(
    vector: np.ndarray,
    weight: np.ndarray,
    bias: np.float32
) -> np.float32:
    """Computes the signed distance of a vector to a hyperplane defined by
    `weight` and `bias`.
    """
    return (vector.dot(weight) + bias)# / np.linalg.norm(weight)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _compute_distances(
    dataset: np.ndarray,
    candidates: np.ndarray,
    query: np.ndarray
) -> np.ndarray:
    """Computes squared L2 distances between query and candidate vectors. For
    ranking purposes, squared distances work the same as regular distances.
    """
    dists = np.empty(len(candidates), dtype=np.float32)
    for i in range(len(candidates)):
        idx = candidates[i]
        dist_sq = 0.0
        for j in range(dataset.shape[1]):
            dist_sq += (dataset[candidates[i],j] - query[j]) ** 2
        dists[i] = dist_sq
    return dists


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _argsort_topk(
    dists: np.ndarray,
    k: int
) -> np.ndarray:
    """Selection algorithm which is faster than full sort for small k.
    """
    n = len(dists)
    indices = np.arange(n)
    for i in range(min(k, n)):
        min_idx = i
        min_val = dists[indices[i]]
        for j in range(i + 1, n):
            if dists[indices[j]] < min_val:
                min_idx = j
                min_val = dists[indices[j]]
        if min_idx != i:
            temp = indices[i]
            indices[i] = indices[min_idx]
            indices[min_idx] = temp
    return indices[:k]


@jit(nopython=True, cache=True, fastmath=True)
def _traverse_tree(
    query: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    cutoffs: np.ndarray,
    children: np.ndarray
) -> np.int64:
    """Traverses tree to find leaf node ID for a query vector using array-based
    representation for numba compatibility.
    """
    nid = 0
    while nid < children.shape[0]:
        val = _hyperplane_distance(query, weights[nid,:], biases[nid])
        if val < 0:
            nid = children[nid][0]
        else:
            nid = children[nid][1]
    return nid


class _GANNTree:

    def __init__(self):
        self._buffer = defaultdict(list)

        # Array-based representation for efficient tree traversal
        self._weights = None
        self._biases = None
        self._cutoffs = None   # 2D array of (left, right) cutoffs
        self._children = None  # 2D array of (left, right) children
        self._leaves = defaultdict(lambda: np.array([], dtype=np.int64))

    @property
    def is_indexed(self):
        return self._weights is not None

    def insert(self, key: int, vector: np.ndarray):
        """Single-tree insert function.
        """
        if self.is_indexed:
            queue = [0]
            while queue:
                nid = queue.pop()
                if nid >= self._children.shape[0]:
                    self._buffer[nid].append(key)
                else:
                    val = _hyperplane_distance(vector, self._weights[nid], self._biases[nid])
                    if val <= self._cutoffs[nid][0]:
                        queue.append(self._children[nid][0])
                    if val >= self._cutoffs[nid][1]:
                        queue.append(self._children[nid][1])

    def flush(self):
        """Single-tree flush function.
        """
        for nid, indices in self._buffer.items():
            self._leaves[nid] = np.append(self._leaves[nid], indices)
        self._buffer.clear()

    def get_candidates(self, query: np.ndarray) -> np.ndarray:
        """Returns nearest neighbor candidates for a query vector.
        """
        if self._weights is not None:
            # Use numba-optimized traversal
            nid = _traverse_tree(
                query,
                self._weights,
                self._biases,
                self._cutoffs,
                self._children
            )
            return self._leaves[nid]
        else:
            raise ValueError("Tree index not built yet.")

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "weights.npy", self._weights)
        np.save(path / "biases.npy", self._biases)
        np.save(path / "cutoffs.npy", self._cutoffs)
        np.save(path / "children.npy", self._children)
        with open(path / "leaves.json", "w") as f:
            json.dump(self._leaves, f, cls=ExtendedJSONEncoder)

    @classmethod
    def load(cls, path: str | Path) -> _GANNTree:
        path = Path(path)
        tree = cls()
        tree._weights = np.load(path / "weights.npy")
        tree._biases = np.load(path / "biases.npy")
        tree._cutoffs = np.load(path / "cutoffs.npy")
        tree._children = np.load(path / "children.npy")
        # Do not load leaves; they should be populated during GANN load
        #with open(path / "leaves.json", "r") as f:
        #    tree._leaves = json.load(f)
        #    for nid, idxs in tree._leaves.items():
        #        tree._leaves[nid] = np.array(idxs, dtype=np.int64)
        return tree

    @classmethod
    def from_dataset(
        cls,
        dataset: np.ndarray,
        spill: float = 0.0,
        verbose: bool = False
    ) -> _GANNTree:

        # Initialize GKMeans
        device = torch_auto_device()
        gkmeans = GKMeans(
            n_clusters=2,
            device=device,
            verbose=verbose
        )

        # Initialize tree structure
        tree = cls()
        weights = []
        biases = []
        cutoffs = []
        children = []
        leaves = [np.arange(dataset.shape[0], dtype=np.int64)]

        while True:
            gkmeans.fit(dataset, groups=np.vstack(leaves))
            centers = gkmeans.cluster_centers_

            new_leaves = []
            for n, leaf in enumerate(leaves):
                vectors = dataset[leaf,:]

                # Compute hyperplane parameters (weight, bias)
                weight = centers[n,1,:] - centers[n,0,:]
                bias = -((centers[n,1,:] + centers[n,0,:]).dot(weight)) / 2.0
                dists = _hyperplane_distance(vectors, weight, bias)
                idxs_by_dist = np.argsort(dists)
                child_size = int(vectors.shape[0] * (0.5 + spill))

                # Set node parameters used to traverse the tree
                weights.append(weight)
                biases.append(bias)
                cutoffs.append((
                    dists[idxs_by_dist[child_size-1]],  # left cutoff
                    dists[idxs_by_dist[-child_size]]    # right cutoff
                ))
                children.append((
                    2*len(children)+1,
                    2*len(children)+2
                ))

                # Update the leaves for the next iteration
                new_leaves.append(leaf[idxs_by_dist[:child_size]])   # left child
                new_leaves.append(leaf[idxs_by_dist[-child_size:]])  # right child

            leaves = new_leaves

            if verbose:
                print(f"Num leaves: {len(leaves)}")

            mean_leaf_size = np.mean([len(leaf) for leaf in leaves])
            if mean_leaf_size < MAX_LEAF_SIZE:
                if verbose:
                    print(f"Done, avg leaf size {mean_leaf_size}")
                    print()
                break

        tree._weights = np.array(weights, dtype=np.float32)
        tree._biases = np.array(biases, dtype=np.float32)
        tree._cutoffs = np.array(cutoffs, dtype=np.float32)
        tree._children = np.array(children, dtype=np.int32)
        tree._leaves = {(n + len(children)): leaf for n, leaf in enumerate(leaves)}

        return tree


class GANN:

    def __init__(
        self,
        dim : int,
        n_trees: int = 1,
        spill: float = 0.0,
        verbose: bool = False
    ):
        self._dim = dim
        self._n_trees = n_trees
        self._spill = spill
        self._verbose = verbose
        self._buffer = []
        self._dataset = np.empty((0, dim), dtype=np.float32)
        self._trees = []

    @property
    def n_trees(self):
        return self._n_trees

    def insert(self, vector: np.ndarray):
        """Inserts a vector into the index (all trees).
        """
        self._buffer.append(vector)
        if self._trees:
            for tree in self._trees:
                key = len(self._dataset) + len(self._buffer) - 1
                tree.insert(key, vector)

    def flush(self):
        """Flushes all of the vectors in the buffer to the dataset.
        """
        buffer = np.array(self._buffer, dtype=np.float32)
        self._dataset = np.append(self._dataset, buffer, axis=0)
        self._buffer.clear()
        for n, tree in enumerate(self._trees):
            tree.flush()

    def index(self):
        self.flush()
        self._trees = [
            _GANNTree.from_dataset(
                self._dataset,
                spill=self._spill,
                verbose=self._verbose
            ) for _ in range(self._n_trees)
        ]

    def search(self, query: np.ndarray, top_k: int = 10) -> np.ndarray:
        if not self._trees:
            raise ValueError("Build the index first.")

        # Gather candidates from all trees
        all_candidates = [tree.get_candidates(query) for tree in self._trees]
        candidates = np.unique(np.hstack(all_candidates))

        # Determine the top k closest candidates
        dists = _compute_distances(self._dataset, candidates, query)
        best = _argsort_topk(dists, top_k)
        #best = np.argpartition(dists, top_k)[:top_k]
        return candidates[best]
    
    def search_original(self, query: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Original non-optimized search for comparison."""
        if not self._trees:
            raise ValueError("Build the index first.")

        # Gather candidates from all trees
        all_candidates = []
        for tree in self._trees:
            all_candidates.extend(tree.get_candidates(query))
        candidates = np.unique(np.array(all_candidates, dtype=np.int64))

        # Original numpy-based distance computation
        best = np.linalg.norm(self._dataset[candidates] - query, axis=1).argsort()
        return candidates[best[:top_k]]

    def save(self, path: str) -> None:
        """Save the index to a directory at `path`.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "meta.json").open("w") as f:
            json.dump({
                    "dim": self._dim,
                    "n_trees": self._n_trees,
                    "spill": self._spill,
                    "verbose": self._verbose
                }, f)
        for n, tree in enumerate(self._trees):
            tree.save(str(path / f"tree_{n}"))
        np.savetxt(str(path / "dataset.txt"), np.array(self._dataset))

    @classmethod
    def load(cls, path: str) -> GANN:
        """Load the index from a directory at `path`.
        """
        path = Path(path)

        # Load metadata and create GANN instance
        with (path / "meta.json").open("r") as f:
            meta = json.load(f)
        gann = cls(**meta)

        # Load trees
        for n in range(gann.n_trees):
            gann._trees.append(_GANNTree.load(path / f"tree_{n}"))
        
        # Load and insert dataset vectors
        dataset = np.loadtxt(str(path / "dataset.txt"), dtype=np.float32)
        if dataset.ndim == 1:
            dataset = dataset.reshape(1, -1)
        for n, vector in enumerate(dataset):
            gann.insert(vector)
        gann.flush()

        return gann
