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


@jit(nopython=True, cache=True, fastmath=True)
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


@jit(nopython=True, cache=True, fastmath=True)
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


@dataclass
class _GANNNode:
    _id: int
    indices: np.ndarray | None = None
    left: np.int32 | _GANNNode = None
    right: np.int32 | _GANNNode = None
    weight: np.ndarray | None = None
    bias: np.float32 | None = None
    left_cutoff: np.float32 | None = None
    right_cutoff: np.float32 | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is not None and self.right is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "_id": self._id,
            "indices": _maybe_apply(lambda x: x.tolist(), self.indices),
            "left": _maybe_apply(int, self.left._id),
            "right": _maybe_apply(int, self.right._id),
            "weight": _maybe_apply(lambda x: x.tolist(), self.weight),
            "bias": _maybe_apply(float, self.bias),
            "left_cutoff": _maybe_apply(float, self.left_cutoff),
            "right_cutoff": _maybe_apply(float, self.right_cutoff)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _GANNNode:
        return cls(
            _id=data["_id"],
            indices=_maybe_apply(np.int64, data["indices"]),
            left=_maybe_apply(np.int32, data["left"]),
            right=_maybe_apply(np.int32, data["right"]),
            weight=_maybe_apply(np.float32, data["weight"]),
            bias=_maybe_apply(np.float32, data["bias"]),
            left_cutoff=_maybe_apply(np.float32, data["left_cutoff"]),
            right_cutoff=_maybe_apply(np.float32, data["right_cutoff"])
        )

class _GANNTree:

    def __init__(self):
        self._root = None
        self._nodes = []
        self._buffer = defaultdict(list)

    @property
    def is_indexed(self):
        return self._root is not None

    def insert(self, key: int, vector: np.ndarray) :
        """Inserts a vector into the index.
        """
        if self.is_indexed:
            queue = [self._root]
            while queue:
                node = queue.pop()
                if node.is_leaf:
                    self._buffer[node._id].append(key)
                else:
                    val = _hyperplane_distance(vector, node.weight, node.bias)
                    if val <= node.left_cutoff:
                        queue.append(self._nodes[node.left])
                    if val >= node.right_cutoff:
                        queue.append(self._nodes[node.right])
        

    def flush(self):
        """Flushes all of the indices in the buffer to the nodes.
        """
        for nid, indices in self._buffer.items():
            node = self._nodes[nid]
            node.indices = np.append(node.indices, indices)
        self._buffer.clear()

    def get_candidates(self, query: np.ndarray) -> np.ndarray:
        """Returns nearest neighbor candidates for a query vector.
        """
        node = self._root
        while not node.is_leaf:
            val = _hyperplane_distance(query, node.weight, node.bias)
            node = node.left if val < node.left_cutoff else node.right
        return node.indices

    def to_file(self, path: str):
        with open(path, "w") as f:
            for node in self._nodes:
                json.dump(node.to_dict(), f, cls=ExtendedJSONEncoder)
                f.write("\n")

    @classmethod
    def from_file(cls, path: str) -> _GANNTree:
        tree = cls()
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                tree._nodes.append(_GANNNode.from_dict(data))
        tree._root = tree._nodes[0]

        # Relink child nodes
        for node in tree._nodes:
            if not node.is_leaf:
                node.left = tree._nodes[node.left]
                node.right = tree._nodes[node.right]

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
        root = _GANNNode(_id=0, indices=np.arange(dataset.shape[0], dtype=np.int64))
        tree._nodes.append(root)

        leaves = [root]

        while True:
            groups = np.vstack([node.indices for node in leaves])
            gkmeans.fit(dataset, groups=groups)
            centers = gkmeans.cluster_centers_

            new_leaves = []
            for n, node in enumerate(leaves):
                vectors = dataset[node.indices,:]

                # Compute hyperplane parameters (weight, bias)
                weight = centers[n,1,:] - centers[n,0,:]
                bias = -((centers[n,1,:] + centers[n,0,:]).dot(weight)) / 2.0
                dists = _hyperplane_distance(vectors, weight, bias)

                # Determine the left and right child node indices
                child_size = int(vectors.shape[0] * (0.5 + spill))
                idxs_by_dist = np.argsort(dists)
                left_indices = node.indices[idxs_by_dist[:child_size]]
                right_indices = node.indices[idxs_by_dist[-child_size:]]

                # Set node parameters used to traverse the tree
                node.weight = weight
                node.bias = np.float32(bias)
                node.left_cutoff = np.float32(dists[idxs_by_dist[child_size-1]])
                node.right_cutoff = np.float32(dists[idxs_by_dist[-child_size]])

                # Create child nodes with sequential IDs
                node.left = _GANNNode(_id=len(tree._nodes), indices=left_indices)
                node.right = _GANNNode(_id=len(tree._nodes), indices=right_indices)
                tree._nodes.append(node.left)
                tree._nodes.append(node.right)

                node.indices = None
                new_leaves.append(node.left)
                new_leaves.append(node.right)

            leaves = new_leaves

            if verbose:
                print(f"Num leaves: {len(leaves)}")

            mean_leaf_size = np.mean([len(node.indices) for node in leaves])
            if mean_leaf_size < MAX_LEAF_SIZE:
                if verbose:
                    print(f"Done, avg leaf size {mean_leaf_size}")
                    print()
                break

        tree._root = root
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
        """Inserts a vector into the index.
        """
        self._buffer.append(vector)
        if self._trees:
            for tree in self._trees:
                tree.insert(len(self._dataset), vector)

    def flush(self):
        """Flushes all of the indices in the buffer to the nodes.
        """
        self._dataset = np.append(self._dataset, self._buffer, axis=0)
        for tree in self._trees:
            tree.flush()

    def index(self):
        dataset = np.array(self._dataset, dtype=np.float32)
        self._trees = [
            _GANNTree.from_dataset(
                dataset,
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
        
        return candidates[best]
    
    def search_original(self, query: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Original non-optimized search for comparison."""
        if not self._trees:
            raise ValueError("Build the index first.")

        # Gather candidates from all trees
        all_candidates = []
        for tree in self._trees:
            all_candidates.extend(tree.get_candidates(query))
        candidates = np.unique(np.array(all_candidates, dtype=np.int32))

        # Original numpy-based distance computation
        dataset_array = np.array(self._dataset, dtype=np.float32)
        best = np.linalg.norm(dataset_array[candidates] - query, axis=1).argsort()
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
                }, f, cls=ExtendedJSONEncoder)
        for n, tree in enumerate(self._trees):
            tree.to_file(str(path / f"tree_{n}.json"))
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
            gann._trees.append(_GANNTree.from_file(path / f"tree_{n}.json"))
        
        # Load and insert dataset vectors
        dataset = np.loadtxt(str(path / "dataset.txt"), dtype=np.float32)
        for n, vector in enumerate(dataset):
            gann.insert(vector)
        gann.flush()

        return gann
