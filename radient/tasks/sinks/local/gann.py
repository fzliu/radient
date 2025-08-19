from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, asdict

import numpy as np

from radient.tasks.sinks.local._gkmeans import GKMeans
from radient.tasks.sinks.local._gkmeans import torch_auto_device
from radient.utils.json_tools import NumPyJSONEncoder


MAX_LEAF_SIZE = 200


def _maybe_apply(
    func: Callable[[Any], Any],
    x: Any
) -> Any:
    """Applies `func` to `x` if `x` is not None, otherwise returns None.
    """
    return func(x) if x is not None else None


def _hyperplane_distance(
    vector: np.ndarray,
    weight: np.ndarray,
    bias: np.float32 | np.float64 | float
) -> float:
    """Computes the distance of a vector to a hyperplane defined by `weight` and `bias`.
    """
    return (vector.dot(weight) + bias) / np.linalg.norm(weight)


@dataclass
class _GANNNode:
    indices: list[int] | None = None
    left: _GANNNode | None = None
    right: _GANNNode | None = None
    weight: np.ndarray | None = None
    bias: np.float32 | None = None
    left_cutoff: np.float32 | None = None
    right_cutoff: np.float32 | None = None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _GANNNode:
        return cls(
            indices=[],
            left=_maybe_apply(_GANNNode.from_dict, data["left"]),
            right=_maybe_apply(_GANNNode.from_dict, data["right"]),
            weight=_maybe_apply(np.float32, data["weight"]),
            bias=_maybe_apply(np.float32, data["bias"]),
            left_cutoff=_maybe_apply(np.float32, data["left_cutoff"]),
            right_cutoff=_maybe_apply(np.float32, data["right_cutoff"])
        )


class _GANNTree:

    def __init__(self):
        self._root = None

    @property
    def is_indexed(self):
        return self._root is not None

    def insert(self, key: int, vector: np.ndarray):
        """Inserts a vector into the index."""
        if self.is_indexed:
            queue = [self._root]
            while queue:
                node = queue.pop()
                if node.is_leaf:
                    node.indices.append(key)
                else:
                    val = _hyperplane_distance(vector, node.weight, node.bias)
                    if val <= node.left_cutoff:
                        queue.append(node.left)
                    if val >= node.right_cutoff:
                        queue.append(node.right)

    def get_candidates(self, query: np.ndarray) -> list[int]:
        """Returns nearest neighbor candidates for a query vector.
        """
        node = self._root
        while not node.is_leaf:
            val = _hyperplane_distance(query, node.weight, node.bias)
            node = node.left if val < 0 else node.right
        return node.indices

    def to_dict(self):
        return {
            "root": asdict(self._root)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _GANNTree:
        tree = cls()
        tree._root = _GANNNode.from_dict(data["root"])
        return tree

    @classmethod
    def from_dataset(
        cls,
        dataset: np.ndarray,
        spill: float = 0.0,
        verbose: bool = False
    ) -> _GANNTree:
        gkmeans = GKMeans(
            n_clusters=2,
            device=torch_auto_device(),
            verbose=verbose
        )

        # Initialize root node
        root = _GANNNode(indices=list(range(dataset.shape[0])))
        nodes = [root]

        while True:
            groups = np.array([node.indices for node in nodes])
            gkmeans.fit(dataset, groups=groups)
            centers = gkmeans.cluster_centers_

            new_nodes = []
            for n, node in enumerate(nodes):
                vectors = dataset[node.indices,:]

                # Compute hyperplane parameters (weight, bias)
                weight = centers[n,1,:] - centers[n,0,:]
                bias = -((centers[n,1,:] + centers[n,0,:]).dot(weight)) / 2.0
                dists = _hyperplane_distance(vectors, weight, bias)

                # Determine the left and right child node indices
                child_size = int(vectors.shape[0] * (0.5 + spill))
                idxs_by_dist = np.argsort(dists)
                left_indices = [node.indices[i] for i in idxs_by_dist[:child_size]]
                right_indices = [node.indices[i] for i in idxs_by_dist[-child_size:]]

                # Set node parameters used to traverse the tree
                node.weight = weight
                node.bias = float(bias)
                node.left_cutoff = float(dists[idxs_by_dist[child_size-1]])
                node.right_cutoff = float(dists[idxs_by_dist[-child_size]])

                node.left = _GANNNode(indices=left_indices)
                node.right = _GANNNode(indices=right_indices)
                node.indices = []
                new_nodes.append(node.left)
                new_nodes.append(node.right)

            nodes = new_nodes

            if verbose:
                print(f"Num leaves: {len(nodes)}")

            mean_leaf_size = np.mean([len(node.indices) for node in nodes])
            if mean_leaf_size < MAX_LEAF_SIZE:
                if verbose:
                    print(f"Done, avg leaf size {mean_leaf_size}")
                    print()
                break

        tree = cls()
        tree._root = root
        return tree


class GANN:

    def __init__(
        self,
        n_trees: int = 1,
        spill: float = 0.0,
        verbose: bool = False
    ):
        self._n_trees = n_trees
        self._spill = spill
        self._verbose = verbose
        self._dataset = []
        self._trees = None

    @property
    def n_trees(self):
        return self._n_trees

    def insert(self, vector: np.ndarray):
        """Inserts a vector into the index.
        """
        if self._trees:
            for tree in self._trees:
                tree.insert(len(self._dataset), vector)
        self._dataset.append(vector)

    def index(self):
        dataset = np.array(self._dataset, dtype=np.float32)
        self._trees = [
            _GANNTree.from_dataset(
                dataset,
                spill=self._spill,
                verbose=self._verbose
            ) for _ in range(self._n_trees)
        ]

    def search(self, query: np.ndarray, top_k: int = 10) -> list[int]:
        if not self._trees:
            raise ValueError("Build the index first.")
        candidates = set()
        candidates.update(*[t.get_candidates(query) for t in self._trees])
        candidates = np.array(list(candidates))
        vectors = np.array([self._dataset[idx] for idx in candidates])
        best = np.linalg.norm(vectors - query, axis=1).argsort()
        return candidates[best[:top_k]]

    def save(self, path: str) -> None:
        """Save the index to a directory at `path`.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            str(path / "dataset.txt"),
            np.array(self._dataset),
            delimiter=","
        )
        with (path / "index.json").open("w") as f:
            json.dump({
                    "n_trees": self._n_trees,
                    "spill": self._spill,
                    "verbose": self._verbose,
                    "trees": [t.to_dict() for t in self._trees],
                }, f, cls=NumPyJSONEncoder)

    @classmethod
    def load(cls, path: str) -> GANN:
        """Load the index from a directory at `path`.
        """
        path = Path(path)
        with (path / "index.json").open("r") as f:
            meta = json.load(f)
        gann = cls(
            n_trees=meta["n_trees"],
            spill=meta["spill"],
            verbose=meta["verbose"]
        )
        gann._trees = [_GANNTree.from_dict(t) for t in meta["trees"]]
        dataset = np.loadtxt(
            str(path / "dataset.txt"),
            delimiter=",",
            dtype=np.float32
        )
        for vector in dataset:
            gann.insert(vector)
        return gann
