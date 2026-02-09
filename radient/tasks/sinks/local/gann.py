from __future__ import annotations

from collections import defaultdict
from functools import partial
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path

from numba import njit, prange
import numpy as np

from radient.tasks.sinks.local._gkmeans import GKMeans
from radient.tasks.sinks.local._gkmeans import torch_auto_device
from radient.utils.json_tools import ExtendedJSONEncoder


MAX_LEAF_SIZE = 200


def _empty_int64_array() -> np.ndarray:
    return np.array([], dtype=np.int64)


@njit(cache=True, fastmath=True)
def _unique(array: np.ndarray) -> np.ndarray:
    s = set()
    for v in array:
        s.add(v)

    out = np.empty(len(s), array.dtype)
    for n, v in enumerate(s):
        out[n] = v
    return out


@njit(cache=True, fastmath=True)
def _hyperplane_distance(
    vector: np.ndarray,
    weight: np.ndarray,
    bias: np.float32
) -> np.float32:
    """Computes the signed distance of a vector to a hyperplane defined by
    `weight` and `bias`.
    """
    return (vector.dot(weight) + bias)# / np.linalg.norm(weight)


@njit(parallel=True, cache=True, fastmath=True)
def _compute_distances(
    dataset: np.ndarray,
    candidates: np.ndarray,
    query: np.ndarray
) -> np.ndarray:
    """Computes squared L2 distances between query and candidate vectors. For
    ranking purposes, squared distances work the same as regular distances.
    """
    dists = np.empty(len(candidates), dtype=np.float32)
    for n in prange(len(candidates)):
        dists[n] = np.sum((dataset[candidates[n]] - query) ** 2)
    return dists


@njit(cache=True, fastmath=True)
def _query_tree(
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
            nid = children[nid,0]
        else:
            nid = children[nid,1]
    return nid


def _load_tree(
    n: int,
    path: Path | str,
    shm_name: str,
    shape: tuple[int, int],
    dtype: np.dtype
) -> _GANNTree:

    # Load tree structure and dataset from shared memory
    tree = _GANNTree.load(path / f"tree_{n}")
    raw_data = mp.shared_memory.SharedMemory(name=shm_name)
    dataset = np.ndarray(shape, dtype=dtype, buffer=raw_data.buf)

    # Insert dataset into tree
    entities = {np.int64(n): dataset[n,:] for n in range(dataset.shape[0])}
    tree.insert(entities)
    raw_data.close()

    return tree

class _GANNTree:

    def __init__(self):
        # Array-based representation for efficient tree traversal
        self._weights = None
        self._biases = None
        self._cutoffs = None   # 2D array of (left, right) cutoffs
        self._children = None  # 2D array of (left, right) children
        # Use a pickle-safe factory; multiprocessing.pool cannot pickle lambdas
        self._leaves = defaultdict(_empty_int64_array)

    @property
    def is_indexed(self):
        return self._weights is not None

    def insert(self, entities: dict[np.int64, np.ndarray]):
        """Single-tree insert function.
        """

        # Output of this operation is a data "buffer" that maps node IDs to a
        # list of keys which should be subsequently added to the tree
        buffer = defaultdict(list)
        for key, vector in entities.items():
            queue = [0]
            while queue:
                nid = queue.pop()
                if nid >= self._children.shape[0]:
                    buffer[nid].append(key)
                else:
                    val = _hyperplane_distance(
                        vector,
                        self._weights[nid,:],
                        self._biases[nid]
                    )
                    if val <= self._cutoffs[nid,0]:
                        queue.append(self._children[nid,0])
                    if val >= self._cutoffs[nid,1]:
                        queue.append(self._children[nid,1])

        # Append the new keys to the leaves
        for nid, vals in buffer.items():
            self._leaves[nid] = np.append(self._leaves[nid], vals)

    def get_candidates(self, query: np.ndarray) -> np.ndarray:
        """Returns nearest neighbor candidates for a query vector.
        """
        if self._weights is not None:
            # Use numba-optimized traversal
            nid = _query_tree(
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
        tree._children = np.array(children, dtype=np.int64)
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
        self._dataset = np.empty((0, dim), dtype=np.float32)
        self._trees = []

    @property
    def n_trees(self):
        return self._n_trees

    def insert(self, dataset: np.ndarray):
        """Inserts a set of vectors into the index (all trees).
        """
        dataset = dataset.astype(np.float32, copy=False)
        if self._trees:
            entities = {}
            for n in range(dataset.shape[0]):
                key = np.int64(self._dataset.shape[0] + n)
                entities[key] = dataset[n,:]
            for tree in self._trees:
                tree.insert(entities)
        self._dataset = np.append(self._dataset, dataset, axis=0)

    def index(self):
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
        candidates = _unique(np.concatenate(all_candidates))

        # Determine the top k closest candidates
        dists = _compute_distances(self._dataset, candidates, query)
        best = np.argpartition(dists, top_k)[:top_k]
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
        np.save(path / "dataset.npy", np.array(self._dataset, dtype=np.float32))

    @classmethod
    def load(cls, path: str) -> GANN:
        """Load the index from a directory at `path`.
        """
        path = Path(path)

        # Load metadata and create GANN instance
        with (path / "meta.json").open("r") as f:
            meta = json.load(f)
        gann = cls(**meta)

        # Load dataset vectors
        dataset = np.load(path / "dataset.npy").astype(np.float32, copy=False)

        # Load trees and insert vectors
        #for n in range(gann.n_trees):
        #    gann._trees.append(_GANNTree.load(path / f"tree_{n}"))
        #gann.insert(dataset)

        shm = mp.shared_memory.SharedMemory(create=True, size=dataset.nbytes)
        shm_array = np.ndarray(dataset.shape, dtype=dataset.dtype, buffer=shm.buf)
        shm_array[:] = dataset[:]
        
        try:
            worker = partial(
                _load_tree,
                path=path,
                shm_name=shm.name,
                shape=dataset.shape,
                dtype=dataset.dtype
            )
            mp_ctx = mp.get_context("spawn")
            with mp_ctx.Pool(12) as pool:
                gann._trees = pool.map(worker, range(gann.n_trees))
        finally:
            shm.close()
            shm.unlink()
        
        gann._dataset = dataset

        return gann
