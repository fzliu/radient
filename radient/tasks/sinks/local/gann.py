from __future__ import annotations

import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np

from radient.tasks.sinks.local._gkmeans import GKMeans
from radient.tasks.sinks.local._gkmeans import torch_auto_device
from radient.utils.json_tools import ExtendedJSONEncoder


MAX_LEAF_SIZE = 200

_GANN_C_DIR = Path(__file__).parent / "_gann_c_src"
_GANN_LIB_PATH = _GANN_C_DIR / "libgann.so"


def _hyperplane_distance(
    vector: np.ndarray,
    weight: np.ndarray,
    bias: np.float32
) -> np.float32:
    """Computes the signed distance of a vector to a hyperplane defined by
    `weight` and `bias`.
    """
    return (vector.dot(weight) + bias)


class _GANNResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int64),
        ("distance", ctypes.c_float),
    ]


class _GANNTree:

    _lib = None

    def __init__(self):
        # C library resources for loading and searching the tree
        self._c_index = None
        self._c_ctx = None

    @classmethod
    def _get_lib(cls):
        if cls._lib is None:
            if not _GANN_LIB_PATH.exists():
                subprocess.check_call(["make", "-C", str(_GANN_C_DIR)])
            lib = ctypes.CDLL(str(_GANN_LIB_PATH), mode=ctypes.RTLD_LOCAL)
            lib.gann_load.argtypes = [ctypes.c_char_p]
            lib.gann_load.restype = ctypes.c_void_p
            lib.gann_search_ctx_create.argtypes = [ctypes.c_void_p]
            lib.gann_search_ctx_create.restype = ctypes.c_void_p
            lib.gann_search.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(_GANNResult),
            ]
            lib.gann_search.restype = ctypes.c_int
            lib.gann_search_ctx_free.argtypes = [ctypes.c_void_p]
            lib.gann_search_ctx_free.restype = None
            lib.gann_free.argtypes = [ctypes.c_void_p]
            lib.gann_free.restype = None
            cls._lib = lib
        return cls._lib

    @property
    def is_indexed(self):
        return self._c_index is not None

    def search(self, query: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Search for nearest neighbors using the C library."""
        lib = self._get_lib()
        query = np.ascontiguousarray(query, dtype=np.float32)
        results = (_GANNResult * top_k)()
        n = lib.gann_search(
            self._c_index, self._c_ctx,
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            top_k, results,
        )
        return np.array([results[i].id for i in range(n)], dtype=np.int64)

    def close(self):
        """Free C library resources."""
        if self._lib is not None:
            if self._c_ctx is not None:
                self._lib.gann_search_ctx_free(self._c_ctx)
                self._c_ctx = None
            if self._c_index is not None:
                self._lib.gann_free(self._c_index)
                self._c_index = None

    def __del__(self):
        self.close()

    @classmethod
    def from_path(cls, path: str | Path) -> _GANNTree:
        """Load an index from a directory using the C library."""
        tree = cls()
        lib = cls._get_lib()
        tree._c_index = lib.gann_load(str(path).encode("utf-8"))
        if not tree._c_index:
            raise RuntimeError(f"Failed to load GANN index from {path}")
        tree._c_ctx = lib.gann_search_ctx_create(tree._c_index)
        return tree

    @classmethod
    def from_dataset(
        cls,
        dataset: np.ndarray,
        out_path: str | Path,
        spill: float = 0.0,
        verbose: bool = False,
    ) -> _GANNTree:

        # Initialize GKMeans
        device = torch_auto_device()
        gkmeans = GKMeans(
            n_clusters=2,
            device=device,
            verbose=verbose
        )

        # Initialize tree structure
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

        # Save the tree structure to disk for loading with the C library
        path = Path(out_path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "weights.npy", np.array(weights, dtype=np.float32))
        np.save(path / "biases.npy", np.array(biases, dtype=np.float32))
        np.save(path / "cutoffs.npy", np.array(cutoffs, dtype=np.float32))
        np.save(path / "children.npy", np.array(children, dtype=np.int64))
        with open(path / "leaves.json", "w") as f:
            json.dump(
                {(n + len(children)): leaf for n, leaf in enumerate(leaves)},
                f, cls=ExtendedJSONEncoder
            )

        return _GANNTree.from_path(out_path)


class GANN:

    def __init__(
        self,
        dim: int,
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

    def insert(self, vectors: np.ndarray):
        """Inserts multiple vectors into the index.
        """
        vectors = vectors.astype(np.float32, copy=False)
        self._dataset = np.append(self._dataset, vectors, axis=0)

    def index(self, out_path: str | Path):
        path = Path(out_path)
        self._trees = [
            _GANNTree.from_dataset(
                self._dataset,
                spill=self._spill,
                verbose=self._verbose,
                out_path=path
            ) for _ in range(self._n_trees)
        ]
        np.save(path / "dataset.npy", self._dataset)

    def search(self, query: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Search for nearest neighbors using the C library.
        """
        if not self._trees:
            raise ValueError("Load an index first.")
        return self._trees[0].search(query, top_k)

    @classmethod
    def load(cls, path: str) -> GANN:
        """Load the index from a directory at `path` using the C library.
        """
        path = Path(path)
        dataset = np.load(path / "dataset.npy").astype(np.float32, copy=False)
        gann = cls(dim=dataset.shape[1])
        gann._dataset = dataset
        gann._trees = [_GANNTree.from_path(path)]
        return gann

    def close(self):
        """Free C library resources."""
        for tree in self._trees:
            tree.close()
        self._trees = []

    def __del__(self):
        self.close()
