import numpy as np

from radient.tasks.sinks.local._gkmeans import KMeans


class GANN():

    def __init__(
        self,
        centroid: np.ndarray | None = None,
        indexes: np.ndarray | None = None,
        dataset: np.ndarray | None = None,
        sealed: bool = False,
        **kwargs
    ):
        super().__init__()
        self._centroid = centroid
        self._indexes = indexes if indexes is not None else []
        self._dataset = dataset if dataset is not None else []
        self._sealed = sealed
        self._children = None
    
    @property
    def centroid(self):
        return self._centroid

    @property
    def children(self):
        return self._children

    @property
    def sealed(self):
        return self._sealed

    @property
    def vectors(self):
        if self._sealed:
            return self._dataset[self._indexes]
        raise ValueError("Index must be sealed.")

    def insert(self, vector: np.ndarray):
        """Inserts a vector into an unsealed index.
        """
        if self._sealed:
            raise ValueError("Cannot insert into a sealed index.")
        self._dataset.append(vector)
        self._indexes.append(len(self._dataset) - 1)

    def seal(self):
        """Seals the index.
        """
        if self._sealed:
            raise ValueError("Index is already sealed.")
        self._sealed = True
        self._dataset = np.array(self._dataset)
        self._indexes = np.array(self._indexes)

    def build(self):
        """Builds the index.
        """
        if not self._sealed:
            raise ValueError("Index must be sealed.")
        if self._indexes.size < 500:
            return

        vectors = self.vectors
        self._sealed = True
        print("Building index with", len(vectors), "vectors.")

        # Get indexes for each cluster
        kmeans = KMeans(n_clusters=2)
        a = kmeans.fit_predict(vectors)
        idxs_C = [np.where(a==n)[0] for n in range(2)]

        # Determine the hyperplane that separates the two clusters
        # (i.e. the two centroids)
        C = list(kmeans.cluster_centers_)
        w = C[1] - C[0]
        b = -(C[1] + C[0]).dot(w) / 2.0

        # Compute each point's distance to the hyperplane
        d = (vectors.dot(w) + b) / np.linalg.norm(w)

        # For each of the two clusters, add 1/3 of the points that are in the
        # other cluster but are very close to the separating hyperplane
        n_add = int(len(vectors) / 6.0)
        idxs_C_add = [
            np.where(d > 0, d, np.inf).argsort()[:n_add],
            np.where(d <= 0, d, -np.inf).argsort()[-n_add:]
        ]
        idxs_C = [np.concatenate([idxs_C[n], idxs_C_add[n]]) for n in range(2)]

        # Create child nodes
        del vectors
        self._children = [
            GANN(
                centroid=C[n],
                indexes=self._indexes[idxs_C[n]],
                dataset=self._dataset,
                sealed=True
            ) for n in range(2)]
        for child in self._children:
            child.build()

    def search(self, vector: np.ndarray, top_k: int = 10):
        """Searches for the nearest vector in the index.
        """
        if not self._sealed:
            raise ValueError("Build the index before searching.")
        
        if self._children is None:
            vectors = self.vectors
            d = np.linalg.norm(vectors - vector, axis=1)
            return self._indexes[np.argsort(d)[:top_k]]
    
        # Determine which child node to search
        d = [np.linalg.norm(node.centroid - vector) for node in self._children]
        return self._children[np.argmin(d)].search(vector, top_k=top_k)
