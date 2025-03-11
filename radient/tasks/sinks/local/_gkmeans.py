from typing import TYPE_CHECKING

import numpy as np

from radient.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
else:
    torch = LazyImport("torch")
    nn = LazyImport("torch.nn")
    optim = LazyImport("torch.optim")
    DataLoader = LazyImport("torch.utils.data", attribute="DataLoader")
    TensorDataset = LazyImport("torch.utils.data", attribute="TensorDataset")


WARMUP_EPOCHS = 5


def _euclidean_distance(
    A: torch.Tensor,
    B: torch.Tensor
):
    A_norm = (A**2).sum(dim=1, keepdim=True)
    B_norm = (B**2).sum(dim=1, keepdim=True).T
    dist = A_norm + B_norm - 2.0 * A @ B.T
    return dist.clamp(min=0.0).sqrt()


def _lp_norm_distance(
    A: torch.Tensor,
    B: torch.Tensor,
    p: float = 2
):
    return torch.cdist(A.unsqueeze(0), B.unsqueeze(0), p=p).squeeze(0)


def _cosine_distance(
    A: torch.Tensor,
    B: torch.Tensor
):
    A_norm = (A**2).sum(dim=1, keepdim=True).sqrt()
    B_norm = (B**2).sum(dim=1, keepdim=True).sqrt().T
    return 1.0 - (A @ B.T) / (A_norm * B_norm)


class GKMeans(nn.Module):
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 50,
        tol: float = 1e-3,
        random_state: int | None = None,
        distance_metric: str = "lp-norm",
        learning_rate: float = 1e-4,
        batch_size: int = 4096,
        size_decay: float = 4.0,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol

        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._size_decay = size_decay
        self._verbose = verbose

        # Set distance metric
        if distance_metric == "euclidean":
            self._calculate_distances = _euclidean_distance
        elif distance_metric == "cosine":
            self._calculate_distances = _cosine_distance
        elif distance_metric == "lp-norm":
            self._calculate_distances = _lp_norm_distance
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

        # Set seed
        if random_state:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

    @property
    def cluster_centers_(self):
        return self._C.detach().numpy()

    def _init_centers(self, X: torch.Tensor):
        # K-means++ initialization
        shape = (self._n_clusters, X.shape[1])
        C_init = torch.empty(shape, dtype=torch.float32)
        C_init[0,:] = X[np.random.choice(X.shape[0])]
        for n in range(1, self._n_clusters):
            d, _ = self._calculate_distances(X, C_init[:n,:]).min(dim=1)
            p = (d**2 / d.pow(2).sum()).numpy()
            C_init[n,:] = X[np.random.choice(X.shape[0], p=p)]
        self._C = nn.Parameter(C_init)

    def _lr_lambda(self, epoch: int):
        if epoch < WARMUP_EPOCHS:
            # Exponential warm-up
            return np.e ** (epoch - WARMUP_EPOCHS)
        else:
            # Cosine decay
            decay_epochs = self._max_iter - WARMUP_EPOCHS
            return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / decay_epochs))

    def forward(self, X: torch.Tensor):
        return self._calculate_distances(X, self._C)

    def forward_loss(self, X: torch.Tensor):
        d = self.forward(X) ** 2
        a = torch.softmax(-1/1.0*d, dim=1)
        s = (a.sum(dim=0) - X.shape[0]/self._n_clusters) ** 2
        return ((a * d).sum() + self._size_decay * s.sum()) / X.shape[0]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None
    ):
        X = torch.from_numpy(X)
        self._init_centers(X)

        # Create dataloader, optimizer, and scheduler
        dataloader = DataLoader(
            TensorDataset(X),
            batch_size=self._batch_size,
            shuffle=True
        )

        optimizer = optim.SGD(self.parameters(), lr=self._learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_lambda)

        # Training loop
        for epoch in range(self._max_iter):
            for batch, in dataloader:
                optimizer.zero_grad()
                loss = self.forward_loss(batch)
                loss.backward()
                optimizer.step()
                #print(f"    Train loss: {loss.item():.3f}")
            scheduler.step()
            if self._verbose:
                with torch.no_grad():
                    loss = self.forward_loss(X)
                    print(f"Epoch {epoch+1}, loss: {loss.item():.5f}")

    def predict(self, X: np.ndarray):
        X = torch.from_numpy(X)
        a = self.forward(X).argmin(dim=1)
        return a.numpy()

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None
    ):
        self.fit(X, y=y, sample_weight=sample_weight)
        return self.predict(X)
