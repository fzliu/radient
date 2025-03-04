from typing import Optional, TYPE_CHECKING

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

WARMUP_EPOCHS = 10


def calculate_distances(
    A: torch.Tensor,
    B: torch.Tensor,
    kind: str = "Lp-norm",
    p: float = 2.0
):
    if kind == "Lp-norm":
        return torch.cdist(A.unsqueeze(0), B.unsqueeze(0), p=p).squeeze(dim=0)
    else:
        raise NotImplementedError(f"Distance metric {kind} not implemented.")


class KMeans(nn.Module):
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 600,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
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

        # Set seed
        if random_state:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

    @property
    def cluster_centers_(self):
        return self.C.detach().numpy()

    def _init_centers(self, X: torch.Tensor):
        # K-means++ initialization
        C_init = torch.empty((self._n_clusters, X.shape[1]), dtype=torch.float32)
        C_init[0,:] = X[np.random.choice(X.shape[0])]
        for n in range(1, self._n_clusters):
            d, _ = calculate_distances(X, C_init[:n]).min(dim=1)
            p = (d**2 / d.pow(2).sum()).numpy()
            C_init[n,:] = X[np.random.choice(X.shape[0], p=p)]
        self.C = nn.Parameter(C_init)

    def _lr_lambda(self, epoch: int):
        if epoch < WARMUP_EPOCHS:
            # Exponential warm-up
            return np.e ** (epoch - WARMUP_EPOCHS)
        else:
            # Cosine decay
            decay_epochs = self._max_iter - WARMUP_EPOCHS
            return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / decay_epochs))

    def forward(self, X: torch.Tensor):
        return calculate_distances(X, self.C)

    def forward_loss(self, X: torch.Tensor):
        d = self.forward(X) ** 2
        a = torch.softmax(-1/1.0*d, dim=1)
        s = (a.sum(dim=0) - X.shape[0]/self._n_clusters) ** 2
        return ((a * d).sum() + self._size_decay * s.sum()) / X.shape[0]

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.array] = None,
        sample_weight: Optional[np.array] = None
    ):
        
        X = torch.tensor(X, dtype=torch.float32)
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
            for n, (batch,) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.forward_loss(batch)
                loss.backward()
                optimizer.step()
                #print(f"    Train loss: {loss.item():.3f}")
            loss = self.forward_loss(X)
            scheduler.step()
            if self._verbose:
                print(f"Epoch {epoch+1}, loss: {loss.item():.5f}")

    def predict(self, X: np.array):
        X = torch.tensor(X, dtype=torch.float32)
        return self.forward(X).argmin(dim=1).numpy()

    def fit_predict(
        self,
        X: np.ndarray,
        y: Optional[np.array] = None,
        sample_weight: Optional[np.array] = None
    ):
        self.fit(X, y=y, sample_weight=sample_weight)
        return self.predict(X)
