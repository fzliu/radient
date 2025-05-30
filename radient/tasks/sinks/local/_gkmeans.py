from typing import TYPE_CHECKING

import numpy as np

from radient.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim
else:
    torch = LazyImport("torch")
    nn = LazyImport("torch.nn")
    optim = LazyImport("torch.optim")


WARMUP_EPOCHS = 10


def torch_auto_device(
    device: str | torch.device | None = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    
    device_type = torch.get_default_device().type
    if "cuda" in device_type and torch.cuda.is_bf16_supported():
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float32)


def _torch_bincount(
    x: torch.Tensor,
    dim: int = -1
):
    dim = dim % x.dim()
    shape = list(x.shape)
    shape[dim] = x.max().item() + 1
    count = torch.zeros(shape, dtype=x.dtype)
    return count.scatter_add_(dim, x, src=torch.ones_like(x))


def _torch_masked_softmax(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1
):
    x_exp = x.exp()
    if mask is not None:
        x_exp *= mask
    return x_exp / x_exp.sum(dim=dim)


def _torch_euclidean_distance(
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    dists = ((A**2).sum(dim=-1, keepdim=True) +
             (B**2).sum(dim=-1, keepdim=True) -
             2.0 * torch.bmm(A, B.transpose(-2, -1)))
    dists.clamp_(min=0.0).sqrt_()
    return dists


def _torch_lp_norm_distance(
    A: torch.Tensor,
    B: torch.Tensor,
    p: float = 2
) -> torch.Tensor:
    return torch.cdist(A, B, p=p)


def _torch_cosine_distance(
    A: torch.Tensor,
    B: torch.Tensor
):
    A_norm = torch.nn.functional.normalize(A, p=2, dim=-1)
    B_norm = torch.nn.functional.normalize(B, p=2, dim=-1)
    return 1.0 - torch.bmm(A_norm, B_norm.transpose(-2, -1))


class GKMeans(nn.Module):
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 600,
        tol: float = 1e-3,
        random_state: int | None = None,
        distance_metric: str = "lp-norm",
        size_decay: float = 1.0,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol

        self._size_decay = size_decay
        self._verbose = verbose

        # Set distance metric
        if distance_metric == "euclidean":
            self.forward = _torch_euclidean_distance
        elif distance_metric == "cosine":
            self.forward = _torch_cosine_distance
        elif distance_metric == "lp-norm":
            self.forward = _torch_lp_norm_distance
        else:
            raise ValueError(f"invalid distance metric: {distance_metric}")

        # Set seed
        if random_state:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._C.numpy()

    def _create_batched_dataset(
        self,
        X: torch.Tensor,
        groups: np.ndarray | None = None
    ) -> torch.Tensor:
        """Takes a flat 2d dataset specified by `X` and adds a batch dimension,
        where each batch corresponds to a pre-existing subgroup of indexes into
        `X` (specified by `groups`).
        """
        if groups is None:
            return X.unsqueeze(0)
        else:
            X_out = torch.empty(groups.shape + X.shape[1:2], dtype=X.dtype)
            for (n, idxs) in enumerate(groups):
                X_out[n,:len(idxs),:] = X[idxs]
        return X_out

    def _lr_lambda(self, epoch: int):
        if epoch < WARMUP_EPOCHS:
            # Exponential warm-up
            return np.e ** (epoch - WARMUP_EPOCHS)
        else:
            # Cosine decay
            decay_epochs = self._max_iter - WARMUP_EPOCHS
            return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / decay_epochs))

    def forward_loss(
        self,
        X: torch.Tensor,
        C: torch.Tensor
    ):
        d = self.forward(X, C) ** 2
        c = X.shape[1]
        l_a = (-1.0*d).softmax(dim=2)
        l_s = (l_a.sum(dim=1) - c/self._n_clusters)**2
        l = ((l_a * d).sum(dim=1) + self._size_decay * l_s) / c
        return l.sum()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        groups: list[list[int]] | None = None
    ):
        """Generates cluster centers using the input data. If `groups` is
        `None`, then this function mimics the normal usage of clustering
        algorithms within `scikit-learn`.

        If `groups` is not `None`, then this function will take existing
        groups of points and create a new set of cluster centers (totaling
        `len(groups) * n_clusters`), treating each group as an independent
        dataset. Groups are expected to be a list of lists, where each inner
        list contains the indices of the points in the group.
        """

        if groups is None:
            groups = [list(np.arange(X.shape[0]))]

        # Create data and cluster center tensors
        X = torch.from_numpy(X).to(
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype()
        )
        C = torch.empty((len(groups), self._n_clusters, X.shape[1]))

        to_run = list(range(groups.shape[0]))
        while len(to_run) > 0:

            # Initialize cluster centers using k-means++
            for n in to_run:
                X_n = X[groups[n],:]
                C_n = C[n,:,:]
                C_n[0,:] = X_n[np.random.choice(X_n.shape[0]),:]
                for m in range(1, self._n_clusters):
                    d, _ = self.forward(X_n, C_n[:m,:]).min(dim=1)
                    p = d.to(torch.float32).cpu().numpy()**2
                    p /= p.sum()
                    C_n[m,:] = X_n[np.random.choice(X_n.shape[0], p=p),:]

            # Create dataset, optimizer, and scheduler
            C_ = C[to_run,:,:].requires_grad_()
            X_ = self._create_batched_dataset(X, groups=groups[to_run])
            optimizer = optim.Adam([C_], lr=1.0/X_.shape[1])

            # Training loop
            # TODO: batching for large vector datasets
            for epoch in range(self._max_iter):
                optimizer.zero_grad()
                loss = self.forward_loss(X_, C_)
                loss.backward()
                optimizer.step()
                if self._verbose and epoch % 25 == 0:
                    with torch.inference_mode():
                        #loss = self.forward_loss(X_, C_)
                        print(f"Epoch {epoch}, loss: {loss.item():.5f}")

            # Post-training cleanup
            C_ = C_.detach()
            C[to_run,:,:] = C_
            self.zero_grad()

            # Determine whether the output clusters are imbalanced
            a = self.forward(X_, C_).argmin(dim=2)
            c = _torch_bincount(a, dim=1)
            b = (c.max(dim=1)[0] - c.min(dim=1)[0]) / a.shape[1]
            to_run = [to_run[n] for n in range(b.numel()) if b[n] > 0.03]
            if self._verbose:
                print(f"Average imbalance: {b.mean():.5f}")
                if to_run:
                    print(
                        f"{len(to_run)} / {len(groups)} "
                        "groups are imbalanced, rerunning on these groups"
                    )

        self._C = C


    def predict(
        self,
        X: np.ndarray,
        groups: np.ndarray | None = None
    ):
        X = torch.from_numpy(X).to(
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype()
        )
        (X_, _) = self._create_batched_dataset(X, groups=groups)
        a = self.forward(X_, self._C).argmin(dim=2)
        return a.numpy()

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None
    ):
        self.fit(X, y=y, sample_weight=sample_weight, groups=groups)
        return self.predict(X, groups=groups)
