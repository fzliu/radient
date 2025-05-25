from typing import TYPE_CHECKING
import warnings

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


WARMUP_EPOCHS = 5


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
        max_iter: int = 100,
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
            raise ValueError(f"Invalid distance metric: {distance_metric}")

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
        groups: list[list[int]] | None = None
    ) -> torch.Tensor:
        """Takes a flat 2d dataset specified by `X` and adds a batch dimension,
        where each batch corresponds to a pre-existing subgroup of indexes into
        `X` (specified by `groups`).
        """
        if not groups:
            X_ = X.unsqueeze(0)
            M_ = torch.ones_like(X, dtype=torch.bool)

        else:
            sz = max([len(cl) for cl in groups])
            X_ = torch.zeros((len(groups), sz, X.shape[1]), dtype=X.dtype)
            M_ = torch.ones((len(groups), sz), dtype=torch.bool)
            for (n, idxs) in enumerate(groups):
                X_[n,:len(idxs),:] = X[idxs]
                M_[n,len(idxs):] = False

        return (X_, M_)

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
        C: torch.Tensor,
        M: torch.Tensor | None = None
    ):
        d = self.forward(X, C) ** 2
        c = X.shape[1]
        if M is not None:
            d *= M.unsqueeze(dim=2)
            c = M.sum(dim=1, keepdim=True)
        l_a = (-10.0*d).softmax(dim=2)
        l_s = (l_a.sum(dim=1) - c/self._n_clusters)**2
        l = ((l_a * d).sum(dim=1) + self._size_decay * l_s) / c
        return l.sum() / X.shape[0]

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

        X = torch.from_numpy(X)
        if groups is None:
            groups = [list(np.arange(X.shape[0]))]

        sh_C = (len(groups), self._n_clusters, X.shape[1])
        C = torch.empty(sh_C, dtype=torch.float32)

        group_indices_to_run = list(range(len(groups)))
        while len(group_indices_to_run) > 0:

            # Initialize cluster centers using k-means++
            for n in group_indices_to_run:
                X_n = X[groups[n],:]
                C_n = C[n,:,:]
                C_n[0,:] = X_n[np.random.choice(X_n.shape[0]),:]
                for m in range(1, self._n_clusters):
                    d, _ = self.forward(X_n, C_n[:m,:]).min(dim=1)
                    p = (d**2 / d.pow(2).sum()).numpy()
                    C_n[m,:] = X_n[np.random.choice(X_n.shape[0], p=p),:]

            # Create dataset, optimizer, and scheduler
            #ldr = DataLoader(TensorDataset(X), batch_size=self._batch_size, shuffle=True)
            groups_ = [groups[n] for n in group_indices_to_run]
            C_ = C[group_indices_to_run,:,:].requires_grad_()
            (X_, M_) = self._create_batched_dataset(X, groups=groups_)
            opt = optim.SGD([C_], lr=0.1/X_.shape[1], momentum=0.0)
            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=self._lr_lambda)

            # Training loop
            # TODO: batching for large vector datasets
            for epoch in range(self._max_iter):
                opt.zero_grad()
                loss = self.forward_loss(X_, C_, M_)
                loss.backward()
                opt.step()
                sch.step()
                if self._verbose and epoch % 5 == 0:
                    with torch.inference_mode():
                        #loss = self.forward_loss(X_, C_, M_)
                        print(f"Epoch {epoch}, loss: {loss.item():.5f}")
            C[group_indices_to_run,:,:] = C_.detach()

            # Determine whether the output clusters are imbalanced
            imbalanced_group_indices = []
            with torch.inference_mode():
                a = self.forward(X_, C_).argmin(dim=-1)
                for n in range(a.shape[0]):
                    a_n = a[n,:].masked_select(M_[n,:])
                    c_n = a_n.bincount()
                    #if (c_n.max() - c_n.min()) > c_n.sum().sqrt():
                    if (c_n.max() - c_n.min()) / c_n.sum() > 0.1:
                        imbalanced_group_indices.append(group_indices_to_run[n])
            if len(imbalanced_group_indices) == 0:
                break
            if self._verbose:
                print(
                    f"{len(imbalanced_group_indices)} / {len(groups)} "
                    "groups are imbalanced"
                )
            group_indices_to_run = imbalanced_group_indices

        self._C = C
        self.zero_grad()

    def predict(
        self,
        X: np.ndarray,
        groups: list[list[int]] | None = None
    ):
        X = torch.from_numpy(X)
        (X_, _) = self._create_batched_dataset(X, groups=groups)
        a = self.forward(X_, self._C).argmin(dim=-1)
        return a.numpy()

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        groups: list[list[int]] | None = None
    ):
        self.fit(X, y=y, sample_weight=sample_weight, groups=groups)
        return self.predict(X, groups=groups)
