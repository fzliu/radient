from typing import TYPE_CHECKING

import numpy as np

from radient.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    torch = LazyImport("torch")
    nn = LazyImport("torch.nn")


WARMUP_EPOCHS = 10


def torch_auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"


def _min_samples_needed(
    dim: int,
    error: float = 0.01,
    uncertainty: float = 0.05
) -> int:
    """Blumer, Ehrenfeucht, Haussler, and Warmuth (1989)
    Learnability and the Vapnik-Chervonenkis Dimension
    """
    m0 = 4 / (error * np.log(2 / uncertainty))
    m1 = 8 * dim / (error * np.log(13 / error))
    return int(np.ceil(max(m0, m1)))


def _torch_hardmax(
    x: torch.Tensor,
    dim: int = -1
):
    x_soft = x.softmax(dim=dim)
    x_soft_detached = x_soft.detach()
    x_hard_detached = torch.nn.functional.one_hot(
        torch.argmax(x_soft_detached, dim=dim),
        num_classes=x.shape[dim]
    ).to(x_soft_detached.dtype)
    return x_hard_detached - x_soft_detached + x_soft


def _torch_bincount(
    x: torch.Tensor,
    dim: int = -1
):
    dim = dim % x.dim()
    shape = list(x.shape)
    shape[dim] = x.max().item() + 1
    count = x.new_zeros(shape)
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
             (B**2).sum(dim=-1, keepdim=True).transpose(-2, -1) -
             2.0 * torch.matmul(A, B.transpose(-2, -1)))
    return dists.clamp_(min=1e-9).sqrt_()


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
        n_clusters: int = 2,
        max_iter: int = 600,
        tol: float = 1e-3,
        random_state: int | None = None,
        distance_metric: str = "euclidean",
        size_decay: float = 1.0,
        device: str | torch.device | None = "cpu",
        dtype: torch.dtype | None = torch.float32,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol

        self._size_decay = size_decay
        self._device = device
        self._dtype = dtype
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
        X: np.ndarray,
        groups: np.ndarray | None = None
    ) -> torch.Tensor:
        """Takes a flat 2d dataset specified by `X` and adds a batch dimension,
        where each batch corresponds to a pre-existing subgroup of indexes into
        `X` (specified by `groups`).
        """
        if groups is None:
            groups = np.arange(X.shape[0])[np.newaxis,:]

        X_out = torch.empty(
            groups.shape + X.shape[1:2],
            device=self._device,
            dtype=self._dtype
        )
        for (n, idxs) in enumerate(groups):
            X_out[n,:len(idxs),:] = torch.from_numpy(X[idxs])
        return X_out

    def forward_loss(
        self,
        X: torch.Tensor,
        C: torch.Tensor
    ):
        d = self.forward(X, C)**2
        #l_a = (-1.0*d).softmax(dim=2)
        l_a = _torch_hardmax(-1.0*d, dim=2)
        l_s = (l_a.sum(dim=1) - X.shape[1]/self._n_clusters)**2
        l = ((l_a * d).sum(dim=1) + self._size_decay * l_s) / X.shape[1]
        return l.sum()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None
    ):
        """Generates cluster centers using the input data. If `groups` is
        `None`, then this function mimics the normal usage of clustering
        algorithms within `scikit-learn`.

        If `groups` is not `None`, then this function will take existing
        groups of points and create a new set of cluster centers (totaling
        `len(groups) * n_clusters`), treating each group as an independent
        dataset. Groups are expected to be an integer ndarray with two
        dimensions, where each inner array contains the indices of the points
        in the group.
        """


        if groups is None:
            groups = np.arange(X.shape[0])[np.newaxis,:]
        b = _min_samples_needed(X.shape[1])
        if groups.shape[1] > b:
            groups_ = np.empty((groups.shape[0], b), dtype=groups.dtype)
        else:
            groups_ = groups

        to_run = list(range(groups_.shape[0]))

        # Create empty cluster centers
        C = torch.empty((
                groups.shape[0],   # num_groups
                self._n_clusters,  # num_clusters
                X.shape[1]),       # vector_dim
            dtype=torch.float32,
            device="cpu"
        )

        while len(to_run) > 0:

            # Sample each group according to the theoretical minimum number
            # of samples needed
            if groups.shape[1] > b:
                for n in to_run:
                    groups_[n,:] = np.random.choice(groups[n], size=b, replace=False)

            # Initialize cluster centers using k-means++
            for n in to_run:
                C_n = C[n,:,:]
                X_n = torch.from_numpy(X[groups_[n],:])
                C_n[0,:] = X_n[np.random.choice(X_n.shape[0]),:]
                for m in range(1, self._n_clusters):
                    d = self.forward(X_n, C_n[:m,:]).min(dim=1)[0]
                    i = torch.multinomial(d**2, 1)
                    C_n[m,:] = X_n[i,:]

            # Create dataset, optimizer, and scheduler
            C_ = C[to_run,:,:].to(device=self._device, dtype=self._dtype)
            X_ = self._create_batched_dataset(X, groups=groups_[to_run])
            C_.requires_grad_()
            optimizer = torch.optim.Adam([C_], lr=0.0001, betas=(0.9, 0.9999))

            # Training loop
            # TODO: batching for large vector datasets
            #prev_loss = np.inf
            for epoch in range(self._max_iter):
                loss = self.forward_loss(X_, C_)
                if epoch % 50 == 0:
                    if self._verbose:
                        print(f"Epoch {epoch}, loss: {loss.item():.5f}")
                    #if prev_loss > loss.item() and prev_loss - loss.item() < self._tol:
                    #    break
                    #prev_loss = loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Post-training cleanup
            C_ = C_.detach()
            C[to_run,:,:] = C_.to(device=C.device, dtype=C.dtype)
            self.zero_grad()

            # Determine whether the output clusters are imbalanced (0.01)
            a = self.forward(X_, C_).argmin(dim=2)
            c = _torch_bincount(a, dim=1)
            d = (c.max(dim=1)[0] - c.min(dim=1)[0]) / a.shape[1]
            to_run = [to_run[n] for n in range(d.numel()) if d[n] > 0.01]
            if self._verbose:
                print(f"Average imbalance: {d.mean():.5f}")
                if to_run:
                    print(
                        f"{len(to_run)} / {len(groups_)} "
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
