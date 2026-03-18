import torch
from torch import Tensor
from torch.utils.data import Dataset


def _index_exclude(x: torch.Tensor, dim: int, exclude_idx: torch.Tensor) -> torch.Tensor:
    n = x.size(dim)
    device = x.device

    keep_mask = torch.ones(n, dtype=torch.bool, device=device)
    keep_mask[exclude_idx.long()] = False

    keep_idx = torch.arange(n, device=device)[keep_mask]
    return x.index_select(dim, keep_idx)


class Task(Dataset):

    def __init__(self, Kxx: Tensor, Y: Tensor, Kxi: Tensor):
        self.Kxx = Kxx          # N x N
        self.Y = Y              # N x ...
        self.Kxi = Kxi          # N x I

    def __len__(self):
        return self.Kxx.shape[0]

    def _to_index_tensor(self, idx):
        n = len(self)
        device = self.Kxx.device

        if isinstance(idx, slice):
            return torch.arange(n, device=device)[idx]

        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx, device=device)
        else:
            idx = idx.to(device)

        if idx.ndim == 0:
            idx = idx.unsqueeze(0)

        if idx.dtype == torch.bool:
            if idx.ndim != 1 or idx.numel() != n:
                raise ValueError("Boolean idx must have shape (N,).")
            return torch.nonzero(idx, as_tuple=False).squeeze(1)

        idx = idx.long()
        if idx.ndim != 1:
            raise ValueError("idx must be a scalar, slice, 1D indices, or boolean mask.")
        if torch.any((idx < 0) | (idx >= n)):
            raise IndexError("idx contains out-of-range values.")
        return idx
    
    def __getitem__(self, idx):
        idx = self._to_index_tensor(idx)
        Kxx_subset = self.Kxx.index_select(0, idx).index_select(1, idx)
        return Kxx_subset, self.Y.index_select(0, idx), self.Kxi.index_select(0, idx)
    
    def train_test(self, idx_train):
        idx_train = self._to_index_tensor(idx_train)                    # M
        Kxx_train = self.Kxx.index_select(1, idx_train)             # N x M
        Kxx_train_train = Kxx_train.index_select(0, idx_train)      # M x M
        Kxx_test_train = _index_exclude(Kxx_train, 0, idx_train)    # (N-M) x M
        Y_train = self.Y.index_select(0, idx_train)                 # M x ...
        Y_test = _index_exclude(self.Y, 0, idx_train)               # (N-M) x ...
        Kxi_train = self.Kxi.index_select(0, idx_train)             # M x I
        Kxi_test = _index_exclude(self.Kxi, 0, idx_train)           # (N-M) x I
        return (Kxx_train_train, Y_train, Kxi_train), (Kxx_test_train, Y_test, Kxi_test)
    
    def sample_train_test(self, train_proportion: float = 0.8, generator: torch.Generator | None = None):
        n = len(self)
        if n < 2:
            raise ValueError("Need at least 2 samples to create a train/test split.")
        if not (0.0 < train_proportion < 1.0):
            raise ValueError("train_proportion must be in (0, 1).")

        n_train = int(round(train_proportion * n))
        n_train = max(1, min(n - 1, n_train))

        perm = torch.randperm(n, generator=generator, device=self.Kxx.device)
        idx_train = perm[:n_train]
        return self.train_test(idx_train)

