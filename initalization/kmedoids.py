import torch
from torch import Tensor


def k_medoids(
    pairwise_matrix: Tensor,
    k: int,
    max_iter: int = 100,
    balance_mode: str = "none",
    balance_weight: float = 0.5,
) -> Tensor:
    """
    K-medoids clustering from a pairwise matrix where smaller values are better.

    This implementation supports:
    - "none": standard k-medoids assignment
    - "soft": size-penalized assignment that encourages similar cluster sizes
    - "hard": capacity-constrained assignment with near-equal cluster sizes

    If you have a similarity matrix (higher is better), pass its negative.
    """
    if pairwise_matrix.ndim != 2 or pairwise_matrix.size(0) != pairwise_matrix.size(1):
        raise ValueError("pairwise_matrix must be square (N x N)")
    n = pairwise_matrix.size(0)
    if k <= 0 or k > n:
        raise ValueError("k must satisfy 1 <= k <= N")
    if balance_mode not in {"none", "soft", "hard"}:
        raise ValueError("balance_mode must be one of: 'none', 'soft', 'hard'")
    if balance_weight < 0:
        raise ValueError("balance_weight must be nonnegative")

    # Step 1: select initial medoids via normalized column-sum priority scores.
    row_sums = pairwise_matrix.sum(dim=1).clamp_min(1e-12)
    priority_scores = -(pairwise_matrix / row_sums.unsqueeze(1)).sum(dim=0)
    _, indices = priority_scores.topk(k)

    base, remainder = divmod(n, k)
    capacities = torch.full((k,), base, device=pairwise_matrix.device, dtype=torch.long)
    capacities[:remainder] += 1

    def _point_order(dist_to_medoids: Tensor) -> Tensor:
        if k < 2:
            return torch.arange(n, device=pairwise_matrix.device)
        best2 = torch.topk(dist_to_medoids, k=2, largest=False, dim=1).values
        margin = best2[:, 1] - best2[:, 0]
        return torch.argsort(margin, descending=True)

    def _assign_none(current_indices: Tensor) -> tuple[Tensor, Tensor]:
        dist_to_medoids = pairwise_matrix[:, current_indices]
        assignment = dist_to_medoids.argmin(dim=1)
        row_ids = torch.arange(n, device=pairwise_matrix.device)
        total_cost = dist_to_medoids[row_ids, assignment].sum()
        return assignment, total_cost

    def _assign_soft(current_indices: Tensor) -> tuple[Tensor, Tensor]:
        dist_to_medoids = pairwise_matrix[:, current_indices]
        order = _point_order(dist_to_medoids)
        counts = torch.zeros(k, device=pairwise_matrix.device, dtype=torch.long)
        assignment = torch.empty(n, device=pairwise_matrix.device, dtype=torch.long)
        for point_idx in order.tolist():
            costs = dist_to_medoids[point_idx] + balance_weight * counts.float()
            cluster = int(costs.argmin().item())
            assignment[point_idx] = cluster
            counts[cluster] += 1
        row_ids = torch.arange(n, device=pairwise_matrix.device)
        total_cost = dist_to_medoids[row_ids, assignment].sum()
        return assignment, total_cost

    def _assign_hard(current_indices: Tensor) -> tuple[Tensor, Tensor]:
        dist_to_medoids = pairwise_matrix[:, current_indices]
        order = _point_order(dist_to_medoids)
        counts = torch.zeros(k, device=pairwise_matrix.device, dtype=torch.long)
        assignment = torch.empty(n, device=pairwise_matrix.device, dtype=torch.long)
        for point_idx in order.tolist():
            prefs = torch.argsort(dist_to_medoids[point_idx])
            for cluster in prefs.tolist():
                if counts[cluster] < capacities[cluster]:
                    assignment[point_idx] = cluster
                    counts[cluster] += 1
                    break
        row_ids = torch.arange(n, device=pairwise_matrix.device)
        total_cost = dist_to_medoids[row_ids, assignment].sum()
        return assignment, total_cost

    def _assign(current_indices: Tensor) -> tuple[Tensor, Tensor]:
        if balance_mode == "none":
            return _assign_none(current_indices)
        if balance_mode == "soft":
            return _assign_soft(current_indices)
        return _assign_hard(current_indices)

    def _update(current_indices: Tensor, assignment: Tensor) -> None:
        for i in range(k):
            sub = (assignment == i).nonzero(as_tuple=True)[0]
            if sub.numel() == 0:
                continue
            sub_dist = pairwise_matrix[sub][:, sub]
            current_indices[i] = sub[sub_dist.sum(dim=1).argmin()]

    cluster_assignment, min_cost = _assign(indices)
    for _ in range(max_iter):
        _update(indices, cluster_assignment)
        cluster_assignment, total_cost = _assign(indices)
        if total_cost >= min_cost:
            break
        min_cost = total_cost

    return indices
