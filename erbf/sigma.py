"""
Adaptive bandwidth (σ) computation for ERBF.

This module provides functions for computing per-point bandwidths using k-NN
distances, as well as global (constant) bandwidth estimation.

Uses PyTorch for GPU-accelerated computation when CUDA is available.

The key insight is that **larger k** values produce larger σ values which in
turn yield better-conditioned kernel matrices, enabling stable exact
interpolation.

Automatic k Selection Heuristic
--------------------------------
    k ≈ 1.5 √N

where N is the number of training points.  This balances locality (small k
captures local density) against stability (large k smooths the kernel).
"""

from __future__ import annotations

import math

import torch

from erbf.kernel import _default_device, _to_tensor

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no-op tqdm if not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable


def auto_select_k(N: int, *, multiplier: float = 1.5, minimum: int = 10) -> int:
    """Select the number of neighbours for adaptive σ computation.

    The heuristic ``k = max(minimum, int(multiplier * √N))`` was found
    empirically to produce well-conditioned kernel matrices.

    Parameters
    ----------
    N : int
        Number of training samples.
    multiplier : float, default=1.5
        Scaling factor applied to √N.
    minimum : int, default=10
        Hard lower bound to prevent degenerate σ estimates when N is small.

    Returns
    -------
    k : int
    """
    return max(minimum, int(math.sqrt(N) * multiplier))


def compute_local_sigmas(
    X: torch.Tensor,
    k_neighbors: int | None = None,
    *,
    k_multiplier: float = 1.5,
    k_minimum: int = 10,
    min_sigma: float | None = None,
    max_sigma: float | None = None,
    metric: str = "euclidean",
    chunk_size: int = 500,
    verbose: bool = False,
    show_progress: bool = False,
) -> torch.Tensor:
    """Compute per-point adaptive σ using k-NN mean distances.

    For each point *x_i* the bandwidth is the mean Euclidean distance to its
    *k* nearest neighbours, optionally clipped to ``[min_sigma, max_sigma]``.

    Uses chunking to keep memory usage constant regardless of N.

    Parameters
    ----------
    X : Tensor of shape (N, d)
        Data matrix.
    k_neighbors : int or None
        Number of neighbours.  If ``None``, automatically selected via
        :func:`auto_select_k`.
    k_multiplier : float, default=1.5
        Passed to :func:`auto_select_k` when *k_neighbors* is ``None``.
    k_minimum : int, default=10
        Passed to :func:`auto_select_k` when *k_neighbors* is ``None``.
    min_sigma : float or None
        Floor for computed σ values.
    max_sigma : float or None
        Ceiling for computed σ values.
    metric : str, default="euclidean"
        Distance metric (currently only "euclidean" is supported with torch).
    chunk_size : int, default=500
        Number of rows to process at a time. Controls memory usage.
    verbose : bool, default=False
        Print diagnostic information.
    show_progress : bool, default=False
        Show progress bar during computation.

    Returns
    -------
    sigmas : Tensor of shape (N,)
        Per-point bandwidth values.
    """
    device = X.device
    N = X.shape[0]

    if k_neighbors is None:
        k_neighbors = auto_select_k(N, multiplier=k_multiplier, minimum=k_minimum)

    if verbose:
        print(
            f"[sigma] Computing adaptive σ: N={N}, k={k_neighbors}, "
            f"metric={metric}, chunk_size={chunk_size}"
        )

    sigmas = torch.zeros(N, dtype=torch.float64, device=device)

    # Precompute ||x_j||^2 for all training points
    YY = (X * X).sum(dim=1)  # (N,)

    # Process in chunks to keep memory constant
    n_chunks = (N + chunk_size - 1) // chunk_size
    chunk_iter = range(n_chunks)
    if show_progress:
        chunk_iter = tqdm(chunk_iter, desc="Computing σ", unit="chunk")

    for chunk_idx in chunk_iter:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)
        X_chunk = X[start:end]

        # Matmul-based squared distances, then sqrt
        XX = (X_chunk * X_chunk).sum(dim=1, keepdim=True)  # (chunk, 1)
        D_chunk = torch.sqrt(torch.clamp(XX + YY.unsqueeze(0) - 2.0 * (X_chunk @ X.T), min=0.0))

        for i, global_i in enumerate(range(start, end)):
            row = D_chunk[i].clone()
            row[global_i] = float("inf")
            # Get k smallest distances
            topk_vals, _ = torch.topk(row, k_neighbors, largest=False)
            sigmas[global_i] = topk_vals.mean()

    if min_sigma is not None:
        sigmas = torch.clamp(sigmas, min=min_sigma)
    if max_sigma is not None:
        sigmas = torch.clamp(sigmas, max=max_sigma)

    if verbose:
        print(
            f"[sigma] σ range: [{sigmas.min().item():.4f}, {sigmas.max().item():.4f}], "
            f"mean={sigmas.mean().item():.4f}"
        )

    if torch.isinf(sigmas).any():
        raise ValueError(
            "Some σ values are infinite. This may indicate that "
            "k_neighbors is too large for the dataset size."
        )
    return sigmas


def compute_global_sigma(
    X: torch.Tensor,
    *,
    method: str = "median",
    metric: str = "euclidean",
    chunk_size: int = 500,
    show_progress: bool = False,
) -> float:
    """Compute a single global σ for all points.

    Uses chunking to keep memory usage constant regardless of N.

    Parameters
    ----------
    X : Tensor of shape (N, d)
    method : {"median", "mean", "max"}
        Aggregation applied to pairwise distances.
    metric : str, default="euclidean"
    chunk_size : int, default=500
        Number of rows to process at a time. Controls memory usage.
    show_progress : bool, default=False
        Show progress bar during computation.

    Returns
    -------
    sigma : float
    """
    N = X.shape[0]

    # For small datasets, use the simple approach
    if N <= chunk_size:
        D = torch.cdist(X, X, p=2.0)
        D.fill_diagonal_(float("nan"))
        if method == "median":
            return float(torch.nanmedian(D).item())
        elif method == "mean":
            return float(torch.nanmean(D).item())
        elif method == "max":
            D.fill_diagonal_(0.0)
            return float(D.max().item())
        else:
            raise ValueError(f"Unknown method '{method}'; use 'median', 'mean', or 'max'")

    # For large datasets, use chunked computation
    if method == "max":
        running_max = 0.0
        n_chunks = (N + chunk_size - 1) // chunk_size
        chunk_iter = range(n_chunks)
        if show_progress:
            chunk_iter = tqdm(chunk_iter, desc="Computing global σ", unit="chunk")

        for chunk_idx in chunk_iter:
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, N)
            D_chunk = torch.cdist(X[start:end], X, p=2.0)
            # Exclude self-distances
            for i, global_i in enumerate(range(start, end)):
                D_chunk[i, global_i] = 0.0
            chunk_max = float(D_chunk.max().item())
            running_max = max(running_max, chunk_max)
        return running_max

    elif method == "mean":
        running_sum = 0.0
        count = 0
        n_chunks = (N + chunk_size - 1) // chunk_size
        chunk_iter = range(n_chunks)
        if show_progress:
            chunk_iter = tqdm(chunk_iter, desc="Computing global σ", unit="chunk")

        for chunk_idx in chunk_iter:
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, N)
            D_chunk = torch.cdist(X[start:end], X, p=2.0)
            # Exclude self-distances
            for i, global_i in enumerate(range(start, end)):
                D_chunk[i, global_i] = float("nan")
            valid_mask = ~torch.isnan(D_chunk)
            running_sum += float(D_chunk[valid_mask].sum().item())
            count += int(valid_mask.sum().item())
        return running_sum / count

    elif method == "median":
        all_dists = []
        n_chunks = (N + chunk_size - 1) // chunk_size
        chunk_iter = range(n_chunks)
        if show_progress:
            chunk_iter = tqdm(chunk_iter, desc="Computing global σ", unit="chunk")

        for chunk_idx in chunk_iter:
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, N)
            D_chunk = torch.cdist(X[start:end], X[start:], p=2.0)
            for i, global_i in enumerate(range(start, end)):
                local_col_start = global_i - start
                D_chunk[i, : local_col_start + 1] = float("nan")
            valid = D_chunk[~torch.isnan(D_chunk)]
            all_dists.append(valid.cpu())
        all_dists_t = torch.cat(all_dists)
        return float(torch.median(all_dists_t).item())

    else:
        raise ValueError(f"Unknown method '{method}'; use 'median', 'mean', or 'max'")
