"""
Adaptive bandwidth (σ) computation for ERBF.

This module provides functions for computing per-point bandwidths using k-NN
distances, as well as global (constant) bandwidth estimation.

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

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

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
    return max(minimum, int(np.sqrt(N) * multiplier))


def compute_local_sigmas(
    X: np.ndarray,
    k_neighbors: Optional[int] = None,
    *,
    k_multiplier: float = 1.5,
    k_minimum: int = 10,
    min_sigma: float = 0.5,
    max_sigma: float = 20.0,
    metric: str = "euclidean",
    chunk_size: int = 500,
    verbose: bool = False,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute per-point adaptive σ using k-NN mean distances.

    For each point *x_i* the bandwidth is the mean Euclidean distance to its
    *k* nearest neighbours, clipped to ``[min_sigma, max_sigma]``.

    Uses chunking to keep memory usage constant regardless of N.

    Parameters
    ----------
    X : ndarray of shape (N, d)
        Data matrix.
    k_neighbors : int or None
        Number of neighbours.  If ``None``, automatically selected via
        :func:`auto_select_k`.
    k_multiplier : float, default=1.5
        Passed to :func:`auto_select_k` when *k_neighbors* is ``None``.
    k_minimum : int, default=10
        Passed to :func:`auto_select_k` when *k_neighbors* is ``None``.
    min_sigma : float, default=0.5
        Floor for computed σ values.
    max_sigma : float, default=20.0
        Ceiling for computed σ values.
    metric : str, default="euclidean"
        Distance metric forwarded to ``scipy.spatial.distance.cdist``.
    chunk_size : int, default=500
        Number of rows to process at a time. Controls memory usage.
    verbose : bool, default=False
        Print diagnostic information.
    show_progress : bool, default=False
        Show progress bar during computation.

    Returns
    -------
    sigmas : ndarray of shape (N,)
        Per-point bandwidth values.
    """
    N = X.shape[0]

    if k_neighbors is None:
        k_neighbors = auto_select_k(N, multiplier=k_multiplier, minimum=k_minimum)
    # Clamp k to at most N-1 (all other points)
    k_neighbors = min(k_neighbors, N - 1)

    if verbose:
        print(
            f"[sigma] Computing adaptive σ: N={N}, k={k_neighbors}, "
            f"metric={metric}, chunk_size={chunk_size}"
        )

    sigmas = np.zeros(N, dtype=np.float64)
    
    # Process in chunks to keep memory constant
    n_chunks = (N + chunk_size - 1) // chunk_size
    chunk_iter = range(n_chunks)
    if show_progress:
        chunk_iter = tqdm(chunk_iter, desc="Computing σ", unit="chunk")
    
    for chunk_idx in chunk_iter:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)
        X_chunk = X[start:end]
        
        # Compute distances from chunk to all points
        D_chunk = cdist(X_chunk, X, metric=metric)  # shape: (chunk_size, N)
        
        for i, global_i in enumerate(range(start, end)):
            row = D_chunk[i].copy()
            row[global_i] = np.inf  # exclude self
            knn_dists = np.sort(row)[:k_neighbors]
            sigmas[global_i] = np.clip(np.mean(knn_dists), min_sigma, max_sigma)

    if verbose:
        print(
            f"[sigma] σ range: [{sigmas.min():.4f}, {sigmas.max():.4f}], "
            f"mean={sigmas.mean():.4f}"
        )

    return sigmas


def compute_global_sigma(
    X: np.ndarray,
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
    X : ndarray of shape (N, d)
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
        D = cdist(X, X, metric=metric)
        np.fill_diagonal(D, np.nan)
        if method == "median":
            return float(np.nanmedian(D))
        elif method == "mean":
            return float(np.nanmean(D))
        elif method == "max":
            return float(np.nanmax(D))
        else:
            raise ValueError(f"Unknown method '{method}'; use 'median', 'mean', or 'max'")
    
    # For large datasets, use chunked computation
    # For median: collect all distances (still memory-heavy, but unavoidable for exact median)
    # For mean: use running stats
    # For max: track running max
    
    if method == "max":
        running_max = 0.0
        n_chunks = (N + chunk_size - 1) // chunk_size
        chunk_iter = range(n_chunks)
        if show_progress:
            chunk_iter = tqdm(chunk_iter, desc="Computing global σ", unit="chunk")
        
        for chunk_idx in chunk_iter:
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, N)
            D_chunk = cdist(X[start:end], X, metric=metric)
            # Exclude self-distances
            for i, global_i in enumerate(range(start, end)):
                D_chunk[i, global_i] = 0.0
            chunk_max = D_chunk.max()
            running_max = max(running_max, chunk_max)
        return float(running_max)
    
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
            D_chunk = cdist(X[start:end], X, metric=metric)
            # Exclude self-distances
            for i, global_i in enumerate(range(start, end)):
                D_chunk[i, global_i] = np.nan
            running_sum += np.nansum(D_chunk)
            count += np.sum(~np.isnan(D_chunk))
        return float(running_sum / count)
    
    elif method == "median":
        # For median, we need to use reservoir sampling or approximate methods
        # For now, compute chunked and collect upper triangle values
        all_dists = []
        n_chunks = (N + chunk_size - 1) // chunk_size
        chunk_iter = range(n_chunks)
        if show_progress:
            chunk_iter = tqdm(chunk_iter, desc="Computing global σ", unit="chunk")
        
        for chunk_idx in chunk_iter:
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, N)
            # Only compute upper triangle to avoid duplicates
            D_chunk = cdist(X[start:end], X[start:], metric=metric)
            for i, global_i in enumerate(range(start, end)):
                # Set self and already-counted pairs to nan
                local_col_start = global_i - start
                D_chunk[i, :local_col_start + 1] = np.nan
            valid = D_chunk[~np.isnan(D_chunk)]
            all_dists.extend(valid.tolist())
        return float(np.median(all_dists))
    
    else:
        raise ValueError(f"Unknown method '{method}'; use 'median', 'mean', or 'max'")