"""
General utility functions for the ERBF library.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import numpy as np


@contextmanager
def timer(label: str = "Elapsed") -> Generator[None, None, None]:
    """Context manager that prints elapsed wall-clock time.

    Usage::

        with timer("Training"):
            clf.fit(X, y)
        # prints: Training: 1.23s
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"{label}: {elapsed:.2f}s")


def normalize(X: np.ndarray, *, method: str = "minmax") -> np.ndarray:
    """Normalise data matrix.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    method : {"minmax", "zscore"}

    Returns
    -------
    X_norm : ndarray
    """
    if method == "minmax":
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        return (X - mins) / denom
    elif method == "zscore":
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        return (X - mu) / sigma
    else:
        raise ValueError(f"Unknown normalisation method '{method}'")


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/test subsets.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    y : ndarray of shape (n,)
    test_size : float, default=0.2
    seed : int, default=42

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_test = max(1, int(n * test_size))
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def check_array(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    """Validate and convert to float64 ndarray."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {X.shape}")
    if np.any(np.isnan(X)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError(f"{name} contains infinite values")
    return X