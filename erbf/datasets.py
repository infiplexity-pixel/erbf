"""
Dataset loading and synthetic data generation utilities.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from typing import Tuple
import numpy as np
import torch
from torchvision import datasets, transforms


def load_mnist_subset(
    n_train: int = 200,
    n_test: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a random subset of MNIST using torchvision, normalised to [0, 1].

    Parameters
    ----------
    n_train : int, default=200
    n_test : int, default=200
    seed : int, default=42

    Returns
    -------
    X_train : ndarray of shape (n_train, 784)
    y_train : ndarray of shape (n_train,)
    X_test : ndarray of shape (n_test, 784)
    y_test : ndarray of shape (n_test,)
    """

    rng = np.random.RandomState(seed)

    transform = transforms.ToTensor()  # Converts to [0,1] float32 tensor

    # Download/load datasets
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    # Convert full datasets to tensors
    X_train_full = train_dataset.data.float() / 255.0
    y_train_full = train_dataset.targets

    X_test_full = test_dataset.data.float() / 255.0
    y_test_full = test_dataset.targets

    # Flatten to (N, 784)
    X_train_full = X_train_full.view(-1, 784)
    X_test_full = X_test_full.view(-1, 784)

    # Random subset selection
    train_idx = rng.permutation(len(X_train_full))[:n_train]
    test_idx = rng.permutation(len(X_test_full))[:n_test]

    X_train = X_train_full[train_idx].numpy()
    y_train = y_train_full[train_idx].numpy()

    X_test = X_test_full[test_idx].numpy()
    y_test = y_test_full[test_idx].numpy()

    return X_train, y_train, X_test, y_test


def make_classification_demo(
    n_samples: int = 300,
    n_features: int = 2,
    n_classes: int = 3,
    separation: float = 2.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple multiclass classification dataset.

    Generates *n_classes* Gaussian clusters.

    Parameters
    ----------
    n_samples : int, default=300
    n_features : int, default=2
    n_classes : int, default=3
    separation : float, default=2.0
        Distance between cluster centres.
    seed : int, default=42

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    per_class = n_samples // n_classes

    centers = rng.uniform(-separation, separation, size=(n_classes, n_features))
    X_list, y_list = [], []

    for c in range(n_classes):
        X_c = rng.normal(loc=centers[c], scale=0.5, size=(per_class, n_features))
        X_list.append(X_c)
        y_list.append(np.full(per_class, c, dtype=np.int32))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_regression_demo(
    n_samples: int = 200,
    n_features: int = 1,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple nonlinear regression dataset.

    ``y = sin(2πx) + noise`` for 1D, or sum-of-sines for higher dimensions.

    Parameters
    ----------
    n_samples : int, default=200
    n_features : int, default=1
    noise : float, default=0.1
    seed : int, default=42

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n_samples, n_features))
    y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, noise, n_samples)
    return X, y