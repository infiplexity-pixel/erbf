"""
ERBF regressor for continuous-valued interpolation.

Solves ``K · w = y`` exactly so that the training targets are interpolated
with zero residual (up to floating-point precision).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import solve

from erbf.kernel import BaseKernel, build_kernel_matrix, chunked_kernel_matmul
from erbf.sigma import compute_local_sigmas

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class ERBFRegressor:
    """Exact RBF interpolation regressor.

    Parameters
    ----------
    k_neighbors : int or None
        Number of neighbours for adaptive σ. ``None`` → automatic.
    k_multiplier : float, default=1.5
        Multiplier for automatic k selection.
    k_minimum : int, default=10
        Minimum k.
    min_sigma : float, default=0.5
        Lower bound for σ.
    max_sigma : float, default=20.0
        Upper bound for σ.
    P : int, default=2
        Distance exponent.
    lambda_reg : float, default=1e-10
        Diagonal regularisation.
    kernel : str or BaseKernel, default="gaussian"
        Kernel function.
    chunk_size : int, default=500
        Number of samples to process at a time. Controls memory usage.
    show_progress : bool, default=False
        Show tqdm progress bars during training and prediction.
    verbose : bool, default=False

    Attributes
    ----------
    weights_ : ndarray of shape (n_targets, n_train) or (n_train,)
        Interpolation weights.
    sigmas_ : ndarray of shape (n_train,)
    K_train_ : ndarray of shape (n_train, n_train)
    condition_number_ : float

    Examples
    --------
    >>> from erbf import ERBFRegressor
    >>> reg = ERBFRegressor(show_progress=True)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    """

    def __init__(
        self,
        k_neighbors: Optional[int] = None,
        k_multiplier: float = 1.5,
        k_minimum: int = 10,
        min_sigma: float = 0.5,
        max_sigma: float = 20.0,
        P: int = 2,
        lambda_reg: float = 1e-10,
        kernel: str | BaseKernel = "gaussian",
        chunk_size: int = 500,
        show_progress: bool = False,
        verbose: bool = False,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.k_multiplier = k_multiplier
        self.k_minimum = k_minimum
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.P = P
        self.lambda_reg = lambda_reg
        self.kernel = kernel
        self.chunk_size = chunk_size
        self.show_progress = show_progress
        self.verbose = verbose

        self.X_train_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.sigmas_: Optional[np.ndarray] = None
        self.K_train_: Optional[np.ndarray] = None
        self.condition_number_: float = 0.0
        self._multi_target: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ERBFRegressor":
        """Fit the ERBF regressor.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._multi_target = y.ndim == 2
        if y.ndim == 1:
            y = y[:, None]

        self.X_train_ = X
        N = X.shape[0]

        self.sigmas_ = compute_local_sigmas(
            X,
            k_neighbors=self.k_neighbors,
            k_multiplier=self.k_multiplier,
            k_minimum=self.k_minimum,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            chunk_size=self.chunk_size,
            show_progress=self.show_progress,
            verbose=self.verbose,
        )

        self.K_train_ = build_kernel_matrix(
            X, X, self.sigmas_, self.sigmas_,
            kernel=self.kernel, P=self.P, lambda_reg=self.lambda_reg,
            symmetric=True,
        )
        self.condition_number_ = float(np.linalg.cond(self.K_train_))

        if self.verbose:
            print(f"[fit] cond(K): {self.condition_number_:.2e}")

        # Solve for each target column
        self.weights_ = solve(self.K_train_, y, assume_a="sym")  # (N, n_targets)
        self.weights_ = self.weights_.T  # (n_targets, N)

        if self.verbose:
            recon = self.K_train_ @ self.weights_.T
            max_err = np.max(np.abs(recon - y))
            print(f"[fit] max interpolation error: {max_err:.2e}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target(s) for *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
        """
        if self.X_train_ is None or self.weights_ is None:
            raise RuntimeError("ERBFRegressor is not fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        # Use chunked kernel-weight multiplication to control memory
        y_pred = chunked_kernel_matmul(
            X, self.X_train_, self.sigmas_, self.sigmas_,
            self.weights_,
            kernel=self.kernel,
            P=self.P,
            lambda_reg=self.lambda_reg,
            chunk_size=self.chunk_size,
            show_progress=self.show_progress,
        )
        if not self._multi_target:
            y_pred = y_pred.ravel()
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score."""
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-15))

    def get_params(self) -> dict:
        return {
            "k_neighbors": self.k_neighbors,
            "k_multiplier": self.k_multiplier,
            "k_minimum": self.k_minimum,
            "min_sigma": self.min_sigma,
            "max_sigma": self.max_sigma,
            "P": self.P,
            "lambda_reg": self.lambda_reg,
            "kernel": self.kernel,
            "chunk_size": self.chunk_size,
            "show_progress": self.show_progress,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "ERBFRegressor":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}'")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"ERBFRegressor(k_neighbors={self.k_neighbors}, "
            f"k_multiplier={self.k_multiplier}, P={self.P}, "
            f"kernel={self.kernel!r})"
        )