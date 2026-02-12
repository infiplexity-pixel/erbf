"""
ERBF multiclass classifier.

Implements one-vs-all classification via exact kernel interpolation.  For each
class *c* an indicator vector ``I_c`` is constructed (1 where the label equals
*c*, 0 otherwise) and the linear system ``K · w_c = I_c`` is solved exactly.

At prediction time the class with the highest interpolated score is selected::

    predict(x) = argmax_c  Σ_i  w_c[i] · K(x, x_i)

Because the kernel system is solved exactly, training accuracy is **guaranteed
to be 100 %** (up to floating-point precision).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import solve

from erbf.kernel import BaseKernel, GaussianKernel, build_kernel_matrix, chunked_kernel_matmul
from erbf.sigma import compute_local_sigmas, auto_select_k

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no-op tqdm if not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable


class ERBFClassifier:
    """Exact RBF interpolation classifier.

    Parameters
    ----------
    k_neighbors : int or None
        Number of neighbours for adaptive σ.  ``None`` → automatic.
    k_multiplier : float, default=1.5
        Multiplier for automatic k selection (k ≈ multiplier·√N).
    k_minimum : int, default=10
        Minimum k when using automatic selection.
    min_sigma : float, default=0.5
        Lower bound for per-point σ.
    max_sigma : float, default=20.0
        Upper bound for per-point σ.
    P : int, default=2
        Distance exponent.
    lambda_reg : float, default=1e-10
        Diagonal regularisation for the kernel matrix.
    kernel : str or BaseKernel, default="gaussian"
        Kernel function.
    chunk_size : int, default=500
        Number of samples to process at a time during training (sigma 
        computation) and prediction. Controls memory usage.
    show_progress : bool, default=False
        Show tqdm progress bars during training and prediction.
    verbose : bool, default=False
        Print diagnostic information during fit/predict.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels discovered during ``fit``.
    weights_ : ndarray of shape (n_classes, n_train)
        Interpolation weights per class.
    sigmas_ : ndarray of shape (n_train,)
        Adaptive bandwidths.
    K_train_ : ndarray of shape (n_train, n_train)
        Training kernel matrix (cached for diagnostics).
    interpolation_errors_ : dict[int, float]
        Max interpolation error per class after training.
    condition_number_ : float
        Condition number of the training kernel matrix.

    Examples
    --------
    >>> from erbf import ERBFClassifier
    >>> clf = ERBFClassifier(verbose=True, show_progress=True)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> print(f"Accuracy: {(y_pred == y_test).mean():.2%}")
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

        # Fitted state
        self.X_train_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.sigmas_: Optional[np.ndarray] = None
        self.K_train_: Optional[np.ndarray] = None
        self.interpolation_errors_: dict[int, float] = {}
        self.condition_number_: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ERBFClassifier":
        """Fit the ERBF classifier on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        self.X_train_ = X
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        N = X.shape[0]

        # 1. Compute adaptive bandwidths
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

        # 2. Build symmetric training kernel
        self.K_train_ = build_kernel_matrix(
            X, X, self.sigmas_, self.sigmas_,
            kernel=self.kernel,
            P=self.P,
            lambda_reg=self.lambda_reg,
            symmetric=True,
        )
        self.condition_number_ = float(np.linalg.cond(self.K_train_))

        if self.verbose:
            print(f"[fit] Kernel shape: {self.K_train_.shape}")
            print(f"[fit] σ range: [{self.sigmas_.min():.4f}, {self.sigmas_.max():.4f}]")
            print(f"[fit] cond(K): {self.condition_number_:.2e}")

        # 3. Solve one-vs-all systems
        self.weights_ = np.zeros((n_classes, N), dtype=np.float64)
        self.interpolation_errors_ = {}

        for idx, c in enumerate(self.classes_):
            target = (y == c).astype(np.float64)
            w = solve(self.K_train_, target, assume_a="sym")
            self.weights_[idx] = w

            # Verify interpolation quality
            reconstruction = self.K_train_ @ w
            max_err = float(np.max(np.abs(reconstruction - target)))
            self.interpolation_errors_[int(c)] = max_err

            if self.verbose:
                print(f"[fit] Class {c}: max interpolation error = {max_err:.2e}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        scores = self.decision_function(X)
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute per-class interpolation scores.

        Uses chunking to keep memory usage constant regardless of the number
        of training centers.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        # Use chunked kernel-weight multiplication to control memory
        return chunked_kernel_matmul(
            X, self.X_train_, self.sigmas_, self.sigmas_,
            self.weights_,
            kernel=self.kernel,
            P=self.P,
            lambda_reg=self.lambda_reg,
            chunk_size=self.chunk_size,
            show_progress=self.show_progress,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax-normalised class probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        scores = self.decision_function(X)
        # Numerically stable softmax
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy on (*X*, *y*).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        accuracy : float
        """
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def get_params(self) -> dict:
        """Return estimator parameters (sklearn-compatible)."""
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

    def set_params(self, **params) -> "ERBFClassifier":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for ERBFClassifier")
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.weights_ is None or self.X_train_ is None:
            raise RuntimeError(
                "ERBFClassifier is not fitted yet. Call .fit(X, y) first."
            )

    def __repr__(self) -> str:
        return (
            f"ERBFClassifier(k_neighbors={self.k_neighbors}, "
            f"k_multiplier={self.k_multiplier}, P={self.P}, "
            f"kernel={self.kernel!r}, lambda_reg={self.lambda_reg})"
        )