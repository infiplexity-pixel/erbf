"""
Kernel functions for ERBF interpolation.

This module provides kernel classes and a factory function for building
kernel matrices from training data. All kernels support local (adaptive)
bandwidth via per-point sigma values.

Supported Kernels:
    - Gaussian (default): K(r) = exp(-r^P / (2σ²))
    - Multiquadric: K(r) = √(1 + (r/σ)²)
    - Inverse Multiquadric: K(r) = 1 / √(1 + (r/σ)²)
    - Thin Plate Spline: K(r) = r² log(r)
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no-op tqdm if not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable


class BaseKernel(abc.ABC):
    """Abstract base class for all ERBF kernel functions.

    Parameters
    ----------
    P : int, default=2
        Exponent applied to pairwise distances before passing through the
        kernel function. ``P=2`` corresponds to squared Euclidean distance.
    lambda_reg : float, default=1e-10
        Tikhonov regularisation added to the diagonal of the kernel matrix to
        ensure positive-definiteness and numerical stability.
    """

    def __init__(self, P: int = 2, lambda_reg: float = 1e-10) -> None:
        if P < 1:
            raise ValueError(f"P must be >= 1, got {P}")
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be >= 0, got {lambda_reg}")
        self.P = P
        self.lambda_reg = lambda_reg

    @abc.abstractmethod
    def _evaluate(
        self,
        D_P: np.ndarray,
        sigma_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute raw kernel values given distance and sigma matrices.

        Parameters
        ----------
        D_P : ndarray of shape (n, m)
            Pairwise distances raised to the power ``self.P``.
        sigma_matrix : ndarray of shape (n, m)
            Element-wise bandwidth matrix (geometric mean of per-point sigmas).

        Returns
        -------
        K : ndarray of shape (n, m)
        """

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sigmas_X: np.ndarray,
        sigmas_Y: Optional[np.ndarray] = None,
        *,
        symmetric: bool = False,
    ) -> np.ndarray:
        """Build the kernel matrix between *X* and *Y*.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Query or training points (rows of the output matrix).
        Y : ndarray of shape (m, d)
            Training points (columns of the output matrix).
        sigmas_X : ndarray of shape (n,)
            Per-point bandwidths for rows of *X*.
        sigmas_Y : ndarray of shape (m,) or None
            Per-point bandwidths for rows of *Y*.  If ``None``, uses
            *sigmas_X* (valid only when X is Y, i.e. the training kernel).
        symmetric : bool, default=False
            If ``True``, the matrix is square and self-similarity diagonal
            entries are forced to 1.0, and Tikhonov regularisation is added.

        Returns
        -------
        K : ndarray of shape (n, m)
        """
        if sigmas_Y is None:
            sigmas_Y = sigmas_X

        n = X.shape[0]
        m = Y.shape[0]

        D = cdist(X, Y)
        D_P = D ** self.P

        # ── Determine row sigmas ──────────────────────────────────
        # sigmas_X is provided by the caller.  Two cases:
        #   Symmetric (training):  len(sigmas_X) == n == m   → use directly
        #   Asymmetric (predict):  len(sigmas_X) may equal m (training sigmas
        #       were forwarded for both arguments).  When that happens the
        #       query points don't have their own trained sigmas, so we
        #       broadcast the training sigmas across each query row.
        if sigmas_X.shape[0] == n:
            row_sigmas = sigmas_X                       # shape (n,)
        else:
            # Query points have no trained sigma → use mean of training sigmas
            # so the geometric mean still yields reasonable per-pair values.
            row_sigmas = np.full(n, sigmas_Y.mean())    # shape (n,)

        # col_sigmas always corresponds to Y (training) points
        col_sigmas = sigmas_Y                           # shape (m,)

        # Geometric mean bandwidth σ_ij = √(σ_row_i · σ_col_j)
        # Result shape: (n, m) — always matches D_P
        sigma_matrix = np.sqrt(row_sigmas[:, None] * col_sigmas[None, :])
        sigma_matrix = np.maximum(sigma_matrix, 1e-8)

        K = self._evaluate(D_P, sigma_matrix)

        if symmetric:
            np.fill_diagonal(K, 1.0)
            K = K + self.lambda_reg * np.eye(n)

        return K

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(P={self.P}, lambda_reg={self.lambda_reg})"


class GaussianKernel(BaseKernel):
    r"""Gaussian (squared-exponential) RBF kernel.

    .. math::
        K(x_i, x_j) = \exp\!\left(-\frac{\|x_i - x_j\|^P}{2\,\sigma_{ij}^2}\right)

    where :math:`\sigma_{ij} = \sqrt{\sigma_i \sigma_j}` (geometric mean).
    """

    def _evaluate(self, D_P: np.ndarray, sigma_matrix: np.ndarray) -> np.ndarray:
        return np.exp(-D_P / (2.0 * sigma_matrix ** 2))


class MultiquadricKernel(BaseKernel):
    r"""Multiquadric kernel.

    .. math::
        K(x_i, x_j) = \sqrt{1 + \left(\frac{\|x_i - x_j\|}{\sigma_{ij}}\right)^P}
    """

    def _evaluate(self, D_P: np.ndarray, sigma_matrix: np.ndarray) -> np.ndarray:
        return np.sqrt(1.0 + D_P / sigma_matrix ** 2)


class InverseMultiquadricKernel(BaseKernel):
    r"""Inverse multiquadric kernel.

    .. math::
        K(x_i, x_j) = \frac{1}{\sqrt{1 + \left(\frac{\|x_i - x_j\|}{\sigma_{ij}}\right)^P}}
    """

    def _evaluate(self, D_P: np.ndarray, sigma_matrix: np.ndarray) -> np.ndarray:
        return 1.0 / np.sqrt(1.0 + D_P / sigma_matrix ** 2)


class ThinPlateSplineKernel(BaseKernel):
    r"""Thin plate spline kernel.

    .. math::
        K(x_i, x_j) = \|x_i - x_j\|^2 \log(\|x_i - x_j\| + \epsilon)

    Note: sigma values are **not** used by this kernel but accepted for API
    compatibility.
    """

    def _evaluate(self, D_P: np.ndarray, sigma_matrix: np.ndarray) -> np.ndarray:
        D = np.power(D_P, 1.0 / self.P)  # recover original distance
        eps = 1e-12
        return D ** 2 * np.log(D + eps)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

_KERNEL_REGISTRY: dict[str, type[BaseKernel]] = {
    "gaussian": GaussianKernel,
    "multiquadric": MultiquadricKernel,
    "inverse_multiquadric": InverseMultiquadricKernel,
    "thin_plate_spline": ThinPlateSplineKernel,
}


def build_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    sigmas_X: np.ndarray,
    sigmas_Y: Optional[np.ndarray] = None,
    *,
    kernel: str | BaseKernel = "gaussian",
    P: int = 2,
    lambda_reg: float = 1e-10,
    symmetric: bool = False,
) -> np.ndarray:
    """Build a kernel matrix between *X* and *Y*.

    This is the primary high-level entry point for kernel construction.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    Y : ndarray of shape (m, d)
    sigmas_X : ndarray of shape (n,)
        Per-point bandwidths for *X*.  During prediction this may be the
        training sigmas (shape ``(m,)``); the kernel will handle the
        mismatch gracefully.
    sigmas_Y : ndarray of shape (m,) or None
        Per-point bandwidths for *Y*.  If ``None``, defaults to *sigmas_X*.
    kernel : str or BaseKernel instance, default="gaussian"
        Kernel function to use.
    P : int, default=2
        Distance exponent.
    lambda_reg : float, default=1e-10
        Diagonal regularisation.
    symmetric : bool, default=False
        Whether the resulting matrix is symmetric (training kernel).

    Returns
    -------
    K : ndarray of shape (n, m)
    """
    if isinstance(kernel, str):
        kernel_name = kernel.lower().replace("-", "_").replace(" ", "_")
        if kernel_name not in _KERNEL_REGISTRY:
            raise ValueError(
                f"Unknown kernel '{kernel}'. "
                f"Choose from: {list(_KERNEL_REGISTRY.keys())}"
            )
        kernel_obj = _KERNEL_REGISTRY[kernel_name](P=P, lambda_reg=lambda_reg)
    elif isinstance(kernel, BaseKernel):
        kernel_obj = kernel
    else:
        raise TypeError(f"kernel must be str or BaseKernel, got {type(kernel)}")

    return kernel_obj(X, Y, sigmas_X, sigmas_Y, symmetric=symmetric)


def chunked_kernel_matmul(
    X: np.ndarray,
    Y: np.ndarray,
    sigmas_X: np.ndarray,
    sigmas_Y: np.ndarray,
    weights: np.ndarray,
    *,
    kernel: str | BaseKernel = "gaussian",
    P: int = 2,
    lambda_reg: float = 1e-10,
    chunk_size: int = 500,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute K(X, Y) @ weights using chunking to control memory usage.

    Instead of materializing the full (n_query, n_train) kernel matrix,
    this function processes queries in chunks, keeping memory usage constant
    regardless of the number of centers (training points).

    Parameters
    ----------
    X : ndarray of shape (n_query, d)
        Query points.
    Y : ndarray of shape (n_train, d)
        Training/center points.
    sigmas_X : ndarray of shape (n_query,) or (n_train,)
        Per-point bandwidths for query. If shape doesn't match X, uses
        mean of sigmas_Y for all query points.
    sigmas_Y : ndarray of shape (n_train,)
        Per-point bandwidths for training centers.
    weights : ndarray of shape (n_classes, n_train) or (n_train,)
        Interpolation weights from training.
    kernel : str or BaseKernel instance, default="gaussian"
        Kernel function to use.
    P : int, default=2
        Distance exponent.
    lambda_reg : float, default=1e-10
        Diagonal regularisation (not used for asymmetric kernel).
    chunk_size : int, default=500
        Number of query points to process at a time.
    show_progress : bool, default=False
        Show tqdm progress bar.

    Returns
    -------
    result : ndarray of shape (n_query, n_classes) or (n_query,)
        The result of K @ weights^T (if weights is 2D) or K @ weights.
    """
    # Build the kernel object
    if isinstance(kernel, str):
        kernel_name = kernel.lower().replace("-", "_").replace(" ", "_")
        if kernel_name not in _KERNEL_REGISTRY:
            raise ValueError(
                f"Unknown kernel '{kernel}'. "
                f"Choose from: {list(_KERNEL_REGISTRY.keys())}"
            )
        kernel_obj = _KERNEL_REGISTRY[kernel_name](P=P, lambda_reg=lambda_reg)
    elif isinstance(kernel, BaseKernel):
        kernel_obj = kernel
    else:
        raise TypeError(f"kernel must be str or BaseKernel, got {type(kernel)}")

    n_query = X.shape[0]
    n_train = Y.shape[0]
    
    # Handle weights shape
    weights = np.asarray(weights)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    n_classes = weights.shape[0]
    
    # Pre-allocate output
    result = np.zeros((n_query, n_classes), dtype=np.float64)
    
    # Process in chunks
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    chunk_iter = range(n_chunks)
    if show_progress:
        chunk_iter = tqdm(chunk_iter, desc="Predicting", unit="chunk", 
                         total=n_chunks)
    
    for chunk_idx in chunk_iter:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_query)
        X_chunk = X[start:end]
        
        # Compute kernel chunk: shape (chunk_size, n_train)
        K_chunk = kernel_obj(X_chunk, Y, sigmas_X, sigmas_Y, symmetric=False)
        
        # Compute matmul: (chunk_size, n_train) @ (n_train, n_classes)
        result[start:end] = K_chunk @ weights.T
    
    if squeeze_output:
        return result.squeeze(axis=1)
    return result