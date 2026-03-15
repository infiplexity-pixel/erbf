"""
Kernel functions for ERBF interpolation.

This module provides kernel classes and a factory function for building
kernel matrices from training data. All kernels support local (adaptive)
bandwidth via per-point sigma values.

Uses PyTorch for GPU-accelerated computation when CUDA is available.

Supported Kernels:
    - Gaussian (default): K(r) = exp(-r^P / (2σ²))
    - Multiquadric: K(r) = √(1 + (r/σ)²)
    - Inverse Multiquadric: K(r) = 1 / √(1 + (r/σ)²)
    - Thin Plate Spline: K(r) = r² log(r)
"""

from __future__ import annotations

import abc

import torch

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no-op tqdm if not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _default_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x: object, device: torch.device | None = None) -> torch.Tensor:
    """Convert input to a float64 torch tensor on the given device."""
    if device is None:
        device = _default_device()
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float64)
    return torch.as_tensor(x, dtype=torch.float64, device=device)


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
        D_P: torch.Tensor,
        sigma_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw kernel values given distance and sigma matrices.

        Parameters
        ----------
        D_P : Tensor of shape (n, m)
            Pairwise distances raised to the power ``self.P``.
        sigma_matrix : Tensor of shape (n, m)
            Element-wise bandwidth matrix (geometric mean of per-point sigmas).

        Returns
        -------
        K : Tensor of shape (n, m)
        """

    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        sigmas_X: torch.Tensor,
        sigmas_Y: torch.Tensor | None = None,
        *,
        symmetric: bool = False,
    ) -> torch.Tensor:
        """Build the kernel matrix between *X* and *Y*.

        Parameters
        ----------
        X : Tensor of shape (n, d)
            Query or training points (rows of the output matrix).
        Y : Tensor of shape (m, d)
            Training points (columns of the output matrix).
        sigmas_X : Tensor of shape (n,)
            Per-point bandwidths for rows of *X*.
        sigmas_Y : Tensor of shape (m,) or None
            Per-point bandwidths for rows of *Y*.  If ``None``, uses
            *sigmas_X* (valid only when X is Y, i.e. the training kernel).
        symmetric : bool, default=False
            If ``True``, the matrix is square and self-similarity diagonal
            entries are forced to 1.0, and Tikhonov regularisation is added.

        Returns
        -------
        K : Tensor of shape (n, m)
        """
        device = X.device

        if sigmas_Y is None:
            sigmas_Y = sigmas_X

        n = X.shape[0]

        if self.P == 2:
            XX = (X * X).sum(dim=1, keepdim=True)       # (n, 1)
            YY = (Y * Y).sum(dim=1, keepdim=True).T     # (1, m)
            D_P = torch.clamp(XX + YY - 2.0 * (X @ Y.T), min=0.0)
        else:
            D = torch.cdist(X, Y, p=2.0)
            D_P = D.pow(self.P)

        if sigmas_X.shape[0] == n:
            row_sigmas = sigmas_X
        else:
            row_sigmas = sigmas_Y.mean().expand(n)

        col_sigmas = sigmas_Y

        sigma_matrix = torch.sqrt(row_sigmas.unsqueeze(1) * col_sigmas.unsqueeze(0))
        sigma_matrix = torch.clamp(sigma_matrix, min=1e-8)

        K = self._evaluate(D_P, sigma_matrix)

        if symmetric:
            K.fill_diagonal_(1.0)
            K = K + self.lambda_reg * torch.eye(n, device=device, dtype=K.dtype)

        return K

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(P={self.P}, lambda_reg={self.lambda_reg})"


class GaussianKernel(BaseKernel):
    r"""Gaussian (squared-exponential) RBF kernel.

    .. math::
        K(x_i, x_j) = \exp\!\left(-\frac{\|x_i - x_j\|^P}{2\,\sigma_{ij}^2}\right)

    where :math:`\sigma_{ij} = \sqrt{\sigma_i \sigma_j}` (geometric mean).
    """

    def _evaluate(self, D_P: torch.Tensor, sigma_matrix: torch.Tensor) -> torch.Tensor:
        return torch.exp(-D_P / (2.0 * sigma_matrix ** 2))


class MultiquadricKernel(BaseKernel):
    r"""Multiquadric kernel.

    .. math::
        K(x_i, x_j) = \sqrt{1 + \left(\frac{\|x_i - x_j\|}{\sigma_{ij}}\right)^P}
    """

    def _evaluate(self, D_P: torch.Tensor, sigma_matrix: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 + D_P / sigma_matrix ** 2)


class InverseMultiquadricKernel(BaseKernel):
    r"""Inverse multiquadric kernel.

    .. math::
        K(x_i, x_j) = \frac{1}{\sqrt{1 + \left(\frac{\|x_i - x_j\|}{\sigma_{ij}}\right)^P}}
    """

    def _evaluate(self, D_P: torch.Tensor, sigma_matrix: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(1.0 + D_P / sigma_matrix ** 2)


class ThinPlateSplineKernel(BaseKernel):
    r"""Thin plate spline kernel.

    .. math::
        K(x_i, x_j) = \|x_i - x_j\|^2 \log(\|x_i - x_j\| + \epsilon)

    Note: sigma values are **not** used by this kernel but accepted for API
    compatibility.
    """

    def _evaluate(self, D_P: torch.Tensor, sigma_matrix: torch.Tensor) -> torch.Tensor:
        D = torch.pow(D_P, 1.0 / self.P)
        eps = 1e-12
        return D ** 2 * torch.log(D + eps)


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
    X: torch.Tensor,
    Y: torch.Tensor,
    sigmas_X: torch.Tensor,
    sigmas_Y: torch.Tensor | None = None,
    *,
    kernel: str | BaseKernel = "gaussian",
    P: int = 2,
    lambda_reg: float = 1e-10,
    symmetric: bool = False,
) -> torch.Tensor:
    """Build a kernel matrix between *X* and *Y*.

    This is the primary high-level entry point for kernel construction.

    Parameters
    ----------
    X : Tensor of shape (n, d)
    Y : Tensor of shape (m, d)
    sigmas_X : Tensor of shape (n,)
        Per-point bandwidths for *X*.  During prediction this may be the
        training sigmas (shape ``(m,)``); the kernel will handle the
        mismatch gracefully.
    sigmas_Y : Tensor of shape (m,) or None
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
    K : Tensor of shape (n, m)
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
    X: torch.Tensor,
    Y: torch.Tensor,
    sigmas_X: torch.Tensor,
    sigmas_Y: torch.Tensor,
    weights: torch.Tensor,
    *,
    kernel: str | BaseKernel = "gaussian",
    P: int = 2,
    lambda_reg: float = 1e-10,
    chunk_size: int = 500,
    show_progress: bool = False,
) -> torch.Tensor:
    """Compute K(X, Y) @ weights using chunking to control memory usage.

    Instead of materializing the full (n_query, n_train) kernel matrix,
    this function processes queries in chunks, keeping memory usage constant
    regardless of the number of centers (training points).

    Parameters
    ----------
    X : Tensor of shape (n_query, d)
        Query points.
    Y : Tensor of shape (n_train, d)
        Training/center points.
    sigmas_X : Tensor of shape (n_query,) or (n_train,)
        Per-point bandwidths for query. If shape doesn't match X, uses
        mean of sigmas_Y for all query points.
    sigmas_Y : Tensor of shape (n_train,)
        Per-point bandwidths for training centers.
    weights : Tensor of shape (n_classes, n_train) or (n_train,)
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
    result : Tensor of shape (n_query, n_classes) or (n_query,)
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
    device = X.device

    # Handle weights shape
    if not isinstance(weights, torch.Tensor):
        weights = torch.as_tensor(weights, dtype=torch.float64, device=device)
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    n_classes = weights.shape[0]

    # Pre-allocate output
    result = torch.zeros(n_query, n_classes, dtype=torch.float64, device=device)

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
        return result.squeeze(1)
    return result
