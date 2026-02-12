"""
Visualization utilities for ERBF models.

All functions return matplotlib ``Figure`` objects so they can be displayed
or saved without side effects.  Matplotlib is imported lazily to avoid a hard
dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure


def _import_matplotlib():
    """Lazy-import matplotlib; raises a helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_kernel_matrix(
    K: np.ndarray,
    *,
    title: str = "ERBF Kernel Matrix",
    cmap: str = "viridis",
    figsize: tuple[int, int] = (8, 6),
) -> "matplotlib.figure.Figure":
    """Visualise a kernel matrix as a heatmap.

    Parameters
    ----------
    K : ndarray of shape (n, n)
    title : str
    cmap : str
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(K, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")
    fig.colorbar(im, ax=ax, label="K(x_i, x_j)")
    fig.tight_layout()
    return fig


def plot_sigma_distribution(
    sigmas: np.ndarray,
    *,
    title: str = "Adaptive σ Distribution",
    bins: int = 30,
    figsize: tuple[int, int] = (8, 5),
) -> "matplotlib.figure.Figure":
    """Plot histogram of adaptive bandwidth values.

    Parameters
    ----------
    sigmas : ndarray of shape (n,)
    title : str
    bins : int
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(sigmas, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(sigmas.mean(), color="red", linestyle="--",
               label=f"mean = {sigmas.mean():.3f}")
    ax.set_xlabel("σ")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

