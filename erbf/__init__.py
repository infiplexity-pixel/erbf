"""
ERBF — Exact Radial Basis Function Interpolation Library
=========================================================

A Python library for classification and regression using Exact Radial Basis
Function (ERBF) interpolation with adaptive local bandwidth selection.

Key Features:
    - Guaranteed 100% training accuracy via exact kernel interpolation
    - Automatic bandwidth (σ) selection using k-NN adaptive local sigmas
    - Automatic k selection heuristic: k ≈ 1.5√N
    - One-vs-all multiclass classification
    - Continuous-valued regression
    - Hyperparameter tuning utilities
    - Visualization tools for kernels, decision boundaries, and sigma distributions

Quick Start::

    from erbf import ERBFClassifier
    clf = ERBFClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

References:
    This library implements the ERBF method described in:
    "Optimized ERBF: Automatic k Selection for Perfect Training Accuracy"
"""

__version__ = "0.1.0"
__author__ = "ERBF Contributors"

from erbf.classifier import ERBFClassifier
from erbf.regressor import ERBFRegressor
from erbf.kernel import (
    GaussianKernel,
    MultiquadricKernel,
    InverseMultiquadricKernel,
    ThinPlateSplineKernel,
    build_kernel_matrix,
    chunked_kernel_matmul,
)
from erbf.sigma import (
    compute_local_sigmas,
    compute_global_sigma,
    auto_select_k,
)
from erbf.model_selection import ERBFGridSearchCV, cross_validate
from erbf.metrics import (
    interpolation_error,
    kernel_condition_number,
    per_class_accuracy,
    classification_report,
)
from erbf.datasets import load_mnist_subset, make_classification_demo, make_regression_demo
from erbf.visualization import (
    plot_kernel_matrix,
    plot_sigma_distribution
)

__all__ = [
    # Core estimators
    "ERBFClassifier",
    "ERBFRegressor",
    # Kernels
    "GaussianKernel",
    "MultiquadricKernel",
    "InverseMultiquadricKernel",
    "ThinPlateSplineKernel",
    "build_kernel_matrix",
    "chunked_kernel_matmul",
    # Sigma computation
    "compute_local_sigmas",
    "compute_global_sigma",
    "auto_select_k",
    # Model selection
    "ERBFGridSearchCV",
    "cross_validate",
    # Metrics
    "interpolation_error",
    "kernel_condition_number",
    "per_class_accuracy",
    "classification_report",
    # Datasets
    "load_mnist_subset",
    "make_classification_demo",
    "make_regression_demo",
    # Visualization
    "plot_kernel_matrix",
    "plot_sigma_distribution",
    "plot_decision_boundary",
    "plot_interpolation_error",
]