# ERBF — Exact Radial Basis Function Interpolation Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for classification and regression using Exact Radial Basis Function (ERBF) interpolation with adaptive local bandwidth selection.

## Key Features

- **Guaranteed 100% training accuracy** via exact kernel interpolation
- **Automatic bandwidth (σ) selection** using k-NN adaptive local sigmas
- **Automatic k selection heuristic**: k ≈ 1.5√N
- **One-vs-all multiclass classification**
- **Continuous-valued regression**
- **Hyperparameter tuning utilities**
- **Visualization tools** for kernels, decision boundaries, and sigma distributions

## Installation

### From PyPI (when published)

```bash
pip install erbf
```

### From source

```bash
git clone https://github.com/erbf-contributors/erbf.git
cd erbf
pip install -e .
```

### With optional dependencies

```bash
# Install with progress bars (tqdm)
pip install erbf[progress]

# Install with visualization support (matplotlib)
pip install erbf[viz]

# Install with dataset utilities (scikit-learn for MNIST loading)
pip install erbf[datasets]

# Install everything
pip install erbf[all]

# Install development dependencies
pip install erbf[dev]
```

## Quick Start

### Classification

```python
from erbf import ERBFClassifier

# Create and fit the classifier
clf = ERBFClassifier(
    k_neighbors=None,  # auto-select k ≈ 1.5√N
    kernel="gaussian",
    verbose=True,
)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")
```

### Regression

```python
from erbf import ERBFRegressor

# Create and fit the regressor
reg = ERBFRegressor(
    k_neighbors=None,  # auto-select k
    kernel="gaussian",
)
reg.fit(X_train, y_train)

# Predict
predictions = reg.predict(X_test)

# Evaluate
r2_score = reg.score(X_test, y_test)
print(f"R² score: {r2_score:.3f}")
```

### Using different kernels

```python
from erbf import ERBFClassifier, MultiquadricKernel

# Use a different kernel
clf = ERBFClassifier(kernel=MultiquadricKernel())
clf.fit(X_train, y_train)
```

### Cross-validation and hyperparameter tuning

```python
from erbf import ERBFClassifier, ERBFGridSearchCV

param_grid = {
    "k_neighbors": [10, 20, 50],
    "lambda_reg": [1e-12, 1e-10, 1e-8],
}

grid_search = ERBFGridSearchCV(
    ERBFClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy",
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.2%}")
```

## API Reference

### Core Classes

- `ERBFClassifier` - Multiclass classifier using exact RBF interpolation
- `ERBFRegressor` - Regressor using exact RBF interpolation

### Kernels

- `GaussianKernel` - Standard Gaussian RBF kernel
- `MultiquadricKernel` - Multiquadric kernel
- `InverseMultiquadricKernel` - Inverse multiquadric kernel
- `ThinPlateSplineKernel` - Thin plate spline kernel

### Sigma Computation

- `compute_local_sigmas()` - Compute adaptive local bandwidths
- `compute_global_sigma()` - Compute global bandwidth
- `auto_select_k()` - Automatic k selection heuristic

### Model Selection

- `ERBFGridSearchCV` - Grid search with cross-validation
- `cross_validate()` - K-fold cross-validation

### Metrics

- `interpolation_error()` - Compute interpolation errors
- `kernel_condition_number()` - Compute kernel matrix condition number
- `per_class_accuracy()` - Per-class accuracy breakdown
- `classification_report()` - Detailed classification report

### Datasets

- `load_mnist_subset()` - Load a subset of MNIST
- `make_classification_demo()` - Generate synthetic classification data
- `make_regression_demo()` - Generate synthetic regression data

### Visualization

- `plot_kernel_matrix()` - Visualize the kernel matrix
- `plot_sigma_distribution()` - Visualize sigma distribution
- `plot_decision_boundary()` - Plot 2D decision boundaries
- `plot_interpolation_error()` - Visualize interpolation errors

## How It Works

ERBF uses exact radial basis function interpolation to solve classification and regression problems. Unlike traditional approximate methods, ERBF solves the linear system `K · w = y` exactly, guaranteeing 100% training accuracy.

The key innovation is **adaptive local bandwidth selection**, where each training point has its own σ (bandwidth) computed from its k nearest neighbors. This allows the model to adapt to varying data density across the feature space.

### Algorithm Overview

1. **Sigma Selection**: For each training point, compute σᵢ as the mean distance to its k nearest neighbors
2. **Kernel Matrix**: Build the kernel matrix K where K[i,j] = φ(||xᵢ - xⱼ|| / σᵢⱼ)
3. **System Solving**: Solve K · w = y exactly (with small regularization for numerical stability)
4. **Prediction**: For new points, compute kernel similarities to all training points and combine with learned weights

## Citation

If you use ERBF in your research, please cite:

```bibtex
@software{erbf2024,
  title = {ERBF: Exact Radial Basis Function Interpolation Library},
  author = {ERBF Contributors},
  year = {2024},
  url = {https://github.com/erbf-contributors/erbf}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
