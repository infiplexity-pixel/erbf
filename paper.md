---
title: "ERBF: Exact Radial Basis Function Interpolation with Adaptive Local Bandwidths"
tags:
  - Python
  - machine learning
  - radial basis functions
  - kernel methods
  - interpolation
authors:
  - name: Ansh Mathur
    orcid: 0009-0009-7720-0733
    affiliation: 1
  - name: Supratik Dey
    orcid: 0009-0005-8878-6857
    affiliation: 1
  - name: Atrishman Mukherjee
    orcid: 0009-0008-7844-1419
    affiliation: 1
affiliations:
  - name: SRM Institute of Science and Technology, Chennai, Tamil Nadu, India
    index: 1
date: 12 February 2026
bibliography: paper.bib
---

# Summary

`ERBF` is an open-source Python library that implements exact radial basis function (RBF) interpolation for supervised learning tasks. The project provides a practical implementation of deterministic kernel interpolation with automatic local bandwidth selection and a scikit-learn-compatible interface. The source code is available at https://github.com/infiplexity-pixel/erbf.

Unlike many commonly used kernel methods that solve regularized optimization problems, exact RBF interpolation reconstructs the training data exactly. `ERBF` focuses on making this approach usable in modern machine-learning workflows by providing automatic bandwidth estimation, numerical stability safeguards, and integration with standard Python machine learning tooling.

The library implements a local bandwidth strategy in which each training sample receives its own kernel scale parameter based on the distance to its k-nearest neighbors. A simple heuristic automatically selects the neighborhood size as

\[
k = \max(10, \lfloor 1.5\sqrt{N} \rfloor)
\]

which avoids manual hyperparameter tuning in many practical settings. The resulting system constructs a kernel matrix using point-specific bandwidths and solves the interpolation problem directly.

# Statement of Need

Kernel methods are widely used in machine learning and scientific computing. Popular approaches such as support vector machines and kernel ridge regression approximate solutions through regularized optimization, while Gaussian processes provide probabilistic modeling but require computationally expensive hyperparameter estimation.

However, practitioners seeking deterministic **exact interpolation with radial basis functions** often encounter limited tooling in the Python ecosystem.

Existing implementations provide only partial support:

- **SciPy's `RBFInterpolator`** focuses primarily on regression with a global bandwidth parameter and does not support classification or scikit-learn integration.
- **Kernel ridge regression** approaches exact interpolation only as the regularization parameter approaches zero, which can lead to numerical instability.
- **Gaussian process implementations** offer noise-free interpolation but require costly marginal likelihood optimization and introduce additional modeling complexity.

These limitations make it difficult to use exact RBF interpolation in practical machine learning workflows.

Exact interpolation can be useful in several contexts. Small-sample learning problems may benefit from preserving all training information without regularization loss. In anomaly detection settings, perfect reconstruction of training data can simplify the identification of outliers through reconstruction error. Some scientific workflows involving irregularly sampled data may also benefit from locally adaptive kernel bandwidths that reflect variations in data density.

`ERBF` addresses this gap by providing an accessible implementation of exact RBF interpolation with adaptive local bandwidths and a familiar scikit-learn-style interface.

# Software Design

`ERBF` implements kernel interpolation for classification and regression. Given a dataset of \(N\) training samples, a kernel matrix is constructed as

\[
K_{ij} = \phi\left(\frac{\|x_i - x_j\|}{\sigma_j}\right)
\]

where \(\phi\) is a radial basis function kernel (Gaussian by default) and \(\sigma_j\) is a point-specific bandwidth parameter. The interpolation coefficients are obtained by solving the linear system

\[
K\alpha = y
\]

which ensures that the model reconstructs the training labels exactly.

Bandwidths are determined from local data density using the distance to the \(k\)-th nearest neighbor,

\[
\sigma_i = c \cdot d_k(x_i)
\]

where \(c\) is a scaling constant and \(d_k(x_i)\) denotes the distance to the \(k\)-th nearest neighbor of sample \(x_i\).

The implementation includes several features intended to support reliable use in practical workflows, including condition number diagnostics, numerical stability safeguards during kernel matrix inversion, and compatibility with common Python machine-learning interfaces.

# Software Quality and Availability

`ERBF` is developed as an open-source Python package with an emphasis on reproducibility and maintainability. The repository includes automated testing, continuous integration across multiple Python versions, and documentation with worked examples demonstrating typical workflows.

The library provides a simple API compatible with scikit-learn conventions:

```python
from erbf import ERBFClassifier

clf = ERBFClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```
The project is released under the MIT license and is intended to support both research and practical experimentation with exact RBF interpolation methods.
# References
