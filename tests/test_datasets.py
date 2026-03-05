"""Tests for dataset utilities."""

from __future__ import annotations

import numpy as np

from erbf.datasets import make_classification_demo, make_regression_demo


class TestMakeClassificationDemo:
    def test_shape(self):
        X, y = make_classification_demo(n_samples=60, n_features=3, n_classes=3)
        assert X.shape == (60, 3)
        assert y.shape == (60,)

    def test_num_classes(self):
        X, y = make_classification_demo(n_samples=60, n_classes=3)
        assert len(np.unique(y)) == 3

    def test_deterministic(self):
        X1, y1 = make_classification_demo(seed=42)
        X2, y2 = make_classification_demo(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds(self):
        X1, _ = make_classification_demo(seed=42)
        X2, _ = make_classification_demo(seed=99)
        assert not np.array_equal(X1, X2)


class TestMakeRegressionDemo:
    def test_shape(self):
        X, y = make_regression_demo(n_samples=100, n_features=2)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_deterministic(self):
        X1, y1 = make_regression_demo(seed=42)
        X2, y2 = make_regression_demo(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_no_noise(self):
        X, y = make_regression_demo(n_features=1, noise=0.0, seed=42)
        expected = np.sin(2 * np.pi * X).ravel()
        np.testing.assert_allclose(y, expected, atol=1e-12)
