"""Tests for utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from erbf.utils import check_array, normalize, train_test_split


class TestNormalize:
    def test_minmax(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_norm = normalize(X, method="minmax")
        np.testing.assert_allclose(X_norm.min(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(X_norm.max(axis=0), 1.0, atol=1e-12)

    def test_zscore(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_norm = normalize(X, method="zscore")
        np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=1e-12)

    def test_constant_column(self):
        X = np.array([[1.0, 5.0], [1.0, 10.0], [1.0, 15.0]])
        X_norm = normalize(X, method="minmax")
        assert np.all(np.isfinite(X_norm))

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown normalisation"):
            normalize(np.ones((3, 2)), method="invalid")


class TestTrainTestSplit:
    def test_shapes(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        assert X_tr.shape[0] + X_te.shape[0] == 50
        assert y_tr.shape[0] + y_te.shape[0] == 50

    def test_no_overlap(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)
        all_y = np.concatenate([y_tr, y_te])
        assert len(set(all_y)) == 10

    def test_deterministic(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        X_tr1, X_te1, _, _ = train_test_split(X, y, seed=42)
        X_tr2, X_te2, _, _ = train_test_split(X, y, seed=42)
        np.testing.assert_array_equal(X_tr1, X_tr2)
        np.testing.assert_array_equal(X_te1, X_te2)


class TestCheckArray:
    def test_converts_to_float64(self):
        X = np.array([[1, 2], [3, 4]])
        result = check_array(X)
        assert result.dtype == np.float64

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            check_array(np.array([1, 2, 3]))

    def test_rejects_nan(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="contains NaN"):
            check_array(X)

    def test_rejects_inf(self):
        X = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(ValueError, match="contains infinite"):
            check_array(X)
