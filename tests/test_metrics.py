"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from erbf.metrics import (
    classification_report,
    interpolation_error,
    kernel_condition_number,
    per_class_accuracy,
)


class TestInterpolationError:
    def test_perfect_interpolation(self):
        # Class 0 has samples at indices 0,1; class 1 at indices 2,3
        y = np.array([0, 0, 1, 1])
        classes = np.array([0, 1])
        # Weights that perfectly reconstruct the indicator vectors with identity K
        # For class 0: target = [1,1,0,0], so weights[0] = [1,1,0,0]
        # For class 1: target = [0,0,1,1], so weights[1] = [0,0,1,1]
        K = np.eye(4)
        weights = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float64)
        errors = interpolation_error(K, weights, y, classes)
        assert all(err < 1e-10 for err in errors.values())

    def test_returns_dict(self):
        K = np.eye(4)
        weights = np.eye(2, 4)
        y = np.array([0, 0, 1, 1])
        classes = np.array([0, 1])
        errors = interpolation_error(K, weights, y, classes)
        assert isinstance(errors, dict)
        assert set(errors.keys()) == {0, 1}


class TestKernelConditionNumber:
    def test_identity(self):
        K = np.eye(10)
        cond = kernel_condition_number(K)
        np.testing.assert_allclose(cond, 1.0, atol=1e-10)

    def test_returns_float(self):
        rng = np.random.default_rng(42)
        K = rng.standard_normal((5, 5))
        K = K @ K.T + np.eye(5)
        cond = kernel_condition_number(K)
        assert isinstance(cond, float)
        assert cond >= 1.0


class TestPerClassAccuracy:
    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        report = per_class_accuracy(y_true, y_pred)
        for info in report.values():
            assert info["accuracy"] == 1.0

    def test_partial(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        report = per_class_accuracy(y_true, y_pred)
        assert report[0]["accuracy"] == 0.5
        assert report[1]["accuracy"] == 1.0

    def test_custom_classes(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        report = per_class_accuracy(y_true, y_pred, classes=np.array([0, 1, 2]))
        assert 0 in report
        assert 1 in report
        assert 2 not in report  # no samples for class 2


class TestClassificationReport:
    def test_returns_string(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = classification_report(y_true, y_pred)
        assert isinstance(result, str)
        assert "Accuracy" in result

    def test_returns_dict(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = classification_report(y_true, y_pred, return_string=False)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert result["accuracy"] == 1.0
