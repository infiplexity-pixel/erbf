"""Tests for the ERBF classifier."""

from __future__ import annotations

import numpy as np
import pytest

from erbf.classifier import ERBFClassifier
from erbf.datasets import make_classification_demo


@pytest.fixture
def classification_data():
    X, y = make_classification_demo(n_samples=60, n_features=2, n_classes=3, seed=42)
    return X, y


class TestERBFClassifier:
    def test_fit_returns_self(self, classification_data):
        X, y = classification_data
        clf = ERBFClassifier()
        result = clf.fit(X, y)
        assert result is clf

    def test_perfect_training_accuracy(self, classification_data):
        X, y = classification_data
        clf = ERBFClassifier()
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert acc == 1.0

    def test_predict_shape(self, classification_data):
        X, y = classification_data
        clf = ERBFClassifier()
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape

    def test_predict_proba_shape(self, classification_data):
        X, y = classification_data
        clf = ERBFClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_non_negative(self, classification_data):
        X, y = classification_data
        clf = ERBFClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all(proba >= 0)

    def test_decision_function_shape(self, classification_data):
        X, y = classification_data
        clf = ERBFClassifier()
        clf.fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(y), 3)

    def test_classes_discovered(self, classification_data):
        X, y = classification_data
        clf = ERBFClassifier()
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1, 2]))

    def test_fitted_attributes(self, classification_data):
        X, y = classification_data
        clf = ERBFClassifier()
        clf.fit(X, y)
        assert clf.weights_ is not None
        assert clf.sigmas_ is not None
        assert clf.K_train_ is not None
        assert clf.condition_number_ > 0
        assert len(clf.interpolation_errors_) == 3

    def test_not_fitted_raises(self):
        clf = ERBFClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(np.zeros((5, 2)))

    def test_get_params(self):
        clf = ERBFClassifier(k_multiplier=2.0, P=3)
        params = clf.get_params()
        assert params["k_multiplier"] == 2.0
        assert params["P"] == 3

    def test_set_params(self):
        clf = ERBFClassifier()
        clf.set_params(k_multiplier=3.0)
        assert clf.k_multiplier == 3.0

    def test_set_invalid_param(self):
        clf = ERBFClassifier()
        with pytest.raises(ValueError, match="Invalid parameter"):
            clf.set_params(nonexistent_param=42)

    def test_repr(self):
        clf = ERBFClassifier()
        assert "ERBFClassifier" in repr(clf)

    def test_verbose(self, classification_data, capsys):
        X, y = classification_data
        clf = ERBFClassifier(verbose=True)
        clf.fit(X, y)
        captured = capsys.readouterr()
        assert "[fit]" in captured.out

    def test_different_kernels(self, classification_data):
        X, y = classification_data
        for kernel_name in ["gaussian", "multiquadric", "inverse_multiquadric"]:
            clf = ERBFClassifier(kernel=kernel_name)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            assert y_pred.shape == y.shape
