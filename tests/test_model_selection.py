"""Tests for model selection utilities."""

from __future__ import annotations

import numpy as np
import pytest

from erbf.classifier import ERBFClassifier
from erbf.datasets import make_classification_demo
from erbf.model_selection import ERBFGridSearchCV, cross_validate


@pytest.fixture
def small_data():
    X, y = make_classification_demo(n_samples=60, n_features=2, n_classes=3, seed=42)
    return X, y


class TestCrossValidate:
    def test_returns_expected_keys(self, small_data):
        X, y = small_data
        clf = ERBFClassifier()
        results = cross_validate(clf, X, y, n_folds=3)
        assert "train_scores" in results
        assert "test_scores" in results
        assert "mean_train" in results
        assert "mean_test" in results
        assert "std_test" in results

    def test_correct_number_of_folds(self, small_data):
        X, y = small_data
        clf = ERBFClassifier()
        results = cross_validate(clf, X, y, n_folds=3)
        assert len(results["train_scores"]) == 3
        assert len(results["test_scores"]) == 3

    def test_train_score_is_perfect(self, small_data):
        X, y = small_data
        clf = ERBFClassifier()
        results = cross_validate(clf, X, y, n_folds=3)
        # ERBF guarantees 100% training accuracy
        np.testing.assert_allclose(results["train_scores"], 1.0, atol=1e-10)


class TestERBFGridSearchCV:
    def test_basic_grid_search(self, small_data):
        X, y = small_data
        clf = ERBFClassifier()
        grid = ERBFGridSearchCV(
            clf,
            param_grid={"k_multiplier": [1.0, 2.0]},
            n_folds=2,
        )
        grid.fit(X, y)
        assert grid.best_params_ is not None
        assert grid.best_score_ > 0
        assert grid.best_estimator_ is not None

    def test_predict_after_fit(self, small_data):
        X, y = small_data
        clf = ERBFClassifier()
        grid = ERBFGridSearchCV(
            clf,
            param_grid={"k_multiplier": [1.5]},
            n_folds=2,
        )
        grid.fit(X, y)
        y_pred = grid.predict(X)
        assert y_pred.shape == y.shape

    def test_not_fitted_raises(self, small_data):
        X, y = small_data
        clf = ERBFClassifier()
        grid = ERBFGridSearchCV(clf, param_grid={"k_multiplier": [1.5]})
        with pytest.raises(RuntimeError, match="not been fitted"):
            grid.predict(X)

    def test_cv_results_populated(self, small_data):
        X, y = small_data
        clf = ERBFClassifier()
        grid = ERBFGridSearchCV(
            clf,
            param_grid={"k_multiplier": [1.0, 1.5]},
            n_folds=2,
        )
        grid.fit(X, y)
        assert len(grid.cv_results_) == 2
