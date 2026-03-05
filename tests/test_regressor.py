"""Tests for the ERBF regressor."""

from __future__ import annotations

import numpy as np
import pytest

from erbf.datasets import make_regression_demo
from erbf.regressor import ERBFRegressor


@pytest.fixture
def regression_data():
    X, y = make_regression_demo(n_samples=50, n_features=1, noise=0.0, seed=42)
    return X, y


class TestERBFRegressor:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        reg = ERBFRegressor()
        result = reg.fit(X, y)
        assert result is reg

    def test_near_perfect_training(self, regression_data):
        X, y = regression_data
        reg = ERBFRegressor()
        reg.fit(X, y)
        r2 = reg.score(X, y)
        assert r2 > 0.99

    def test_predict_shape_1d(self, regression_data):
        X, y = regression_data
        reg = ERBFRegressor()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape

    def test_predict_shape_multi_target(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = rng.standard_normal((30, 3))
        reg = ERBFRegressor()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == (30, 3)

    def test_fitted_attributes(self, regression_data):
        X, y = regression_data
        reg = ERBFRegressor()
        reg.fit(X, y)
        assert reg.weights_ is not None
        assert reg.sigmas_ is not None
        assert reg.K_train_ is not None
        assert reg.condition_number_ > 0

    def test_not_fitted_raises(self):
        reg = ERBFRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            reg.predict(np.zeros((5, 2)))

    def test_get_params(self):
        reg = ERBFRegressor(lambda_reg=1e-8)
        params = reg.get_params()
        assert params["lambda_reg"] == 1e-8

    def test_set_params(self):
        reg = ERBFRegressor()
        reg.set_params(k_multiplier=3.0)
        assert reg.k_multiplier == 3.0

    def test_set_invalid_param(self):
        reg = ERBFRegressor()
        with pytest.raises(ValueError, match="Invalid parameter"):
            reg.set_params(nonexistent=42)

    def test_repr(self):
        reg = ERBFRegressor()
        assert "ERBFRegressor" in repr(reg)

    def test_verbose(self, regression_data, capsys):
        X, y = regression_data
        reg = ERBFRegressor(verbose=True)
        reg.fit(X, y)
        captured = capsys.readouterr()
        assert "[fit]" in captured.out
