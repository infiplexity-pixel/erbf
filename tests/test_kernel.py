"""Tests for ERBF kernel functions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from erbf.kernel import (
    GaussianKernel,
    InverseMultiquadricKernel,
    MultiquadricKernel,
    ThinPlateSplineKernel,
    build_kernel_matrix,
    chunked_kernel_matmul,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_data(rng):
    X = torch.tensor(rng.standard_normal((20, 3)), dtype=torch.float64)
    sigmas = torch.ones(20, dtype=torch.float64)
    return X, sigmas


class TestGaussianKernel:
    def test_self_similarity_is_one(self, simple_data):
        X, sigmas = simple_data
        kernel = GaussianKernel()
        K = kernel(X, X, sigmas, sigmas, symmetric=True)
        # Diagonal should be 1.0 + lambda_reg
        torch.testing.assert_close(
            K.diag(), torch.full((20,), 1.0 + 1e-10, dtype=torch.float64), atol=1e-12, rtol=0
        )

    def test_symmetric(self, simple_data):
        X, sigmas = simple_data
        kernel = GaussianKernel()
        K = kernel(X, X, sigmas, sigmas, symmetric=True)
        torch.testing.assert_close(K, K.T, atol=1e-12, rtol=0)

    def test_values_in_range(self, simple_data):
        X, sigmas = simple_data
        kernel = GaussianKernel()
        K = kernel(X, X, sigmas, sigmas, symmetric=False)
        assert torch.all(K >= 0)
        assert torch.all(K <= 1.0 + 1e-8)

    def test_shape(self, rng):
        X = torch.tensor(rng.standard_normal((10, 5)), dtype=torch.float64)
        Y = torch.tensor(rng.standard_normal((15, 5)), dtype=torch.float64)
        sigmas_x = torch.ones(10, dtype=torch.float64)
        sigmas_y = torch.ones(15, dtype=torch.float64)
        kernel = GaussianKernel()
        K = kernel(X, Y, sigmas_x, sigmas_y)
        assert K.shape == (10, 15)


class TestMultiquadricKernel:
    def test_minimum_value_is_one(self, simple_data):
        X, sigmas = simple_data
        kernel = MultiquadricKernel()
        K = kernel(X, X, sigmas, sigmas, symmetric=False)
        assert torch.all(K >= 1.0)

    def test_symmetric(self, simple_data):
        X, sigmas = simple_data
        kernel = MultiquadricKernel()
        K = kernel(X, X, sigmas, sigmas, symmetric=True)
        torch.testing.assert_close(K, K.T, atol=1e-12, rtol=0)


class TestInverseMultiquadricKernel:
    def test_values_in_range(self, simple_data):
        X, sigmas = simple_data
        kernel = InverseMultiquadricKernel()
        K = kernel(X, X, sigmas, sigmas, symmetric=False)
        assert torch.all(K > 0)
        assert torch.all(K <= 1.0)

    def test_symmetric(self, simple_data):
        X, sigmas = simple_data
        kernel = InverseMultiquadricKernel()
        K = kernel(X, X, sigmas, sigmas, symmetric=True)
        torch.testing.assert_close(K, K.T, atol=1e-12, rtol=0)


class TestThinPlateSplineKernel:
    def test_shape(self, simple_data):
        X, sigmas = simple_data
        kernel = ThinPlateSplineKernel()
        K = kernel(X, X, sigmas, sigmas, symmetric=False)
        assert K.shape == (20, 20)


class TestBaseKernel:
    def test_invalid_P(self):
        with pytest.raises(ValueError, match="P must be >= 1"):
            GaussianKernel(P=0)

    def test_invalid_lambda_reg(self):
        with pytest.raises(ValueError, match="lambda_reg must be >= 0"):
            GaussianKernel(lambda_reg=-1.0)

    def test_repr(self):
        k = GaussianKernel(P=3, lambda_reg=0.01)
        assert "GaussianKernel" in repr(k)
        assert "P=3" in repr(k)


class TestBuildKernelMatrix:
    def test_string_kernel(self, simple_data):
        X, sigmas = simple_data
        K = build_kernel_matrix(X, X, sigmas, sigmas, kernel="gaussian", symmetric=True)
        assert K.shape == (20, 20)

    def test_kernel_object(self, simple_data):
        X, sigmas = simple_data
        kernel = GaussianKernel()
        K = build_kernel_matrix(X, X, sigmas, sigmas, kernel=kernel, symmetric=True)
        assert K.shape == (20, 20)

    def test_unknown_kernel_string(self, simple_data):
        X, sigmas = simple_data
        with pytest.raises(ValueError, match="Unknown kernel"):
            build_kernel_matrix(X, X, sigmas, sigmas, kernel="nonexistent")

    def test_invalid_kernel_type(self, simple_data):
        X, sigmas = simple_data
        with pytest.raises(TypeError, match="kernel must be str or BaseKernel"):
            build_kernel_matrix(X, X, sigmas, sigmas, kernel=42)

    def test_all_kernel_names(self, simple_data):
        X, sigmas = simple_data
        for name in ["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline"]:
            K = build_kernel_matrix(X, X, sigmas, sigmas, kernel=name, symmetric=True)
            assert K.shape == (20, 20)


class TestChunkedKernelMatmul:
    def test_matches_direct_computation(self, rng):
        X = torch.tensor(rng.standard_normal((30, 4)), dtype=torch.float64)
        Y = torch.tensor(rng.standard_normal((20, 4)), dtype=torch.float64)
        sigmas_x = torch.ones(30, dtype=torch.float64)
        sigmas_y = torch.ones(20, dtype=torch.float64)
        weights = torch.tensor(rng.standard_normal((3, 20)), dtype=torch.float64)

        # Direct computation
        K = build_kernel_matrix(X, Y, sigmas_x, sigmas_y, kernel="gaussian")
        expected = K @ weights.T

        # Chunked computation
        result = chunked_kernel_matmul(
            X, Y, sigmas_x, sigmas_y, weights,
            kernel="gaussian", chunk_size=10,
        )
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=0)

    def test_single_weight_vector(self, rng):
        X = torch.tensor(rng.standard_normal((10, 3)), dtype=torch.float64)
        Y = torch.tensor(rng.standard_normal((8, 3)), dtype=torch.float64)
        sigmas_x = torch.ones(10, dtype=torch.float64)
        sigmas_y = torch.ones(8, dtype=torch.float64)
        weights = torch.tensor(rng.standard_normal(8), dtype=torch.float64)

        result = chunked_kernel_matmul(
            X, Y, sigmas_x, sigmas_y, weights,
            kernel="gaussian", chunk_size=5,
        )
        assert result.ndim == 1
        assert result.shape == (10,)
