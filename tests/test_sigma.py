"""Tests for adaptive bandwidth (sigma) computation."""

from __future__ import annotations

import numpy as np
import pytest

from erbf.sigma import auto_select_k, compute_global_sigma, compute_local_sigmas


class TestAutoSelectK:
    def test_basic(self):
        k = auto_select_k(100)
        assert isinstance(k, int)
        assert k >= 10

    def test_minimum_applied(self):
        k = auto_select_k(4, minimum=10)
        assert k == 10

    def test_multiplier(self):
        k = auto_select_k(100, multiplier=2.0, minimum=1)
        assert k == int(np.sqrt(100) * 2.0)

    def test_small_N(self):
        k = auto_select_k(1, minimum=5)
        assert k == 5


class TestComputeLocalSigmas:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((50, 3))

    def test_shape(self, data):
        sigmas = compute_local_sigmas(data)
        assert sigmas.shape == (50,)

    def test_positive(self, data):
        sigmas = compute_local_sigmas(data)
        assert np.all(sigmas > 0)

    def test_clipping(self, data):
        sigmas = compute_local_sigmas(data, min_sigma=1.0, max_sigma=5.0)
        assert np.all(sigmas >= 1.0)
        assert np.all(sigmas <= 5.0)

    def test_explicit_k(self, data):
        sigmas = compute_local_sigmas(data, k_neighbors=5)
        assert sigmas.shape == (50,)

    def test_chunked_matches_unchunked(self, data):
        sigmas_small_chunk = compute_local_sigmas(data, k_neighbors=5, chunk_size=10)
        sigmas_large_chunk = compute_local_sigmas(data, k_neighbors=5, chunk_size=100)
        np.testing.assert_allclose(sigmas_small_chunk, sigmas_large_chunk, atol=1e-12)

    def test_verbose(self, data, capsys):
        compute_local_sigmas(data, verbose=True)
        captured = capsys.readouterr()
        assert "[sigma]" in captured.out


class TestComputeGlobalSigma:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((30, 2))

    def test_median(self, data):
        sigma = compute_global_sigma(data, method="median")
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_mean(self, data):
        sigma = compute_global_sigma(data, method="mean")
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_max(self, data):
        sigma = compute_global_sigma(data, method="max")
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_invalid_method(self, data):
        with pytest.raises(ValueError, match="Unknown method"):
            compute_global_sigma(data, method="invalid")

    def test_max_is_largest(self, data):
        s_mean = compute_global_sigma(data, method="mean")
        s_max = compute_global_sigma(data, method="max")
        assert s_max >= s_mean
