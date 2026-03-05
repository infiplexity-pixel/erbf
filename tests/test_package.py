"""Tests for the top-level package."""

from __future__ import annotations

import erbf


class TestPackage:
    def test_version(self):
        assert erbf.__version__ == "0.1.1"

    def test_all_exports_exist(self):
        for name in erbf.__all__:
            assert hasattr(erbf, name), f"Missing export: {name}"

    def test_core_classes_importable(self):
        from erbf import ERBFClassifier, ERBFRegressor
        assert ERBFClassifier is not None
        assert ERBFRegressor is not None
