"""
Model selection utilities: grid search and cross-validation.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional

import numpy as np

from erbf.classifier import ERBFClassifier
from erbf.regressor import ERBFRegressor


def cross_validate(
    estimator: ERBFClassifier | ERBFRegressor,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """K-fold cross-validation for an ERBF estimator.

    Parameters
    ----------
    estimator : ERBFClassifier or ERBFRegressor
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    n_folds : int, default=5
    shuffle : bool, default=True
    seed : int, default=42
    verbose : bool, default=False

    Returns
    -------
    results : dict
        Keys: ``"train_scores"``, ``"test_scores"``, ``"mean_train"``,
        ``"mean_test"``, ``"std_test"``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    N = X.shape[0]

    indices = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    folds = np.array_split(indices, n_folds)
    train_scores = []
    test_scores = []

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # Clone estimator with same params
        params = estimator.get_params()
        est = estimator.__class__(**params)
        est.fit(X_tr, y_tr)

        tr_score = est.score(X_tr, y_tr)
        te_score = est.score(X_te, y_te)
        train_scores.append(tr_score)
        test_scores.append(te_score)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds}: "
                  f"train={tr_score:.4f}, test={te_score:.4f}")

    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)

    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "mean_train": float(train_scores.mean()),
        "mean_test": float(test_scores.mean()),
        "std_test": float(test_scores.std()),
    }


class ERBFGridSearchCV:
    """Exhaustive grid search with cross-validation for ERBF estimators.

    Parameters
    ----------
    estimator : ERBFClassifier or ERBFRegressor
    param_grid : dict[str, list]
        Dictionary mapping parameter names to lists of values to try.
    n_folds : int, default=5
    verbose : bool, default=False

    Attributes
    ----------
    best_params_ : dict
    best_score_ : float
    cv_results_ : list[dict]

    Examples
    --------
    >>> from erbf import ERBFClassifier, ERBFGridSearchCV
    >>> clf = ERBFClassifier()
    >>> grid = ERBFGridSearchCV(
    ...     clf,
    ...     param_grid={
    ...         "k_multiplier": [1.0, 1.5, 2.0],
    ...         "lambda_reg": [1e-12, 1e-10, 1e-8],
    ...     },
    ...     n_folds=3,
    ...     verbose=True,
    ... )
    >>> grid.fit(X_train, y_train)
    >>> print(grid.best_params_)
    """

    def __init__(
        self,
        estimator: ERBFClassifier | ERBFRegressor,
        param_grid: dict[str, list],
        *,
        n_folds: int = 5,
        verbose: bool = False,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_folds = n_folds
        self.verbose = verbose

        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = -np.inf
        self.best_estimator_: Optional[ERBFClassifier | ERBFRegressor] = None
        self.cv_results_: list[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ERBFGridSearchCV":
        """Run grid search.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))

        if self.verbose:
            print(f"[GridSearch] {len(combinations)} parameter combinations, "
                  f"{self.n_folds}-fold CV")

        for combo in combinations:
            params = dict(zip(keys, combo))

            est = self.estimator.__class__(**self.estimator.get_params())
            est.set_params(**params)

            results = cross_validate(
                est, X, y, n_folds=self.n_folds, verbose=False,
            )

            self.cv_results_.append({
                "params": params,
                "mean_test_score": results["mean_test"],
                "std_test_score": results["std_test"],
                "mean_train_score": results["mean_train"],
            })

            if self.verbose:
                print(f"  {params} → test={results['mean_test']:.4f} "
                      f"± {results['std_test']:.4f}")

            if results["mean_test"] > self.best_score_:
                self.best_score_ = results["mean_test"]
                self.best_params_ = params

        # Refit on full dataset with best params
        self.best_estimator_ = self.estimator.__class__(
            **{**self.estimator.get_params(), **self.best_params_}
        )
        self.best_estimator_.fit(X, y)

        if self.verbose:
            print(f"\n[GridSearch] Best: {self.best_params_} "
                  f"(score={self.best_score_:.4f})")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best estimator."""
        if self.best_estimator_ is None:
            raise RuntimeError("Grid search has not been fitted.")
        return self.best_estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score using the best estimator."""
        if self.best_estimator_ is None:
            raise RuntimeError("Grid search has not been fitted.")
        return self.best_estimator_.score(X, y)