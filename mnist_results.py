"""
MNIST Classification with ERBF
===============================

This example reproduces the original optimized_erbf.py results using
the library's clean API. It demonstrates:

  - Loading a subset of MNIST
  - Fitting the ERBFClassifier with automatic k selection
  - Verifying 100% training accuracy (guaranteed by exact interpolation)
  - Evaluating test accuracy with per-class breakdown
  - Inspecting kernel diagnostics (condition number, interpolation errors)
"""

from erbf import (
    ERBFClassifier,
    load_mnist_subset,
    classification_report,
    per_class_accuracy,
    plot_kernel_matrix,
    plot_sigma_distribution,
)
from erbf.utils import timer

# ── 1.  Load data ──────────────────────────────────────────────────
print("=" * 65)
print("  ERBF Library  ·  MNIST Classification Example")
print("=" * 65)

X_train, y_train, X_test, y_test = load_mnist_subset(
    n_train=5_000, n_test=200, seed=42
)
print(f"\nData loaded: train {X_train.shape}, test {X_test.shape}")
print(f"Label distribution: {dict(zip(*__import__('numpy').unique(y_train, return_counts=True)))}")

# ── 2.  Create and fit the classifier ─────────────────────────────
#
#   k_neighbors=None  →  automatic selection via k ≈ 1.5·√N
#   kernel="gaussian"  →  Gaussian RBF (default)
#   verbose=True       →  print diagnostics during fit
#   chunk_size=500     →  process 500 samples at a time (constant memory)
#   show_progress=True →  show progress bars during sigma computation & prediction
#
clf = ERBFClassifier(
    k_neighbors=1_000_000,       # auto: will pick k = max(10, int(1.5·√200)) = 21
    P=2,                    # squared-Euclidean exponent
    lambda_reg=1e-10,       # tiny Tikhonov regularisation
    kernel="gaussian",
    chunk_size=500,         # keep memory constant regardless of centers
    show_progress=True,     # show loading bars
    verbose=True,
)

print("\n─── Training ───")
with timer("Fit time"):
    clf.fit(X_train, y_train)

# ── 3.  Inspect kernel diagnostics ────────────────────────────────
print(f"\nKernel condition number : {clf.condition_number_:.2e}")
print(f"σ range                : [{clf.sigmas_.min():.3f}, {clf.sigmas_.max():.3f}]")
print(f"σ mean                 : {clf.sigmas_.mean():.3f}")
print("Max interpolation error per class:")
for c, err in sorted(clf.interpolation_errors_.items()):
    print(f"  Class {c}: {err:.2e}")

# ── 4.  Training accuracy (must be 100%) ──────────────────────────
print("\n─── Training Evaluation ───")
train_acc = clf.score(X_train, y_train)
print(f"Training accuracy: {train_acc:.2%}  (expected: 100.00%)")

train_details = per_class_accuracy(y_train, clf.predict(X_train))
all_perfect = all(v["accuracy"] == 1.0 for v in train_details.values())
print(f"All classes perfect: {'✅ YES' if all_perfect else '❌ NO'}")

# ── 5.  Test evaluation ───────────────────────────────────────────
print("\n─── Test Evaluation ───")
y_pred = clf.predict(X_test)
test_acc = clf.score(X_test, y_test)
print(f"Test accuracy: {test_acc:.2%}\n")

print(classification_report(y_test, y_pred))

# ── 6.  Probability estimates ─────────────────────────────────────
print("\n─── Sample Predictions with Confidence ───")
proba = clf.predict_proba(X_test[:5])
for i in range(5):
    pred = y_pred[i]
    conf = proba[i, list(clf.classes_).index(pred)]
    true = y_test[i]
    status = "✓" if pred == true else "✗"
    print(f"  Sample {i}: true={true}, pred={pred}, "
          f"confidence={conf:.3f}  {status}")

# ── 7.  Optional: save visualisations ─────────────────────────────
try:
    fig_k = plot_kernel_matrix(clf.K_train_, title="MNIST ERBF Kernel (N=200)")
    fig_k.savefig("kernel_matrix.png", dpi=150)
    print("\n📊 Saved kernel_matrix.png")

    fig_s = plot_sigma_distribution(clf.sigmas_)
    fig_s.savefig("sigma_distribution.png", dpi=150)
    print("📊 Saved sigma_distribution.png")
except ImportError:
    print("\n(Install matplotlib to generate plots)")

print("\n" + "=" * 65)
print("Done.")