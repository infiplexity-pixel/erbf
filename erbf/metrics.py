"""
Evaluation metrics for ERBF models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from erbf.kernel import build_kernel_matrix


def interpolation_error(
    K: np.ndarray,
    weights: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> dict[int, float]:
    """Compute max interpolation error per class.

    Parameters
    ----------
    K : ndarray of shape (N, N)
        Training kernel matrix.
    weights : ndarray of shape (n_classes, N)
        Class interpolation weights.
    y : ndarray of shape (N,)
        True labels.
    classes : ndarray
        Unique class labels.

    Returns
    -------
    errors : dict[int, float]
        Max |K·w_c - I_c| for each class *c*.
    """
    errors = {}
    for idx, c in enumerate(classes):
        target = (y == c).astype(np.float64)
        reconstruction = K @ weights[idx]
        errors[int(c)] = float(np.max(np.abs(reconstruction - target)))
    return errors


def kernel_condition_number(K: np.ndarray) -> float:
    """Return the 2-norm condition number of kernel matrix *K*."""
    return float(np.linalg.cond(K))


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[np.ndarray] = None,
) -> dict[int, dict[str, float | int]]:
    """Per-class accuracy breakdown.

    Parameters
    ----------
    y_true : ndarray
    y_pred : ndarray
    classes : ndarray or None

    Returns
    -------
    report : dict
        ``{class_label: {"accuracy": float, "correct": int, "total": int}}``
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if classes is None:
        classes = np.unique(y_true)

    report = {}
    for c in classes:
        mask = y_true == c
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int((y_pred[mask] == c).sum())
        report[int(c)] = {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
        }
    return report


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[np.ndarray] = None,
    *,
    return_string: bool = True,
) -> str | dict:
    """Generate a classification report similar to sklearn's.

    Parameters
    ----------
    y_true, y_pred : ndarray
    classes : ndarray or None
    return_string : bool, default=True
        If ``True``, returns a formatted string.

    Returns
    -------
    report : str or dict
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))

    rows = []
    total_correct = 0
    total_count = 0

    for c in classes:
        mask_true = y_true == c
        mask_pred = y_pred == c

        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int(mask_true.sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        rows.append({
            "class": int(c),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
        total_correct += tp
        total_count += support

    overall_acc = total_correct / total_count if total_count > 0 else 0.0

    if not return_string:
        return {"per_class": rows, "accuracy": overall_acc, "total": total_count}

    lines = [
        f"{'Class':>8s}  {'Precision':>9s}  {'Recall':>6s}  {'F1':>6s}  {'Support':>7s}",
        "-" * 48,
    ]
    for r in rows:
        lines.append(
            f"{r['class']:>8d}  {r['precision']:>9.4f}  {r['recall']:>6.4f}  "
            f"{r['f1']:>6.4f}  {r['support']:>7d}"
        )
    lines.append("-" * 48)
    lines.append(f"{'Accuracy':>8s}  {' ':>9s}  {' ':>6s}  {overall_acc:>6.4f}  {total_count:>7d}")

    return "\n".join(lines)