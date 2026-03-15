"""
Evaluation metrics for ERBF models.

Uses PyTorch for kernel-related computations when CUDA is available.
"""

from __future__ import annotations

import numpy as np
import torch


def interpolation_error(
    K: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    y: np.ndarray,
    classes: np.ndarray,
) -> dict[int, float]:
    """Compute max interpolation error per class.

    Parameters
    ----------
    K : ndarray or Tensor of shape (N, N)
        Training kernel matrix.
    weights : ndarray or Tensor of shape (n_classes, N)
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
    if isinstance(K, np.ndarray):
        K = torch.as_tensor(K, dtype=torch.float64)
    if isinstance(weights, np.ndarray):
        weights = torch.as_tensor(weights, dtype=torch.float64)

    errors = {}
    for idx, c in enumerate(classes):
        target = torch.tensor(
            (y == c).astype(np.float64), dtype=torch.float64, device=K.device
        )
        reconstruction = K @ weights[idx]
        errors[int(c)] = float(torch.max(torch.abs(reconstruction - target)).item())
    return errors


def kernel_condition_number(K: np.ndarray | torch.Tensor) -> float:
    """Return the 2-norm condition number of kernel matrix *K*."""
    if isinstance(K, np.ndarray):
        K = torch.as_tensor(K, dtype=torch.float64)
    return float(torch.linalg.cond(K).item())


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray | None = None,
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
    classes: np.ndarray | None = None,
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
