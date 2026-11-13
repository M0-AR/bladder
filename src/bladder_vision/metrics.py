"""Dependency-light classification metrics for research reporting."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall, F1 and support for one class."""

    precision: float
    recall: float
    f1: float
    support: int


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_matrix(
    y_true: Sequence[Hashable], y_pred: Sequence[Hashable], labels: Sequence[Hashable] | None = None
) -> tuple[list[Hashable], list[list[int]]]:
    """Return class labels and a rows=true, columns=predicted confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    resolved = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    index = {label: position for position, label in enumerate(resolved)}
    matrix = [[0 for _ in resolved] for _ in resolved]
    for truth, prediction in zip(y_true, y_pred, strict=True):
        if truth not in index or prediction not in index:
            raise ValueError(f"Unknown label in pair ({truth!r}, {prediction!r})")
        matrix[index[truth]][index[prediction]] += 1
    return resolved, matrix


def classification_report(
    y_true: Sequence[Hashable], y_pred: Sequence[Hashable], labels: Sequence[Hashable] | None = None
) -> dict[str, object]:
    """Compute per-class and aggregate metrics without third-party dependencies."""
    resolved, matrix = confusion_matrix(y_true, y_pred, labels)
    total = sum(sum(row) for row in matrix)
    per_class: dict[str, ClassMetrics] = {}

    for position, label in enumerate(resolved):
        true_positive = matrix[position][position]
        predicted_positive = sum(matrix[row][position] for row in range(len(resolved)))
        actual_positive = sum(matrix[position])
        precision = _safe_divide(true_positive, predicted_positive)
        recall = _safe_divide(true_positive, actual_positive)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[str(label)] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=actual_positive,
        )

    correct = sum(matrix[position][position] for position in range(len(resolved)))
    macro_f1 = _safe_divide(sum(item.f1 for item in per_class.values()), len(per_class))
    weighted_f1 = _safe_divide(sum(item.f1 * item.support for item in per_class.values()), total)
    return {
        "labels": [str(label) for label in resolved],
        "matrix": matrix,
        "per_class": per_class,
        "accuracy": _safe_divide(correct, total),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "support": total,
    }


def binary_screening_metrics(
    y_true: Iterable[bool | int], y_score: Iterable[float], threshold: float = 0.5
) -> dict[str, float | int]:
    """Compute screening metrics for a scored binary problem.

    ``y_true`` uses 1/True for the positive (suspicious) class. The returned
    sensitivity, specificity, PPV and NPV are the four numbers a clinical reader
    needs before trusting a threshold.
    """
    truths = [bool(value) for value in y_true]
    scores = list(y_score)
    if len(truths) != len(scores):
        raise ValueError("y_true and y_score must have the same length")
    if not truths:
        raise ValueError("At least one sample is required")

    predictions = [score >= threshold for score in scores]
    tp = sum(
        1 for truth, prediction in zip(truths, predictions, strict=True) if truth and prediction
    )
    tn = sum(
        1
        for truth, prediction in zip(truths, predictions, strict=True)
        if not truth and not prediction
    )
    fp = sum(
        1 for truth, prediction in zip(truths, predictions, strict=True) if not truth and prediction
    )
    fn = sum(
        1 for truth, prediction in zip(truths, predictions, strict=True) if truth and not prediction
    )

    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    ppv = _safe_divide(tp, tp + fp)
    npv = _safe_divide(tn, tn + fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": _safe_divide(tp + tn, len(truths)),
        "f1": _safe_divide(2 * ppv * sensitivity, ppv + sensitivity),
        "threshold": threshold,
    }


def format_report(report: dict[str, object]) -> str:
    """Render a classification report as an aligned plain-text table."""
    per_class: dict[str, ClassMetrics] = report["per_class"]  # type: ignore[assignment]
    lines = [
        f"{'class':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
        "-" * 56,
    ]
    for label, item in per_class.items():
        lines.append(
            f"{label:<16}{item.precision:>10.3f}{item.recall:>10.3f}"
            f"{item.f1:>10.3f}{item.support:>10d}"
        )
    lines.append("-" * 56)
    lines.append(f"{'accuracy':<16}{float(report['accuracy']):>10.3f}")
    lines.append(f"{'macro f1':<16}{float(report['macro_f1']):>10.3f}")
    lines.append(f"{'weighted f1':<16}{float(report['weighted_f1']):>10.3f}")
    lines.append(f"{'support':<16}{int(report['support']):>10d}")
    return "\n".join(lines)
