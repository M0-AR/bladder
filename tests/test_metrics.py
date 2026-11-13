from __future__ import annotations

import pytest

from bladder_vision.metrics import (
    binary_screening_metrics,
    classification_report,
    confusion_matrix,
    format_report,
)


def test_confusion_matrix_order_and_counts():
    labels, matrix = confusion_matrix(["a", "a", "b", "b"], ["a", "b", "a", "b"])
    assert labels == ["a", "b"]
    assert matrix == [[1, 1], [1, 1]]


def test_classification_report_known_values():
    report = classification_report(["a", "a", "b", "b"], ["a", "b", "a", "b"])
    assert report["accuracy"] == 0.5
    assert report["macro_f1"] == 0.5
    assert report["per_class"]["a"].precision == 0.5
    assert report["per_class"]["b"].recall == 0.5
    assert report["support"] == 4


def test_classification_report_weighted_f1():
    report = classification_report(
        ["a", "a", "a", "b"],
        ["a", "a", "b", "b"],
        labels=["a", "b"],
    )
    assert report["accuracy"] == 0.75
    assert report["per_class"]["a"].support == 3
    assert report["per_class"]["b"].support == 1


def test_binary_screening_metrics():
    metrics = binary_screening_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], threshold=0.5)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["ppv"] == 0.5
    assert metrics["npv"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        confusion_matrix(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        binary_screening_metrics([1], [])


def test_format_report_contains_headings():
    report = classification_report(["a", "b"], ["a", "b"])
    text = format_report(report)
    assert "precision" in text
    assert "accuracy" in text
    assert "macro f1" in text
