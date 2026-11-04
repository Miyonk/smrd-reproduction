import pytest

from src.evaluate import (
    bootstrap_accuracy,
    classification_metrics,
    mcnemar_test,
    report_c_collapse,
    summarize_binary,
    wilson_ci,
)


def test_binary_metrics_known_values():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    m = classification_metrics(y_true, y_pred, 2)
    # TP=2, FP=1, FN=1, TN=1
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["npv"] == pytest.approx(1 / 2)
    assert m["fpr"] == pytest.approx(1 / 2)
    assert m["fnr"] == pytest.approx(1 / 3)


def test_multiclass_macro_f1():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 2]
    m = classification_metrics(y_true, y_pred, 4)
    assert m["accuracy"] == pytest.approx(0.75)
    # class 3 has zero recall and precision; macro averages all four classes.
    assert 0.0 < m["macro_f1"] < 1.0
    assert len(m["per_class"]) == 4


def test_wilson_ci_bounds():
    low, high = wilson_ci(95, 100)
    assert low < 0.95 < high
    assert 0.0 <= low <= high <= 1.0


def test_wilson_ci_empty():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_mcnemar_no_disagreement():
    result = mcnemar_test([0, 1, 1], [0, 1, 1], [0, 1, 1])
    assert result["p_value"] == 1.0


def test_mcnemar_significant():
    truth = [0] * 20
    a = [0] * 20
    b = [1] * 20
    result = mcnemar_test(truth, a, b)
    assert result["b"] == 20
    assert result["c"] == 0
    assert result["p_value"] < 0.001


def test_bootstrap_deterministic():
    a = bootstrap_accuracy([0, 1, 1, 0], [0, 1, 1, 1], n_resamples=200, seed=7)
    b = bootstrap_accuracy([0, 1, 1, 0], [0, 1, 1, 1], n_resamples=200, seed=7)
    assert a == b


def test_report_c_collapse():
    assert report_c_collapse([0, 1, 2, 3]) == [0, 1, 1, 1]


def test_summarize_binary_adds_ci():
    m = summarize_binary([0, 1, 0, 1], [0, 1, 1, 1])
    assert "wilson_ci_95" in m
    assert len(m["wilson_ci_95"]) == 2
