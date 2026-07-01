import numpy as np
import pytest

from calibrain.metric_evaluation import MetricEvaluator
from calibrain.uncertainty_estimation import UncertaintyEstimator


def _make_metric_evaluator(nominal_coverages=None):
    ue = UncertaintyEstimator(
        nominal_coverages=np.array([0.0, 0.5, 0.9, 1.0])
        if nominal_coverages is None
        else np.asarray(nominal_coverages, dtype=float)
    )
    return MetricEvaluator(ue)


def test_calibration_metrics_4_is_zero_for_perfect_calibration():
    nominal = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
    empirical = nominal.copy()

    metrics = MetricEvaluator.calibration_metrics_4(nominal, empirical)

    assert metrics["max_underconfidence_deviation"] == pytest.approx(0.0)
    assert metrics["max_overconfidence_deviation"] == pytest.approx(0.0)
    assert metrics["mean_absolute_deviation"] == pytest.approx(0.0)
    assert metrics["mean_signed_deviation"] == pytest.approx(0.0)


def test_calibration_metrics_4_detects_underconfidence():
    nominal = np.array([0.1, 0.5, 0.9])
    empirical = np.array([0.3, 0.7, 1.0])

    metrics = MetricEvaluator.calibration_metrics_4(nominal, empirical)

    assert metrics["max_underconfidence_deviation"] == pytest.approx(0.0)
    assert metrics["max_overconfidence_deviation"] == pytest.approx(0.2)
    assert metrics["mean_absolute_deviation"] == pytest.approx((0.2 + 0.2 + 0.1) / 3.0)
    assert metrics["mean_signed_deviation"] == pytest.approx((0.2 + 0.2 + 0.1) / 3.0)


def test_calibration_metrics_4_detects_overconfidence():
    nominal = np.array([0.1, 0.5, 0.9])
    empirical = np.array([0.0, 0.2, 0.6])

    metrics = MetricEvaluator.calibration_metrics_4(nominal, empirical)

    assert metrics["max_underconfidence_deviation"] == pytest.approx(0.3)
    assert metrics["max_overconfidence_deviation"] == pytest.approx(0.0)
    assert metrics["mean_absolute_deviation"] == pytest.approx((0.1 + 0.3 + 0.3) / 3.0)
    assert metrics["mean_signed_deviation"] == pytest.approx((-0.1 - 0.3 - 0.3) / 3.0)


def test_calibration_curve_fixed_returns_expected_structure():
    evaluator = _make_metric_evaluator(nominal_coverages=[0.0, 0.5, 1.0])

    x_true = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    x_hat = x_true.copy()
    posterior_cov = np.diag([0.25, 1.0])

    curve = evaluator.calibration_curve(
        x_true=x_true,
        x_hat=x_hat,
        posterior_uncert=posterior_cov,
        setting="fixed",
        mode="aggregated",
    )

    np.testing.assert_allclose(curve["nominal"], np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(curve["empirical"], np.array([1.0, 1.0, 1.0]))
    assert set(curve["metrics_4"]) == {
        "max_underconfidence_deviation",
        "max_overconfidence_deviation",
        "mean_absolute_deviation",
        "mean_signed_deviation",
    }


def test_evaluate_all_fixed_returns_core_metric_blocks():
    evaluator = _make_metric_evaluator(nominal_coverages=[0.0, 0.5, 1.0])

    x_true = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    x_hat = np.array([[1.0, 1.0], [1.5, 1.5]], dtype=float)
    posterior_cov = np.diag([0.25, 1.0])

    results = evaluator.evaluate_all(
        x_true=x_true,
        x_hat=x_hat,
        posterior_uncert=posterior_cov,
        setting="fixed",
        mode="aggregated",
    )

    assert "mse" in results
    assert "mae" in results
    assert "rmse" in results
    assert "rmae" in results
    assert "mean_posterior_std" in results
    assert "calibration" in results
    assert np.isfinite(results["mse"])
    assert np.isfinite(results["mae"])
    assert np.isfinite(results["mean_posterior_std"])
