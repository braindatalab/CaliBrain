import numpy as np
import pytest

from calibrain.metric_evaluation import MetricEvaluator
from calibrain.uncertainty_calibration import UncertaintyCalibrator
from calibrain.uncertainty_estimation import UncertaintyEstimator


def _make_calibrator(nominal_coverages=None):
    ue = UncertaintyEstimator(
        nominal_coverages=np.array([0.0, 0.5, 0.9, 1.0])
        if nominal_coverages is None
        else np.asarray(nominal_coverages, dtype=float)
    )
    me = MetricEvaluator(ue)
    return UncertaintyCalibrator(ue, me)


def _make_fixed_dataset():
    x_true = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=float,
    )
    x_hat = np.array(
        [
            [0.1, -0.1, 0.0],
            [1.2, 0.8, 1.1],
            [2.3, 1.7, 2.1],
        ],
        dtype=float,
    )
    posterior_var = np.array([0.05, 0.1, 0.2], dtype=float)
    return {
        "orientation_type": "fixed",
        "coil_type": None,
        "x_true": x_true,
        "x_hat": x_hat,
        "posterior_var": posterior_var,
        "n_sources": x_true.shape[0],
        "n_times": x_true.shape[1],
    }


def test_apply_calibration_requires_fitted_model():
    calibrator = _make_calibrator()

    with pytest.raises(ValueError, match="Calibrator must be fitted first"):
        calibrator.apply_calibration(np.array([0.1, 0.5, 0.9]))


def test_calibrate_precal_mode_returns_identity_post_block():
    calibrator = _make_calibrator()
    dataset = _make_fixed_dataset()

    results = calibrator.calibrate(test_data=dataset, fit=False)

    assert results["train_empirical_coverages"] is None
    np.testing.assert_allclose(
        results["pre_calibration"]["nominal_coverages"],
        results["post_calibration"]["nominal_coverages"],
    )
    np.testing.assert_allclose(
        results["pre_calibration"]["empirical_coverages"],
        results["post_calibration"]["empirical_coverages"],
    )
    np.testing.assert_allclose(
        results["post_calibration"]["recalibrated_nominal_coverages"],
        results["pre_calibration"]["nominal_coverages"],
    )
    assert results["pre_calibration"]["split_metadata"]["fit"] is False
    assert results["post_calibration"]["interval_type"] == "marginal"


def test_calibrate_fit_mode_returns_monotone_recalibrated_coverages():
    calibrator = _make_calibrator()
    train_data = _make_fixed_dataset()
    test_data = _make_fixed_dataset()

    results = calibrator.calibrate(train_data=train_data, test_data=test_data, fit=True)

    recalibrated = results["post_calibration"]["recalibrated_nominal_coverages"]

    assert calibrator.is_calibrated is True
    assert results["train_empirical_coverages"] is not None
    assert recalibrated.shape == calibrator.nominal_coverages.shape
    assert np.all(recalibrated >= 0.0)
    assert np.all(recalibrated <= 1.0)
    assert np.all(np.diff(recalibrated) >= -1e-12)
    assert results["pre_calibration"]["split_metadata"]["fit"] is True


def test_fit_mapping_then_evaluate_with_mapping_returns_expected_structure():
    calibrator = _make_calibrator()
    train_data = _make_fixed_dataset()
    test_data = _make_fixed_dataset()

    fit_results = calibrator.fit_mapping(train_data=train_data)
    eval_results = calibrator.evaluate_with_mapping(test_data=test_data)

    assert "train_curve" in fit_results
    assert "recalibrated_nominal_coverages" in fit_results
    assert calibrator.is_calibrated is True
    assert eval_results["train_empirical_coverages"] is None
    np.testing.assert_allclose(
        eval_results["post_calibration"]["nominal_coverages"],
        calibrator.nominal_coverages,
    )
    assert eval_results["pre_calibration"]["split_metadata"]["uses_separate_eval_split"] is True


def test_calibrate_rejects_invalid_free_interval_type():
    calibrator = _make_calibrator()
    dataset = _make_fixed_dataset()

    with pytest.raises(ValueError, match="free_interval_type must be 'full_cov' or 'marginal'"):
        calibrator.calibrate(test_data=dataset, fit=False, free_interval_type="invalid")
