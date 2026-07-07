import numpy as np
from mne.io.constants import FIFF

from calibrain.metric_evaluation import MetricEvaluator
from calibrain.uncertainty_calibration import UncertaintyCalibrator
from calibrain.uncertainty_estimation import UncertaintyEstimator


def _make_eeg_free_data():
    x_true = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]],
            [[0.0, 0.0], [1.0, 1.0], [0.2, 0.2]],
        ],
        dtype=float,
    )
    x_hat = x_true.copy()
    posterior_cov_blocks = np.array(
        [
            np.diag([0.3, 0.4, 0.5]),
            np.diag([0.2, 0.3, 0.6]),
        ],
        dtype=float,
    )
    return x_true, x_hat, posterior_cov_blocks


def _make_eeg_free_dataset():
    x_true, x_hat, posterior_cov_blocks = _make_eeg_free_data()
    return {
        "orientation_type": "free",
        "coil_type": FIFF.FIFFV_COIL_EEG,
        "x_true": x_true,
        "x_hat": x_hat,
        "posterior_cov_blocks": posterior_cov_blocks,
        "n_sources": x_true.shape[0],
        "n_times": x_true.shape[2],
    }


def test_eeg_free_componentwise_aggregated_membership_returns_expected_contract():
    ue = UncertaintyEstimator(nominal_coverages=np.array([0.0, 0.5, 1.0]))
    x_true, x_hat, posterior_cov_blocks = _make_eeg_free_data()

    result = ue.aggregated_componentwise_interval_membership_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_uncert=posterior_cov_blocks,
        nominal_coverage=0.5,
        n_orient=3,
    )

    assert result["x_true_agg"].shape == (2, 3)
    assert result["x_hat_agg"].shape == (2, 3)
    assert result["posterior_var"].shape == (2, 3)
    assert result["posterior_var_agg"].shape == (2, 3)
    assert result["within"].shape == (2, 3)
    assert result["empirical_coverage"] == 1.0


def test_eeg_free_ellipsoid_aggregated_membership_returns_expected_contract():
    ue = UncertaintyEstimator(nominal_coverages=np.array([0.0, 0.5, 1.0]))
    x_true, x_hat, posterior_cov_blocks = _make_eeg_free_data()

    result = ue.aggregated_ellipsoid_membership_eeg_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_cov=posterior_cov_blocks,
        nominal_coverage=0.5,
    )

    assert result["x_true_agg"].shape == (2, 3)
    assert result["x_hat_agg"].shape == (2, 3)
    assert result["q_values"].shape == (2,)
    assert result["cov_blocks"].shape == (2, 3, 3)
    assert result["within"].shape == (2,)
    assert result["empirical_coverage"] == 1.0


def test_metric_evaluator_eeg_free_supports_marginal_and_full_cov():
    ue = UncertaintyEstimator(nominal_coverages=np.array([0.0, 0.5, 1.0]))
    me = MetricEvaluator(ue)
    x_true, x_hat, posterior_cov_blocks = _make_eeg_free_data()

    marginal_curve = me.calibration_curve(
        x_true=x_true,
        x_hat=x_hat,
        posterior_uncert=posterior_cov_blocks,
        setting="eeg_free",
        mode="aggregated",
        free_interval_type="marginal",
    )
    full_cov_curve = me.calibration_curve(
        x_true=x_true,
        x_hat=x_hat,
        posterior_uncert=posterior_cov_blocks,
        setting="eeg_free",
        mode="aggregated",
        free_interval_type="full_cov",
    )

    assert marginal_curve["nominal"].shape == (3,)
    assert full_cov_curve["nominal"].shape == (3,)
    assert np.all(marginal_curve["empirical"] >= 0.0)
    assert np.all(marginal_curve["empirical"] <= 1.0)
    assert np.all(full_cov_curve["empirical"] >= 0.0)
    assert np.all(full_cov_curve["empirical"] <= 1.0)


def test_uncertainty_calibrator_eeg_free_precal_supports_both_interval_types():
    ue = UncertaintyEstimator(nominal_coverages=np.array([0.0, 0.5, 1.0]))
    me = MetricEvaluator(ue)
    calibrator = UncertaintyCalibrator(ue, me)
    dataset = _make_eeg_free_dataset()

    marginal_results = calibrator.calibrate(
        test_data=dataset,
        fit=False,
        free_interval_type="marginal",
    )
    full_cov_results = calibrator.calibrate(
        test_data=dataset,
        fit=False,
        free_interval_type="full_cov",
    )

    assert marginal_results["pre_calibration"]["interval_type"] == "marginal"
    assert full_cov_results["pre_calibration"]["interval_type"] == "full_cov"
    np.testing.assert_allclose(
        marginal_results["post_calibration"]["recalibrated_nominal_coverages"],
        ue.nominal_coverages,
    )
    np.testing.assert_allclose(
        full_cov_results["post_calibration"]["recalibrated_nominal_coverages"],
        ue.nominal_coverages,
    )
