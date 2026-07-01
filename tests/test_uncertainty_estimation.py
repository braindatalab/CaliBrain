import numpy as np
import pytest

from calibrain.uncertainty_estimation import UncertaintyEstimator


def test_posterior_variance_from_cov_returns_diagonal():
    posterior_cov = np.array(
        [
            [4.0, 1.5, -2.0],
            [1.5, 9.0, 0.5],
            [-2.0, 0.5, 16.0],
        ]
    )

    posterior_var = UncertaintyEstimator.posterior_variance_from_cov(posterior_cov)

    np.testing.assert_allclose(posterior_var, np.array([4.0, 9.0, 16.0]))


def test_posterior_variance_from_cov_clips_negative_diagonal_entries():
    posterior_cov = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -3.0],
        ]
    )

    posterior_var = UncertaintyEstimator.posterior_variance_from_cov(posterior_cov)

    np.testing.assert_allclose(posterior_var, np.array([0.0, 2.0, 0.0]))


def test_uncertainty_estimator_rejects_invalid_nominal_coverages():
    with pytest.raises(ValueError, match="Nominal coverages must be between 0 and 1"):
        UncertaintyEstimator(nominal_coverages=np.array([-0.1, 0.5, 1.0]))

    with pytest.raises(ValueError, match="Nominal coverages must be between 0 and 1"):
        UncertaintyEstimator(nominal_coverages=np.array([0.0, 0.5, 1.1]))


def test_uncertainty_estimator_sorts_and_deduplicates_nominal_coverages():
    estimator = UncertaintyEstimator(nominal_coverages=np.array([0.9, 0.1, 0.9, 0.5]))

    np.testing.assert_allclose(estimator.nominal_coverages, np.array([0.1, 0.5, 0.9]))


def test_pointwise_interval_membership_returns_full_coverage_for_exact_match():
    estimator = UncertaintyEstimator(nominal_coverages=np.array([0.8]))

    x_true = np.array([[1.0, 1.0], [2.0, 2.0]])
    x_hat = x_true.copy()
    posterior_var = np.array([0.25, 1.0])

    result = estimator.pointwise_interval_membership(
        x_true=x_true,
        x_hat=x_hat,
        posterior_var=posterior_var,
        nominal_coverage=0.8,
    )

    assert result["within"].shape == x_true.shape
    assert result["count_within"] == x_true.size
    assert result["total_count"] == x_true.size
    assert result["empirical_coverage"] == pytest.approx(1.0)
    assert result["n_times"] == 2


def test_aggregated_interval_membership_scales_variance_by_number_of_timepoints():
    estimator = UncertaintyEstimator(nominal_coverages=np.array([0.5]))

    x_true = np.array([[1.0, 3.0], [2.0, 4.0]])
    x_hat = np.array([[1.0, 3.0], [2.0, 4.0]])
    posterior_var = np.array([8.0, 2.0])

    result = estimator.aggregated_interval_membership(
        x_true=x_true,
        x_hat=x_hat,
        posterior_var=posterior_var,
        nominal_coverage=0.5,
    )

    np.testing.assert_allclose(result["x_true_agg"], np.array([2.0, 3.0]))
    np.testing.assert_allclose(result["x_hat_agg"], np.array([2.0, 3.0]))
    np.testing.assert_allclose(result["posterior_var_agg"], np.array([4.0, 1.0]))
    assert result["empirical_coverage"] == pytest.approx(1.0)
