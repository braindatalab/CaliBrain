# =============================================================================
# Direction-aggregated component-wise credible intervals
#         for free-orientation EEG and reduced free-orientation MEG
# =============================================================================
#
# This block is designed to be pasted AFTER the existing UncertaintyEstimator
# class definition.
#
# It adds a parallel uncertainty diagnostic:
#
#   - existing free EEG diagnostic:
#       3D ellipsoid membership
#
#   - existing free MEG diagnostic:
#       2D ellipse membership
#
#   - additional diagnostic added here:
#       scalar marginal credible intervals for each retained local component,
#       with calibration results aggregated over directions/components.
#
# For EEG:
#       x_true, x_hat : (N, 3, T)
#       posterior_cov : (3N, 3N)
#
# For reduced free-orientation MEG:
#       x_true, x_hat : (N, 2, T)
#       posterior_cov : (2N, 2N)
#
# Main idea:
#       1. Extract local marginal variances from diag(posterior_cov).
#       2. Reshape them as (N, K), where K=3 for EEG and K=2 for MEG.
#       3. Build scalar credible intervals component by component.
#       4. Aggregate interval membership over sources, directions, and time.
#       5. Return one calibration curve and one metric vector per setting.
#
# Important:
#       Directions or components are NOT evaluated separately.
#       They are pooled because the local coordinate systems are arbitrary:
#           - EEG x/y/z labels depend on the chosen coordinate system.
#           - MEG SVD component labels are local and voxel-dependent.
#       Therefore, no separate direction-wise curves or scores are computed.
#
#       MEG is evaluated directly in reduced 2D coefficient coordinates.
#       No 2D -> 3D lifting is used inside this diagnostic.
#       No 3D -> 2D projection is performed inside this diagnostic.
# =============================================================================

import numpy as np
from typing import Any, Dict, Optional


def _ue_as_source_major_free_mean(
    posterior_mean: np.ndarray,
    *,
    n_sources: int,
    n_orient: int,
    name: str = "posterior_mean",
) -> np.ndarray:
    """
    Convert a free-orientation posterior mean to source-major shape (N,K,T).

    Accepted input shapes
    ---------------------
    1) (N,K,T)
       Already source-major.

    2) (K*N,T)
       Flattened source-major representation.

    Notes
    -----
    This assumes source-major flattening:

        (N,K,T) -> (N*K,T),

    consistent with a lead-field block reshaping convention such as

        L_block.reshape(M, N*K).

    Parameters
    ----------
    posterior_mean : ndarray
        Posterior mean returned by the inverse solver.
    n_sources : int
        Number of sources N.
    n_orient : int
        Number of local components K.
        EEG free orientation: K=3.
        MEG reduced free orientation: K=2.
    name : str
        Name used in error messages.

    Returns
    -------
    posterior_mean_nkt : ndarray, shape (N,K,T)
    """
    posterior_mean = np.asarray(posterior_mean, dtype=float)

    if posterior_mean.ndim == 3:
        if posterior_mean.shape[0] != n_sources or posterior_mean.shape[1] != n_orient:
            raise ValueError(
                f"{name} must have shape (N,K,T)=({n_sources},{n_orient},T). "
                f"Got {posterior_mean.shape}."
            )
        return posterior_mean

    if posterior_mean.ndim == 2:
        expected_first_dim = n_sources * n_orient
        if posterior_mean.shape[0] != expected_first_dim:
            raise ValueError(
                f"{name} has incompatible flattened shape. Expected first "
                f"dimension {expected_first_dim}, got {posterior_mean.shape[0]}."
            )

        T = posterior_mean.shape[1]
        return posterior_mean.reshape(n_sources, n_orient, T)

    raise ValueError(
        f"{name} must be either (N,K,T) or (K*N,T). Got {posterior_mean.shape}."
    )


def _ue_componentwise_variance_from_cov(
    posterior_cov: np.ndarray,
    *,
    n_sources: int,
    n_orient: int,
    min_var: float = 0.0,
) -> np.ndarray:
    """
    Extract component-wise marginal posterior variances.

    The full posterior covariance has shape (K*N,K*N). For component-wise
    marginal credible intervals, only diagonal entries are used. These are
    reshaped into source-major form:

        posterior_var[i,k] = marginal variance of source i, component k.

    Parameters
    ----------
    posterior_cov : ndarray, shape (K*N,K*N)
        Full posterior covariance.
    n_sources : int
        Number of sources N.
    n_orient : int
        Number of local components K.
    min_var : float
        Optional numerical lower bound for variances.

    Returns
    -------
    posterior_var : ndarray, shape (N,K)
        Component-wise marginal variances.
    """
    posterior_cov = np.asarray(posterior_cov, dtype=float)

    expected_shape = (n_sources * n_orient, n_sources * n_orient)
    if posterior_cov.shape != expected_shape:
        raise ValueError(
            f"posterior_cov must have shape {expected_shape}. "
            f"Got {posterior_cov.shape}."
        )

    posterior_var = np.diag(posterior_cov).astype(float, copy=True)
    posterior_var = posterior_var.reshape(n_sources, n_orient)

    if min_var <= 0.0:
        return np.maximum(posterior_var, 0.0)

    return np.maximum(posterior_var, float(min_var))


def _ue_validate_nominal_coverage(nominal_coverage: float) -> float:
    """
    Validate and return nominal coverage as a float.
    """
    c = float(nominal_coverage)
    if not (0.0 < c < 1.0):
        raise ValueError("nominal_coverage must be strictly between 0 and 1.")
    return c


def _ue_pointwise_componentwise_interval_membership_free(
    self,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    posterior_cov: np.ndarray,
    nominal_coverage: float,
    *,
    n_orient: int,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    """
    Pointwise component-wise credible interval membership for free orientation.

    Each retained local component is treated as a scalar marginal posterior
    quantity. Membership is then pooled over all sources, directions/components,
    and time points.

    EEG usage
    ---------
    x_true        : (N,3,T)
    x_hat         : (N,3,T)
    posterior_cov : (3N,3N)
    n_orient      : 3

    MEG usage
    ---------
    x_true        : (N,2,T)
    x_hat         : (N,2,T)
    posterior_cov : (2N,2N)
    n_orient      : 2

    Returns
    -------
    dict
        Includes scalar interval bounds, membership indicators, and pooled
        empirical coverage aggregated over sources, directions, and time.
    """
    nominal_coverage = _ue_validate_nominal_coverage(nominal_coverage)

    x_true = np.asarray(x_true, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    posterior_cov = np.asarray(posterior_cov, dtype=float)

    if x_true.ndim != 3 or x_true.shape[1] != n_orient:
        raise ValueError(
            f"x_true must have shape (N,{n_orient},T). Got {x_true.shape}."
        )
    if x_hat.shape != x_true.shape:
        raise ValueError(
            f"x_hat must match x_true shape. Got {x_hat.shape} vs {x_true.shape}."
        )

    N, K, T = x_true.shape

    posterior_var = _ue_componentwise_variance_from_cov(
        posterior_cov,
        n_sources=N,
        n_orient=n_orient,
        min_var=min_var,
    )  # (N,K)

    posterior_var_full = np.repeat(posterior_var[:, :, None], T, axis=2)

    ci_lower, ci_upper = self.credible_intervals_normal(
        mean=x_hat,
        variance=posterior_var_full,
        nominal_coverage=nominal_coverage,
    )

    within = (x_true >= ci_lower) & (x_true <= ci_upper)  # (N,K,T)

    count_within = int(np.sum(within))
    total_count = int(within.size)
    empirical_coverage = float(count_within / total_count)

    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "within": within,
        "posterior_var": posterior_var,
        "posterior_var_full": posterior_var_full,
        "z_score": float(self._get_z(nominal_coverage)),

        # Direction-aggregated result
        "count_within": count_within,
        "total_count": total_count,
        "empirical_coverage": empirical_coverage,
        "aggregation_axes": "sources, directions, time",

        "n_sources": int(N),
        "n_orient": int(K),
        "n_times": int(T),
    }


def _ue_aggregated_componentwise_interval_membership_free(
    self,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    posterior_cov: np.ndarray,
    nominal_coverage: float,
    *,
    n_orient: int,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    """
    Temporally aggregated component-wise credible interval membership.

    Aggregation is done by temporal averaging. Since the posterior covariance is
    static over time, each marginal variance is scaled by 1/T. Membership is
    then pooled over all sources and directions/components.
    """
    nominal_coverage = _ue_validate_nominal_coverage(nominal_coverage)

    x_true = np.asarray(x_true, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    posterior_cov = np.asarray(posterior_cov, dtype=float)

    if x_true.ndim != 3 or x_true.shape[1] != n_orient:
        raise ValueError(
            f"x_true must have shape (N,{n_orient},T). Got {x_true.shape}."
        )
    if x_hat.shape != x_true.shape:
        raise ValueError(
            f"x_hat must match x_true shape. Got {x_hat.shape} vs {x_true.shape}."
        )

    N, K, T = x_true.shape

    x_true_agg = np.mean(x_true, axis=2)  # (N,K)
    x_hat_agg = np.mean(x_hat, axis=2)    # (N,K)

    posterior_var = _ue_componentwise_variance_from_cov(
        posterior_cov,
        n_sources=N,
        n_orient=n_orient,
        min_var=min_var,
    )  # (N,K)

    posterior_var_agg = posterior_var / float(T)

    ci_lower, ci_upper = self.credible_intervals_normal(
        mean=x_hat_agg,
        variance=posterior_var_agg,
        nominal_coverage=nominal_coverage,
    )

    within = (x_true_agg >= ci_lower) & (x_true_agg <= ci_upper)  # (N,K)

    count_within = int(np.sum(within))
    total_count = int(within.size)
    empirical_coverage = float(count_within / total_count)

    return {
        "x_true_agg": x_true_agg,
        "x_hat_agg": x_hat_agg,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "within": within,
        "posterior_var": posterior_var,
        "posterior_var_agg": posterior_var_agg,
        "z_score": float(self._get_z(nominal_coverage)),

        # Direction-aggregated result
        "count_within": count_within,
        "total_count": total_count,
        "empirical_coverage": empirical_coverage,
        "aggregation_axes": "sources, directions",

        "n_sources": int(N),
        "n_orient": int(K),
        "n_times": int(T),
    }


def _ue_calibration_curve_componentwise_intervals_pointwise_free(
    self,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    posterior_cov: np.ndarray,
    *,
    n_orient: int,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    """
    Direction-aggregated pointwise component-wise calibration curve.

    For each nominal coverage level, scalar interval membership is pooled over
    sources, directions/components, and time points. Therefore, the output
    contains one empirical calibration curve, not one curve per direction.
    """
    empirical_coverages = []
    counts = []
    totals = []

    nominal_coverages = np.asarray(self.nominal_coverages, dtype=float)

    for c in nominal_coverages:
        out = self.pointwise_componentwise_interval_membership_free(
            x_true=x_true,
            x_hat=x_hat,
            posterior_cov=posterior_cov,
            nominal_coverage=float(c),
            n_orient=n_orient,
            min_var=min_var,
        )

        empirical_coverages.append(out["empirical_coverage"])
        counts.append(out["count_within"])
        totals.append(out["total_count"])

    return {
        "nominal_coverages": nominal_coverages,
        "empirical_coverages": np.asarray(empirical_coverages, dtype=float),
        "ci_counts": np.asarray(counts, dtype=int),
        "total_counts": np.asarray(totals, dtype=int),
        "n_orient": int(n_orient),
        "mode": "pointwise",
        "interval_type": "componentwise_marginal",
        "direction_aggregation": "pooled_over_all_components",
    }


def _ue_calibration_curve_componentwise_intervals_aggregated_free(
    self,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    posterior_cov: np.ndarray,
    *,
    n_orient: int,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    """
    Direction-aggregated temporally aggregated component-wise calibration curve.

    For each nominal coverage level, scalar interval membership is pooled over
    sources and directions/components after temporal averaging. Therefore, the
    output contains one empirical calibration curve, not one curve per direction.
    """
    empirical_coverages = []
    counts = []
    totals = []

    nominal_coverages = np.asarray(self.nominal_coverages, dtype=float)

    for c in nominal_coverages:
        out = self.aggregated_componentwise_interval_membership_free(
            x_true=x_true,
            x_hat=x_hat,
            posterior_cov=posterior_cov,
            nominal_coverage=float(c),
            n_orient=n_orient,
            min_var=min_var,
        )

        empirical_coverages.append(out["empirical_coverage"])
        counts.append(out["count_within"])
        totals.append(out["total_count"])

    return {
        "nominal_coverages": nominal_coverages,
        "empirical_coverages": np.asarray(empirical_coverages, dtype=float),
        "ci_counts": np.asarray(counts, dtype=int),
        "total_counts": np.asarray(totals, dtype=int),
        "n_orient": int(n_orient),
        "mode": "aggregated",
        "interval_type": "componentwise_marginal",
        "direction_aggregation": "pooled_over_all_components",
    }


# =============================================================================
# EEG free-orientation wrappers
# =============================================================================

def _ue_pointwise_componentwise_intervals_eeg_free(
    self,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    posterior_cov: np.ndarray,
    nominal_coverage: float,
    *,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    """
    EEG free-orientation component-wise pointwise intervals.

    Expected shapes
    ---------------
    x_true        : (N,3,T)
    x_hat         : (N,3,T)
    posterior_cov : (3N,3N)
    """
    return self.pointwise_componentwise_interval_membership_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_cov=posterior_cov,
        nominal_coverage=nominal_coverage,
        n_orient=3,
        min_var=min_var,
    )


def _ue_aggregated_componentwise_intervals_eeg_free(
    self,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    posterior_cov: np.ndarray,
    nominal_coverage: float,
    *,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    """
    EEG free-orientation component-wise temporally aggregated intervals.

    Expected shapes
    ---------------
    x_true        : (N,3,T)
    x_hat         : (N,3,T)
    posterior_cov : (3N,3N)
    """
    return self.aggregated_componentwise_interval_membership_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_cov=posterior_cov,
        nominal_coverage=nominal_coverage,
        n_orient=3,
        min_var=min_var,
    )


def _ue_calibration_curve_componentwise_eeg_free_pointwise(
    self,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    posterior_cov: np.ndarray,
    *,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    return self.calibration_curve_componentwise_intervals_pointwise_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_cov=posterior_cov,
        n_orient=3,
        min_var=min_var,
    )


def _ue_calibration_curve_componentwise_eeg_free_aggregated(
    self,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    posterior_cov: np.ndarray,
    *,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    return self.calibration_curve_componentwise_intervals_aggregated_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_cov=posterior_cov,
        n_orient=3,
        min_var=min_var,
    )


# =============================================================================
# MEG reduced free-orientation wrappers
# =============================================================================

def _ue_pointwise_componentwise_intervals_meg_free(
    self,
    x_true_2d: np.ndarray,
    x_hat_2d: np.ndarray,
    posterior_cov_2d: np.ndarray,
    nominal_coverage: float,
    *,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    """
    MEG reduced free-orientation component-wise pointwise intervals.

    Expected shapes
    ---------------
    x_true_2d        : (N,2,T)
    x_hat_2d         : (N,2,T)
    posterior_cov_2d : (2N,2N)

    Important
    ---------
    This function works directly in reduced 2D MEG coefficient coordinates.
    No 2D-to-3D lifting and no 3D-to-2D projection are used here.
    """
    return self.pointwise_componentwise_interval_membership_free(
        x_true=x_true_2d,
        x_hat=x_hat_2d,
        posterior_cov=posterior_cov_2d,
        nominal_coverage=nominal_coverage,
        n_orient=2,
        min_var=min_var,
    )


def _ue_aggregated_componentwise_intervals_meg_free(
    self,
    x_true_2d: np.ndarray,
    x_hat_2d: np.ndarray,
    posterior_cov_2d: np.ndarray,
    nominal_coverage: float,
    *,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    """
    MEG reduced free-orientation component-wise temporally aggregated intervals.

    Expected shapes
    ---------------
    x_true_2d        : (N,2,T)
    x_hat_2d         : (N,2,T)
    posterior_cov_2d : (2N,2N)

    Important
    ---------
    This function works directly in reduced 2D MEG coefficient coordinates.
    No 2D-to-3D lifting and no 3D-to-2D projection are used here.
    """
    return self.aggregated_componentwise_interval_membership_free(
        x_true=x_true_2d,
        x_hat=x_hat_2d,
        posterior_cov=posterior_cov_2d,
        nominal_coverage=nominal_coverage,
        n_orient=2,
        min_var=min_var,
    )


def _ue_calibration_curve_componentwise_meg_free_pointwise(
    self,
    x_true_2d: np.ndarray,
    x_hat_2d: np.ndarray,
    posterior_cov_2d: np.ndarray,
    *,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    return self.calibration_curve_componentwise_intervals_pointwise_free(
        x_true=x_true_2d,
        x_hat=x_hat_2d,
        posterior_cov=posterior_cov_2d,
        n_orient=2,
        min_var=min_var,
    )


def _ue_calibration_curve_componentwise_meg_free_aggregated(
    self,
    x_true_2d: np.ndarray,
    x_hat_2d: np.ndarray,
    posterior_cov_2d: np.ndarray,
    *,
    min_var: float = 0.0,
) -> Dict[str, Any]:
    return self.calibration_curve_componentwise_intervals_aggregated_free(
        x_true=x_true_2d,
        x_hat=x_hat_2d,
        posterior_cov=posterior_cov_2d,
        n_orient=2,
        min_var=min_var,
    )


# =============================================================================
# Attach the add-on functions to UncertaintyEstimator
# =============================================================================
#
# This lets you paste this block after the existing class without editing the
# original class indentation.
# =============================================================================

UncertaintyEstimator.as_source_major_free_mean = staticmethod(_ue_as_source_major_free_mean)
UncertaintyEstimator.componentwise_variance_from_cov = staticmethod(_ue_componentwise_variance_from_cov)

UncertaintyEstimator.pointwise_componentwise_interval_membership_free = (
    _ue_pointwise_componentwise_interval_membership_free
)
UncertaintyEstimator.aggregated_componentwise_interval_membership_free = (
    _ue_aggregated_componentwise_interval_membership_free
)

UncertaintyEstimator.calibration_curve_componentwise_intervals_pointwise_free = (
    _ue_calibration_curve_componentwise_intervals_pointwise_free
)
UncertaintyEstimator.calibration_curve_componentwise_intervals_aggregated_free = (
    _ue_calibration_curve_componentwise_intervals_aggregated_free
)

UncertaintyEstimator.pointwise_componentwise_intervals_eeg_free = (
    _ue_pointwise_componentwise_intervals_eeg_free
)
UncertaintyEstimator.aggregated_componentwise_intervals_eeg_free = (
    _ue_aggregated_componentwise_intervals_eeg_free
)
UncertaintyEstimator.calibration_curve_componentwise_eeg_free_pointwise = (
    _ue_calibration_curve_componentwise_eeg_free_pointwise
)
UncertaintyEstimator.calibration_curve_componentwise_eeg_free_aggregated = (
    _ue_calibration_curve_componentwise_eeg_free_aggregated
)

UncertaintyEstimator.pointwise_componentwise_intervals_meg_free = (
    _ue_pointwise_componentwise_intervals_meg_free
)
UncertaintyEstimator.aggregated_componentwise_intervals_meg_free = (
    _ue_aggregated_componentwise_intervals_meg_free
)
UncertaintyEstimator.calibration_curve_componentwise_meg_free_pointwise = (
    _ue_calibration_curve_componentwise_meg_free_pointwise
)
UncertaintyEstimator.calibration_curve_componentwise_meg_free_aggregated = (
    _ue_calibration_curve_componentwise_meg_free_aggregated
)


# =============================================================================
# Direction-aggregated calibration metrics
# =============================================================================
#
# These metrics are computed from one pooled calibration curve. Directions are
# not evaluated separately.
# =============================================================================

def componentwise_calibration_metrics(curve: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute calibration metrics from a direction-aggregated component-wise curve.

    Expected input
    --------------
    curve["nominal_coverages"] : shape (M,)
        Nominal coverage grid.

    curve["empirical_coverages"] : shape (M,)
        Empirical coverage curve pooled over all retained components/directions.

    Returns
    -------
    metrics : dict
        Scalar calibration metrics for the pooled curve.
    """
    nominal = np.asarray(curve["nominal_coverages"], dtype=float)
    empirical = np.asarray(curve["empirical_coverages"], dtype=float)

    if nominal.ndim != 1 or empirical.ndim != 1:
        raise ValueError(
            "curve['nominal_coverages'] and curve['empirical_coverages'] "
            "must both be one-dimensional."
        )

    if nominal.shape[0] != empirical.shape[0]:
        raise ValueError(
            "Mismatch between nominal coverage grid and empirical coverage curve."
        )

    diff = empirical - nominal

    # Undercoverage means empirical < nominal and corresponds to overconfidence.
    overconfidence = np.maximum(nominal - empirical, 0.0)

    # Overcoverage means empirical > nominal and corresponds to underconfidence.
    underconfidence = np.maximum(empirical - nominal, 0.0)

    return {
        "max_underconfidence_deviation": float(np.max(underconfidence)),
        "max_overconfidence_deviation": float(np.max(overconfidence)),
        "mean_absolute_deviation": float(np.mean(np.abs(diff))),
        "mean_signed_deviation": float(np.mean(diff)),
    }


def print_componentwise_calibration_metrics(
    label: str,
    metrics: Dict[str, float],
) -> None:
    """
    Print direction-aggregated component-wise calibration metrics.
    """
    print(f"\n================ {label} ================")
    print("  max_underconfidence_deviation :", metrics["max_underconfidence_deviation"])
    print("  max_overconfidence_deviation  :", metrics["max_overconfidence_deviation"])
    print("  mean_absolute_deviation       :", metrics["mean_absolute_deviation"])
    print("  mean_signed_deviation         :", metrics["mean_signed_deviation"])


# =============================================================================
# Optional helper: unified runner for one free-orientation estimator output
# =============================================================================
#
# This helper is useful when the solver output dictionary has:
#
#       out_estimator["posterior_mean"]
#       out_estimator["posterior_cov"]
#
# It works for both EEG and reduced free-orientation MEG.
# =============================================================================

def run_componentwise_free_orientation_uncertainty(
    *,
    modality: str,
    x_true: np.ndarray,
    out_estimator: Dict[str, Any],
    nominal_coverage: float = 0.95,
    min_var: float = 1e-12,
    ue: Optional[UncertaintyEstimator] = None,
) -> Dict[str, Any]:
    """
    Run direction-aggregated component-wise free-orientation diagnostics.

    Parameters
    ----------
    modality : {"eeg", "meg"}
        Determines the number of retained components:
            EEG -> K=3
            reduced MEG -> K=2

    x_true : ndarray
        EEG: shape (N,3,T)
        MEG: shape (N,2,T)

    out_estimator : dict
        Must contain:
            "posterior_mean"
            "posterior_cov"

    nominal_coverage : float
        Coverage level for the diagnostic membership output.

    min_var : float
        Numerical lower bound for marginal variances.

    ue : UncertaintyEstimator or None
        Existing uncertainty estimator. If None, a new one is created.

    Returns
    -------
    results : dict
        Contains pointwise diagnostics, temporally aggregated diagnostics,
        direction-aggregated calibration curves, and scalar calibration metrics.
    """
    modality = str(modality).lower()

    if modality not in {"eeg", "meg"}:
        raise ValueError("modality must be either 'eeg' or 'meg'.")

    if ue is None:
        ue = UncertaintyEstimator()

    x_true = np.asarray(x_true, dtype=float)

    if modality == "eeg":
        n_orient = 3
    else:
        n_orient = 2

    if x_true.ndim != 3 or x_true.shape[1] != n_orient:
        raise ValueError(
            f"For modality='{modality}', x_true must have shape "
            f"(N,{n_orient},T). Got {x_true.shape}."
        )

    N = x_true.shape[0]

    x_hat = ue.as_source_major_free_mean(
        out_estimator["posterior_mean"],
        n_sources=N,
        n_orient=n_orient,
        name="out_estimator['posterior_mean']",
    )

    posterior_cov = np.asarray(out_estimator["posterior_cov"], dtype=float)

    diag_point = ue.pointwise_componentwise_interval_membership_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_cov=posterior_cov,
        nominal_coverage=nominal_coverage,
        n_orient=n_orient,
        min_var=min_var,
    )

    diag_agg = ue.aggregated_componentwise_interval_membership_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_cov=posterior_cov,
        nominal_coverage=nominal_coverage,
        n_orient=n_orient,
        min_var=min_var,
    )

    curve_point = ue.calibration_curve_componentwise_intervals_pointwise_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_cov=posterior_cov,
        n_orient=n_orient,
        min_var=min_var,
    )

    curve_agg = ue.calibration_curve_componentwise_intervals_aggregated_free(
        x_true=x_true,
        x_hat=x_hat,
        posterior_cov=posterior_cov,
        n_orient=n_orient,
        min_var=min_var,
    )

    metrics_point = componentwise_calibration_metrics(curve_point)
    metrics_agg = componentwise_calibration_metrics(curve_agg)

    return {
        "ue": ue,
        "modality": modality,
        "n_orient": int(n_orient),
        "x_true": x_true,
        "x_hat": x_hat,
        "posterior_cov": posterior_cov,
        "diag_point": diag_point,
        "diag_agg": diag_agg,
        "curve_point": curve_point,
        "curve_agg": curve_agg,
        "metrics_point": metrics_point,
        "metrics_agg": metrics_agg,
    }


def print_componentwise_uncertainty_summary(
    label: str,
    results: Dict[str, Any],
) -> None:
    """
    Print a compact summary of direction-aggregated component-wise diagnostics.
    """
    print(f"\n================ {label} ================")
    print("  modality                       :", results["modality"])
    print("  n_orient                       :", results["n_orient"])

    print("\nPOINTWISE")
    print("  empirical coverage             :", results["diag_point"]["empirical_coverage"])
    print("  count / total                  :", results["diag_point"]["count_within"], "/", results["diag_point"]["total_count"])
    print("  aggregation axes               :", results["diag_point"]["aggregation_axes"])

    print("\nTEMPORALLY AGGREGATED")
    print("  empirical coverage             :", results["diag_agg"]["empirical_coverage"])
    print("  count / total                  :", results["diag_agg"]["count_within"], "/", results["diag_agg"]["total_count"])
    print("  aggregation axes               :", results["diag_agg"]["aggregation_axes"])

    print_componentwise_calibration_metrics(
        f"{label} / POINTWISE POOLED METRICS",
        results["metrics_point"],
    )

    print_componentwise_calibration_metrics(
        f"{label} / TEMPORALLY AGGREGATED POOLED METRICS",
        results["metrics_agg"],
    )


# =============================================================================
# Example usage
# =============================================================================
#
# ue = UncertaintyEstimator()
#
# # Free EEG / BMN
# # Expected:
# #   x_free_eeg.shape == (N,3,T)
# #   out_free_eeg_bmn["posterior_mean"] is either (N,3,T) or (3N,T)
# #   out_free_eeg_bmn["posterior_cov"].shape == (3N,3N)
#
# results_eeg_bmn_comp = run_componentwise_free_orientation_uncertainty(
#     modality="eeg",
#     x_true=x_free_eeg,
#     out_estimator=out_free_eeg_bmn,
#     nominal_coverage=0.95,
#     min_var=1e-12,
#     ue=ue,
# )
#
# print_componentwise_uncertainty_summary(
#     "FREE EEG / BMN / DIRECTION-AGGREGATED COMPONENT-WISE",
#     results_eeg_bmn_comp,
# )
#
# # Free MEG reduced 2D / BMN
# # Expected:
# #   a_free_meg.shape == (N,2,T)
# #   out_free_meg_bmn["posterior_mean"] is either (N,2,T) or (2N,T)
# #   out_free_meg_bmn["posterior_cov"].shape == (2N,2N)
#
# results_meg_bmn_comp = run_componentwise_free_orientation_uncertainty(
#     modality="meg",
#     x_true=a_free_meg,
#     out_estimator=out_free_meg_bmn,
#     nominal_coverage=0.95,
#     min_var=1e-12,
#     ue=ue,
# )
#
# print_componentwise_uncertainty_summary(
#     "FREE MEG / BMN / DIRECTION-AGGREGATED COMPONENT-WISE",
#     results_meg_bmn_comp,
# )
