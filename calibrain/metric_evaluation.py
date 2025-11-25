from venv import logger
import numpy as np
import pandas as pd
import logging # Import logging
import numpy as np
from mne.io.constants import FIFF
from scipy.spatial.distance import cdist
from mne import read_forward_solution, convert_forward_solution
from ot import emd2  # Earth Mover's Distance (Wasserstein-2)
from mne.inverse_sparse.mxne_inverse import _make_sparse_stc
from sklearn.metrics import jaccard_score, mean_squared_error, f1_score

DEFAULT_CALIBRATION_METRICS = (
    "mean_calibration_error",
    "max_underconfidence_deviation",
    "max_overconfidence_deviation",
    "mean_absolute_deviation",
    "mean_signed_deviation",
)

DEFAULT_EVALUATION_METRICS = (
    "mean_posterior_std",
    "emd",
    "jaccard_error",
    "mse",
    "euclidean_distance",
    "f1",
    "accuracy",
)

class MetricEvaluator:
    def __init__(
        self,
        nominal_coverages: np.ndarray = None,
        evaluation_metrics: list[str] | None = None,
        calibration_metrics: list[str] | tuple[str, ...] | None = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize the MetricEvaluator with confidence levels, metrics, and a logger.

        Parameters
        ----------
        nominal_coverages : np.ndarray
            Nominal confidence levels (c) - what we expect theoretically.
        evaluation_metrics : list[str], optional
            List of metric names (method names) to evaluate (non-calibration).
        calibration_metrics : list[str], optional
            Names of metrics that should be treated as calibration-specific.
        logger : logging.Logger, optional
            Logger instance for logging debug and error messages.
        """
        self.nominal_coverages = nominal_coverages
        self.logger = logger

        if evaluation_metrics is None:
            evaluation_metrics = DEFAULT_EVALUATION_METRICS
        self.evaluation_metrics = tuple(evaluation_metrics)

        if calibration_metrics is None:
            calibration_metrics = DEFAULT_CALIBRATION_METRICS
        self.calibration_metrics = tuple(calibration_metrics)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['logger'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        if not hasattr(self, "evaluation_metrics"):
            legacy = state.get("metrics")
            if legacy is None:
                legacy = ()
            self.evaluation_metrics = tuple(legacy)
        if not hasattr(self, "calibration_metrics") or self.calibration_metrics is None:
            self.calibration_metrics = DEFAULT_CALIBRATION_METRICS

    # Calibration curve metrics
    def mean_calibration_error(self, empirical_coverages, **kwargs):
        """Calculate the area under the curve (AUC) deviation, which measures the average calibration error.
        Parameters
        ----------
        empirical_coverages : np.ndarray
            Empirical coverage values.
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverages': np.ndarray, empirical coverage values.
            
        Returns
        -------
        float
            The AUC deviation value.
        """
        delta_c = np.diff(self.nominal_coverages, prepend=self.nominal_coverages[0])
        abs_dev = np.abs(empirical_coverages - self.nominal_coverages)
        return np.sum(abs_dev * delta_c)

    def max_underconfidence_deviation(self, empirical_coverages, **kwargs):
        """
        Calculate the maximum positive deviation (MUD) from the confidence levels ().
        Parameters:
        
        MUD = max_i(ĉ_i - c_i)
        
        Represents the largest observed deviation due to underconfidence.
        Identifies the most conservative confidence level.        
        ----------
        empirical_coverages : np.ndarray
            Empirical coverage values.
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverages': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The maximum positive deviation value.
        """
        deviation = empirical_coverages - self.nominal_coverages
        return np.max(deviation)

    def max_overconfidence_deviation(self, empirical_coverages, **kwargs):
        """
        Calculate the maximum negative deviation from the confidence levels:
        
        MOD = -min_i(ĉ_i - c_i)
        
        It is Negated Minimal Deviation and represents the largest observed deviation due to overconfidence.
        Identifies the most overconfident confidence level.        
        Parameters
        ----------
        empirical_coverages : np.ndarray
            Empirical coverage values.
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverages': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The maximum negative deviation value.
        """
        deviation = empirical_coverages - self.nominal_coverages
        return -np.min(deviation) #TODO: check whether we need the minus here!

    def mean_absolute_deviation(self, empirical_coverages, **kwargs):
        """
        Calculate the mean absolute deviation from the confidence levels.
        
        MAD = (1/K) * Σ_{i=1}^{K} |ĉ_i - c_i|
        
        Measures the average magnitude of deviation between empirical and nominal coverage.
        Lower values indicate better overall calibration.
        
        ----------
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverages': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The maximum absolute deviation value.
        """
        deviation = empirical_coverages - self.nominal_coverages
        return np.mean(np.abs(deviation))

    def mean_signed_deviation(self, empirical_coverages, **kwargs):
        """
        Calculate the mean signed deviation from the confidence levels.
        Mean Signed Deviation (MSD)
        
        MSD = (1/K) * Σ_{i=1}^{K} (ĉ_i - c_i)
        
        Captures directional bias in calibration.
        - Positive values: systematic underconfidence (intervals are too wide)
        - Negative values: systematic overconfidence (intervals are too narrow)   
        ----------
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverages': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The mean signed deviation value.
        """
        deviation = empirical_coverages - self.nominal_coverages
        return np.mean(deviation)

    def mean_posterior_std(self, posterior_var, **kwargs):
        """
        Calculate the mean posterior standard deviation across all sources.
        
        This provides a single-number summary of the overall uncertainty level.
        Lower values indicate more confident estimates on average.
        
        Parameters
        ----------
        posterior_var : np.ndarray
            Posterior variance vector of shape (n_sources,).
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations
        Returns
        -------
        float
            The mean posterior standard deviation.
        """
        posterior_std = np.sqrt(posterior_var).reshape(-1, 1)
        # If a mask is needed, it should be an attribute of self, e.g., self.active_mask
        # For now, calculating mean over all available std values.
        # if hasattr(self, 'active_mask') and self.active_mask is not None:
        #     return {"mean_posterior_std": np.mean(posterior_std[self.active_mask])}
        return np.mean(posterior_std)


    def emd(self, x, x_hat, orientation_type, subject, fwd_path, **kwargs):
        """
        Compute Earth Mover's Distance (EMD) between true and estimated source activations.
        Adapted from BSI-ZOO
        Parameters:
        - x : (n_sources, n_times) or (n_sources, 3, n_times)
            Ground truth source time courses.
        - x_hat : same shape as x
            Estimated source time courses.
        - orientation_type : str
            'fixed' or 'free' for orientation modeling.
        - subject : str
            Subject ID used to locate the forward model.

        Returns:
        - float
            Earth Mover's Distance between normalized source distributions.
        """
        if orientation_type == "fixed":
            temp_true = np.linalg.norm(x, axis=1)
            a_mask = temp_true != 0
            a = temp_true[a_mask]

            temp_est = np.linalg.norm(x_hat, axis=1)
            b_mask = temp_est != 0
            b = temp_est[b_mask]
            
        elif orientation_type == "free":
            temp_true = np.linalg.norm(x, axis=2)
            temp_true = np.linalg.norm(temp_true, axis=1)
            a_mask = temp_true != 0
            a = temp_true[a_mask]

            temp_est = np.linalg.norm(x_hat, axis=2)
            temp_est = np.linalg.norm(temp_est, axis=1)
            b_mask = temp_est != 0
            b = temp_est[b_mask]
        else:
            raise ValueError(f"Unknown orientation_type: {orientation_type}")

        # Handle edge cases
        if len(a) == 0 or len(b) == 0:
            self.logger.warning("No active sources found for EMD computation")
            return np.inf

        # Load forward solution and extract source locations
        fwd = read_forward_solution(f"{fwd_path}-fwd.fif")
        fwd = convert_forward_solution(fwd, force_fixed=True)
        src = fwd["src"]

        # Create sparse source time courses
        stc_a = _make_sparse_stc(a[:, None], a_mask, fwd, tmin=1, tstep=1)
        stc_b = _make_sparse_stc(b[:, None], b_mask, fwd, tmin=1, tstep=1)

        # Extract source coordinates
        rr_a = np.r_[src[0]["rr"][stc_a.lh_vertno], src[1]["rr"][stc_a.rh_vertno]]
        rr_b = np.r_[src[0]["rr"][stc_b.lh_vertno], src[1]["rr"][stc_b.rh_vertno]]
        
        # Compute distance matrix between source locations
        M = cdist(rr_a, rr_b, metric="euclidean")

        # Normalize amplitudes to form probability distributions
        a_norm = a / a.sum()
        b_norm = b / b.sum()

        # Compute Earth Mover's Distance
        return emd2(a_norm, b_norm, M)

    def jaccard_error(self, x, x_hat, orientation_type=None, **kwargs):
        """
        TODO: To be checked!
        Calculate Jaccard error between true and estimated active source sets.
        
        Parameters
        ----------
        x : np.ndarray
            True source activations
        x_hat : np.ndarray  
            Estimated source activations
        orientation_type : str, optional
            'fixed' or 'free' for orientation modeling
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        float
            Jaccard error (1 - Jaccard index) between active source sets
        """
        # Convert continuous activations to binary (active/inactive)
        if orientation_type == "fixed":
            # For fixed orientation: check if source amplitude > threshold
            x_binary = (np.linalg.norm(x, axis=1) > 1e-10).astype(int)
            x_hat_binary = (np.linalg.norm(x_hat, axis=1) > 1e-10).astype(int)
        elif orientation_type == "free":
            # For free orientation: check if source amplitude > threshold
            x_binary = (np.linalg.norm(np.linalg.norm(x, axis=2), axis=1) > 1e-10).astype(int)
            x_hat_binary = (np.linalg.norm(np.linalg.norm(x_hat, axis=2), axis=1) > 1e-10).astype(int)
        
        # Compute Jaccard score for binary arrays
        jaccard_score_value = jaccard_score(x_binary, x_hat_binary, average='binary')
        
        return 1 - jaccard_score_value  # Convert to error (lower is better)

    def mse(self, x, x_hat, orientation_type, **kwargs):
        if orientation_type == "free":
            x = np.linalg.norm(x, axis=2)
            x_hat = np.linalg.norm(x_hat, axis=2)

        return mean_squared_error(x, x_hat)

    def _get_active_nnz(self, x, x_hat, orientation_type, subject, fwd_path, nnz):
        "adapted from BSI-ZOO"
        fwd = read_forward_solution(f"{fwd_path}-fwd.fif")

        if orientation_type == "fixed":
            fwd = convert_forward_solution(fwd, force_fixed=True)

            active_set = np.linalg.norm(x, axis=1) != 0

            # check if no vertices are estimated
            temp = np.linalg.norm(x_hat, axis=1)
            if len(np.unique(temp)) == 1:
                print("No vertices estimated!")

            temp_ = np.partition(-temp, nnz)
            max_temp = -temp_[:nnz]  # get n(=nnz) max amplitudes

            # remove 0 from list incase less vertices than nnz were estimated
            max_temp = np.delete(max_temp, np.where(max_temp == 0.0))
            active_set_hat = np.array(list(map(max_temp.__contains__, temp)))

            stc = _make_sparse_stc(
                x[active_set], active_set, fwd, tmin=1, tstep=1
            )  # ground truth
            stc_hat = _make_sparse_stc(
                x_hat[active_set_hat], active_set_hat, fwd, tmin=1, tstep=1
            )  # estimate

        elif orientation_type == "free":
            fwd = convert_forward_solution(fwd)

            # temp = np.linalg.norm
            active_set = np.linalg.norm(x, axis=2) != 0

            temp = np.linalg.norm(x_hat, axis=2)
            temp = np.linalg.norm(temp, axis=1)
            temp_ = np.partition(-temp, nnz)
            max_temp = -temp_[:nnz]  # get n(=nnz) max amplitudes
            max_temp = np.delete(max_temp, np.where(max_temp == 0.0))
            active_set_hat = np.array(list(map(max_temp.__contains__, temp)))
            active_set_hat = np.repeat(active_set_hat, 3).reshape(
                active_set_hat.shape[0], -1
            )

            stc = _make_sparse_stc(
                x[active_set], active_set, fwd, tmin=1, tstep=1
            )  # ground truth
            stc_hat = _make_sparse_stc(
                x_hat[active_set_hat], active_set_hat, fwd, tmin=1, tstep=1
            )  # estimate

        return stc, stc_hat, active_set, active_set_hat, fwd
    
    def euclidean_distance(self, x, x_hat, orientation_type, subject, nnz, fwd_path, **kwargs):
        "adapted from BSI-ZOO"
        stc, stc_hat, _, _, fwd = self._get_active_nnz(x, x_hat, orientation_type, subject, fwd_path, nnz)

        # euclidean distance check
        lh_coordinates = fwd["src"][0]["rr"][stc.lh_vertno]
        lh_coordinates_hat = fwd["src"][0]["rr"][stc_hat.lh_vertno]
        rh_coordinates = fwd["src"][1]["rr"][stc.rh_vertno]
        rh_coordinates_hat = fwd["src"][1]["rr"][stc_hat.rh_vertno]
        coordinates = np.concatenate([lh_coordinates, rh_coordinates], axis=0)
        coordinates_hat = np.concatenate([lh_coordinates_hat, rh_coordinates_hat], axis=0)
        euclidean_distance = np.linalg.norm(
            coordinates[: coordinates_hat.shape[0], :] - coordinates_hat, axis=1
        )

        return np.mean(euclidean_distance)

    def f1(self, x, x_hat, orientation_type, **kwargs):
        "adapted from BSI-ZOO"
        if orientation_type == "fixed":
            active_set = np.linalg.norm(x, axis=1) != 0
            active_set_hat = np.linalg.norm(x_hat, axis=1) != 0

        elif orientation_type == "free":
            temp = np.linalg.norm(x, axis=2)
            active_set = np.linalg.norm(temp, axis=1) != 0

            temp = np.linalg.norm(x_hat, axis=2)
            active_set_hat = np.linalg.norm(temp, axis=1) != 0

        return f1_score(active_set, active_set_hat)


    def accuracy(self, x, x_hat, orientation_type, **kwargs):
        """
        Calculate accuracy between true and estimated active source sets.
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        where:
        - TP: True Positives (correctly identified active sources)
        - TN: True Negatives (correctly identified inactive sources)  
        - FP: False Positives (incorrectly identified as active)
        - FN: False Negatives (missed active sources)
        
        Parameters
        ----------
        x : np.ndarray
            True source activations (n_sources, n_times) or (n_sources, 3, n_times)
        x_hat : np.ndarray  
            Estimated source activations (same shape as x)
        orientation_type : str
            'fixed' or 'free' for orientation modeling
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        float
            Accuracy score between 0.0 and 1.0 (higher is better)
        """
        # Convert continuous activations to binary (active/inactive)
        if orientation_type == "fixed":
            # For fixed orientation: check if source amplitude > threshold
            true_active = (np.linalg.norm(x, axis=1) > 1e-10).astype(int)
            pred_active = (np.linalg.norm(x_hat, axis=1) > 1e-10).astype(int)
        elif orientation_type == "free":
            # For free orientation: check if source amplitude > threshold
            temp_true = np.linalg.norm(x, axis=2)
            true_active = (np.linalg.norm(temp_true, axis=1) > 1e-10).astype(int)
            
            temp_pred = np.linalg.norm(x_hat, axis=2)
            pred_active = (np.linalg.norm(temp_pred, axis=1) > 1e-10).astype(int)
        else:
            raise ValueError(f"Unknown orientation_type: {orientation_type}")
        
        # Calculate confusion matrix components
        tp = np.sum((true_active == 1) & (pred_active == 1))  # True Positives
        tn = np.sum((true_active == 0) & (pred_active == 0))  # True Negatives
        fp = np.sum((true_active == 0) & (pred_active == 1))  # False Positives
        fn = np.sum((true_active == 1) & (pred_active == 0))  # False Negatives
        
        # Calculate accuracy
        total = tp + tn + fp + fn
        if total == 0:
            return 1.0  # Perfect accuracy when no sources exist
        
        accuracy_score = (tp + tn) / total
        return accuracy_score

    # Evaluate metrics and return results as a dictionary (side-effect free)
    def evaluate_metrics(self, which: str = "evaluation", **kwargs) -> dict:
        """Evaluate configured metrics and return a dict mapping metric names to values.

        Parameters
        ----------
        which : {"evaluation", "calibration", "all"}
            Which set of metrics to evaluate.
        kwargs : dict
            Keyword arguments passed to metric methods (e.g., empirical_coverages, cov, x, x_hat, orientation_type, subject, fwd_path, nnz).

        Returns
        -------
        dict
            Dictionary mapping metric names (or metric_name_error) to computed results.
        """
        results = {}

        groups = {
            "evaluation": self.evaluation_metrics,
            "calibration": self.calibration_metrics,
            "all": self.evaluation_metrics + self.calibration_metrics,
        }
        which = (which or "all").lower()
        metric_names = groups.get(which)

        if not metric_names:
            self.logger.warning("Unknown metric selection '%s'.", which)
            return results

        for metric_name_str in metric_names:
            try:
                if hasattr(self, metric_name_str):
                    method = getattr(self, metric_name_str)

                    if callable(method):
                        self.logger.debug(f"Calling metric method: {metric_name_str}")
                        result = method(**kwargs)
                        results[metric_name_str] = result
                    else:
                        self.logger.error(
                            f"Attribute '{metric_name_str}' found in {type(self).__name__} but it is not callable. Skipping."
                        )
                        results[f"{metric_name_str}_error"] = "Attribute not callable"
                else:
                    self.logger.error(
                        f"Metric method '{metric_name_str}' not found in {type(self).__name__}. Skipping."
                    )
                    results[f"{metric_name_str}_error"] = "Method not found"

            except Exception as e:
                logger.error(
                    f"Unexpected error evaluating metric method {metric_name_str} on '{type(self).__name__}': {e}",
                    exc_info=True,
                )
                results[f"{metric_name_str}_error"] = f"Execution error: {str(e)}"

        return results
