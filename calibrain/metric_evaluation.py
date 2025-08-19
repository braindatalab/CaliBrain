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

class MetricEvaluator:
    def __init__(self, confidence_levels : np.ndarray = None, metrics : list[str] = None, logger : logging.Logger = None):
        """
        Initialize the MetricEvaluator with confidence levels, metrics, and a logger.
        Parameters
        ----------
        confidence_levels : np.ndarray, optional
            Array of confidence levels to evaluate metrics against.
        metrics : list[str], optional
            List of metric names (method names) to evaluate.
        logger : logging.Logger, optional
            Logger instance for logging debug and error messages.
        """
        self.confidence_levels = confidence_levels
        self.metrics = metrics if metrics is not None else []
        self.logger = logger

    # Calibration curve metrics
    def mean_calibration_error(self, empirical_coverage, **kwargs):
        """Calculate the area under the curve (AUC) deviation, which measures the average calibration error.
        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverage': np.ndarray, empirical coverage values.
            
        Returns
        -------
        float
            The AUC deviation value.
        """
        delta_c = np.diff(self.confidence_levels, prepend=self.confidence_levels[0])
        abs_dev = np.abs(empirical_coverage - self.confidence_levels)
        return np.sum(abs_dev * delta_c)

    def max_underconfidence_deviation(self, empirical_coverage, **kwargs):
        """Calculate the maximum positive deviation from the confidence levels ().
        Parameters
        ----------
        empirical_coverage : np.ndarray
            Empirical coverage values.
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverage': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The maximum positive deviation value.
        """
        deviation = empirical_coverage - self.confidence_levels
        return np.max(deviation)

    def max_overconfidence_deviation(self, empirical_coverage, **kwargs):
        """Calculate the maximum negative deviation from the confidence levels.
        Parameters
        ----------
        empirical_coverage : np.ndarray
            Empirical coverage values.
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverage': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The maximum negative deviation value.
        """
        deviation = empirical_coverage - self.confidence_levels
        return -np.min(deviation) #TODO: check whether we need the minus here!

    def mean_absolute_deviation(self, empirical_coverage, **kwargs):
        """Calculate the mean absolute deviation from the confidence levels.
        ----------
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverage': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The maximum absolute deviation value.
        """
        deviation = empirical_coverage - self.confidence_levels
        return np.mean(np.abs(deviation))

    def mean_signed_deviation(self, empirical_coverage, **kwargs):
        """Calculate the mean signed deviation from the confidence levels.
        ----------
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverage': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The mean signed deviation value.
        """
        deviation = empirical_coverage - self.confidence_levels
        return np.mean(deviation)

    def mean_posterior_std(self, cov, **kwargs):
        """Calculate the mean posterior standard deviation.
        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'cov': np.ndarray, covariance matrix for uncertainty metrics.
        Returns
        -------
        float
            The mean posterior standard deviation.
        """
        posterior_std = np.sqrt(np.diag(cov))
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
            temp = np.linalg.norm(x, axis=1)
            a_mask = temp != 0
            a = temp[a_mask]

            temp = np.linalg.norm(x_hat, axis=1)
            b_mask = temp != 0
            b = temp[b_mask]
            # temp_ = np.partition(-temp, nnz)
            # b = -temp_[:nnz]  # get n(=nnz) max amplitudes
            # b = -temp_[:nnz]  # get n(=nnz) max amplitudes
        
        elif orientation_type == "free":
            temp = np.linalg.norm(x, axis=2)
            temp = np.linalg.norm(temp, axis=1)
            a_mask = temp != 0
            a = temp[a_mask]

            temp = np.linalg.norm(x_hat, axis=2)
            temp = np.linalg.norm(temp, axis=1)
            b_mask = temp != 0
            b = temp[b_mask]
            # temp_ = np.partition(-temp, nnz)
            # b = -temp_[:nnz]  # get n(=nnz) max amplitudes

        
        # Step 3: Load the forward solution and extract source locations
        fwd = read_forward_solution(f"{fwd_path}/{subject}-fwd.fif")

        # if orientation_type == "fixed":
        #     if fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
        #         fwd = convert_forward_solution(fwd, force_fixed=True) # surf_ori=True

        fwd = convert_forward_solution(fwd, force_fixed=True)
        src = fwd["src"]

        stc_a = _make_sparse_stc(a[:, None], a_mask, fwd, tmin=1, tstep=1)
        stc_b = _make_sparse_stc(b[:, None], b_mask, fwd, tmin=1, tstep=1)

        rr_a = np.r_[src[0]["rr"][stc_a.lh_vertno], src[1]["rr"][stc_a.rh_vertno]]
        rr_b = np.r_[src[0]["rr"][stc_b.lh_vertno], src[1]["rr"][stc_b.rh_vertno]]
        M = cdist(rr_a, rr_b, metric="euclidean")

        # Normalize a and b as EMD is defined between probability distributions
        a /= a.sum()
        b /= b.sum()

        return emd2(a, b, M)

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
        fwd = read_forward_solution(f"{fwd_path}/{subject}-fwd.fif")

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

    # Evaluate and store metrics
    def evaluate_and_store_metrics(self, current_results_dict : dict, metric_suffix="", **kwargs):
        """Evaluate metrics and update the results dictionary.

        Parameters
        ----------
        current_results_dict : dict
            Dictionary to store the results of the metrics.
        metric_suffix : str, optional
            Suffix to add to metric keys (e.g., "_all_sources", "_active_indices").
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverage': np.ndarray, empirical coverage values.
            - 'cov': np.ndarray, covariance matrix for uncertainty metrics.
        """
        if not self.metrics: # Handles if self.metrics is an empty list
            self.logger.info(f"No metrics to call for suffix '{metric_suffix}' (self.metrics is empty).")
            return

        self.logger.debug(
            f"Evaluating metrics with suffix: '{metric_suffix}' from self.metrics: {self.metrics} "
            f"for instance of {type(self).__name__}"
        )

        for metric_name_str in self.metrics:
            metric_output = {} # Initialize for each metric
            
            try:
                if hasattr(self, metric_name_str):
                    method = getattr(self, metric_name_str)
                    
                    if callable(method):
                        self.logger.debug(f"Calling metric method: {metric_name_str} with suffix '{metric_suffix}'")

                        # Call the metric method with kwargs, which should contain necessary parameters
                        result = method(**kwargs)

                        # Wrap scalar outputs into a dict
                        metric_output = {f"{metric_name_str}{metric_suffix}": result}
                    else:
                        self.logger.error(
                            f"Attribute '{metric_name_str}' found in {type(self).__name__} but it is not callable "
                            f"(suffix: '{metric_suffix}'). Skipping."
                        )
                        metric_output = {f"{metric_name_str}{metric_suffix}_error": "Attribute not callable"}
                else:
                    self.logger.error(
                        f"Metric method '{metric_name_str}' not found in {type(self).__name__} "
                        f"(suffix: '{metric_suffix}'). Skipping."
                    )
                    metric_output = {f"{metric_name_str}{metric_suffix}_error": "Method not found"}
            
            except Exception as e:
                self.logger.error(
                    f"Unexpected error evaluating metric method {metric_name_str} (suffix: '{metric_suffix}') "
                    f"on '{type(self).__name__}': {e}", exc_info=True
                )
                metric_output = {f"{metric_name_str}{metric_suffix}_error": f"Execution error: {str(e)}"}
            
            current_results_dict.update(metric_output)