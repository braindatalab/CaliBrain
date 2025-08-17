import numpy as np
import pandas as pd
import logging # Import logging

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
    def mean_calibration_error(self, **kwargs):
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
        empirical_coverage = kwargs.get('empirical_coverage', None)
        delta_c = np.diff(self.confidence_levels, prepend=self.confidence_levels[0])
        abs_dev = np.abs(empirical_coverage - self.confidence_levels)
        return np.sum(abs_dev * delta_c)

    def max_underconfidence_deviation(self, **kwargs):
        """Calculate the maximum positive deviation from the confidence levels ().
        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverage': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The maximum positive deviation value.
        """
        empirical_coverage = kwargs.get('empirical_coverage', None)
        deviation = empirical_coverage - self.confidence_levels
        return np.max(deviation)

    def max_overconfidence_deviation(self, **kwargs):
        """Calculate the maximum negative deviation from the confidence levels.
        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments that may be needed for metric calculations:
            - 'empirical_coverage': np.ndarray, empirical coverage values.
        Returns
        -------
        float
            The maximum negative deviation value.
        """
        empirical_coverage = kwargs.get('empirical_coverage', None)
        deviation = empirical_coverage - self.confidence_levels
        return -np.min(deviation) #TODO: check whether we need the minus here!

    def mean_absolute_deviation(self, **kwargs):
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
        empirical_coverage = kwargs.get('empirical_coverage', None)
        deviation = empirical_coverage - self.confidence_levels
        return np.mean(np.abs(deviation))

    def mean_signed_deviation(self, **kwargs):
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
        empirical_coverage = kwargs.get('empirical_coverage', None)
        deviation = empirical_coverage - self.confidence_levels
        return np.mean(deviation)

    def mean_posterior_std(self, **kwargs):
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
        cov = kwargs.get('cov', None)
        posterior_std = np.sqrt(np.diag(cov))
        # If a mask is needed, it should be an attribute of self, e.g., self.active_mask
        # For now, calculating mean over all available std values.
        # if hasattr(self, 'active_mask') and self.active_mask is not None:
        #     return {"mean_posterior_std": np.mean(posterior_std[self.active_mask])}
        return np.mean(posterior_std)

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