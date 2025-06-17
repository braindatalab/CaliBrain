import numpy as np
import pandas as pd
import logging # Import logging

class EvaluationMetrics:
    def __init__(
        self,
        x=None,
        x_hat=None,
        subject=None,
        orientation_type=None,
        nnz=None,
        y=None,
        L=None,
        cov=None,
        confidence_levels=None,
        empirical_coverage=None,
        metrics_to_call=None, # List of method name strings
        logger=None
    ):
        self.x = x
        self.x_hat = x_hat
        self.subject = subject
        self.orientation_type = orientation_type
        self.nnz = nnz
        self.y = y
        self.L = L
        self.cov = cov
        self.confidence_levels = confidence_levels
        self.empirical_coverage = empirical_coverage
        self.metrics_to_call = metrics_to_call if metrics_to_call is not None else []
        self.logger = logger
        
    def calibration_curve_metrics(self):
        delta_c = np.diff(self.confidence_levels, prepend=self.confidence_levels[0])
        deviation = self.empirical_coverage - self.confidence_levels
        abs_deviation = np.abs(deviation)

        auc_deviation = np.sum(abs_deviation * delta_c)
        max_pos_dev = np.max(deviation) # underconfidence
        max_neg_dev = np.min(deviation) # overconfidence
        max_abs_dev = np.max(abs_deviation)

        return {
            "AUC_deviation": auc_deviation,
            "max_positive_deviation": max_pos_dev,
            "max_negative_deviation": max_neg_dev,
            "max_absolute_deviation": max_abs_dev,
        }

    def mean_posterior_std(self):
        posterior_std = np.sqrt(np.diag(self.cov))
        # If a mask is needed, it should be an attribute of self, e.g., self.active_mask
        # For now, calculating mean over all available std values.
        # if hasattr(self, 'active_mask') and self.active_mask is not None:
        #     return {"mean_posterior_std": np.mean(posterior_std[self.active_mask])}
        return {"mean_posterior_std": np.mean(posterior_std)}
    
    def evaluate_and_store_metrics(self, current_results_dict, metric_suffix=""):
        """
        Evaluates methods listed in self.metrics_to_call and updates current_results_dict.
        
        Parameters:
        - current_results_dict (dict): The dictionary to update with metric scores.
        - metric_suffix (str): Suffix to add to metric keys (e.g., "_active_set").
        """
        if not self.metrics_to_call: # Handles if self.metrics_to_call is an empty list
            self.logger.info(f"No metrics to call for suffix '{metric_suffix}' (self.metrics_to_call is empty).")
            return

        self.logger.info(
            f"Evaluating metrics with suffix: '{metric_suffix}' from self.metrics_to_call: {self.metrics_to_call} "
            f"for instance of {type(self).__name__}"
        )

        for metric_name_str in self.metrics_to_call: # metric_name_str is a string from the list
            metric_output_dict_for_update = {} # Initialize for each metric
            try:
                if hasattr(self, metric_name_str):
                    method_to_call = getattr(self, metric_name_str)
                    if callable(method_to_call):
                        self.logger.debug(f"Calling metric method: {metric_name_str} with suffix '{metric_suffix}'")
                        returned_value = method_to_call() # "Evaluate the string" by calling the method

                        if isinstance(returned_value, dict):
                            # Add suffix to each key in the returned dictionary
                            metric_output_dict_for_update = {k + metric_suffix: v for k, v in returned_value.items()}
                        else:
                            # This case should ideally be minimized by ensuring metric methods return dicts.
                            self.logger.warning(
                                f"Method {metric_name_str} (suffix: {metric_suffix}) on "
                                f"{type(self).__name__} did not return a dictionary as expected. "
                                f"Returned: {returned_value}. Wrapping it as '{metric_name_str + metric_suffix}'."
                            )
                            metric_output_dict_for_update = {metric_name_str + metric_suffix: returned_value}
                    else:
                        self.logger.error(
                            f"Attribute '{metric_name_str}' found in {type(self).__name__} but it is not callable "
                            f"(suffix: '{metric_suffix}'). Skipping."
                        )
                        metric_output_dict_for_update = {f"{metric_name_str}{metric_suffix}_error": "Attribute not callable"}
                else:
                    self.logger.error(
                        f"Metric method '{metric_name_str}' not found in {type(self).__name__} "
                        f"(suffix: '{metric_suffix}'). Skipping."
                    )
                    metric_output_dict_for_update = {f"{metric_name_str}{metric_suffix}_error": "Method not found"}
            
            except Exception as e:
                self.logger.error(
                    f"Unexpected error evaluating metric method {metric_name_str} (suffix: '{metric_suffix}') "
                    f"on '{type(self).__name__}': {e}", exc_info=True
                )
                metric_output_dict_for_update = {f"{metric_name_str}{metric_suffix}_error": f"Execution error: {str(e)}"}
            
            current_results_dict.update(metric_output_dict_for_update)