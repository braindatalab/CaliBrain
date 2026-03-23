from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import runpy
import numpy as np

from calibrain import MetricEvaluator, UncertaintyEstimator


def build_uncertainty_components(
    *, nominal_coverages: Iterable[float] | None, logger
) -> Tuple[UncertaintyEstimator, MetricEvaluator]:
    coverages = (
        np.asarray(list(nominal_coverages), dtype=float)
        if nominal_coverages is not None
        else np.linspace(0.1, 0.999, num=10)
    )
    uncertainty_estimator = UncertaintyEstimator(
        nominal_coverages=coverages,
        logger=logger,
    )
    metrics_config = {
        "evaluation": (
            "mean_posterior_std",
            "emd",
        ),
        "calibration": (
            "mean_signed_deviation",
            "mean_absolute_deviation",
            "max_underconfidence_deviation",
            "max_overconfidence_deviation",
        ),
    }
    metric_evaluator = MetricEvaluator(
        nominal_coverages=coverages,
        evaluation_metrics=metrics_config["evaluation"],
        calibration_metrics=metrics_config["calibration"],
        logger=logger,
    )
    return uncertainty_estimator, metric_evaluator


def load_python_config(config_path: Path | str) -> Dict[str, Any]:
    namespace = runpy.run_path(str(config_path))
    if "CONFIG" not in namespace:
        raise ValueError(f"Config {config_path} must define a CONFIG dict.")
    return namespace["CONFIG"]
