from calibrain.leadfield_builder import LeadfieldBuilder
from calibrain.source_simulation import SourceSimulator
from calibrain.sensor_simulation import SensorSimulator
from calibrain.source_estimation import SourceEstimator, gamma_map_sflex, BMN, gamma_lambda_map_sflex, BMN_joint
from calibrain.uncertainty_estimation import UncertaintyEstimator
from calibrain.metric_evaluation import MetricEvaluator
from calibrain.uncertainty_calibration import UncertaintyCalibrator
from calibrain.visualization import Visualizer
from calibrain.data_generation import DataGenerator
from calibrain.datasets import fetch_example_data

__all__ = [
    "DataGenerator",
    "fetch_example_data",
    "extract_leadfields",
    "reduce_forward_solutions",
    "LeadfieldBuilder",
    "SourceSimulator",
    "SensorSimulator",
    "SourceEstimator",
    "UncertaintyEstimator",
    "UncertaintyCalibrator",
    "MetricEvaluator",
    "Visualizer",
    "gamma_map_sflex",
    "BMN",
    "BMN_joint",
    "gamma_lambda_map_sflex",
]

__version__ = "1.1.0"

# These two are imported on first use. ``python -m calibrain.reduce_fwd`` and
# ``python -m calibrain.extract_leadfield`` run those modules as scripts, and
# importing them here first would make Python warn that they were already imported.
_LAZY = {
    "extract_leadfields": "calibrain.extract_leadfield",
    "reduce_forward_solutions": "calibrain.reduce_fwd",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'calibrain' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
