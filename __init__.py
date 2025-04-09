from .calibrain.benchmark import Benchmark
from .calibrain.leadfield_simulation import DipoleSimulation
from .calibrain.inverse_estimation import SourceEstimator

__all__ = [
    "Benchmark",
    "DipoleSimulation",
    "SourceEstimator",
]
__version__ = "1.0.0"