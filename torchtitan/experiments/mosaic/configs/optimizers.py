from dataclasses import dataclass
from torchtitan.config import Optimizer as BaseOptimizer

@dataclass
class MosaicOptimizerConfig(BaseOptimizer):
    """
    Mosaic-specific optimizer config with additional hyperparameters.
    """
    v1: float = 0.0
    """v1 hyperparameter for quasi-hyperbolic optimizers"""