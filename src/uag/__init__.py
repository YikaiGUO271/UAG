"""Universal Accelerated Gradient method for unconstrained smooth optimization."""
from ._api import OptimizeResult, WorkStats, minimize

__version__ = "0.1.0"
__all__ = ["minimize", "OptimizeResult", "WorkStats", "__version__"]
