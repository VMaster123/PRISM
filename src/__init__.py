"""
PRISM: Physics-Representation Interpretability for Structural Mechanics / PDE-Constrained Engineering Design
"""

from .pde_solver import HeatPDE2D, SIMPOptimizer, generate_pde_dataset
from .models import SpatialAutoencoder, SparseAutoencoder
from .interpretability import PRISMInterventionEngine

__all__ = [
    "HeatPDE2D",
    "SIMPOptimizer",
    "generate_pde_dataset",
    "SpatialAutoencoder",
    "SparseAutoencoder",
    "PRISMInterventionEngine",
]
