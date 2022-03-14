from . import multioutput
from .base import Combination, Kernel, Product, Sum
from .changepoints import ChangePoints
from .convolutional import Convolutional
from .linears import Linear, Polynomial
from .misc import ArcCosine, Coregion
from .multioutput import (
    IndependentLatent,
    LinearCoregionalization,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
)
from .periodic import Periodic
from .statics import Constant, Static, White

from .stationaries import (  # isort:skip
    # base classes:
    Stationary,
    IsotropicStationary,
    AnisotropicStationary,
    # actual kernel classes:
    Cosine,
    Exponential,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    SquaredExponential,
)

Bias = Constant
RBF = SquaredExponential

__all__ = [
    "AnisotropicStationary",
    "ArcCosine",
    "Bias",
    "ChangePoints",
    "Combination",
    "Constant",
    "Convolutional",
    "Coregion",
    "Cosine",
    "Exponential",
    "IndependentLatent",
    "IsotropicStationary",
    "Kernel",
    "Linear",
    "LinearCoregionalization",
    "Matern12",
    "Matern32",
    "Matern52",
    "MultioutputKernel",
    "Periodic",
    "Polynomial",
    "Product",
    "RBF",
    "RationalQuadratic",
    "SeparateIndependent",
    "SharedIndependent",
    "SquaredExponential",
    "Static",
    "Stationary",
    "Sum",
    "White",
    "base",
    "changepoints",
    "convolutional",
    "linears",
    "misc",
    "multioutput",
    "periodic",
    "statics",
    "stationaries",
]
