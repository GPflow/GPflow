from .base import Combination, Kernel, Product, Sum
from .convolutional import Convolutional
from .changepoints import ChangePoints
from .linears import Linear, Polynomial
from .misc import ArcCosine, Coregion
from . import multioutput

from .multioutput import (
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
    IndependentLatent,
    LinearCoregionalization,
)
from .periodic import Periodic
from .statics import Constant, Static, White
from .stationaries import (
    SquaredExponential,
    Cosine,
    Exponential,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    Stationary,
    IsotropicStationary,
    AnisotropicStationary,
)

Bias = Constant
RBF = SquaredExponential
