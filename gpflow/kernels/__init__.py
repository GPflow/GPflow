from .base import Combination, Kernel, Product, Sum
from .linears import Linear, Polynomial
from .misc import ArcCosine, Coregion, Periodic
from .statics import Constant, Static, White
from .stationaries import (RBF, Cosine, Exponential, Matern12, Matern32,
                           Matern52, RationalQuadratic, Stationary)

Bias = Constant
SquaredExponential = RBF