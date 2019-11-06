from .base import Combination, Kernel, Product, Sum
from .changepoints import ChangePoints
from .linears import Linear, Polynomial
from .misc import ArcCosine, Coregion, Periodic
from .mo_kernels import (MultioutputKernel, SeparateIndependent, SharedIndependent, IndependentLatent, LinearCoregionalization)
from .statics import Constant, Static, White
from .stationaries import (SquaredExponential, Cosine, Exponential, Matern12, Matern32, Matern52, RationalQuadratic, Stationary)
from .convolutional import Convolutional

Bias = Constant
RBF = SquaredExponential
