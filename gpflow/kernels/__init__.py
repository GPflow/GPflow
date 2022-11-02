r"""
:class:`Kernel <gpflow.kernels.Kernel>` s form a core component of GPflow models and allow prior information to
be encoded about a latent function of interest.
For an introduction to kernels, see `Kernels <https://gpflow.github.io/GPflow/develop/notebooks/getting_started/kernels.html>`_
in our Getting Started guide. The effect of choosing
different kernels, and how it is possible to combine multiple kernels is shown
in the `"Using kernels in GPflow" notebook <notebooks/kernels.html>`_.

Broadcasting over leading dimensions:
`kernel.K(X1, X2)` returns the kernel evaluated on every pair in X1 and X2.
E.g. if X1 has shape [S1, N1, D] and X2 has shape [S2, N2, D], kernel.K(X1, X2)
will return a tensor of shape [S1, N1, S2, N2]. Similarly, kernel.K(X1, X1)
returns a tensor of shape [S1, N1, S1, N1]. In contrast, the return shape of
kernel.K(X1) is [S1, N1, N1]. (Without leading dimensions, the behaviour of
kernel.K(X, None) is identical to kernel.K(X, X).)
"""

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
