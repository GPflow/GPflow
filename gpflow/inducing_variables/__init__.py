from .inducing_variables import InducingVariables, InducingPoints, Multiscale
from .inducing_patch import InducingPatches
from .multioutput import inducing_variables as mo_inducing_variables
from .multioutput.inducing_variables import (
    MultioutputInducingVariables,
    FallbackSharedIndependentInducingVariables,
    FallbackSeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
    SeparateIndependentInducingVariables,
)
