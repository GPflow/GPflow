from . import multioutput
from .inducing_patch import InducingPatches
from .inducing_variables import InducingPoints, InducingVariables, Multiscale
from .multioutput import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    MultioutputInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)

__all__ = [
    "FallbackSeparateIndependentInducingVariables",
    "FallbackSharedIndependentInducingVariables",
    "InducingPatches",
    "InducingPoints",
    "InducingVariables",
    "MultioutputInducingVariables",
    "Multiscale",
    "SeparateIndependentInducingVariables",
    "SharedIndependentInducingVariables",
    "inducing_patch",
    "inducing_variables",
    "multioutput",
]
