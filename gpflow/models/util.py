import numpy as np

from ..inducing_variables import InducingVariables, InducingPoints


def inducingpoint_wrapper(inducing_variable):
    """
    This wrapper allows transparently passing either an InducingVariables
    object or an array specifying InducingPoints positions.
    """
    if not isinstance(inducing_variable, InducingVariables):
        inducing_variable = InducingPoints(inducing_variable)
    return inducing_variable
