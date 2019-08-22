import numpy as np

from ..inducing_variables import InducingPoints


def inducingpoint_wrapper(inducing_variable):
    """
    Models which used to take only Z can now pass `inducing_variable` and `Z` to this method. This method will
    check for consistency and return the correct inducing_variable. This allows backwards compatibility in
    for the methods.
    """
    if isinstance(inducing_variable, np.ndarray):
        inducing_variable = InducingPoints(inducing_variable)
    return inducing_variable
