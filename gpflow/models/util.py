import numpy as np

from ..features import InducingPoints


def inducingpoint_wrapper(feature):
    """
    Models which used to take only Z can now pass `feature` and `Z` to this method. This method will
    check for consistency and return the correct feature. This allows backwards compatibility in
    for the methods.
    """
    if isinstance(feature, np.ndarray):
        feature = InducingPoints(feature)
    return feature
