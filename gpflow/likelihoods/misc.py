from .base import MonteCarloLikelihood
from .scalar_continuous import Gaussian


class GaussianMC(MonteCarloLikelihood, Gaussian):
    """
    Stochastic version of Gaussian likelihood for demonstration purposes only.
    """

    pass
