from multipledispatch import dispatch

from . import features, kernels, mean_functions, GPflowError
from .quadrature import mvnquad


class ProbabilityDistribution:
    pass


class Gaussian(ProbabilityDistribution):
    def __init__(self, mu, cov):
        self.mu = mu  # N x D
        self.cov = cov  # N x D x D


class DiagonalGaussian(ProbabilityDistribution):
    def __init__(self, mu, vars):
        self.mu = mu  # N x D
        self.vars = vars  # N x D


class Uniform(ProbabilityDistribution):
    def __init__(self, a, b):
        self.a = a  # N x D
        self.b = b


def get_eval_func(obj):
    if type(obj) is tuple:
        # Feature + kernel combination
        if not isinstance(obj[0], features.InducingFeature) or not isinstance(obj[1], kernels.Kern):
            raise GPflowError("Tuples should contain an InducingFeature and Kern, in that order.")

        return lambda x, *args, **kwargs: obj[0].Kuf(obj[1], x, *args, **kwargs)
    elif isinstance(obj, mean_functions.MeanFunction):
        return lambda x, *args, **kwargs: obj(x, *args, **kwargs)
    elif isinstance(obj, kernels.Kern):
        return lambda x, *args, **kwargs: obj.K(x, *args, **kwargs)
    elif obj is None:
        return lambda _, *args, *kwargs: 1.0
    else:
        raise NotImplementedError()


@dispatch(object, object, Gaussian)
def expectation(obj1, obj2, p):
    eval_func = lambda x: get_eval_func(obj1)(x) * get_eval_func(obj2)(x)

    mvnquad(eval_func, p.mu, p.covs, 20, p.mu.shape[1].value)
