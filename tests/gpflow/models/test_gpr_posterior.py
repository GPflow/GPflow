from itertools import product
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.models.gpr import GP_deprecated, GP_with_posterior
from gpflow.posteriors import PrecomputeCacheType

input_dim = 7
output_dim = 1

def make_models(regression_data):
    k = gpflow.kernels.Matern52()

    mold = GP_deprecated(data=regression_data, kernel=k)
    mnew = GP_with_posterior(data=regression_data, kernel=k)
    return mold, mnew


@pytest.mark.parametrize("cache_type", [PrecomputeCacheType.TENSOR, PrecomputeCacheType.VARIABLE])
def test_old_vs_new_gp_vs_new_with_posterior(cache_type:PrecomputeCacheType):
    X = np.random.randn(100, input_dim)
    Y = np.random.randn(100, output_dim)
    Xt = tf.convert_to_tensor(X)
    Yt = tf.convert_to_tensor(Y)
    mold, mnew = make_models((Xt,Yt))
    X_new = np.random.randn(100, input_dim)
    Xt_new = tf.convert_to_tensor(X_new)

    for full_cov in (True, False):
        for full_output_cov in (True, False):
            mu_old, var2_old = mold.predict_f(Xt_new, full_cov=full_cov, full_output_cov=full_output_cov)
            mu_new_fuse, var2_new_fuse = mnew.predict_f(Xt_new, full_cov=full_cov, full_output_cov=full_output_cov)
            mu_new_cache, var2_new_cache = mnew.posterior(cache_type).predict_f(Xt_new, full_cov=full_cov, full_output_cov=full_output_cov)
            # check new fuse is same as old version
            np.testing.assert_allclose(mu_new_fuse, mu_old)
            np.testing.assert_allclose(var2_new_fuse, var2_old)
            # check new cache is same as old version
            np.testing.assert_allclose(mu_old,mu_new_cache)
            np.testing.assert_allclose(var2_old,var2_new_cache)
