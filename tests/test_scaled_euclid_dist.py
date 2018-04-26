import pytest
import numpy as np
import tensorflow as tf
from gpflow.test_util import session_tf
from gpflow.kernels import Matern12, Matern32, Matern52, Exponential, Cosine
from gpflow import settings
float_type = settings.float_type

rng = np.random.RandomState(0)

class Datum:
    num_data = 100
    D = 100
    X = rng.rand(num_data, D)*100

@pytest.mark.parametrize('kernel', ['Matern12', 'Matern32', 'Matern52', 'Exponential', 'Cosine'])
def test_error(session_tf, kernel):
    '''
    Tests output & gradients of kernels that are a function of the (scaled) euclidean distance
    of the points. We test on a high dimensional space, which can generate very small distances
    causing the scaled_square_dist to generate some negative values.
    '''
    kernels = {'Matern12': Matern12, 'Matern32': Matern32,
               'Matern52': Matern52, 'Exponential': Exponential,
               'Cosine': Cosine}

    k = kernels[kernel](Datum.D)

    K = k.compute_K_symm(Datum.X)
    assert not np.isnan(K).any(), 'There are NaNs in the output of the ' + kernel + ' kernel.'

    X = tf.placeholder(float_type)
    dK = session_tf.run(tf.gradients(k.K(X, X), X)[0], feed_dict={X: Datum.X})
    assert not np.isnan(dK).any(), 'There are NaNs in the gradient of the ' + kernel + ' kernel.'
