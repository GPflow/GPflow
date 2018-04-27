# Copyright 2018 the GPflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
import tensorflow as tf
import gpflow.kernels as kernels
from gpflow.test_util import session_tf
from gpflow import settings


rng = np.random.RandomState(0)


class Datum:
    num_data = 100
    D = 100
    X = rng.rand(num_data, D) * 100


@pytest.mark.parametrize('kernel', [kernels.Matern12, kernels.Matern32, kernels.Matern52, kernels.Exponential, kernels.Cosine])
def test_kernel_euclidean_distance(session_tf, kernel):
    '''
    Tests output & gradients of kernels that are a function of the (scaled) euclidean distance
    of the points. We test on a high dimensional space, which can generate very small distances
    causing the scaled_square_dist to generate some negative values.
    '''
    
    k = kernel(Datum.D)
    K = k.compute_K_symm(Datum.X)
    assert not np.isnan(K).any(), 'There are NaNs in the output of the ' + kernel.__name__ + ' kernel.'
    assert np.isfinite(K).all(), 'There are Infs in the output of the ' + kernel.__name__ + ' kernel.'

    X = tf.placeholder(settings.float_type)
    dK = session_tf.run(tf.gradients(k.K(X, X), X)[0], feed_dict={X: Datum.X})
    assert not np.isnan(dK).any(), 'There are NaNs in the gradient of the ' + kernel.__name__ + ' kernel.'
    assert np.isfinite(dK).all(), 'There are Infs in the output of the ' + kernel.__name__ + ' kernel.'
