# Copyright 2017 the GPflow authors.
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

import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose
import pytest

import gpflow
from gpflow.test_util import session_tf
from gpflow import settings
from gpflow import kernels

# TODO: kernels.Coregion

Kerns = [
    # Static
    kernels.White,
    kernels.Constant,

    # Stationary
    kernels.RBF,
    kernels.RationalQuadratic,
    kernels.Exponential,
    kernels.Matern12,
    kernels.Matern32,
    kernels.Matern52,
    kernels.Cosine,

    kernels.Linear,
    kernels.Polynomial,
    kernels.ArcCosine,
    kernels.Periodic,
]

def _test_no_active_dims(Kern, sess):
    S, N, M, D = 5, 4, 3, 2
    X1 = tf.identity(np.random.randn(S, N, D))
    X2 = tf.identity(np.random.randn(S, M, D))
    kern = Kern(D) + gpflow.kernels.White(2)

    compare_vs_map(X1, X2, kern, sess)

def _test_slice_active_dims(Kern, sess):
    S, N, M, D = 5, 4, 3, 4
    d = 2
    X1 = tf.identity(np.random.randn(S, N, D))
    X2 = tf.identity(np.random.randn(S, M, D))
    kern = Kern(d, active_dims=slice(1, 1+d))

    compare_vs_map(X1, X2, kern, sess)

def _test_indices_active_dims(Kern, sess):
    S, N, M, D = 5, 4, 3, 4

    X1 = tf.identity(np.random.randn(S, N, D))
    X2 = tf.identity(np.random.randn(S, M, D))
    kern = Kern(2, active_dims=[1, 3])

    compare_vs_map(X1, X2, kern, sess)


def compare_vs_map(X1, X2, kern, sess):
    K12_map = tf.map_fn(lambda x: kern.K(x[0], x[1]), [X1, X2], dtype=settings.float_type)
    K12_native = kern.K(X1, X2)
    assert_allclose(*sess.run([K12_map, K12_native]))

    K_map = tf.map_fn(kern.K, X1, dtype=settings.float_type)
    K_native = kern.K(X1)
    assert_allclose(*sess.run([K_map, K_native]))

    Kdiag_map = tf.map_fn(kern.Kdiag, X1, dtype=settings.float_type)
    Kdiag_native = kern.Kdiag(X1)
    assert_allclose(*sess.run([Kdiag_map, Kdiag_native]))

def test_rbf_no_active_dims(session_tf):
    _test_no_active_dims(gpflow.kernels.RBF, session_tf)

def test_rbf_slice_active_dims(session_tf):
    _test_slice_active_dims(gpflow.kernels.RBF, session_tf)

def test_rbf_indices_active_dims(session_tf):
    _test_indices_active_dims(gpflow.kernels.RBF, session_tf)

@pytest.mark.parametrize("Kern", Kerns)
def test_all_no_active_dims(session_tf, Kern):
    _test_no_active_dims(Kern, session_tf)

