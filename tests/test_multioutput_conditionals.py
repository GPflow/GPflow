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
# limitations under the License.from __future__ import print_function

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import settings
from gpflow.conditionals import conditional
from gpflow.multikernels import ListKernel
from gpflow.test_util import GPflowTestCase
from numpy.testing import assert_allclose


class IdenticalOutputTest(GPflowTestCase):
    def prepare(self):
        num_latent = 2
        num_data = 3
        k = gpflow.kernels.Matern32(1)
        listk = ListKernel([gpflow.kernels.Matern32(1) for _ in range(num_latent)])
        X = tf.placeholder(settings.float_type, shape=[None, None])
        mu = tf.placeholder(settings.float_type, shape=[None, None])
        Xs = tf.placeholder(settings.float_type, shape=[None, None])
        sqrt = tf.placeholder(settings.float_type, shape=[num_latent, num_data, num_data])

        rng = np.random.RandomState(0)
        X_data = rng.randn(num_data, 1)
        mu_data = rng.randn(num_data, num_latent)
        sqrt_data = rng.randn(num_latent, num_data, num_data)
        Xs_data = rng.randn(50, 1)

        feed_dict = {X: X_data, Xs: Xs_data, mu: mu_data, sqrt: sqrt_data}
        k.compile()

        # the chols are diagonal matrices, with the same entries as the diag representation.
        return Xs, X, k, listk, mu, sqrt, feed_dict

    def testIdenticalEquality(self):
        with self.test_context() as sess:
            Xs, X, k, listk, mu, sqrt, feed_dict = self.prepare()

            _fm1, _fv1 = conditional(Xs, X, k, mu, q_sqrt=sqrt)
            _fm2, _fv2 = conditional(Xs, X, listk, mu, q_sqrt=sqrt)

            fm1, fm2, fv1, fv2 = sess.run([_fm1, _fm2, _fv1, _fv2], feed_dict=feed_dict)

            assert_allclose(fm1, fm2)
            assert_allclose(fv1, fv2)

            # _fm1, _fv1 = conditional(Xs, X, k, mu, q_sqrt=sqrt, full_cov_output=True)
            # _fm2, _fv2 = conditional(Xs, X, listk, mu, q_sqrt=sqrt, full_cov_output=True)
            #
            # fm1fc, fm2fc, fv1fc, fv2fc = sess.run([_fm1, _fm2, _fv1, _fv2], feed_dict=feed_dict)
            #
            # assert_allclose(fm1, fm2)
            # assert_allclose(fv1, fv2)
            # assert_allclose(fm1, np.diag(fm1fc))


class IndependentOutputTest(GPflowTestCase):
    def prepare(self):
        rng = np.random.RandomState(0)
        num_latent = 2
        num_data = 3

        k = gpflow.kernels.Matern32(1)
        kernlist = [gpflow.kernels.Matern32(1) for _ in range(num_latent)]
        for k in kernlist:
            k.lengthscales = rng.rand(1)[0] * 2.0
            k.variance = rng.rand(1)[0] * 2.0
        listk = ListKernel(kernlist)
        print(listk)

        X = tf.placeholder(settings.float_type, shape=[None, None])
        mu = tf.placeholder(settings.float_type, shape=[None, None])
        Xs = tf.placeholder(settings.float_type, shape=[None, None])
        sqrt = tf.placeholder(settings.float_type, shape=[num_latent, num_data, num_data])

        X_data = rng.randn(num_data, 1)
        mu_data = rng.randn(num_data, num_latent)
        sqrt_data = rng.randn(num_latent, num_data, num_data)
        Xs_data = rng.randn(50, 1)

        feed_dict = {X: X_data, Xs: Xs_data, mu: mu_data, sqrt: sqrt_data}
        k.compile()

        # the chols are diagonal matrices, with the same entries as the diag representation.
        return Xs, X, k, listk, mu, sqrt, feed_dict

    def testIdenticalEquality(self):
        with self.test_context() as sess:
            Xs, X, k, listk, mu, sqrt, feed_dict = self.prepare()

            _separate = [conditional(Xs, X, listk.kern_list[i], mu[:, i, None], q_sqrt=sqrt[None, i, :, :])
                         for i in range(len(listk.kern_list))]
            separate = sess.run(_separate, feed_dict=feed_dict)
            separate_mus = np.hstack([sep[0] for sep in separate])
            separate_vars = np.hstack([sep[1] for sep in separate])

            _together = conditional(Xs, X, listk, mu, q_sqrt=sqrt)
            together = sess.run(_together, feed_dict=feed_dict)

            assert_allclose(separate_mus, together[0])
            assert_allclose(separate_vars, together[1])
