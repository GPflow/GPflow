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
from numpy.testing import assert_almost_equal, assert_allclose

import gpflow
from gpflow.test_util import GPflowTestCase


class SampleGaussianTest(GPflowTestCase):
    class Gauss(gpflow.models.Model):
        def __init__(self, **kwargs):
            super(SampleGaussianTest.Gauss, self).__init__(**kwargs)
            self.x = gpflow.Param(np.zeros(3))
        @gpflow.params_as_tensors
        def build_objective(self):
            return 0.5 * tf.reduce_sum(tf.square(self.x))
        def _build_likelihood(self):
            return tf.constant(0.0, dtype=gpflow.settings.float_type)

    @gpflow.defer_build()
    def setUp(self):
        with self.test_context():
            tf.set_random_seed(1)
            self.m = SampleGaussianTest.Gauss()
            self.hmc = gpflow.train.HMC()

    def test_mean_cov(self):
        with self.test_context():
            self.m.compile()
            num_samples = 1000
            samples = self.hmc.sample(self.m, num_samples=num_samples,
                                      lmin=10, lmax=21, epsilon=0.05)
            self.assertEqual(samples.shape, (num_samples, 2))
            xs = np.array(samples[self.m.x.pathname].tolist(), dtype=np.float32)
            mean = xs.mean(0)
            cov = np.cov(xs.T)
            cov_standard = np.eye(cov.shape[0])

            # TODO(@awav): inspite of the fact that we set up graph's random seed,
            # the operation seed is still assigned by tensorflow automatically
            # and hence sample output numbers are not deterministic.
            #
            # self.assertTrue(np.sum(np.abs(mean) < 0.1) >= mean.size/2)
            # assert_allclose(cov, cov_standard, rtol=1e-1, atol=1e-1)

    def test_rng(self):
        """
        Make sure all randomness can be atributed to the rng
        """
        def get_samples():
            num_samples = 100
            m = SampleGaussianTest.Gauss()
            m.compile()
            hmc = gpflow.train.HMC()
            samples = hmc.sample(m, num_samples=num_samples, epsilon=0.05,
                                 lmin=10, lmax=21, thin=10)
            return np.array(samples[m.x.pathname].values.tolist(), dtype=np.float32)

        with self.test_context():
            tf.set_random_seed(1)
            s1 = get_samples()

        with self.test_context():
            tf.set_random_seed(2)
            s2 = get_samples()

        with self.test_context():
            tf.set_random_seed(3)
            s3 = get_samples()

        self.assertFalse(np.all(s1 == s2))
        self.assertFalse(np.all(s1 == s3))

    def test_burn(self):
        with self.test_context():
            self.m.compile()
            num_samples = 10
            x0 = list(self.m.read_trainables().values())[0].copy()
            samples = self.hmc.sample(self.m, num_samples=num_samples,
                                      lmin=10, lmax=21, epsilon=0.05,
                                      burn=10, logprobs=False)

            x = samples.iloc[-1][0]
            self.assertEqual(samples.shape, (10, 1))
            self.assertEqual(x.shape, (3,))
            self.assertFalse(np.all(x == x0))

    def test_columns_names(self):
        with self.test_session():
            self.m.compile()
            num_samples = 10
            samples = self.hmc.sample(self.m, num_samples=num_samples,
                                      lmin=10, lmax=21, epsilon=0.05)
            names = [p.pathname for p in self.m.parameters]
            names.append('logprobs')
            names = set(names)
            col_names = set(samples.columns)
            self.assertEqual(col_names, names)


class Quadratic(gpflow.models.Model):
    def __init__(self):
        super(Quadratic, self).__init__()
        rng = np.random.RandomState(0)
        self.x = gpflow.Param(rng.randn(2), dtype=gpflow.settings.float_type)
    @gpflow.params_as_tensors
    def _build_likelihood(self):
        return -tf.reduce_sum(tf.square(self.x))


class SampleModelTest(GPflowTestCase):
    """
    Create a very simple model and make sure samples form is make sense.
    """

    def setUp(self):
        tf.set_random_seed(1)

    def test_mean(self):
        with self.test_context():
            m = Quadratic()
            hmc = gpflow.train.HMC()
            num_samples = 400
            samples = hmc.sample(m, num_samples=num_samples,
                                 epsilon=0.05, lmin=10, lmax=20, thin=10)
            xs = np.array(samples[m.x.pathname].tolist(), dtype=np.float32)
            self.assertEqual(samples.shape, (400, 2))
            self.assertEqual(xs.shape, (400, 2))
            assert_almost_equal(xs.mean(0), np.zeros(2), decimal=1)

            llh = [m.compute_log_likelihood(feed_dict=m.sample_feed_dict(s))
                   for i, s in samples.iterrows()]
            assert_allclose(llh, - (xs**2).sum(1), atol=1e-6)


class CheckTrainingVariableState(GPflowTestCase):
    def model(self):
        X, Y = np.random.randn(2, 10, 1)
        return gpflow.models.GPMC(
            X, Y,
            kern=gpflow.kernels.Matern32(1),
            likelihood=gpflow.likelihoods.StudentT())

    def test_last_update(self):
        with self.test_context():
            m = self.model()
            hmc = gpflow.train.HMC()
            samples = hmc.sample(m, num_samples=10, lmin=1, lmax=10, epsilon=0.05, thin=10)
            self.check_last_variables_state(m, samples)

    def test_with_fixed(self):
        with self.test_context():
            m = self.model()
            m.kern.lengthscales.trainable = False
            hmc = gpflow.train.HMC()
            samples = hmc.sample(m, num_samples=10, lmax=10, epsilon=0.05)
            missing_param = m.kern.lengthscales.pathname
            self.assertTrue(missing_param not in samples)
            self.check_last_variables_state(m, samples)

    def test_multiple_runs(self):
        with self.test_context():
            m = self.model()
            hmc = gpflow.train.HMC()
            for n in [1, 2]:
                samples = hmc.sample(m, num_samples=n, lmax=10, epsilon=0.05)
                self.check_last_variables_state(m, samples)

    def check_last_variables_state(self, m, samples):
        xs = samples.drop('logprobs', axis=1)
        params = {p.pathname: p for p in m.trainable_parameters}
        self.assertEqual(set(params.keys()), set(xs.columns))
        last = xs.iloc[-1]
        for col in last.index:
            assert_almost_equal(last[col], params[col].read_value())


if __name__ == '__main__':
    tf.test.main()
