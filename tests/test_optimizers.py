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

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.actions import Loop
from gpflow.test_util import GPflowTestCase, session_tf
from gpflow.training import GradientDescentOptimizer
from gpflow.training.natgrad_optimizer import (NatGradOptimizer, XiSqrtMeanVar,
                                               expectation_to_meanvarsqrt,
                                               meanvarsqrt_to_expectation,
                                               natural_to_expectation)
from gpflow.training.optimizer import Optimizer
from gpflow.training.tensorflow_optimizer import _TensorFlowOptimizer


class Empty(gpflow.models.Model):
    @gpflow.params_as_tensors
    def _build_likelihood(self):
        return tf.constant(0.0, dtype=gpflow.settings.float_type)


class Demo(gpflow.models.Model):
    def __init__(self, add_to_inits=[], add_to_trainables=[], name=None):
        super().__init__(name=name)
        data = np.random.randn(10, 10)
        self.a = gpflow.Param(data, dtype=gpflow.settings.float_type)
        self.init_vars = add_to_inits
        self.trainable_vars = add_to_trainables

    @property
    def initializables(self):
        return super().initializables + self.init_vars

    @property
    def trainable_tensors(self):
        return super().trainable_tensors + self.trainable_vars

    @gpflow.params_as_tensors
    def _build_likelihood(self):
        return tf.reduce_sum(self.a) + sum(map(tf.reduce_prod, self.trainable_vars))


class NonOptimizer(gpflow.training.tensorflow_optimizer._TensorFlowOptimizer):
    pass


class TestSimpleOptimizerInterface(GPflowTestCase):
    def test_non_existing_optimizer(self):
        with self.assertRaises(TypeError):
            _ = NonOptimizer()


class OptimizerCase:
    optimizer = None

    # pylint: disable=E1101,E1102

    def test_different_sessions(self):
        with self.test_context() as session:
            demo = Demo()

        # Force initialization.
        with self.test_context() as session1:
            gpflow.reset_default_session()
            opt = self.optimizer()
            demo.initialize(session1, force=True)
            opt.minimize(demo, maxiter=1)

        # Mild initialization requirement: default changed session case.
        with self.test_context() as session2:
            self.assertFalse(session1 == session2)
            gpflow.reset_default_session()
            opt = self.optimizer()
            opt.minimize(demo, maxiter=1, initialize=True)

        # Mild initialization requirement: pass session case.
        with self.test_context() as session3:
            opt = self.optimizer()
            opt.minimize(demo, maxiter=1, session=session3, initialize=True)

    def test_optimizer_with_var_list(self):
        with self.test_context():
            demo = Demo()
            dfloat = gpflow.settings.float_type
            var1 = tf.get_variable('var_a1', shape=(), dtype=dfloat)
            var2 = tf.get_variable('var_b2', shape=(), trainable=False, dtype=dfloat)
            var3 = tf.get_variable('var_c3', shape=(), dtype=dfloat)

        with self.test_context() as session:
            opt = self.optimizer()
            demo.initialize(session)

            # No var list variables and empty feed_dict
            opt.minimize(demo, maxiter=1, initialize=False, anchor=False)

            # Var list is empty
            with self.assertRaises(ValueError):
                opt.minimize(Empty(), var_list=[], maxiter=1)

            # Var list variable
            session.run(var1.initializer)
            placeholder = tf.placeholder(gpflow.settings.float_type)
            opt.minimize(demo,
                         var_list=[var1],
                         feed_dict={placeholder: [1., 2]},
                         maxiter=5, initialize=False, anchor=False)

            # Var list variable is not trainable
            session.run(var2.initializer)
            opt.minimize(demo, var_list=[var2], maxiter=1, initialize=False)

            # NOTE(@awav): TensorFlow optimizer skips uninitialized values which
            # are not present in objective.
            demo._objective += var3
            with self.assertRaises(tf.errors.FailedPreconditionError):
                opt.minimize(demo, session=session, var_list=[var3], maxiter=1,
                             initialize=False, anchor=False)

    def test_optimizer_tensors(self):
        with self.test_context():
            opt = self.optimizer()
            if not isinstance(opt, gpflow.train.ScipyOptimizer):
                demo = Demo()
                opt.minimize(demo, maxiter=0)
                self.assertEqual(opt.model, demo)

                opt.model = Demo()
                self.assertNotEqual(opt.model, demo)
                self.assertEqual(opt.minimize_operation, None)
                self.assertEqual(opt.optimizer, None)

    def test_non_gpflow_model(self):
        with self.test_context():
            opt = self.optimizer()
            with self.assertRaises(ValueError):
                opt.minimize(None, maxiter=0)


    def test_external_variables_in_model(self):
        with self.test_context():
            dfloat = gpflow.settings.float_type

            var1_init = np.array(1.0, dtype=dfloat)
            var2_init = np.array(2.0, dtype=dfloat)
            var3_init = np.array(3.0, dtype=dfloat)

            var1 = tf.get_variable('var1', initializer=var1_init, dtype=dfloat)
            var2 = tf.get_variable('var2', initializer=var2_init, dtype=dfloat)
            var3 = tf.get_variable('var3', initializer=var3_init, trainable=False, dtype=dfloat)

            opt = self.optimizer()

        # Just initialize variable, but do not use it in training
        with self.test_context() as session:
            gpflow.reset_default_session()
            demo1 = Demo(add_to_inits=[var1], add_to_trainables=[], name="demo1")
            opt.minimize(demo1, maxiter=10)
            self.assertEqual(session.run(var1), var1_init)

        with self.test_context() as session:
            gpflow.reset_default_session()
            with self.assertRaises(tf.errors.FailedPreconditionError):
                demo2 = Demo(add_to_inits=[], add_to_trainables=[var2], name="demo2")
                opt.minimize(demo2, maxiter=10)

        with self.test_context() as session:
            gpflow.reset_default_session()
            demo3 = Demo(add_to_inits=[var1, var2, var3],
                         add_to_trainables=[var2, var3], name="demo3")
            opt.minimize(demo3, maxiter=10)

            self.assertEqual(session.run(var1), var1_init)
            self.assertFalse(session.run(var3) == var3_init)
            self.assertFalse(session.run(var2) == var2_init)


class TestScipyOptimizer(GPflowTestCase, OptimizerCase):
    optimizer = gpflow.train.ScipyOptimizer


class TestGradientDescentOptimizer(GPflowTestCase, OptimizerCase):
    optimizer = lambda _self: gpflow.train.GradientDescentOptimizer(0.1)


class TestAdamOptimizer(GPflowTestCase, OptimizerCase):
    optimizer = lambda _self: gpflow.train.AdamOptimizer(0.1)


class TestMomentumOptimizer(GPflowTestCase, OptimizerCase):
    optimizer = lambda _self: gpflow.train.MomentumOptimizer(0.1, 0.9)


class TestAdadeltaOptimizer(GPflowTestCase, OptimizerCase):
    optimizer = lambda _self: gpflow.train.AdadeltaOptimizer(0.1)


class TestRMSPropOptimizer(GPflowTestCase, OptimizerCase):
    optimizer = lambda _self: gpflow.train.RMSPropOptimizer(0.1)


class TestFtrlOptimizer(GPflowTestCase, OptimizerCase):
    optimizer = lambda _self: gpflow.train.FtrlOptimizer(0.1)

# class TestProximalAdagradOptimizer(GPflowTestCase, OptimizerCase):
#     optimizer = lambda _self: gpflow.train.ProximalAdagradOptimizer(0.1)
#
#                 raise TypeError(
#                     "%s type %s of argument '%s'." %
#                     (prefix, dtypes.as_dtype(attrs[input_arg.type_attr]).name,
# >                    inferred_from[input_arg.type_attr]))
# E               TypeError: Input 'lr' of 'ApplyProximalAdagrad' Op has type float32 that does not match type float64 of argument 'var'.
#
# ../../anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:546: TypeError



def test_scipy_optimizer_options(session_tf):
    np.random.seed(12345)
    X = np.random.randn(10, 1)
    Y = np.sin(X) + np.random.randn(*X.shape)
    m = gpflow.models.SGPR(X, Y, gpflow.kernels.RBF(1), Z=X)
    gtol = 'gtol'
    gtol_value = 1.303e-6
    o1 = gpflow.train.ScipyOptimizer(options={gtol: gtol_value})
    o2 = gpflow.train.ScipyOptimizer()
    o1.minimize(m, maxiter=0)
    o2.minimize(m, maxiter=0)
    assert gtol in o1.optimizer.optimizer_kwargs['options']
    assert o1.optimizer.optimizer_kwargs['options'][gtol] == gtol_value
    assert gtol not in o2.optimizer.optimizer_kwargs['options']

def test_small_q_sqrt_handeled_correctly(session_tf):
    """
    This is an extra test to make sure things still work when q_sqrt is small. This was breaking (#767)
    """
    N, D = 3, 2
    X = np.random.randn(N, D)
    Y = np.random.randn(N, 1)
    kern = gpflow.kernels.RBF(D)
    lik_var = 0.1
    lik = gpflow.likelihoods.Gaussian()
    lik.variance = lik_var

    m_vgp = gpflow.models.VGP(X, Y, kern, lik)
    m_gpr = gpflow.models.GPR(X, Y, kern)
    m_gpr.likelihood.variance = lik_var

    m_vgp.set_trainable(False)
    m_vgp.q_mu.set_trainable(True)
    m_vgp.q_sqrt.set_trainable(True)
    m_vgp.q_mu = np.random.randn(N, 1)
    m_vgp.q_sqrt = np.eye(N)[None, :, :] * 1e-3
    NatGradOptimizer(1.).minimize(m_vgp, [(m_vgp.q_mu, m_vgp.q_sqrt)], maxiter=1)

    assert_allclose(m_gpr.compute_log_likelihood(),
                    m_vgp.compute_log_likelihood(), atol=1e-4)

def test_VGP_vs_GPR(session_tf):
    """
    With a Gaussian likelihood the Gaussian variational (VGP) model should be equivalent to the exact 
     regression model (GPR) after a single nat grad step of size 1
    """
    N, D = 3, 2
    X = np.random.randn(N, D)
    Y = np.random.randn(N, 1)
    kern = gpflow.kernels.RBF(D)
    lik_var = 0.1
    lik = gpflow.likelihoods.Gaussian()
    lik.variance = lik_var

    m_vgp = gpflow.models.VGP(X, Y, kern, lik)
    m_gpr = gpflow.models.GPR(X, Y, kern)
    m_gpr.likelihood.variance = lik_var

    m_vgp.set_trainable(False)
    m_vgp.q_mu.set_trainable(True)
    m_vgp.q_sqrt.set_trainable(True)
    NatGradOptimizer(1.).minimize(m_vgp, [(m_vgp.q_mu, m_vgp.q_sqrt)], maxiter=1)

    assert_allclose(m_gpr.compute_log_likelihood(),
                    m_vgp.compute_log_likelihood(), atol=1e-4)


def test_other_XiTransform_VGP_vs_GPR(session_tf, xi_transform=XiSqrtMeanVar()):
    """
    With other transforms the solution is not given in a single step, but it should still give the same answer
    after a number of smaller steps. 
    """
    N, D = 3, 2
    X = np.random.randn(N, D)
    Y = np.random.randn(N, 1)
    kern = gpflow.kernels.RBF(D)
    lik_var = 0.1
    lik = gpflow.likelihoods.Gaussian()
    lik.variance = lik_var

    m_vgp = gpflow.models.VGP(X, Y, kern, lik)
    m_gpr = gpflow.models.GPR(X, Y, kern)
    m_gpr.likelihood.variance = lik_var

    m_vgp.set_trainable(False)
    m_vgp.q_mu.set_trainable(True)
    m_vgp.q_sqrt.set_trainable(True)
    NatGradOptimizer(0.01).minimize(m_vgp, [[m_vgp.q_mu, m_vgp.q_sqrt, xi_transform]], maxiter=500)

    assert_allclose(m_gpr.compute_log_likelihood(),
                    m_vgp.compute_log_likelihood(), atol=1e-4)


def test_XiEtas_VGP_vs_GPR(session_tf):
    class XiEta:
        def meanvarsqrt_to_xi(self, mean, varsqrt):
            return meanvarsqrt_to_expectation(mean, varsqrt)

        def xi_to_meanvarsqrt(self, xi_1, xi_2):
            return expectation_to_meanvarsqrt(xi_1, xi_2)

        def naturals_to_xi(self, nat_1, nat_2):
            return natural_to_expectation(nat_1, nat_2)

    test_other_XiTransform_VGP_vs_GPR(session_tf, xi_transform=XiEta())


def test_SVGP_vs_SGPR(session_tf):
    """
    With a Gaussian likelihood the sparse Gaussian variational (SVGP) model should be equivalent to the analytically 
     optimial sparse regression model (SGPR) after a single nat grad step of size 1
    """
    N, M, D = 4, 3, 2
    X = np.random.randn(N, D)
    Z = np.random.randn(M, D)
    Y = np.random.randn(N, 1)
    kern = gpflow.kernels.RBF(D)
    lik_var = 0.1
    lik = gpflow.likelihoods.Gaussian()
    lik.variance = lik_var

    m_svgp = gpflow.models.SVGP(X, Y, kern, lik, Z=Z)
    m_sgpr = gpflow.models.SGPR(X, Y, kern, Z=Z)
    m_sgpr.likelihood.variance = lik_var

    m_svgp.set_trainable(False)
    m_svgp.q_mu.set_trainable(True)
    m_svgp.q_sqrt.set_trainable(True)
    NatGradOptimizer(1.).minimize(m_svgp, [[m_svgp.q_mu, m_svgp.q_sqrt]], maxiter=1)

    assert_allclose(m_sgpr.compute_log_likelihood(),
                    m_svgp.compute_log_likelihood(), atol=1e-5)


class CombinationOptimizer(Optimizer):
    """
    A class that applies one step of each of multiple optimizers in a loop.
    """
    def __init__(self, optimizers_with_kwargs):
        self.name = None
        super().__init__()
        self.optimizers_with_kwargs = optimizers_with_kwargs

    def minimize(self, model, session=None, maxiter=1000, feed_dict=None, anchor=True):
        session = model.enquire_session(session)

        minimize_ops = []
        for optimizer, kwargs in self.optimizers_with_kwargs:
            # hack to init the optimizer operation
            optimizer.minimize(model, maxiter=0, anchor=anchor, **kwargs)

            # this is what we will call in the loop
            minimize_ops.append(optimizer.minimize_operation)

        with session.graph.as_default(), tf.name_scope(self.name):
            feed_dict = self._gen_feed_dict(model, feed_dict)

            for _i in range(maxiter):
                for op in minimize_ops:
                    session.run(op, feed_dict=feed_dict)

        if anchor:
            model.anchor(session)


class Datum:
    N, M, D = 4, 3, 2
    X = np.random.randn(N, D)
    Z = np.random.randn(M, D)
    Y = np.random.randn(N, 1)
    learning_rate = 0.01
    lik_var = 0.1
    gamma = 1.0


@pytest.fixture
def svgp(session_tf):
    rbf = gpflow.kernels.RBF(Datum.D)
    lik = gpflow.likelihoods.Gaussian()
    lik.variance = Datum.lik_var
    return gpflow.models.SVGP(Datum.X, Datum.Y, rbf, lik, Z=Datum.Z)


@pytest.fixture
def sgpr(session_tf):
    rbf = gpflow.kernels.RBF(Datum.D)
    m = gpflow.models.SGPR(Datum.X, Datum.Y, rbf, Z=Datum.Z)
    m.likelihood.variance = Datum.lik_var
    return m


def test_hypers_SVGP_vs_SGPR(session_tf, svgp, sgpr):
    """
    Test SVGP vs SGPR. Combined optimization.

    The logic is as follows:

    SVGP is given on nat grad step with gamma=1. Now it is identical to SGPR (which has
    analytic optimal variational distribution)

    We then take an ordinary gradient step on the hyperparameters (and inducing locations Z)

    Finally we update the variational parameters to their optimal values with another nat grad
    step with gamma=1.

    These three steps are equivalent to an ordinary gradient step on the parameters of SGPR

    In this test we simply make the variational parameters trainable=False, so they are not
    updated by the ordinary gradient step
    """
    anchor = False
    variationals = [(svgp.q_mu, svgp.q_sqrt)]

    svgp.q_mu.trainable = False
    svgp.q_sqrt.trainable = False

    opt = NatGradOptimizer(Datum.gamma)
    opt.minimize(svgp, var_list=variationals, maxiter=1, anchor=anchor)

    sgpr_likelihood = sgpr.compute_log_likelihood()
    svgp_likelihood = svgp.compute_log_likelihood()
    assert_allclose(sgpr_likelihood, svgp_likelihood, atol=1e-5)

    # combination (doing GD first as we've already done the nat grad step
    a1 = GradientDescentOptimizer(Datum.learning_rate).make_optimize_action(svgp)
    a2 = NatGradOptimizer(Datum.gamma).make_optimize_action(svgp, var_list=variationals)
    Loop([a1, a2]).with_settings(stop=1)()

    GradientDescentOptimizer(Datum.learning_rate).minimize(sgpr, maxiter=1, anchor=anchor)

    sgpr_likelihood = sgpr.compute_log_likelihood()
    svgp_likelihood = svgp.compute_log_likelihood()
    assert_allclose(sgpr_likelihood, svgp_likelihood, atol=1e-5)


def test_hypers_SVGP_vs_SGPR_tensors(session_tf, svgp, sgpr):
    """
    Test SVGP vs SGPR. Running optimization as tensors w/o GPflow wrapper.

    """
    anchor = False
    variationals = [(svgp.q_mu, svgp.q_sqrt)]

    svgp.q_mu.trainable = False
    svgp.q_sqrt.trainable = False

    o1 = NatGradOptimizer(Datum.gamma)
    o1_tensor = o1.make_optimize_tensor(svgp, var_list=variationals)

    o2 = GradientDescentOptimizer(Datum.learning_rate)
    o2_tensor = o2.make_optimize_tensor(svgp)

    o3 = NatGradOptimizer(Datum.gamma)
    o3_tensor = o3.make_optimize_tensor(svgp, var_list=variationals)

    session_tf.run(o1_tensor)

    sgpr_likelihood = sgpr.compute_log_likelihood()
    svgp_likelihood = svgp.compute_log_likelihood()
    assert_allclose(sgpr_likelihood, svgp_likelihood, atol=1e-5)

    session_tf.run(o2_tensor)
    session_tf.run(o3_tensor)

    GradientDescentOptimizer(Datum.learning_rate).minimize(sgpr, maxiter=1, anchor=anchor)

    sgpr_likelihood = sgpr.compute_log_likelihood()
    svgp_likelihood = svgp.compute_log_likelihood()
    assert_allclose(sgpr_likelihood, svgp_likelihood, atol=1e-5)
