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
import tensorflow as tf
import pytest

import gpflow
from gpflow.test_util import session_tf
from gpflow.test_util import GPflowTestCase
from gpflow.training.natgrad_optimizer import NatGradOptimizer
from gpflow.training import GradientDescentOptimizer
from gpflow.training.optimizer import Optimizer
from gpflow.training.tensorflow_optimizer import _TensorFlowOptimizer

from numpy.testing import assert_allclose

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

def test_VGP_vs_GPR(session_tf):
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
    NatGradOptimizer(1.).minimize(m_vgp, [[m_vgp.q_mu, m_vgp.q_sqrt]], maxiter=1)

    assert_allclose(m_gpr.compute_log_likelihood(),
                    m_vgp.compute_log_likelihood(), atol=1e-5)

def test_SVGP_vs_SGPR(session_tf):
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
    A class that applies one step of each of multiple optimizers in a loop
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
            optimizer.minimize(model, maxiter=0, **kwargs)

            # this is what we will call in the loop
            minimize_ops.append(optimizer.minimize_operation)

        with session.graph.as_default(), tf.name_scope(self.name):
            feed_dict = self._gen_feed_dict(model, feed_dict)

            for _i in range(maxiter):
                for op in minimize_ops:
                    session.run(op, feed_dict=feed_dict)

        if anchor:
            model.anchor(session)

# # TODO(@hugh) this needs to be fixed
def test_hypers_SVGP_vs_SGPR(session_tf):
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

    m_svgp.q_mu.set_trainable(False)
    m_svgp.q_sqrt.set_trainable(False)

    NatGradOptimizer(1.).minimize(m_svgp, [[m_svgp.q_mu, m_svgp.q_sqrt]], maxiter=1)

    # this is fine
    assert_allclose(m_sgpr.compute_log_likelihood(),
                        m_svgp.compute_log_likelihood(), atol=1e-5)

    # combination (doing GD first as we've already done the nat grad step
    o1 = [GradientDescentOptimizer(0.01), {}]
    o2 = [NatGradOptimizer(1.), {'var_list':[[m_svgp.q_mu, m_svgp.q_sqrt]]}]
    nag_grads_with_gd_optimizer = CombinationOptimizer([o1, o2])
    nag_grads_with_gd_optimizer.minimize(m_svgp,  maxiter=1)

    # this should be the same as
    GradientDescentOptimizer(0.01).minimize(m_sgpr, maxiter=1)

    # but isn't...
    with pytest.raises(AssertionError):
        assert_allclose(m_sgpr.compute_log_likelihood(),
                        m_svgp.compute_log_likelihood(), atol=1e-5)
    # the trouble seems to be that the [q_mu, q_sqrt] haven't updated as far as o1 is concerned


class ExcludedGradientDescentOptimizer(_TensorFlowOptimizer):
    def __init__(self, *args, excluded_params=[], **kwargs):
        Optimizer.__init__(self)
        self._model = None
        self._optimizer = tf.train.GradientDescentOptimizer(*args, **kwargs)
        self._minimize_operation = None
        self.excluded_params = excluded_params

    def _gen_var_list(self, model, var_list):
        var_list = var_list or []
        p = set(model.trainable_tensors)
        p -= set([t.unconstrained_tensor for t in self.excluded_params])
        return list(p.union(var_list))

def test_hypers_SVGP_vs_SGPR_with_excluded_vars(session_tf):
    N, M, D = 4, 3, 2
    X = np.random.randn(N, D)
    Z = np.random.randn(M, D)
    Y = np.random.randn(N, 1)

    lik_var = 0.1
    lik = gpflow.likelihoods.Gaussian()
    lik.variance = lik_var

    m_svgp = gpflow.models.SVGP(X, Y, gpflow.kernels.RBF(D), lik, Z=Z)

    m_sgpr = gpflow.models.SGPR(X, Y, gpflow.kernels.RBF(D), Z=Z)
    m_sgpr.likelihood.variance = lik_var

    lr = 0.1

    # combination (doing GD first as we've already done the nat grad step
    p = [[m_svgp.q_mu, m_svgp.q_sqrt]]
    o1 = [NatGradOptimizer(1.), {'var_list':p}]
    o2 = [ExcludedGradientDescentOptimizer(lr, excluded_params=p[0]), {}]
    o3 = [NatGradOptimizer(1.), {'var_list':p}]

    nag_grads_with_gd_optimizer = CombinationOptimizer([o1, o2, o3])
    nag_grads_with_gd_optimizer.minimize(m_svgp, maxiter=1)

    # this should be the same as
    GradientDescentOptimizer(lr).minimize(m_sgpr, maxiter=1)

    assert_allclose(m_sgpr.compute_log_likelihood(),
                    m_svgp.compute_log_likelihood(), atol=1e-5)
    # now it works...