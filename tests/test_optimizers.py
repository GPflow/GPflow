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

import gpflow
from gpflow.test_util import GPflowTestCase

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
