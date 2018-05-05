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


# pylint: disable=W0212

import numpy as np
import tensorflow as tf

import gpflow
from gpflow import settings
from gpflow import session_manager
from gpflow.test_util import GPflowTestCase


class TestSessionConfiguration(GPflowTestCase):

    def prepare(self):
        with gpflow.defer_build():
            return gpflow.models.GPR(
                np.ones((1, 1)),
                np.ones((1, 1)),
                kern=gpflow.kernels.Matern52(1))

    def test_option_persistance(self):
        '''
        Test configuration options are passed to tensorflow session
        '''
        dop = 3
        settings.session.intra_op_parallelism_threads = dop
        settings.session.inter_op_parallelism_threads = dop
        settings.session.allow_soft_placement = True
        session = gpflow.session_manager.get_session()
        self.assertTrue(session._config.inter_op_parallelism_threads == dop)
        self.assertTrue(isinstance(session._config.inter_op_parallelism_threads, int))
        self.assertTrue(session._config.allow_soft_placement)
        self.assertTrue(isinstance(session._config.allow_soft_placement, bool))
        # m = self.prepare()
        # m.compile()
        # opt = gpflow.train.ScipyOptimizer()
        # opt.minimize(m, maxiter=1)

    def test_option_mutability(self):
        '''
        Test configuration options are passed to tensorflow session
        '''
        dop = 33
        settings.session.intra_op_parallelism_threads = dop
        settings.session.inter_op_parallelism_threads = dop
        graph = tf.Graph()
        tf_session = session_manager.get_session(
            graph=graph,
            output_file_name=settings.profiling.output_file_name + "_objective",
            output_directory=settings.profiling.output_directory,
            each_time=settings.profiling.each_time)
        self.assertTrue(tf_session._config.intra_op_parallelism_threads == dop)
        self.assertTrue(tf_session._config.inter_op_parallelism_threads == dop)
        tf_session.close()

        # change maximum degree of parallelism
        dopOverride = 12
        tf_session = session_manager.get_session(
            graph=graph,
            output_file_name=settings.profiling.output_file_name + "_objective",
            output_directory=settings.profiling.output_directory,
            each_time=settings.profiling.each_time,
            config=tf.ConfigProto(intra_op_parallelism_threads=dopOverride,
                                  inter_op_parallelism_threads=dopOverride))
        self.assertTrue(tf_session._config.intra_op_parallelism_threads == dopOverride)
        self.assertTrue(tf_session._config.inter_op_parallelism_threads == dopOverride)
        tf_session.close()

    def test_session_default_graph(self):
        tf_session = session_manager.get_session()
        self.assertEqual(tf_session.graph, tf.get_default_graph())
        tf_session.close()


if __name__ == '__main__':
    tf.test.main()
