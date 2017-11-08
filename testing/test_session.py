
# pylint: disable=W0212

import unittest
import numpy as np
import gpflow
import tensorflow as tf

from gpflow import settings
from gpflow import session_manager
from gpflow.test_util import GPflowTestCase


class TestSessionConfiguration(GPflowTestCase):

    @gpflow.defer_build()
    def prepare(self):
        m = gpflow.models.GPR(
            np.ones((1, 1)),
            np.ones((1, 1)),
            kern=gpflow.kernels.Matern52(1))
        return m

    def test_option_persistance(self):
        '''
        Test configuration options are passed to tensorflow session
        '''
        m = self.prepare()
        dop = 3
        settings.session.intra_op_parallelism_threads = dop
        settings.session.inter_op_parallelism_threads = dop
        settings.session.allow_soft_placement = True
        m.compile()
        session = gpflow.session_manager.get_default_session()
        self.assertTrue(session._config.inter_op_parallelism_threads == dop)
        self.assertTrue(isinstance(session._config.inter_op_parallelism_threads, int))
        self.assertTrue(session._config.allow_soft_placement)
        self.assertTrue(isinstance(session._config.allow_soft_placement, bool))
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m, maxiter=1)

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
    unittest.main()
