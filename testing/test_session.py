import unittest
import numpy as np
import GPflow
from GPflow import session
from GPflow import settings
import tensorflow as tf
from .parallel import ParallelTestCase


class TestSessionConfiguration(ParallelTestCase):
    def setUp(self):
        self.m = GPflow.gpr.GPR(np.ones((1, 1)), np.ones((1, 1)), kern=GPflow.kernels.Matern52(1))

    def test_option_persistance(self):
        '''
        Test configuration options are passed to tensorflow session
        '''
        dop = 3
        settings.session.intra_op_parallelism_threads = dop
        settings.session.inter_op_parallelism_threads = dop
        settings.session.allow_soft_placement = True
        self.m._compile()
        self.assertTrue(self.m._session._config.intra_op_parallelism_threads == dop)
        self.assertTrue(self.m._session._config.inter_op_parallelism_threads == dop)
        self.assertTrue(isinstance(self.m._session._config.inter_op_parallelism_threads, int))
        self.assertTrue(self.m._session._config.allow_soft_placement)
        self.assertTrue(isinstance(self.m._session._config.allow_soft_placement, bool))
        self.m.optimize(maxiter=1)

    def test_option_mutability(self):
        '''
        Test configuration options are passed to tensorflow session
        '''
        dop = 33
        settings.session.intra_op_parallelism_threads = dop
        settings.session.inter_op_parallelism_threads = dop
        graph = tf.Graph()
        tf_session = session.get_session(graph=graph,
                                         output_file_name=settings.profiling.output_file_name + "_objective",
                                         output_directory=settings.profiling.output_directory,
                                         each_time=settings.profiling.each_time)
        self.assertTrue(tf_session._config.intra_op_parallelism_threads == dop)
        self.assertTrue(tf_session._config.inter_op_parallelism_threads == dop)
        # change maximum degree of parallelism
        dopOverride = 12
        tf_session = session.get_session(graph=graph,
                                         output_file_name=settings.profiling.output_file_name + "_objective",
                                         output_directory=settings.profiling.output_directory,
                                         each_time=settings.profiling.each_time,
                                         config=tf.ConfigProto(intra_op_parallelism_threads=dopOverride,
                                                               inter_op_parallelism_threads=dopOverride))
        self.assertTrue(tf_session._config.intra_op_parallelism_threads == dopOverride)
        self.assertTrue(tf_session._config.inter_op_parallelism_threads == dopOverride)

    def test_autoflow(self):
        dop = 4
        settings.session.intra_op_parallelism_threads = dop
        settings.session.inter_op_parallelism_threads = dop
        self.m._compile()  # clear pick up new settings
        self.m.compute_log_likelihood()  # causes Autoflow to create log likelihood graph
        afsession = self.m.__dict__['_compute_log_likelihood_AF_storage']['session']
        self.assertTrue(afsession._config.intra_op_parallelism_threads == dop)
        self.assertTrue(afsession._config.inter_op_parallelism_threads == dop)


if __name__ == '__main__':
    unittest.main()
