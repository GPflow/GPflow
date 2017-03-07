import unittest
import numpy as np
import GPflow


class TestSessionConfiguration(unittest.TestCase):
    def test_options(self):
        '''
        Test configuration options are passed to tensorflow session
        '''
        dop = 3
        GPflow.settings.session.intra_op_parallelism_threads = dop
        GPflow.settings.session.inter_op_parallelism_threads = dop
        GPflow.settings.session.allow_soft_placement = True
        m = GPflow.gpr.GPR(np.ones((1, 1)), np.ones((1, 1)), kern=GPflow.kernels.Matern52(1))
        m._compile()
        self.assertTrue(m._session._config.intra_op_parallelism_threads == dop)
        self.assertTrue(m._session._config.inter_op_parallelism_threads == dop)
        self.assertTrue(isinstance(m._session._config.inter_op_parallelism_threads, int))
        self.assertTrue(m._session._config.allow_soft_placement)
        self.assertTrue(isinstance(m._session._config.allow_soft_placement, bool))
        m.optimize(maxiter=1)
