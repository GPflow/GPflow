import glob
import os

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.test_util import GPflowTestCase


class TestProfiling(GPflowTestCase):
    @gpflow.autobuild(False)
    def setup(self):
        X = np.random.rand(100, 1)
        Y = np.sin(X) + np.random.randn(*X.shape) * 0.01
        k = gpflow.kernels.RBF(1)
        return gpflow.models.GPR(X, Y, k)

    def test_profile(self):
        m = self.setup()
        s = gpflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_directory = os.path.dirname(__file__)
        s.profiling.output_file_name = 'test_trace_profile'

        with gpflow.settings.temp_settings(s):
            with gpflow.session_manager.get_session():
                m.compile()
                opt = gpflow.train.ScipyOptimizer()
                opt.minimize(m, maxiter=10)

        expected_file = os.path.join(s.profiling.output_directory,
                                     s.profiling.output_file_name + '.json')

        self.assertTrue(os.path.exists(expected_file))
        os.remove(expected_file)


    def test_autoflow(self):
        m = self.setup()
        s = gpflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_directory = os.path.dirname(__file__)
        s.profiling.output_file_name = 'test_trace_autoflow'

        with gpflow.settings.temp_settings(s):
            m.compile()
            m.kern.compute_K_symm(m.X.read_value())

        expected_file = os.path.join(s.profiling.output_directory,
                                     s.profiling.output_file_name + '.json')
        self.assertTrue(os.path.exists(expected_file))
        os.remove(expected_file)

        m.clear()
        s.profiling.output_directory = __file__
        m.compile()
        # TODO(@awav): CHECK IT
        # with self.assertRaises(IOError):
        #     with gpflow.settings.temp_settings(s):
        #        m.kern.compute_K_symm(m.X.read_value())

    def test_eachtime(self):
        m = self.setup()
        s = gpflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.each_time = True
        s.profiling.output_directory = os.path.dirname(__file__) + '/each_time/'
        name = 'test_eachtime'
        s.profiling.output_file_name = name
        with gpflow.settings.temp_settings(s):
            with gpflow.session_manager.get_session():
                m.compile()
                opt = gpflow.train.ScipyOptimizer()
                opt.minimize(m, maxiter=2)

        pattern = s.profiling.output_directory + '/{name}*.json'.format(name=name)
        for filename in glob.glob(pattern):
            os.remove(filename)

        if os.path.exists(s.profiling.output_directory):
            os.rmdir(s.profiling.output_directory)
