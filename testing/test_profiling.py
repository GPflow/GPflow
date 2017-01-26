import os
import unittest

import numpy as np

import GPflow


class TestProfiling(unittest.TestCase):
    def setUp(self):
        X = np.random.rand(100, 1)
        Y = np.sin(X) + np.random.randn(*X.shape) * 0.01
        k = GPflow.kernels.RBF(1)
        self.m = GPflow.gpr.GPR(X, Y, k)

    def test_profile(self):
        s = GPflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_directory = './testing/'
        with GPflow.settings.temp_settings(s):
            self.m._compile()
            self.m._objective(self.m.get_free_state())

        expected_file = s.profiling.output_directory + s.profiling.output_file_name + "_objective.json"
        self.assertTrue(os.path.exists(expected_file))
        if os.path.exists(expected_file):
            os.remove(expected_file)

    def test_autoflow(self):
        s = GPflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_directory = './testing/'
        with GPflow.settings.temp_settings(s):
            self.m.kern.compute_K_symm(self.m.X.value)

        expected_file = s.profiling.output_directory + s.profiling.output_file_name + "_compute_K_symm.json"
        self.assertTrue(os.path.exists(expected_file))
        if os.path.exists(expected_file):
            os.remove(expected_file)
