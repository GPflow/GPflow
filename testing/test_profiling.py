import glob
import os
import unittest

import numpy as np

import gpflow


class TestProfiling(unittest.TestCase):
    def setUp(self):
        X = np.random.rand(100, 1)
        Y = np.sin(X) + np.random.randn(*X.shape) * 0.01
        k = gpflow.kernels.RBF(1)
        self.m = gpflow.gpr.GPR(X, Y, k)

    def test_profile(self):
        s = gpflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_directory = './testing/'
        with gpflow.settings.temp_settings(s):
            self.m._compile()
            self.m._objective(self.m.get_free_state())

        expected_file = s.profiling.output_directory + s.profiling.output_file_name + "_objective.json"
        self.assertTrue(os.path.exists(expected_file))
        if os.path.exists(expected_file):
            os.remove(expected_file)

    def test_autoflow(self):
        s = gpflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_directory = './testing/'
        with gpflow.settings.temp_settings(s):
            self.m.kern.compute_K_symm(self.m.X.value)

        expected_file = s.profiling.output_directory + s.profiling.output_file_name + "_compute_K_symm.json"
        self.assertTrue(os.path.exists(expected_file))
        if os.path.exists(expected_file):
            os.remove(expected_file)

        s.profiling.output_directory = './testing/__init__.py'
        self.m.kern._kill_autoflow()
        with self.assertRaises(IOError):
            with gpflow.settings.temp_settings(s):
                self.m.kern.compute_K_symm(self.m.X.value)

    def test_eachtime(self):
        s = gpflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.each_time = True
        s.profiling.output_directory = './testing/each_time/'
        with gpflow.settings.temp_settings(s):
            self.m._compile()
            self.m._objective(self.m.get_free_state())
            self.m._objective(self.m.get_free_state())

        for f in glob.glob(s.profiling.output_directory + "timeline_objective_*.json"):
            os.remove(f)

        if os.path.exists(s.profiling.output_directory):
            os.rmdir(s.profiling.output_directory)

