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

import glob
import os

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.test_util import GPflowTestCase


class TestProfiling(GPflowTestCase):
    def prepare(self):
        with gpflow.defer_build():
            X = np.random.rand(100, 1)
            Y = np.sin(X) + np.random.randn(*X.shape) * 0.01
            k = gpflow.kernels.RBF(1)
            return gpflow.models.GPR(X, Y, k)

    def test_profile(self):
        m = self.prepare()
        s = gpflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_directory = tf.test.get_temp_dir()
        s.profiling.output_file_name = 'test_trace_profile'

        with gpflow.settings.temp_settings(s):
            with gpflow.session_manager.get_session().as_default():
                m.compile()
                opt = gpflow.train.ScipyOptimizer()
                opt.minimize(m, maxiter=10)

        expected_file = os.path.join(s.profiling.output_directory,
                                     s.profiling.output_file_name + '.json')

        self.assertTrue(os.path.exists(expected_file))
        os.remove(expected_file)


    def test_autoflow(self):
        m = self.prepare()
        s = gpflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_directory = tf.test.get_temp_dir()
        s.profiling.output_file_name = 'test_trace_autoflow'

        with gpflow.settings.temp_settings(s):
            with gpflow.session_manager.get_session().as_default():
                m.compile()
                m.kern.compute_K_symm(m.X.read_value())

        directory = s.profiling.output_directory
        filename = s.profiling.output_file_name + '.json'
        expected_file = os.path.join(directory, filename)
        self.assertTrue(os.path.exists(expected_file))
        os.remove(expected_file)

        m.clear()
        s.profiling.output_directory = tf.test.get_temp_dir()
        m.compile()

        # TODO(@awav): CHECK IT
        # with self.assertRaises(IOError):
        #     with gpflow.settings.temp_settings(s):
        #        m.kern.compute_K_symm(m.X.read_value())

    def test_eachtime(self):
        m = self.prepare()
        s = gpflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.each_time = True
        s.profiling.output_directory = tf.test.get_temp_dir() + '/each_time/'
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

if __name__ == "__main__":
    tf.test.main()
