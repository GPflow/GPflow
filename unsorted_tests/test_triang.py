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

import tensorflow as tf

import numpy as np
from numpy.testing import assert_array_almost_equal


from gpflow import misc
from gpflow.test_util import GPflowTestCase


class TestVecToTri(GPflowTestCase):
    def reference_inverse(self, matrices):
		# This is the inverse operation of the vec_to_tri op being tested.
        D, N, _ = matrices.shape
        M = (N * (N + 1)) // 2
        tril_indices = np.tril_indices(N)
        output = np.zeros((D, M))
        for vector_index in range(D):
            matrix = matrices[vector_index, :]
            output[vector_index, :] = matrix[tril_indices]
        return output

    def get_example_matrices(self, D, N ):
        rng = np.random.RandomState(1)
        random_matrices = rng.randn(D, N, N)
        for matrix_index in range(D):
            for row_index in range(N):
                for col_index in range(N):
                    if col_index > row_index:
                        random_matrices[matrix_index, row_index, col_index] = 0.
        return random_matrices

    def test_basic_functionality(self):
        with self.test_context() as sess:
            N = 3
            D = 3
            reference_matrices = self.get_example_matrices(D, N)
            input_vector_tensor = tf.constant(self.reference_inverse(reference_matrices))

            test_matrices_tensor = misc.vec_to_tri(input_vector_tensor, N)
            test_matrices = sess.run(test_matrices_tensor)
            assert_array_almost_equal(reference_matrices, test_matrices)

    def test_differentiable(self):
        with self.test_context() as sess:
            N = 3
            D = 3
            reference_matrices = self.get_example_matrices(D, N)
            input_vector_array = self.reference_inverse(reference_matrices)
            input_vector_tensor = tf.constant(input_vector_array)

            test_matrices_tensor = misc.vec_to_tri(input_vector_tensor, N)
            reduced_sum = tf.reduce_sum(test_matrices_tensor)
            gradient = tf.gradients(reduced_sum, input_vector_tensor)[0]
            reference_gradient = np.ones_like(input_vector_array)
            test_gradient = sess.run(gradient)
            assert_array_almost_equal(reference_gradient, test_gradient)


if __name__ == "__main__":
    tf.test.main()
