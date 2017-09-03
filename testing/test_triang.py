import unittest
from gpflow.tf_wraps import vec_to_tri
import tensorflow as tf
import numpy as np

from testing.gpflow_testcase import GPflowTestCase
from gpflow.tf_wraps import vec_to_tri


class TestVecToTri(GPflowTestCase):
    def referenceInverse(self, matrices):
		#this is the inverse operation of the vec_to_tri
		#op being tested.
        D, N, _ = matrices.shape
        M = (N * (N + 1)) // 2
        tril_indices = np.tril_indices(N)
        output = np.zeros((D, M))
        for vector_index in range(D):
            matrix = matrices[vector_index, :]
            output[vector_index, :] = matrix[tril_indices]
        return output

    def getExampleMatrices(self, D, N ):
        rng = np.random.RandomState(1)
        random_matrices = rng.randn(D, N, N)
        for matrix_index in range(D):
            for row_index in range(N):
                for col_index in range(N):
                    if col_index > row_index:
                        random_matrices[matrix_index, row_index, col_index] = 0.
        return random_matrices

    def testBasicFunctionality(self):
        with self.test_session() as sess:
            N = 3
            D = 3
            reference_matrices = self.getExampleMatrices(D, N)
            input_vector_tensor = tf.constant(self.referenceInverse(reference_matrices))

            test_matrices_tensor = vec_to_tri(input_vector_tensor, N)
            test_matrices = sess.run(test_matrices_tensor)
            np.testing.assert_array_almost_equal(reference_matrices, test_matrices)

    def testDifferentiable(self):
        with self.test_session() as sess:
            N = 3
            D = 3
            reference_matrices = self.getExampleMatrices(D, N)
            input_vector_array = self.referenceInverse(reference_matrices)
            input_vector_tensor = tf.constant(input_vector_array)

            test_matrices_tensor = vec_to_tri(input_vector_tensor, N)
            reduced_sum = tf.reduce_sum(test_matrices_tensor)
            gradient = tf.gradients(reduced_sum, input_vector_tensor)[0]
            reference_gradient = np.ones_like(input_vector_array)
            test_gradient = sess.run(gradient)
            np.testing.assert_array_almost_equal(reference_gradient, test_gradient)

if __name__ == "__main__":
    unittest.main()
