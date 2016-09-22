import tensorflow as tf
import GPflow.kernels
from GPflow.tf_wraps import eye


class RBF(GPflow.kernels.RBF):
    def eKdiag(self, X, Xcov=None):
        """
        Also known as phi_0.
        :param X:
        :return: N
        """
        return self.Kdiag(X)

    def eKxz(self, Z, Xmu, Xcov):
        """
        Also known as phi_1: <K_{x, Z}>_{q(x)}.
        :param Z: MxD inducing inputs
        :param Xmu: X mean (NxD)
        :param Xcov: NxDxD  # TODO code needs to be extended to return 2x(N+1)xDxD
        :return: NxM
        """
        # use only active dimensions
        Xmu, Xcov = self._slice(Xmu, Xcov)
        Z, _ = self._slice(Z, None)
        M = tf.shape(Z)[0]
        vec = tf.expand_dims(Xmu, 1) - tf.expand_dims(Z, 0)  # NxMxD
        scalemat = tf.expand_dims(tf.diag(self.lengthscales ** 2.0), 0) + Xcov  # NxDxD
        rsm = tf.tile(tf.expand_dims(scalemat, 1), (1, M, 1, 1))  # Reshaped scalemat
        smIvec = tf.batch_matrix_solve(rsm, tf.expand_dims(vec, 3))[:, :, :, 0]  # NxMxD
        q = tf.reduce_sum(smIvec * vec, [2])  # NxM
        det = tf.batch_matrix_determinant(
            tf.expand_dims(eye(self.input_dim), 0) + tf.reshape(self.lengthscales ** -2.0, (1, 1, -1)) * Xcov
        )  # N
        return self.variance * tf.expand_dims(det ** -0.5, 1) * tf.exp(-0.5 * q)

    def exKxz(self, Z, Xmu, Xcov):
        """
        <x_t K_{x_{t-1}, Z}>_q_{x_{t-1:t}}
        :param Z: MxD inducing inputs
        :param Xmu: X mean (N+1xD)
        :param Xcov: 2x(N+1)xDxD
        :return: NxMxD
        """
        tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[1:3], name="assert_Xmu_Xcov_shape")
        # use only active dimensions
        Xmu, Xcov = self._slice(Xmu, Xcov)
        Z, _ = self._slice(Z, None)

        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0] - 1
        Xsigmb = tf.slice(Xcov, [0, 0, 0, 0], tf.pack([-1, N, -1, -1]))
        Xsigm = Xsigmb[0, :, :, :]  # NxDxD
        Xsigmc = Xsigmb[1, :, :, :]  # NxDxD
        Xmum = tf.slice(Xmu, [0, 0], tf.pack([N, -1]))
        Xmup = Xmu[1:, :]
        scalemat = tf.expand_dims(tf.diag(self.lengthscales ** 2.0), 0) + Xsigm  # NxDxD

        det = tf.batch_matrix_determinant(
            tf.expand_dims(eye(self.input_dim), 0) + tf.reshape(self.lengthscales ** -2.0, (1, 1, -1)) * Xsigm
        )  # N

        vec = tf.expand_dims(Z, 0) - tf.expand_dims(Xmum, 1)  # NxMxD

        rsm = tf.tile(tf.expand_dims(scalemat, 1), (1, M, 1, 1))  # Reshaped scalemat
        smIvec = tf.batch_matrix_solve(rsm, tf.expand_dims(vec, 3))[:, :, :, 0]  # NxMxDx1
        q = tf.reduce_sum(smIvec * vec, [2])  # NxM

        addvec = tf.batch_matmul(
            tf.tile(tf.expand_dims(Xsigmc, 1), (1, M, 1, 1)),
            tf.expand_dims(smIvec, 3),
            adj_x=True
        )[:, :, :, 0] + tf.expand_dims(Xmup, 1)  # NxMxD

        return self.variance * addvec * tf.reshape(det ** -0.5, (N, 1, 1)) * tf.expand_dims(tf.exp(-0.5 * q), 2)

    def eKzxKxz(self, Z, Xmu, Xcov):
        """
        Also known as Phi_2.
        :param Z: MxD
        :param Xmu: X mean (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :return: NxMxM
        """
        # use only active dimensions
        Xmu, Xcov = self._slice(Xmu, Xcov)
        Z, _ = self._slice(Z, None)
        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0]
        D = self.input_dim
        Kmms = tf.sqrt(self.K(Z)) / self.variance ** 0.5
        scalemat = tf.expand_dims(eye(D), 0) + 2 * Xcov * tf.reshape(self.lengthscales ** -2.0, [1, 1, -1])  # NxDxD
        det = tf.batch_matrix_determinant(scalemat)

        mat = Xcov + 0.5 * tf.expand_dims(tf.diag(self.lengthscales ** 2.0), 0)  # NxDxD
        cm = tf.batch_cholesky(mat)  # NxDxD
        vec = 0.5 * (tf.reshape(Z, [1, M, 1, D]) +
                     tf.reshape(Z, [1, 1, M, D])) - tf.reshape(Xmu, [N, 1, 1, D])  # NxMxMxD
        cmr = tf.tile(tf.reshape(cm, [N, 1, 1, D, D]), [1, M, M, 1, 1])  # NxMxMxDxD
        smI_z = tf.batch_matrix_triangular_solve(cmr, tf.expand_dims(vec, 4))  # NxMxMxDx1
        fs = tf.reduce_sum(tf.square(smI_z), [3, 4])

        return self.variance ** 2.0 * tf.expand_dims(Kmms, 0) * tf.exp(-0.5 * fs) * tf.reshape(det ** -0.5, [N, 1, 1])


class Linear(GPflow.kernels.Linear):
    def eKdiag(self, X, Xcov):
        # use only active dimensions
        X, Xcov = self._slice(X, Xcov)
        return self.variance * (tf.reduce_sum(tf.square(X), 1) + tf.reduce_sum(tf.batch_matrix_diag_part(Xcov), 1))

    def eKxz(self, Z, Xmu, Xcov):
        # use only active dimensions
        Xmu, Xcov = self._slice(Xmu, Xcov)
        Z, _ = self._slice(Z, None)
        return self.variance * tf.matmul(Xmu, tf.transpose(Z))

    def exKxz(self, Z, Xmu, Xcov):
        # use only active dimensions
        Xmu, Xcov = self._slice(Xmu, Xcov)
        Z, _ = self._slice(Z, None)
        N = tf.shape(Xmu)[0] - 1
        Xmum = Xmu[:-1, :]
        Xmup = Xmu[1:, :]
        op = tf.expand_dims(Xmum, 2) * tf.expand_dims(Xmup, 1) + Xcov[1, :-1, :, :]  # NxDxD
        return self.variance*tf.batch_matmul(tf.tile(tf.expand_dims(Z, 0), (N, 1, 1)), op)

    def eKzxKxz(self, Z, Xmu, Xcov):
        """
        exKxz
        :param Z: MxD
        :param Xmu: NxD
        :param Xcov: NxDxD
        :return:
        """
        # use only active dimensions
        Xmu, Xcov = self._slice(Xmu, Xcov)
        Z, _ = self._slice(Z, None)
        N = tf.shape(Xmu)[0]
        mom2 = tf.expand_dims(Xmu, 1) * tf.expand_dims(Xmu, 2) + Xcov  # NxDxD
        eZ = tf.tile(tf.expand_dims(Z, 0), (N, 1, 1))  # NxMxD
        return self.variance ** 2.0 * tf.batch_matmul(tf.batch_matmul(eZ, mom2), eZ, adj_y=True)
