from functools import reduce
import numpy as np
import tensorflow as tf
import GPflow.kernels
from GPflow.tf_wraps import eye
from ._settings import settings

float_type = settings.dtypes.float_type


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
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        M = tf.shape(Z)[0]
        D = tf.shape(Xmu)[1]
        lengthscales = self.lengthscales if self.ARD else tf.zeros((D,), dtype=float_type) + self.lengthscales

        vec = tf.expand_dims(Xmu, 1) - tf.expand_dims(Z, 0)  # NxMxD
        scalemat = tf.expand_dims(tf.diag(lengthscales ** 2.0), 0) + Xcov  # NxDxD
        rsm = tf.tile(tf.expand_dims(scalemat, 1), (1, M, 1, 1))  # Reshaped scalemat
        smIvec = tf.matrix_solve(rsm, tf.expand_dims(vec, 3))[:, :, :, 0]  # NxMxD
        q = tf.reduce_sum(smIvec * vec, [2])  # NxM
        det = tf.matrix_determinant(
            tf.expand_dims(eye(D), 0) + tf.reshape(lengthscales ** -2.0, (1, 1, -1)) * Xcov
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
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(Xmu)[1], self.input_dim, message="Currently cannot handle slicing in exKxz."),
            tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[1:3], name="assert_Xmu_Xcov_shape")
        ]):
            Xmu = tf.identity(Xmu)

        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0] - 1
        D = tf.shape(Xmu)[1]
        Xsigmb = tf.slice(Xcov, [0, 0, 0, 0], tf.pack([-1, N, -1, -1]))
        Xsigm = Xsigmb[0, :, :, :]  # NxDxD
        Xsigmc = Xsigmb[1, :, :, :]  # NxDxD
        Xmum = tf.slice(Xmu, [0, 0], tf.pack([N, -1]))
        Xmup = Xmu[1:, :]
        lengthscales = self.lengthscales if self.ARD else tf.zeros((D,), dtype=float_type) + self.lengthscales
        scalemat = tf.expand_dims(tf.diag(lengthscales ** 2.0), 0) + Xsigm  # NxDxD

        det = tf.matrix_determinant(
            tf.expand_dims(eye(tf.shape(Xmu)[1]), 0) + tf.reshape(lengthscales ** -2.0, (1, 1, -1)) * Xsigm
        )  # N

        vec = tf.expand_dims(Z, 0) - tf.expand_dims(Xmum, 1)  # NxMxD

        rsm = tf.tile(tf.expand_dims(scalemat, 1), (1, M, 1, 1))  # Reshaped scalemat
        smIvec = tf.matrix_solve(rsm, tf.expand_dims(vec, 3))[:, :, :, 0]  # NxMxDx1
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
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]
        lengthscales = self.lengthscales if self.ARD else tf.zeros((D,), dtype=float_type) + self.lengthscales

        Kmms = tf.sqrt(self.K(Z, presliced=True)) / self.variance ** 0.5
        scalemat = tf.expand_dims(eye(D), 0) + 2 * Xcov * tf.reshape(lengthscales ** -2.0, [1, 1, -1])  # NxDxD
        det = tf.matrix_determinant(scalemat)

        mat = Xcov + 0.5 * tf.expand_dims(tf.diag(lengthscales ** 2.0), 0)  # NxDxD
        cm = tf.cholesky(mat)  # NxDxD
        vec = 0.5 * (tf.reshape(Z, [1, M, 1, D]) +
                     tf.reshape(Z, [1, 1, M, D])) - tf.reshape(Xmu, [N, 1, 1, D])  # NxMxMxD
        cmr = tf.tile(tf.reshape(cm, [N, 1, 1, D, D]), [1, M, M, 1, 1])  # NxMxMxDxD
        smI_z = tf.matrix_triangular_solve(cmr, tf.expand_dims(vec, 4))  # NxMxMxDx1
        fs = tf.reduce_sum(tf.square(smI_z), [3, 4])

        return self.variance ** 2.0 * tf.expand_dims(Kmms, 0) * tf.exp(-0.5 * fs) * tf.reshape(det ** -0.5, [N, 1, 1])


class Linear(GPflow.kernels.Linear):
    def eKdiag(self, X, Xcov):
        if self.ARD:
            # TODO: Implement ARD
            raise NotImplementedError
        # use only active dimensions
        X, _ = self._slice(X, None)
        Xcov = self._slice_cov(Xcov)
        return self.variance * (tf.reduce_sum(tf.square(X), 1) + tf.reduce_sum(tf.matrix_diag_part(Xcov), 1))

    def eKxz(self, Z, Xmu, Xcov):
        if self.ARD:
            # TODO: Implement ARD
            raise NotImplementedError
        # use only active dimensions
        Z, Xmu = self._slice(Z, Xmu)
        return self.variance * tf.batch_matmul(Xmu, tf.transpose(Z))

    def exKxz(self, Z, Xmu, Xcov):
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(Xmu)[1], self.input_dim, message="Currently cannot handle slicing in exKxz."),
            tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[1:3], name="assert_Xmu_Xcov_shape")
        ]):
            Xmu = tf.identity(Xmu)

        N = tf.shape(Xmu)[0] - 1
        Xmum = Xmu[:-1, :]
        Xmup = Xmu[1:, :]
        op = tf.expand_dims(Xmum, 2) * tf.expand_dims(Xmup, 1) + Xcov[1, :-1, :, :]  # NxDxD
        return self.variance * tf.batch_matmul(tf.tile(tf.expand_dims(Z, 0), (N, 1, 1)), op)

    def eKzxKxz(self, Z, Xmu, Xcov):
        """
        exKxz
        :param Z: MxD
        :param Xmu: NxD
        :param Xcov: NxDxD
        :return:
        """
        # use only active dimensions
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        N = tf.shape(Xmu)[0]
        mom2 = tf.expand_dims(Xmu, 1) * tf.expand_dims(Xmu, 2) + Xcov  # NxDxD
        eZ = tf.tile(tf.expand_dims(Z, 0), (N, 1, 1))  # NxMxD
        return self.variance ** 2.0 * tf.batch_matmul(tf.batch_matmul(eZ, mom2), eZ, adj_y=True)


class Add(GPflow.kernels.Add):
    """
    Add
    This version of Add will call the corresponding kernel expectations for each of the summed kernels. This will be
    much better for kernels with analytically calculated kernel expectations. If quadrature is to be used, it's probably
    better to do quadrature on the summed kernel function using `GPflow.kernels.Add` instead.
    """

    def eKdiag(self, X, Xcov):
        return reduce(tf.add, [k.eKdiag(X, Xcov) for k in self.kern_list])

    def eKxz(self, Z, Xmu, Xcov):
        return reduce(tf.add, [k.eKxz(Z, Xmu, Xcov) for k in self.kern_list])

    def exKxz(self, Z, Xmu, Xcov):
        return reduce(tf.add, [k.exKxz(Z, Xmu, Xcov) for k in self.kern_list])

    def eKzxKxz(self, Z, Xmu, Xcov):
        all_sum = reduce(tf.add, [k.eKzxKxz(Z, Xmu, Xcov) for k in self.kern_list])
        eKxzs = [k.eKxz(Z, Xmu, Xcov) for k in self.kern_list]
        crossmeans = []
        for i, Ka in enumerate(eKxzs):
            for Kb in eKxzs[i + 1:]:
                op = Ka[:, None, :] * Kb[:, :, None]
                ct = tf.transpose(op, [0, 2, 1]) + op
                crossmeans.append(ct)
        crossmean = reduce(tf.add, crossmeans)

        if self.on_separate_dimensions and Xcov.get_shape().ndims == 2:
            # If we're on separate dimensions and the covariances are diagonal, we don't need Cov[Kzx1Kxz2].
            return all_sum + crossmean
        else:
            # Quadrature for Cov[(Kzx1 - eKzx1)(kxz2 - eKxz2)]
            Xmu, Z = self._slice(Xmu, Z)
            Xcov = self._slice_cov(Xcov)
            N, M, HpowD = tf.shape(Xmu)[0], tf.shape(Z)[0], self.num_gauss_hermite_points ** self.input_dim
            X, wn = GPflow.kernels.mvhermgauss(Xmu, Xcov, self.num_gauss_hermite_points,
                                               self.input_dim)  # (H**DxNxD, H**D)

            cKxzs = [tf.reshape(
                k.K(tf.reshape(X, (-1, self.input_dim)), Z, presliced=False),
                (HpowD, N, M)
            ) - eK[None, :, :] for k, eK in zip(self.kern_list, eKxzs)]  # Centred Kxz
            crosscovs = []
            for i, cKa in enumerate(cKxzs):
                for cKb in cKxzs[i + 1:]:
                    cc = tf.reduce_sum(cKa[:, :, None, :] * cKb[:, :, :, None] * wn[:, None, None, None], 0)
                    crosscovs.append(cc + tf.transpose(cc, [0, 2, 1]))
            crosscov = reduce(tf.add, crosscovs)
            return all_sum + crossmean + crosscov

    @property
    def on_separate_dimensions(self):
        if np.any([isinstance(k.active_dims, slice) for k in self.kern_list]):
            # Be conservative in the case of a slice object
            return False
        else:
            dimlist = [k.active_dims for k in self.kern_list]
            overlapping = False
            for i, dims_i in enumerate(dimlist):
                for dims_j in dimlist[i + 1:]:
                    if np.any(dims_i.reshape(-1, 1) == dims_j.reshape(1, -1)):
                        overlapping = True
            return not overlapping
