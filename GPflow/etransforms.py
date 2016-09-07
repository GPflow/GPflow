import numpy as np
import tensorflow as tf
import GPflow.transforms


def index_block(y, x, D):
    return np.s_[y * D:(y + 1) * D, x * D:(x + 1) * D]


class TriDiagonalBlockRep(GPflow.transforms.Transform):
    """
    Transforms an unconstrained representation of a PSD block tri diagonal matrix to its PSD block representation.
    """

    def __init__(self):
        GPflow.transforms.Transform.__init__(self)

    def forward(self, x):
        """
        Transforms from the free state to the matrix of blocks.
        :param x: Unconstrained state (Nx2DxD), where D is the block size.
        :return: Return PSD blocks (2xNxDxD)
        """
        N, D = x.shape[0], x.shape[2]
        diagblocks = np.einsum('nij,nik->njk', x, x)
        ob = np.einsum('nij,nik->njk', x[:-1, :, :], x[1:, :, :])
        # ob = np.einsum('nij,njk->nik', x[:-1, :, :].transpose([0, 2, 1]), x[1:, :, :])
        offblocks = np.vstack((ob, np.zeros((1, D, D))))
        return np.array([diagblocks, offblocks])

    def tf_forward(self, x):
        N, D = tf.shape(x)[0], tf.shape(x)[2]
        xm = tf.slice(x, [0, 0, 0], tf.pack([N - 1, -1, -1]))
        xp = x[1:, :, :]
        diagblocks = tf.batch_matmul(x, x, adj_x=True)
        offblocks = tf.concat(0, [tf.batch_matmul(xm, xp, adj_x=True), tf.zeros((1, D, D), dtype=tf.float64)])
        return tf.pack([diagblocks, offblocks])

    # This class has no backwards transform and no jacobian.

    def forward_fullmat(self, x):
        """
        Utility function to transform from free state to the full block matrix representation.
        :param x: Unconstrained state (Nx2DxD), where D is the block size.
        :return: Return a PSD matrix (NDxND)
        """
        blockrep = self.forward(x)
        return self.block_to_fullmat(blockrep)

    def block_to_fullmat(self, blockrep):
        N, D = blockrep.shape[1], blockrep.shape[2]
        fullmat = np.zeros((N * D, N * D))
        # First do the tri-diagonal part
        for n in range(N):
            fullmat[index_block(n, n, D)] = blockrep[0, n, :, :]
            if n < N - 1:
                fullmat[index_block(n, n + 1, D)] = blockrep[1, n, :, :]
                fullmat[index_block(n + 1, n, D)] = blockrep[1, n, :, :].T

        # Then fill in the rest of the matrix, one block off-diagonal strip at a time
        for diag in range(2, N):
            for n in range(N - diag):
                b_d = fullmat[index_block(n + 1, n + diag, D)]  # block down
                b_dl = fullmat[index_block(n + 1, n + diag - 1, D)]  # block down left
                b_l = fullmat[index_block(n, n + diag - 1, D)].T  # block up left
                block = np.dot(np.linalg.solve(b_dl, b_d).T, b_l)
                fullmat[index_block(n, n + diag, D)] = block.T
                fullmat[index_block(n + diag, n, D)] = block
        return fullmat

    @staticmethod
    def sample(blockmat, num_samples):
        """
        :param blockmat: 2xNxDxD
        :param num_samples: Number of samples
        :return: samplesxNxD
        """
        N = blockmat.shape[1]
        D = blockmat.shape[2]
        samples = np.zeros((num_samples, N, D))
        samples[:, 0, :] = np.dot(np.linalg.cholesky(blockmat[0, 0, :, :]), np.random.randn(D, num_samples)).T
        for n in range(1, N):
            s = np.linalg.solve(blockmat[0, n - 1, :, :], blockmat[1, n - 1, :, :]).T
            mu = np.dot(s, samples[:, n - 1, :].T).T  # samplesxD
            cov = blockmat[0, n, :, :] - np.dot(s, blockmat[1, n - 1, :, :])
            ccov = np.linalg.cholesky(cov)  # DxD
            samples[:, n, :] = mu + np.dot(ccov, np.random.randn(D, num_samples)).T
        return samples

    def __str__(self):
        return "BlockTriDiagonal"
