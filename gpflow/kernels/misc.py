import numpy as np
import tensorflow as tf

from ..base import Parameter
from ..config import default_float
from ..utilities import positive
from .base import Kernel


class ArcCosine(Kernel):
    """
    The Arc-cosine family of kernels which mimics the computation in neural
    networks. The order parameter specifies the assumed activation function.
    The Multi Layer Perceptron (MLP) kernel is closely related to the ArcCosine
    kernel of order 0. The key reference is

    ::

        @incollection{NIPS2009_3628,
            title = {Kernel Methods for Deep Learning},
            author = {Youngmin Cho and Lawrence K. Saul},
            booktitle = {Advances in Neural Information Processing Systems 22},
            year = {2009},
            url = {http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf}
        }
    """

    implemented_orders = {0, 1, 2}

    def __init__(self, order=0, variance=1.0, weight_variances=1., bias_variance=1., active_dims=None):
        """
        :param order: specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order
        :param variance: the (initial) value for the variance parameter
        :param weight_variances: the (initial) value for the weight_variances parameter,
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param bias_variance: the (initial) value for the bias_variance parameter
            defaults to 1.0
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(active_dims)

        if order not in self.implemented_orders:
            raise ValueError('Requested kernel order is not implemented.')
        self.order = order

        self.variance = Parameter(variance, transform=positive())
        self.bias_variance = Parameter(bias_variance, transform=positive())
        self.weight_variances = Parameter(weight_variances, transform=positive())
        self._validate_ard_active_dims(self.weight_variances)

    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.weight_variances.shape.ndims > 0

    def _weighted_product(self, X, X2=None):
        if X2 is None:
            return tf.reduce_sum(self.weight_variances * tf.square(X), axis=1) + self.bias_variance
        return tf.linalg.matmul((self.weight_variances * X), X2, transpose_b=True) + self.bias_variance

    def _J(self, theta):
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
        elif self.order == 2:
            return 3. * tf.sin(theta) * tf.cos(theta) + (np.pi - theta) * (1. + 2. * tf.cos(theta)**2)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)

        X_denominator = tf.sqrt(self._weighted_product(X))
        if X2 is None:
            X2 = X
            X2_denominator = X_denominator
        else:
            X2_denominator = tf.sqrt(self._weighted_product(X2))

        numerator = self._weighted_product(X, X2)
        cos_theta = numerator / X_denominator[:, None] / X2_denominator[None, :]
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return self.variance * (1. / np.pi) * self._J(theta) * \
               X_denominator[:, None] ** self.order * \
               X2_denominator[None, :] ** self.order

    def K_diag(self, X, presliced=False):
        if not presliced:
            X, _ = self.slice(X, None)

        X_product = self._weighted_product(X)
        const = tf.cast((1. / np.pi) * self._J(0.), default_float())
        return self.variance * const * X_product**self.order


class Coregion(Kernel):
    def __init__(self, output_dim, rank, active_dims=None):
        """
        A Coregionalization kernel. The inputs to this kernel are _integers_
        (we cast them from floats as needed) which usually specify the
        *outputs* of a Coregionalization model.

        The parameters of this kernel, W, kappa, specify a positive-definite
        matrix B.

          B = W W^T + diag(kappa) .

        The kernel function is then an indexing of this matrix, so

          K(x, y) = B[x, y] .

        We refer to the size of B as "num_outputs x num_outputs", since this is
        the number of outputs in a coregionalization model. We refer to the
        number of columns on W as 'rank': it is the number of degrees of
        correlation between the outputs.

        NB. There is a symmetry between the elements of W, which creates a
        local minimum at W=0. To avoid this, it's recommended to initialize the
        optimization (or MCMC chain) using a random W.
        """

        # assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(active_dims)

        self.output_dim = output_dim
        self.rank = rank
        W = np.zeros((self.output_dim, self.rank))
        kappa = np.ones(self.output_dim)
        self.W = Parameter(W)
        self.kappa = Parameter(kappa, transform=positive())

    def K(self, X, X2=None, presliced=False):
        X, X2 = self.slice(X, X2)
        X = tf.cast(X[:, 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[:, 0], tf.int32)
        B = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(self.kappa)
        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    def K_diag(self, X, presliced=False):
        X, _ = self.slice(X, None)
        X = tf.cast(X[:, 0], tf.int32)
        Bdiag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        return tf.gather(Bdiag, X)
