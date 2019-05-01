import numpy as np
import tensorflow as tf
from ..base import Parameter, positive
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

    def __init__(self,
                 order=0,
                 variance=1.0,
                 weight_variances=1.,
                 bias_variance=1.,
                 active_dims=None,
                 ard=None):
        """
        - input_dim is the dimension of the input to the kernel
        - order specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order.
        - variance is the initial value for the variance parameter
        - weight_variances is the initial value for the weight_variances parameter
          defaults to 1.0 (ard=False) or np.ones(input_dim) (ard=True).
        - bias_variance is the initial value for the bias_variance parameter
          defaults to 1.0.
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ard specifies whether the kernel has one weight_variance per dimension
          (ard=True) or a single weight_variance (ard=False).
        """
        super().__init__(active_dims)

        if order not in self.implemented_orders:
            raise ValueError('Requested kernel order is not implemented.')
        self.order = order

        self.variance = Parameter(variance, transform=positive())
        self.bias_variance = Parameter(bias_variance, transform=positive())
        # weight_variances, self.ard = self._validate_ard_shape("weight_variances", weight_variances, ard)
        self.ard = ard
        self.weight_variances = Parameter(weight_variances,
                                          transform=positive())

    def _weighted_product(self, X, X2=None):
        if X2 is None:
            return tf.reduce_sum(self.weight_variances * tf.square(X),
                                 axis=1) + self.bias_variance
        return tf.linalg.matmul(
            (self.weight_variances * X), X2,
            transpose_b=True) + self.bias_variance

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
            return 3. * tf.sin(theta) * tf.cos(theta) + \
                   (np.pi - theta) * (1. + 2. * tf.cos(theta) ** 2)

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
        return self.variance * (1. /
                                np.pi) * self._J(0.) * X_product**self.order


class Periodic(Kernel):
    """
    The periodic kernel. Defined in  Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    Derived using an RBF kernel once mapped the original inputs through
    the mapping u=(cos(x), sin(x)).

    The resulting periodic kernel can be expressed as:
        k(r) =  σ² exp{ -0.5 sin²(π r / γ) / ℓ²}

    where:
    r  is the Euclidean distance between the input points
    ℓ is the lengthscale parameter,
    σ² is the variance parameter,
    γ is the period parameter.

    (note that usually we have a factor of 4 instead of 0.5 in front but this is absorbed into lengthscale
    hyperparameter).
    """

    def __init__(self,
                 period=1.0,
                 variance=1.0,
                 lengthscale=1.0,
                 active_dims=None):
        # No ard support for lengthscale or period yet
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive())
        self.lengthscale = Parameter(lengthscale, transform=positive())
        self.ard = False
        self.period = Parameter(period, transform=positive())

    def K_diag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now [N, 1, D]
        f2 = tf.expand_dims(X2, 0)  # now [1, M, D]

        r = np.pi * (f - f2) / self.period
        r = tf.reduce_sum(tf.square(tf.sin(r) / self.lengthscale), 2)

        return self.variance * tf.exp(-0.5 * r)


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
        B = tf.linalg.matmul(self.W, self.W,
                             transpose_b=True) + tf.linalg.diag(self.kappa)
        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    def K_diag(self, X, presliced=False):
        X, _ = self.slice(X, None)
        X = tf.cast(X[:, 0], tf.int32)
        Bdiag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        return tf.gather(Bdiag, X)
