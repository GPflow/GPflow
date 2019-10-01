# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews
# Copyright 2017 Artem Artemev @awav
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
r"""
Kernels form a core component of GPflow models and allow prior information to
be encoded about a latent function of interest. The effect of choosing
different kernels, and how it is possible to combine multiple kernels is shown
in the `"Using kernels in GPflow" notebook <notebooks/kernels.html>`_.

Broadcasting over leading dimensions:
`kernel.K(X1, X2)` returns the kernel evaluated on every pair in X1 and X2.
E.g. if X1 has shape [S1, N1, D] and X2 has shape [S2, N2, D], kernel.K(X1, X2)
will return a tensor of shape [S1, N1, S2, N2]. Similarly, kernel.K(X1, X1)
returns a tensor of shape [S1, N1, S1, N1]. In contrast, the return shape of
kernel.K(X1) is [S1, N1, N1]. (Without leading dimensions, the behaviour of
kernel.K(X, None) is identical to kernel.K(X, X).)
"""

from functools import reduce, lru_cache
import warnings

import tensorflow as tf
import numpy as np

from . import transforms
from . import settings

from .decors import params_as_tensors, autoflow
from .misc import _broadcasting_elementwise_op
from .params import Parameter, Parameterized, ParamList


class Kernel(Parameterized):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        """
        input dim is an integer
        active dims is either an iterable of integers or None.

        Input dim is the number of input dimensions to the kernel. If the
        kernel is computed on a matrix X which has more columns than input_dim,
        then by default, only the first input_dim columns are used. If
        different columns are required, then they may be specified by
        active_dims.

        If active dims is None, it effectively defaults to range(input_dim),
        but we store it as a slice for efficiency.
        """
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        if active_dims is None:
            self.active_dims = slice(input_dim)
        elif isinstance(active_dims, slice):
            self.active_dims = active_dims
            if active_dims.start is not None and active_dims.stop is not None and active_dims.step is not None:
                assert len(
                    range(active_dims.start, active_dims.stop, active_dims.step)) == input_dim  # pragma: no cover
        else:
            self.active_dims = np.array(active_dims, dtype=np.int32)
            assert len(active_dims) == input_dim

    def _validate_ard_shape(self, name, value, ARD=None):
        """
        Validates the shape of a potentially ARD hyperparameter

        :param name: The name of the parameter (used for error messages)
        :param value: A scalar or an array.
        :param ARD: None, False, or True. If None, infers ARD from shape of value.
        :return: Tuple (value, ARD), where _value_ is a scalar if input_dim==1 or not ARD, array otherwise.
            The _ARD_ is False if input_dim==1 or not ARD, True otherwise.
        """
        if ARD is None:
            ARD = np.asarray(value).squeeze().shape != ()

        if ARD:
            # accept float or array:
            value = value * np.ones(self.input_dim, dtype=settings.float_type)

        if self.input_dim == 1 or not ARD:
            correct_shape = ()
        else:
            correct_shape = (self.input_dim,)

        if np.asarray(value).squeeze().shape != correct_shape:
            raise ValueError("shape of {} does not match input_dim".format(name))

        return value, ARD

    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None]))
    def compute_K(self, X, Z):
        return self.K(X, Z)

    @autoflow((settings.float_type, [None, None]))
    def compute_K_symm(self, X):
        return self.K(X)

    @autoflow((settings.float_type, [None, None]))
    def compute_Kdiag(self, X):
        return self.Kdiag(X)

    def on_separate_dims(self, other_kernel):
        """
        Checks if the dimensions, over which the kernels are specified, overlap.
        Returns True if they are defined on different/separate dimensions and False otherwise.
        """
        if isinstance(self.active_dims, slice) or isinstance(other_kernel.active_dims, slice):
            # Be very conservative for kernels defined over slices of dimensions
            return False

        if np.any(self.active_dims.reshape(-1, 1) == other_kernel.active_dims.reshape(1, -1)):
            return False

        return True

    def _slice(self, X, X2):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims`.
        :param X: Input 1 (NxD).
        :param X2: Input 2 (MxD), may be None.
        :return: Sliced X, X2, (Nxself.input_dim).
        """
        if isinstance(self.active_dims, slice):
            X = X[..., self.active_dims]
            if X2 is not None:
                X2 = X2[..., self.active_dims]
        else:
            X = tf.gather(X, self.active_dims, axis=-1)
            if X2 is not None:
                X2 = tf.gather(X2, self.active_dims, axis=-1)

        input_dim_shape = tf.shape(X)[-1]
        input_dim = tf.convert_to_tensor(self.input_dim, dtype=settings.int_type)
        with tf.control_dependencies([tf.assert_equal(input_dim_shape, input_dim)]):
            X = tf.identity(X)

        return X, X2

    def _slice_cov(self, cov):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims` for covariance matrices. This requires slicing the
        rows *and* columns. This will also turn flattened diagonal
        matrices into a tensor of full diagonal matrices.
        :param cov: Tensor of covariance matrices (NxDxD or NxD).
        :return: N x self.input_dim x self.input_dim.
        """
        cov = tf.cond(tf.equal(tf.rank(cov), 2), lambda: tf.matrix_diag(cov), lambda: cov)

        if isinstance(self.active_dims, slice):
            cov = cov[..., self.active_dims, self.active_dims]
        else:
            cov_shape = tf.shape(cov)
            covr = tf.reshape(cov, [-1, cov_shape[-1], cov_shape[-1]])
            gather1 = tf.gather(tf.transpose(covr, [2, 1, 0]), self.active_dims)
            gather2 = tf.gather(tf.transpose(gather1, [1, 0, 2]), self.active_dims)
            cov = tf.reshape(tf.transpose(gather2, [2, 0, 1]),
                             tf.concat([cov_shape[:-2], [len(self.active_dims), len(self.active_dims)]], 0))
        return cov

    def __add__(self, other):
        return Sum([self, other])

    def __mul__(self, other):
        return Product([self, other])


class Static(Kernel):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance, σ².
    """

    def __init__(self, input_dim, variance=1.0, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance, transform=transforms.positive,
                                  dtype=settings.float_type)

    @params_as_tensors
    def Kdiag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


class White(Static):
    """
    The White kernel: this kernel produces 'white noise'. The kernel equation is

        k(x_n, x_m) = δ(n, m) σ²

    where:
    δ(.,.) is the Kronecker delta,
    σ²  is the variance parameter.
    """

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            d = tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
            return tf.matrix_diag(d)
        else:
            shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], 0)
            return tf.zeros(shape, settings.float_type)


class Constant(Static):
    """
    The Constant (aka Bias) kernel. Functions drawn from a GP with this kernel
    are constant, i.e. f(x) = c, with c ~ N(0, σ^2). The kernel equation is

        k(x, y) = σ²

    where:
    σ²  is the variance parameter.
    """

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            shape = tf.concat([tf.shape(X)[:-2],
                               tf.reshape(tf.shape(X)[-2], [1]),
                               tf.reshape(tf.shape(X)[-2], [1])], 0)
        else:
            shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], 0)
        return tf.fill(shape, tf.squeeze(self.variance))


class Bias(Constant):
    """
    Another name for the Constant kernel, included for convenience.
    """
    pass


class Stationary(Kernel):
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=1.0,
                 active_dims=None, ARD=None, name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - if ARD is not None, it specifies whether the kernel has one
          lengthscale per dimension (ARD=True) or a single lengthscale
          (ARD=False). Otherwise, inferred from shape of lengthscales.
        """
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance, transform=transforms.positive,
                                  dtype=settings.float_type)

        lengthscales, self.ARD = self._validate_ard_shape("lengthscales", lengthscales, ARD)
        self.lengthscales = Parameter(lengthscales, transform=transforms.positive,
                                      dtype=settings.float_type)

    @params_as_tensors
    def _scaled_square_dist(self, X, X2):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.

        This function can deal with leading dimensions in X and X2. 
        In the sample case, where X and X2 are both 2 dimensional, 
        for example, X is [N, D] and X2 is [M, D], then a tensor of shape 
        [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D] 
        then the output will be [N1, S1, N2, S2].
        """

        X = X / self.lengthscales

        if X2 is None:
            Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += Xs + tf.matrix_transpose(Xs)
            return dist

        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        X2 = X2 / self.lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        dist = -2 * tf.tensordot(X, X2, [[-1], [-1]])
        dist += _broadcasting_elementwise_op(tf.add, Xs, X2s)
        return dist

    @staticmethod
    def _clipped_sqrt(r2):
        # Clipping around the (single) float precision which is ~1e-45.
        return tf.sqrt(tf.maximum(r2, 1e-36))

    def scaled_square_dist(self, X, X2):  # pragma: no cover
        return self._scaled_square_dist(X, X2)

    def scaled_euclid_dist(self, X, X2):  # pragma: no cover
        """
        Returns |(X - X2ᵀ)/lengthscales| (L2-norm).
        """
        warnings.warn('scaled_euclid_dist is deprecated and will be removed '
                      'in GPflow version 1.4.0. For stationary kernels, '
                      'define K_r(r) instead.',
                      DeprecationWarning)
        r2 = self.scaled_square_dist(X, X2)
        return self._clipped_sqrt(r2)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        """
        Calculates the kernel matrix K(X, X2) (or K(X, X) if X2 is None).
        Handles the slicing as well as scaling and computes k(x, x') = k(r),
        where r² = ((x - x')/lengthscales)².

        Internally, this calls self.K_r2(r²), which in turn computes the
        square-root and calls self.K_r(r). Classes implementing stationary
        kernels can either overwrite `K_r2(r2)` if they only depend on the
        squared distance, or `K_r(r)` if they need the actual radial distance.
        """
        if not presliced:
            X, X2 = self._slice(X, X2)
        return self.K_r2(self.scaled_square_dist(X, X2))

    @params_as_tensors
    def K_r(self, r):
        """
        Returns the kernel evaluated on `r`, which is the scaled Euclidean distance
        Should operate element-wise on r
        """
        raise NotImplementedError

    def K_r2(self, r2):
        """
        Returns the kernel evaluated on `r2`, which is the scaled squared distance.
        Will call self.K_r(r=sqrt(r2)), or can be overwritten directly (and should operate element-wise on r2).
        """
        r = self._clipped_sqrt(r2)
        return self.K_r(r)


class SquaredExponential(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is

        k(r) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    @params_as_tensors
    def K_r2(self, r2):
        return self.variance * tf.exp(-r2 / 2.)


RBF = SquaredExponential


class RationalQuadratic(Stationary):
    """
    Rational Quadratic kernel. The kernel equation is

    k(r) = σ² (1 + r² / 2α)^(-α)

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter,
    α  determines relative weighting of small-scale and large-scale fluctuations.

    For α → ∞, the RQ kernel becomes equivalent to the squared exponential.
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=1.0, alpha=1.0,
                 active_dims=None, ARD=None, name=None):
        super().__init__(input_dim, variance, lengthscales, active_dims, ARD, name)
        self.alpha = Parameter(alpha, transform=transforms.positive,
                               dtype=settings.float_type)

    @params_as_tensors
    def K_r2(self, r2):
        return self.variance * (1 + r2 / (2 * self.alpha)) ** (- self.alpha)


class Linear(Kernel):
    """
    The linear kernel. Functions drawn from a GP with this kernel are linear, i.e. f(x) = cx. The kernel equation is

        k(x, y) = σ²xy

    where:
    σ²  is the variance parameter.
    """

    def __init__(self, input_dim, variance=1.0, active_dims=None, ARD=None, name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter(s)
          if ARD=True, there is one variance per input
        - active_dims is a list of length input_dim which controls
          which columns of X are used.
        """
        super().__init__(input_dim, active_dims, name=name)

        variance, self.ARD = self._validate_ard_shape("variance", variance, ARD)
        self.variance = Parameter(variance, transform=transforms.positive,
                                  dtype=settings.float_type)

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            return tf.matmul(X * self.variance, X, transpose_b=True)
        else:
            return tf.tensordot(X * self.variance, X2, [[-1], [-1]])

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.reduce_sum(tf.square(X) * self.variance, -1)


class Polynomial(Linear):
    """
    The Polynomial kernel. Functions drawn from a GP with this kernel are
    polynomials of degree `d`. The kernel equation is

        k(x, y) = (σ²xy + γ) ^ d

    where:
    σ² is the variance parameter,
    γ is the offset parameter,
    d is the degree parameter.
    """

    def __init__(self, input_dim,
                 degree=3.0,
                 variance=1.0,
                 offset=1.0,
                 active_dims=None,
                 ARD=None,
                 name=None):
        """
        :param input_dim: the dimension of the input to the kernel
        :param variance: the (initial) value for the variance parameter(s)
                         if ARD=True, there is one variance per input
        :param degree: the degree of the polynomial
        :param active_dims: a list of length input_dim which controls
          which columns of X are used.
        :param ARD: use variance as described
        """
        super().__init__(input_dim, variance, active_dims, ARD, name=name)
        self.degree = degree
        self.offset = Parameter(offset, transform=transforms.positive,
                                dtype=settings.float_type)

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        return (Linear.K(self, X, X2, presliced=presliced) + self.offset) ** self.degree

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return (Linear.Kdiag(self, X, presliced=presliced) + self.offset) ** self.degree


class Exponential(Stationary):
    """
    The Exponential kernel. It is equivalent to a Matern12 kernel with doubled lengthscales.
    """

    @params_as_tensors
    def K_r(self, r):
        return self.variance * tf.exp(-0.5 * r)


class Matern12(Stationary):
    """
    The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
    differentiable anywhere. The kernel equation is

    k(r) = σ² exp{-r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
    σ² is the variance parameter
    """

    @params_as_tensors
    def K_r(self, r):
        return self.variance * tf.exp(-r)


class Matern32(Stationary):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
    """

    @params_as_tensors
    def K_r(self, r):
        sqrt3 = np.sqrt(3.)
        return self.variance * (1. + sqrt3 * r) * tf.exp(-sqrt3 * r)


class Matern52(Stationary):
    """
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
    differentiable. The kernel equation is

    k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
    """

    @params_as_tensors
    def K_r(self, r):
        sqrt5 = np.sqrt(5.)
        return self.variance * (1.0 + sqrt5 * r + 5. / 3. * tf.square(r)) * tf.exp(-sqrt5 * r)


class Cosine(Stationary):
    """
    The Cosine kernel. Functions drawn from a GP with this kernel are sinusoids
    (with a random phase).  The kernel equation is

        k(r) =  σ² cos{r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
    """

    @params_as_tensors
    def K_r(self, r):
        return self.variance * tf.cos(r)


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

    Note: broadcasting over leading dimensions has not yet been implemented for
    the ArcCosine kernel.
    """

    implemented_orders = {0, 1, 2}

    def __init__(self, input_dim,
                 order=0,
                 variance=1.0, weight_variances=1., bias_variance=1.,
                 active_dims=None, ARD=None, name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - order specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order.
        - variance is the initial value for the variance parameter
        - weight_variances is the initial value for the weight_variances parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - bias_variance is the initial value for the bias_variance parameter
          defaults to 1.0.
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one weight_variance per dimension
          (ARD=True) or a single weight_variance (ARD=False).
        """
        super().__init__(input_dim, active_dims, name=name)

        if order not in self.implemented_orders:
            raise ValueError('Requested kernel order is not implemented.')
        self.order = order

        self.variance = Parameter(variance, transform=transforms.positive,
                                  dtype=settings.float_type)
        self.bias_variance = Parameter(bias_variance, transform=transforms.positive,
                                       dtype=settings.float_type)
        weight_variances, self.ARD = self._validate_ard_shape("weight_variances", weight_variances, ARD)
        self.weight_variances = Parameter(weight_variances, transform=transforms.positive,
                                          dtype=settings.float_type)

    @params_as_tensors
    def _weighted_product(self, X, X2=None):
        if X2 is None:
            return tf.reduce_sum(self.weight_variances * tf.square(X), axis=-1) + self.bias_variance
        return tf.matmul((self.weight_variances * X), X2, transpose_b=True) + self.bias_variance

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

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)

        X_denominator = tf.sqrt(self._weighted_product(X))
        if X2 is None:
            X2 = X
            X2_denominator = X_denominator
        else:
            X2_denominator = tf.sqrt(self._weighted_product(X2))

        numerator = self._weighted_product(X, X2)
        X_denominator = tf.expand_dims(X_denominator, -1)
        X2_denominator = tf.matrix_transpose(tf.expand_dims(X2_denominator, -1))
        cos_theta = numerator / X_denominator / X2_denominator
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return self.variance * (1. / np.pi) * self._J(theta) * \
               X_denominator ** self.order * \
               X2_denominator ** self.order

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)

        X_product = self._weighted_product(X)
        theta = tf.constant(0., settings.float_type)
        return self.variance * (1. / np.pi) * self._J(theta) * X_product ** self.order


class Periodic(Kernel):
    """
    The periodic family of kernels. Can be used to wrap any Stationary kernel
    to transform it into a periodic version. The canonical form (based on the
    SquaredExponential kernel) can be found in Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    The derivation can be achieved by mapping the original inputs through the
    transformation u = (cos(x), sin(x)).

    For the SquaredExponential base kernel, the result can be expressed as:
        k(r) = σ² exp{ -0.5 sin²(π r / γ) / ℓ² }

    where:
    r is the Euclidean distance between the input points,
    ℓ is the lengthscale parameter,
    σ² is the variance parameter,
    γ is the period parameter.

    (note that usually we have a factor of 4 instead of 0.5 in front but this
    is absorbed into lengthscale hyperparameter).
    """
    def __init__(self, input_dim=None, period=1.0, variance=1.0,
                 lengthscales=1.0, base=None, active_dims=None, name=None):
        """
        The Periodic kernel supports the specification of any Stationary kernel
        as the base through the "base" parameter. If this is provided then
        input_dim, variance, lengthscales and active_dims parameters are
        taken from there.

        If a "base" kernel is not provided, then a SquaredExponential is
        constructed implicitly using the input_dim, variance, lengthscales
        and active_dims parameters.
        """
        if (base is not None) and (input_dim is not None):
            raise ValueError("input_dim should be defined through the base kernel, not explicitly")

        if base is None:
            warnings.warn(
                'Implicit specification of the base kernel for Periodic is '
                'deprecated, use Periodic(base=SquaredExponential({i}, '
                'variance={v}, lengthscales={l}, active_dims={a}), period={p}) '
                'instead.'.format(i=input_dim, v=variance, l=lengthscales,
                                  a=active_dims, p=period),
                DeprecationWarning)
            base = SquaredExponential(
                input_dim,
                variance=variance,
                lengthscales=lengthscales,
                active_dims=active_dims)

        if not isinstance(base, Stationary):
            raise TypeError("Periodic requires a Stationary kernel as the `base`")

        super().__init__(base.input_dim, base.active_dims, name=name)
        self.base = base
        self.period = Parameter(  # No ARD support for period yet
            period, transform=transforms.positive, dtype=settings.float_type)

    @property
    def ARD(self):
        return self.base.ARD

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return self.base.Kdiag(X)

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, -2)  # ... x N x 1 x D
        f2 = tf.expand_dims(X2, -3)  # ... x 1 x M x D

        r = np.pi * (f - f2) / self.period
        scaled_sine = tf.sin(r) / self.base.lengthscales
        try:
            sine_r = tf.reduce_sum(tf.abs(scaled_sine), -1)
            K = self.base.K_r(sine_r)
        except NotImplementedError:
            sine_r2 = tf.reduce_sum(tf.square(scaled_sine), -1)
            K = self.base.K_r2(sine_r2)
        return K


class Coregion(Kernel):
    def __init__(self, input_dim, output_dim, rank, active_dims=None, name=None):
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
        assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(input_dim, active_dims, name=name)

        self.output_dim = output_dim
        self.rank = rank
        self.W = Parameter(np.zeros((self.output_dim, self.rank), dtype=settings.float_type))
        self.kappa = Parameter(np.ones(self.output_dim, dtype=settings.float_type), transform=transforms.positive)

    @params_as_tensors
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        X = tf.cast(X[:, 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[:, 0], tf.int32)
        B = tf.matmul(self.W, self.W, transpose_b=True) + tf.matrix_diag(self.kappa)
        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    @params_as_tensors
    def Kdiag(self, X):
        X, _ = self._slice(X, None)
        X = tf.cast(X[:, 0], tf.int32)
        Bdiag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        return tf.gather(Bdiag, X)


class Combination(Kernel):
    """
    Combine a list of kernels, e.g. by adding or multiplying (see inheriting
    classes).

    The names of the kernels to be combined are generated from their class
    names.
    """

    def __init__(self, kernels, name=None):
        if not all(isinstance(k, Kernel) for k in kernels):
            raise TypeError("can only combine Kernel instances")  # pragma: no cover

        input_dim = np.max([k.input_dim
                            if type(k.active_dims) is slice else
                            np.max(k.active_dims) + 1
                            for k in kernels])
        super().__init__(input_dim=input_dim, name=name)

        # add kernels to a list, flattening out instances of this class therein
        kernels_list = []
        for k in kernels:
            if isinstance(k, self.__class__):
                kernels_list.extend(k.kernels)
            else:
                kernels_list.append(k)
        self.kernels = ParamList(kernels_list)

    @property
    def on_separate_dimensions(self):
        """
        Checks whether the kernels in the combination act on disjoint subsets
        of dimensions. Currently, it is hard to asses whether two slice objects
        will overlap, so this will always return False.
        :return: Boolean indicator.
        """
        if np.any([isinstance(k.active_dims, slice) for k in self.kernels]):
            # Be conservative in the case of a slice object
            return False
        else:
            dimlist = [k.active_dims for k in self.kernels]
            overlapping = False
            for i, dims_i in enumerate(dimlist):
                for dims_j in dimlist[i + 1:]:
                    if np.any(dims_i.reshape(-1, 1) == dims_j.reshape(1, -1)):
                        overlapping = True
            return not overlapping


class Convolutional(Kernel):
    """
    Plain convolutional kernel as described in \citet{vdw2017convgp}. Defines
    a GP f( ) that is constructed from a sum of responses of individual patches
    in an image:
      f(x) = \sum_p x^{[p]}
    where x^{[p]} is the pth patch in the image.

    @incollection{vdw2017convgp,
      title = {Convolutional Gaussian Processes},
      author = {van der Wilk, Mark and Rasmussen, Carl Edward and Hensman, James},
      booktitle = {Advances in Neural Information Processing Systems 30},
      year = {2017},
      url = {http://papers.nips.cc/paper/6877-convolutional-gaussian-processes.pdf}
    }
    """

    def __init__(self, basekern, img_size, patch_size, colour_channels=1):
        Kernel.__init__(self, np.prod(img_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.basekern = basekern
        self.colour_channels = colour_channels

        if self.basekern.input_dim != np.prod(patch_size):
            raise ValueError("Basekern input dimensions must be consistent with patch size.")

    @lru_cache()
    @params_as_tensors
    def get_patches(self, X):
        """
        Extracts patches from the images X. Patches are extracted separately for each of the colour channels.
        :param X: (N x input_dim)
        :return: Patches (N, num_patches, patch_size)
        """
        # Roll the colour channel to the front, so it appears to `tf.extract_image_patches()` as separate images. Then
        # extract patches and reshape to have the first axis the same as the number of images. The separate patches will
        # then be in the second axis.
        castX = tf.transpose(
            tf.reshape(X, [tf.shape(X)[0], -1, self.colour_channels]),
            [0, 2, 1])
        patches = tf.extract_image_patches(
            tf.reshape(castX, [-1, self.img_size[0], self.img_size[1], 1], name="rX"),
            [1, self.patch_size[0], self.patch_size[1], 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1], "VALID")
        shp = tf.shape(patches)  # img x out_rows x out_cols
        return tf.cast(tf.reshape(patches,
                                  [tf.shape(X)[0], self.colour_channels * shp[1] * shp[2], shp[3]]),
                       settings.float_type)

    @params_as_tensors
    def K(self, X, X2=None):
        Xp = self.get_patches(X)  # [N, num_patches, patch_len]
        Xp2 = self.get_patches(X2) if X2 is not None else Xp

        bigK = self.basekern.K(Xp, Xp2)  # [N, num_patches, N, num_patches]
        K = tf.reduce_sum(bigK, [1, 3])
        return K / self.num_patches ** 2.0

    @params_as_tensors
    def Kdiag(self, X):
        Xp = self.get_patches(X)  # [N, num_patches, patch_len]
        return tf.reduce_sum(self.basekern.K(Xp), [1, 2]) / self.num_patches ** 2.0

    @property
    def patch_len(self):
        return np.prod(self.patch_size)

    @property
    def num_patches(self):
        return (self.img_size[0] - self.patch_size[0] + 1) * (
                self.img_size[1] - self.patch_size[1] + 1) * self.colour_channels

    @autoflow((settings.float_type,))
    def compute_patches(self, X):
        return self.get_patches(X)


class WeightedConvolutional(Convolutional):
    """
    Similar to `Convolutional`, but with different weights for each patch:
      f(x) = \sum_p x^{[p]}

     @incollection{vdw2017convgp,
      title = {Convolutional Gaussian Processes},
      author = {van der Wilk, Mark and Rasmussen, Carl Edward and Hensman, James},
      booktitle = {Advances in Neural Information Processing Systems 30},
      year = {2017},
      url = {http://papers.nips.cc/paper/6877-convolutional-gaussian-processes.pdf}
    }
    """
    def __init__(self, basekern, img_size, patch_size, weights=None, colour_channels=1):
        Convolutional.__init__(self, basekern, img_size, patch_size, colour_channels)
        self.weights = Parameter(np.ones(self.num_patches, dtype=settings.float_type) if weights is None
                                 else weights)

    @params_as_tensors
    def K(self, X, X2=None):
        Xp = self.get_patches(X)  # [N, P, patch_len]
        Xp2 = Xp if X2 is None else self.get_patches(X2)

        bigK = self.basekern.K(Xp, Xp2)  # [N, num_patches, N, num_patches]

        W2 = self.weights[:, None] * self.weights[None, :]  # [P, P]
        W2bigK = bigK * W2[None, :, None, :]
        return tf.reduce_sum(W2bigK, [1, 3]) / self.num_patches ** 2.0

    @params_as_tensors
    def Kdiag(self, X):
        Xp = self.get_patches(X)  # N x num_patches x patch_dim
        W2 = self.weights[:, None] * self.weights[None, :]  # [P, P]
        bigK = self.basekern.K(Xp)  # [N, P, P]
        return tf.reduce_sum(bigK * W2[None, :, :], [1, 2]) / self.num_patches ** 2.0


class Sum(Combination):
    def K(self, X, X2=None, presliced=False):
        return reduce(tf.add, [k.K(X, X2) for k in self.kernels])

    def Kdiag(self, X, presliced=False):
        return reduce(tf.add, [k.Kdiag(X) for k in self.kernels])


class Product(Combination):
    def K(self, X, X2=None, presliced=False):
        return reduce(tf.multiply, [k.K(X, X2) for k in self.kernels])

    def Kdiag(self, X, presliced=False):
        return reduce(tf.multiply, [k.Kdiag(X) for k in self.kernels])


def make_deprecated_class(oldname, NewClass):
    """
    Returns a class that raises NotImplementedError on instantiation.
    e.g.:
    >>> Kern = make_deprecated_class("Kern", Kernel)
    """
    msg = ("{module}.{} has been renamed to {module}.{}"
           .format(oldname, NewClass.__name__, module=NewClass.__module__))

    class OldClass(NewClass):
        def __new__(cls, *args, **kwargs):
            raise NotImplementedError(msg)

    OldClass.__doc__ = msg
    OldClass.__qualname__ = OldClass.__name__ = oldname
    return OldClass


Kern = make_deprecated_class("Kern", Kernel)
Add = make_deprecated_class("Add", Sum)
Prod = make_deprecated_class("Prod", Product)
PeriodicKernel = make_deprecated_class("PeriodicKernel", Periodic)
