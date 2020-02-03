# Copyright 2018 the GPflow authors.
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

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.config import default_float
from gpflow.kernels import SquaredExponential, ArcCosine, Linear


rng = np.random.RandomState(1)


def _ref_rbf(X, lengthscale, signal_variance):
    num_data, _ = X.shape
    kernel = np.zeros((num_data, num_data))
    for row_index in range(num_data):
        for column_index in range(num_data):
            vecA = X[row_index, :]
            vecB = X[column_index, :]
            delta = vecA - vecB
            distance_squared = np.dot(delta.T, delta)
            kernel[row_index, column_index] = signal_variance * \
                                              np.exp(-0.5 * distance_squared / lengthscale ** 2)
    return kernel


def _ref_arccosine(X, order, weight_variances, bias_variance, signal_variance):
    num_points = X.shape[0]
    kernel = np.empty((num_points, num_points))
    for row in range(num_points):
        for col in range(num_points):
            x = X[row]
            y = X[col]

            numerator = (weight_variances * x).dot(y) + bias_variance

            x_denominator = np.sqrt((weight_variances * x).dot(x) +
                                    bias_variance)
            y_denominator = np.sqrt((weight_variances * y).dot(y) +
                                    bias_variance)
            denominator = x_denominator * y_denominator

            theta = np.arccos(np.clip(numerator / denominator, -1., 1.))
            if order == 0:
                J = np.pi - theta
            elif order == 1:
                J = np.sin(theta) + (np.pi - theta) * np.cos(theta)
            elif order == 2:
                J = 3. * np.sin(theta) * np.cos(theta)
                J += (np.pi - theta) * (1. + 2. * np.cos(theta) ** 2)

            kernel[row, col] = signal_variance * (1. / np.pi) * J * \
                               x_denominator ** order * \
                               y_denominator ** order
    return kernel


def _ref_periodic(X, base_name, lengthscale, signal_variance, period):
    """
    Calculates K(X) for the periodic kernel based on various base kernels.
    """
    sine_arg = np.pi * (X[:, None, :] - X[None, :, :]) / period
    sine_base = np.sin(sine_arg) / lengthscale
    if base_name in {"RBF", "SquaredExponential"}:
        dist = 0.5 * np.sum(np.square(sine_base), axis=-1)
        exp_dist = np.exp(-dist)
    elif base_name == "Matern12":
        dist = np.sum(np.abs(sine_base), axis=-1)
        exp_dist = np.exp(-dist)
    elif base_name == "Matern32":
        dist = np.sqrt(3) * np.sum(np.abs(sine_base), axis=-1)
        exp_dist = (1 + dist) * np.exp(-dist)
    elif base_name == "Matern52":
        dist = np.sqrt(5) * np.sum(np.abs(sine_base), axis=-1)
        exp_dist = (1 + dist + dist ** 2 / 3) * np.exp(-dist)
    return signal_variance * exp_dist


def _ref_changepoints(X, kernels, locations, steepness):
    """
    Calculates K(X) for each kernel in `kernels`, then multiply by sigmoid functions
    in order to smoothly transition betwen them. The sigmoid transitions are defined
    by a location and a steepness parameter.
    """
    locations = sorted(locations)
    steepness = steepness if isinstance(steepness, list) else [steepness] * len(locations)
    locations = np.array(locations).reshape((1, 1, -1))
    steepness = np.array(steepness).reshape((1, 1, -1))

    sig_X = 1. / (1. + np.exp(-steepness * (X[:, :, None] - locations)))

    starters = sig_X * np.transpose(sig_X, axes=(1, 0, 2))
    stoppers = (1 - sig_X) * np.transpose((1 - sig_X), axes=(1, 0, 2))

    ones = np.ones((X.shape[0], X.shape[0], 1))
    starters = np.concatenate([ones, starters], axis=2)
    stoppers = np.concatenate([stoppers, ones], axis=2)

    kernel_stack = np.stack([k(X) for k in kernels], axis=2)
    return (kernel_stack * starters * stoppers).sum(axis=2)


@pytest.mark.parametrize('variance, lengthscale', [[2.3, 1.4]])
def test_rbf_1d(variance, lengthscale):
    X = rng.randn(3, 1)
    kernel = gpflow.kernels.SquaredExponential(lengthscale=lengthscale, variance=variance)

    gram_matrix = kernel(X)
    reference_gram_matrix = _ref_rbf(X, lengthscale, variance)

    assert_allclose(gram_matrix, reference_gram_matrix)


@pytest.mark.parametrize('variance, lengthscale', [[2.3, 1.4]])
def test_rq_1d(variance, lengthscale):
    kSE = gpflow.kernels.SquaredExponential(lengthscale=lengthscale, variance=variance)
    kRQ = gpflow.kernels.RationalQuadratic(lengthscale=lengthscale,
                                           variance=variance,
                                           alpha=1e8)
    rng = np.random.RandomState(1)
    X = rng.randn(6, 1).astype(default_float())

    gram_matrix_SE = kSE(X)
    gram_matrix_RQ = kRQ(X)
    assert_allclose(gram_matrix_SE, gram_matrix_RQ)


def _assert_arccosine_kern_err(variance, weight_variances, bias_variance,
                               order, X):
    kernel = gpflow.kernels.ArcCosine(order=order,
                                      variance=variance,
                                      weight_variances=weight_variances,
                                      bias_variance=bias_variance)
    gram_matrix = kernel(X)
    reference_gram_matrix = _ref_arccosine(X, order, weight_variances,
                                           bias_variance, variance)
    assert_allclose(gram_matrix, reference_gram_matrix)


@pytest.mark.parametrize('order', gpflow.kernels.ArcCosine.implemented_orders)
@pytest.mark.parametrize('D, weight_variances',
                         [[1, 1.7], [3, 1.7], [3, (1.1, 1.7, 1.9)]])
@pytest.mark.parametrize('N, bias_variance, variance',
                         [[3, 0.6, 2.3]])
def test_arccosine_1d_and_3d(order, D, N, weight_variances, bias_variance,
                             variance):
    X_data = rng.randn(N, D)
    _assert_arccosine_kern_err(variance, weight_variances, bias_variance,
                               order, X_data)


@pytest.mark.parametrize('order', [42])
def test_arccosine_non_implemented_order(order):
    with pytest.raises(ValueError):
        gpflow.kernels.ArcCosine(order=order)


@pytest.mark.parametrize('D, N', [[1, 4]])
def test_arccosine_nan_gradient(D, N):
    X = rng.rand(N, D)
    kernel = gpflow.kernels.ArcCosine()
    with tf.GradientTape() as tape:
        Kff = kernel(X)
    grads = tape.gradient(Kff, kernel.trainable_variables)
    assert not np.any(np.isnan(grads))


def _assert_periodic_kern_err(base_class, lengthscale, variance, period, X):
    base = base_class(lengthscale=lengthscale, variance=variance)
    kernel = gpflow.kernels.Periodic(base, period=period)
    gram_matrix = kernel(X)
    reference_gram_matrix = _ref_periodic(X, base_class.__name__, lengthscale, variance, period)

    assert_allclose(gram_matrix, reference_gram_matrix)


@pytest.mark.parametrize('base_class', [
    gpflow.kernels.SquaredExponential,
    gpflow.kernels.Matern12,
    gpflow.kernels.Matern32,
    gpflow.kernels.Matern52,
])
@pytest.mark.parametrize('D, lengthscale, period', [
    [1, 2., 3.],                  # 1d, single lengthscale, single period
    [2, 11.5, 3.],                # 2d, single lengthscale, single period
    [2, 11.5, (3., 6.)],          # 2d, single lengthscale, ard period
    [2, (11.5, 12.5), 3.],        # 2d, ard lengthscale, single period
    [2, (11.5, 12.5), (3., 6.)],  # 2d, ard lengthscale, ard period
])
@pytest.mark.parametrize('N, variance', [
    [3, 2.3],
    [5, 1.3],
])
def test_periodic(base_class, D, N, lengthscale, variance, period):
    X = rng.randn(N, D) if D == 1 else rng.multivariate_normal(
        np.zeros(D), np.eye(D), N)
    _assert_periodic_kern_err(base_class, lengthscale, variance, period, X)


@pytest.mark.parametrize('base_class', [
    gpflow.kernels.SquaredExponential,
    gpflow.kernels.Matern12,
])
def test_periodic_diag(base_class):
    N, D = 5, 3
    X = rng.multivariate_normal(np.zeros(D), np.eye(D), N)
    base = base_class(lengthscale=2., variance=1.)
    kernel = gpflow.kernels.Periodic(base, period=6.)
    assert_allclose(base(X, full=False), kernel(X, full=False))


def test_periodic_non_stationary_base():
    error_msg = r"Periodic requires a Stationary kernel as the `base`"
    with pytest.raises(TypeError, match=error_msg):
        gpflow.kernels.Periodic(gpflow.kernels.Linear())


def test_periodic_bad_ard_period():
    error_msg = r"Size of `active_dims` \[1 2\] does not match size of ard parameter \(3\)"
    base = gpflow.kernels.RBF(active_dims=[1, 2])
    with pytest.raises(ValueError, match=error_msg):
        gpflow.kernels.Periodic(base, period=[1., 1., 1.])


kernel_setups = [
    kernel() for kernel in gpflow.kernels.Stationary.__subclasses__()
] + [
    gpflow.kernels.Constant(),
    gpflow.kernels.Linear(),
    gpflow.kernels.Polynomial(),
    gpflow.kernels.ArcCosine()
]


@pytest.mark.parametrize('D', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
@pytest.mark.parametrize('N', [10])
def test_kernel_symmetry_1d_and_5d(D, kernel, N):
    X = rng.randn(N, D)
    errors = kernel(X) - kernel(X, X)
    assert np.allclose(errors, 0)


@pytest.mark.parametrize('N, N2, input_dim, output_dim, rank',
                         [[10, 12, 1, 3, 2]])
def test_coregion_shape(N, N2, input_dim, output_dim, rank):
    X = np.random.randint(0, output_dim, (N, input_dim))
    X2 = np.random.randint(0, output_dim, (N2, input_dim))
    kernel = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank)
    kernel.W = rng.randn(output_dim, rank)
    kernel.kappa = rng.randn(output_dim, 1).reshape(-1) + 1.

    Kff2 = kernel(X, X2)
    assert Kff2.shape == (10, 12)
    Kff = kernel(X)
    assert Kff.shape == (10, 10)


@pytest.mark.parametrize('N, input_dim, output_dim, rank', [[10, 1, 3, 2]])
def test_coregion_diag(N, input_dim, output_dim, rank):
    X = np.random.randint(0, output_dim, (N, input_dim))
    kernel = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank)
    kernel.W = rng.randn(output_dim, rank)
    kernel.kappa = rng.randn(output_dim, 1).reshape(-1) + 1.

    K = kernel(X)
    Kdiag = kernel.K_diag(X)
    assert np.allclose(np.diag(K), Kdiag)


@pytest.mark.parametrize('N, input_dim, output_dim, rank', [[10, 1, 3, 2]])
def test_coregion_slice(N, input_dim, output_dim, rank):
    X = np.random.randint(0, output_dim, (N, input_dim))
    X = np.hstack((X, rng.randn(10, 1)))
    kernel1 = gpflow.kernels.Coregion(output_dim=output_dim,
                                      rank=rank,
                                      active_dims=[0])
    # compute another kernel with additinoal inputs,
    # make sure out kernel is still okay.
    kernel2 = gpflow.kernels.SquaredExponential(active_dims=[1])
    kernel_prod = kernel1 * kernel2
    K1 = kernel_prod(X)
    K2 = kernel1(X) * kernel2(X)  # slicing happens inside kernel
    assert np.allclose(K1, K2)


_dim = 3
kernel_setups_extended = kernel_setups + [
    SquaredExponential() + Linear(),
    SquaredExponential() * Linear(),
    SquaredExponential() + Linear(variance=rng.rand(_dim))
] + [ArcCosine(order=order) for order in ArcCosine.implemented_orders]


@pytest.mark.parametrize('kernel', kernel_setups_extended)
@pytest.mark.parametrize('N, dim', [[30, _dim]])
def test_diags(kernel, N, dim):
    X = np.random.randn(N, dim)
    kernel1 = tf.linalg.diag_part(kernel(X, full=True))
    kernel2 = kernel(X, full=False)
    assert np.allclose(kernel1, kernel2)


def test_conv_diag():
    kernel = gpflow.kernels.Convolutional(gpflow.kernels.SquaredExponential(), [3, 3], [2, 2])
    X = np.random.randn(3, 9)
    kernel_full = np.diagonal(kernel(X, full=True))
    kernel_diag = kernel(X, full=False)
    assert np.allclose(kernel_full, kernel_diag)


# Add a rbf and linear kernel, make sure the result is the same as adding the result of
# the kernels separately.
_kernel_setups_add = [
    gpflow.kernels.SquaredExponential(),
    gpflow.kernels.Linear(),
    gpflow.kernels.SquaredExponential() + gpflow.kernels.Linear(),
]


@pytest.mark.parametrize('N, D', [[10, 1]])
def test_add_symmetric(N, D):
    X = rng.randn(N, D)
    Kffs = [kernel(X) for kernel in _kernel_setups_add]

    assert np.allclose(Kffs[0] + Kffs[1], Kffs[2])


@pytest.mark.parametrize('N, M, D', [[10, 12, 1]])
def test_add_asymmetric(N, M, D):
    X, Z = rng.randn(N, D), rng.randn(M, D)
    Kfus = [kernel(X, Z) for kernel in _kernel_setups_add]

    assert np.allclose(Kfus[0] + Kfus[1], Kfus[2])


@pytest.mark.parametrize('N, D', [[10, 1]])
def test_white(N, D):
    """
    The white kernel should not give the same result when called with k(X) and
    k(X, X)
    """
    X = rng.randn(N, D)
    kernel = gpflow.kernels.White()
    Kff_sym = kernel(X)
    Kff_asym = kernel(X, X)

    assert not np.allclose(Kff_sym, Kff_asym)


_kernel_classes_slice = [kernel for kernel in gpflow.kernels.Stationary.__subclasses__()] + \
                        [gpflow.kernels.Constant,
                         gpflow.kernels.Linear,
                         gpflow.kernels.Polynomial]

_kernel_triples_slice = [
    (k1(active_dims=[0]), k2(active_dims=[1]), k3(active_dims=slice(0, 1)))
    for k1, k2, k3 in zip(_kernel_classes_slice, _kernel_classes_slice,
                          _kernel_classes_slice)
]


@pytest.mark.parametrize('kernel_triple', _kernel_triples_slice)
@pytest.mark.parametrize('N, D', [[20, 2]])
def test_slice_symmetric(kernel_triple, N, D):
    X = rng.randn(N, D)
    K1, K3 = kernel_triple[0](X), kernel_triple[2](X[:, :1])
    assert np.allclose(K1, K3)
    K2, K4 = kernel_triple[1](X), kernel_triple[2](X[:, 1:])
    assert np.allclose(K2, K4)


@pytest.mark.parametrize('kernel_triple', _kernel_triples_slice)
@pytest.mark.parametrize('N, M, D', [[10, 12, 2]])
def test_slice_asymmetric(kernel_triple, N, M, D):
    X = rng.randn(N, D)
    Z = rng.randn(M, D)
    K1, K3 = kernel_triple[0](X, Z), kernel_triple[2](X[:, :1], Z[:, :1])
    assert np.allclose(K1, K3)
    K2, K4 = kernel_triple[1](X, Z), kernel_triple[2](X[:, 1:], Z[:, 1:])
    assert np.allclose(K2, K4)


_kernel_setups_prod = [
    gpflow.kernels.Matern32(),
    gpflow.kernels.Matern52(lengthscale=0.3),
    gpflow.kernels.Matern32() * gpflow.kernels.Matern52(lengthscale=0.3),
]


@pytest.mark.parametrize('N, D', [[30, 2]])
def test_product(N, D):
    X = rng.randn(N, D)
    Kffs = [kernel(X) for kernel in _kernel_setups_prod]

    assert np.allclose(Kffs[0] * Kffs[1], Kffs[2])


@pytest.mark.parametrize('N, D', [[30, 4], [10, 7]])
def test_active_product(N, D):
    X = rng.randn(N, D)
    dims, rand_idx, ls = list(range(D)), int(rng.randint(0, D)), rng.uniform(
        1., 7., D)
    active_dims_list = [
        dims[:rand_idx] + dims[rand_idx + 1:], [rand_idx], dims
    ]
    lengthscale_list = [
        np.hstack([ls[:rand_idx], ls[rand_idx + 1:]]), ls[rand_idx], ls
    ]
    kernels = [
        gpflow.kernels.SquaredExponential(lengthscale=lengthscale, active_dims=dims)
        for dims, lengthscale in zip(active_dims_list, lengthscale_list)
    ]
    kernel_prod = kernels[0] * kernels[1]

    Kff = kernels[2](X)
    Kff_prod = kernel_prod(X)

    assert np.allclose(Kff, Kff_prod)


@pytest.mark.parametrize('D', [4, 7])
def test_ard_init_scalar(D):
    """
    For ard kernels, make sure that kernels can be instantiated with a single
    lengthscale or a suitable array of lengthscale
    """
    kernel_1 = gpflow.kernels.SquaredExponential(lengthscale=2.3)
    kernel_2 = gpflow.kernels.SquaredExponential(lengthscale=np.ones(D) * 2.3)
    lengthscale_1 = kernel_1.lengthscale.read_value()
    lengthscale_2 = kernel_2.lengthscale.read_value()
    assert np.allclose(lengthscale_1, lengthscale_2, atol=1e-10)


def test_ard_invalid_active_dims():
    msg = r"Size of `active_dims` \[1\] does not match size of ard parameter \(2\)"
    with pytest.raises(ValueError, match=msg):
        gpflow.kernels.SquaredExponential(lengthscale=np.ones(2), active_dims=[1])


@pytest.mark.parametrize('kernel_class, param_name', [
    [gpflow.kernels.SquaredExponential, "lengthscale"],
    [gpflow.kernels.Linear, "variance"],
    [gpflow.kernels.ArcCosine, "weight_variances"],
])
@pytest.mark.parametrize('param_value, ard', [
    [1., False],
    [[1.], True],
    [[1., 1.], True],
])
def test_ard_property(kernel_class, param_name, param_value, ard):
    kernel = kernel_class(**{param_name: param_value})
    assert kernel.ard is ard


@pytest.mark.parametrize('locations, steepness, error_msg', [
    # 1. Kernels locations dimension mismatch
    [[1.], 1.,
     r"Number of kernels \(3\) must be one more than the number of changepoint locations \(1\)"],

     # 2. Locations steepness dimension mismatch
    [[1., 2.], [1.],
     r"Dimension of steepness \(1\) does not match number of changepoint locations \(2\)"],
])
def test_changepoints_init_fail(locations, steepness, error_msg):
    kernels = [
        gpflow.kernels.Matern12(),
        gpflow.kernels.Linear(),
        gpflow.kernels.Matern32(),
    ]
    with pytest.raises(ValueError, match=error_msg):
        gpflow.kernels.ChangePoints(kernels, locations, steepness)


def _assert_changepoints_kern_err(X, kernels, locations, steepness):
    kernel = gpflow.kernels.ChangePoints(kernels, locations, steepness=steepness)
    reference_gram_matrix = _ref_changepoints(X, kernels, locations, steepness)

    assert_allclose(kernel(X), reference_gram_matrix)
    assert_allclose(kernel.K_diag(X), np.diag(reference_gram_matrix))


@pytest.mark.parametrize('N', [2, 10])
@pytest.mark.parametrize('kernels, locations, steepness', [
    # 1. Single changepoint
    [[gpflow.kernels.Constant(),
      gpflow.kernels.Constant()], [2.], 5.],
    # 2. Two changepoints
    [[gpflow.kernels.Constant(),
      gpflow.kernels.Constant(),
      gpflow.kernels.Constant()], [1., 2.], 5.],
    # 3. Multiple steepness
    [[gpflow.kernels.Constant(),
      gpflow.kernels.Constant(),
      gpflow.kernels.Constant()], [1., 2.], [5., 10.]],
    # 4. Variety of kernels
    [[gpflow.kernels.Matern12(),
      gpflow.kernels.Linear(),
      gpflow.kernels.SquaredExponential(),
      gpflow.kernels.Constant()], [1., 2., 3.], 5.],
])
def test_changepoints(N, kernels, locations, steepness):
    X_data = rng.randn(N, 1)
    _assert_changepoints_kern_err(X_data, kernels, locations, steepness)


@pytest.mark.parametrize('active_dims_1, active_dims_2, is_separate', [
    [[1, 2, 3], None, False],
    [None, [1, 2, 3], False],
    [None, None, False],
    [[1, 2, 3], [3, 4, 5], False],
    [[1, 2, 3], [4, 5, 6], True],
    ])
def test_on_separate_dims(active_dims_1, active_dims_2, is_separate):
    kernel_1 = gpflow.kernels.Linear(active_dims=active_dims_1)
    kernel_2 = gpflow.kernels.SquaredExponential(active_dims=active_dims_2)
    assert kernel_1.on_separate_dims(kernel_2) == is_separate
    assert kernel_2.on_separate_dims(kernel_1) == is_separate
    assert kernel_1.on_separate_dims(kernel_1) is False
    assert kernel_2.on_separate_dims(kernel_2) is False


@pytest.mark.parametrize('kernel', kernel_setups_extended)
def test_kernel_call_diag_and_X2_errors(kernel):
    X = rng.randn(4, 1)
    X2 = rng.randn(5, 1)

    with pytest.raises(ValueError):
        kernel(X, X2, full=False)
