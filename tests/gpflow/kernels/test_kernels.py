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

from typing import Any, Optional, Sequence, Tuple, Type, Union, cast

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow.ci_utils
from gpflow.base import AnyNDArray, TensorType
from gpflow.config import default_float
from gpflow.experimental.check_shapes import check_shapes
from gpflow.kernels import (
    RBF,
    AnisotropicStationary,
    ArcCosine,
    ChangePoints,
    Constant,
    Convolutional,
    Coregion,
    Cosine,
    IsotropicStationary,
    Kernel,
    Linear,
    LinearCoregionalization,
    Matern12,
    Matern32,
    Matern52,
    MultioutputKernel,
    Periodic,
    Polynomial,
    RationalQuadratic,
    SeparateIndependent,
    SharedIndependent,
    SquaredExponential,
    Stationary,
    White,
)
from tests.gpflow.kernels.reference import ref_arccosine_kernel, ref_periodic_kernel, ref_rbf_kernel

rng = np.random.RandomState(1)


@check_shapes(
    "X: [N, D]",
    "locations: [L]",
    "steepness: [broadcast L]",
)
def _ref_changepoints(
    X: AnyNDArray,
    kernels: Sequence[Kernel],
    locations: Sequence[float],
    steepness: Union[float, Sequence[float]],
) -> AnyNDArray:
    """
    Calculates K(X) for each kernel in `kernels`, then multiply by sigmoid functions
    in order to smoothly transition betwen them. The sigmoid transitions are defined
    by a location and a steepness parameter.
    """
    locations = sorted(locations)
    steepness = steepness if isinstance(steepness, Sequence) else [steepness] * len(locations)
    np_locations: AnyNDArray = np.array(locations).reshape((1, 1, -1))
    np_steepness: AnyNDArray = np.array(steepness).reshape((1, 1, -1))

    sig_X = 1.0 / (1.0 + np.exp(-np_steepness * (X[:, :, None] - np_locations)))

    starters = sig_X * np.transpose(sig_X, axes=(1, 0, 2))
    stoppers = (1 - sig_X) * np.transpose((1 - sig_X), axes=(1, 0, 2))

    ones = np.ones((X.shape[0], X.shape[0], 1))
    starters = np.concatenate([ones, starters], axis=2)
    stoppers = np.concatenate([stoppers, ones], axis=2)

    kernel_stack: AnyNDArray = np.stack([k(X) for k in kernels], axis=2)

    return cast(AnyNDArray, (kernel_stack * starters * stoppers).sum(axis=2))


@pytest.mark.parametrize("variance, lengthscales", [[2.3, 1.4]])
def test_rbf_1d(variance: TensorType, lengthscales: TensorType) -> None:
    X = rng.randn(3, 1)
    kernel = SquaredExponential(lengthscales=lengthscales, variance=variance)

    gram_matrix = kernel(X)
    reference_gram_matrix = ref_rbf_kernel(X, lengthscales, variance)

    assert_allclose(gram_matrix, reference_gram_matrix)


@pytest.mark.parametrize("variance, lengthscales", [[2.3, 1.4]])
def test_rq_1d(variance: TensorType, lengthscales: TensorType) -> None:
    kSE = SquaredExponential(lengthscales=lengthscales, variance=variance)
    kRQ = RationalQuadratic(lengthscales=lengthscales, variance=variance, alpha=1e8)
    rng = np.random.RandomState(1)
    X: AnyNDArray = rng.randn(6, 1).astype(default_float())

    gram_matrix_SE = kSE(X)
    gram_matrix_RQ = kRQ(X)
    assert_allclose(gram_matrix_SE, gram_matrix_RQ)


@check_shapes(
    "variance: []",
    "weight_variances: [broadcast n_active_dims]",
    "bias_variance: []",
    "X: [N, D]",
)
def _assert_arccosine_kern_err(
    variance: TensorType,
    weight_variances: TensorType,
    bias_variance: TensorType,
    order: int,
    X: TensorType,
) -> None:
    kernel = ArcCosine(
        order=order,
        variance=variance,
        weight_variances=weight_variances,
        bias_variance=bias_variance,
    )
    gram_matrix = kernel(X)
    reference_gram_matrix = ref_arccosine_kernel(
        X, order, weight_variances, bias_variance, variance
    )
    assert_allclose(gram_matrix, reference_gram_matrix)


@pytest.mark.parametrize("order", ArcCosine.implemented_orders)
@pytest.mark.parametrize("D, weight_variances", [[1, 1.7], [3, 1.7], [3, (1.1, 1.7, 1.9)]])
@pytest.mark.parametrize("N, bias_variance, variance", [[3, 0.6, 2.3]])
def test_arccosine_1d_and_3d(
    order: int,
    D: int,
    N: int,
    weight_variances: Union[float, Sequence[float]],
    bias_variance: float,
    variance: float,
) -> None:
    X_data = rng.randn(N, D)
    _assert_arccosine_kern_err(variance, weight_variances, bias_variance, order, X_data)


@pytest.mark.parametrize("order", [42])
def test_arccosine_non_implemented_order(order: int) -> None:
    with pytest.raises(ValueError):
        ArcCosine(order=order)


@pytest.mark.parametrize("D, N", [[1, 4]])
def test_arccosine_nan_gradient(D: int, N: int) -> None:
    X = rng.rand(N, D)
    kernel = ArcCosine()
    with tf.GradientTape() as tape:
        Kff = kernel(X)
    grads = tape.gradient(Kff, kernel.trainable_variables)
    assert not np.any(np.isnan(grads))


@pytest.mark.parametrize(
    "base_class",
    [
        SquaredExponential,
        Matern12,
        Matern32,
        Matern52,
    ],
)
@pytest.mark.parametrize(
    "D, lengthscales, period",
    [
        [1, 2.0, 3.0],  # 1d, single lengthscale, single period
        [2, 11.5, 3.0],  # 2d, single lengthscale, single period
        [2, 11.5, (3.0, 6.0)],  # 2d, single lengthscale, ard period
        [2, (11.5, 12.5), 3.0],  # 2d, ard lengthscales, single period
        [2, (11.5, 12.5), (3.0, 6.0)],  # 2d, ard lengthscales, ard period
    ],
)
@pytest.mark.parametrize(
    "N, variance",
    [
        [3, 2.3],
        [5, 1.3],
    ],
)
def test_periodic(
    base_class: Type[Stationary],
    D: int,
    N: int,
    lengthscales: TensorType,
    variance: TensorType,
    period: TensorType,
) -> None:
    X = rng.randn(N, D) if D == 1 else rng.multivariate_normal(np.zeros(D), np.eye(D), N)

    base_kernel = base_class(lengthscales=lengthscales, variance=variance)
    kernel = Periodic(base_kernel, period=period)
    gram_matrix = kernel(X)
    reference_gram_matrix = ref_periodic_kernel(
        X, base_class.__name__, lengthscales, variance, period
    )

    assert_allclose(gram_matrix, reference_gram_matrix)


@pytest.mark.parametrize(
    "base_class",
    [
        SquaredExponential,
        Matern12,
    ],
)
def test_periodic_diag(base_class: Type[Stationary]) -> None:
    N, D = 5, 3
    X = rng.multivariate_normal(np.zeros(D), np.eye(D), N)
    base_kernel = base_class(lengthscales=2.0, variance=1.0)
    kernel = Periodic(base_kernel, period=6.0)
    assert_allclose(base_kernel(X, full_cov=False), kernel(X, full_cov=False))


def test_periodic_non_stationary_base_kernel() -> None:
    error_msg = r"Periodic requires an IsotropicStationary kernel as the `base_kernel`"
    with pytest.raises(TypeError, match=error_msg):
        Periodic(Linear())


def test_periodic_bad_ard_period() -> None:
    error_msg = r"Size of `active_dims` \[1 2\] does not match size of ard parameter \(3\)"
    base_kernel = RBF(active_dims=[1, 2])
    with pytest.raises(ValueError, match=error_msg):
        Periodic(base_kernel, period=[1.0, 1.0, 1.0])


kernel_setups: Tuple[Kernel, ...] = tuple(
    kernel()
    for kernel in gpflow.ci_utils.subclasses(Stationary)
    if kernel not in (IsotropicStationary, AnisotropicStationary)
) + (
    Constant(),
    Linear(),
    Polynomial(),
    ArcCosine(),
)


@pytest.mark.parametrize("D", [1, 5])
@pytest.mark.parametrize("kernel", kernel_setups)
@pytest.mark.parametrize("N", [10])
def test_kernel_symmetry_1d_and_5d(D: int, kernel: Kernel, N: int) -> None:
    X = rng.randn(N, D)
    errors = kernel(X) - kernel(X, X)
    assert np.allclose(errors, 0)


@pytest.mark.parametrize("N, N2, input_dim, output_dim, rank", [[10, 12, 1, 3, 2]])
def test_coregion_shape(N: int, N2: int, input_dim: int, output_dim: int, rank: int) -> None:
    X = np.random.randint(0, output_dim, (N, input_dim))
    X2 = np.random.randint(0, output_dim, (N2, input_dim))
    kernel = Coregion(output_dim=output_dim, rank=rank)
    kernel.W.assign(rng.randn(output_dim, rank))
    kernel.kappa.assign(np.exp(rng.randn(output_dim, 1).reshape(-1)))

    Kff2 = kernel(X, X2)
    assert Kff2.shape == (10, 12)
    Kff = kernel(X)
    assert Kff.shape == (10, 10)


@pytest.mark.parametrize("N, input_dim, output_dim, rank", [[10, 1, 3, 2]])
def test_coregion_diag(N: int, input_dim: int, output_dim: int, rank: int) -> None:
    X = np.random.randint(0, output_dim, (N, input_dim))
    kernel = Coregion(output_dim=output_dim, rank=rank)
    kernel.W.assign(rng.randn(output_dim, rank))
    kernel.kappa.assign(np.exp(rng.randn(output_dim, 1).reshape(-1)))

    K = kernel(X)
    Kdiag = kernel.K_diag(X)
    assert np.allclose(np.diag(K), Kdiag)


@pytest.mark.parametrize("N, input_dim, output_dim, rank", [[10, 1, 3, 2]])
def test_coregion_slice(N: int, input_dim: int, output_dim: int, rank: int) -> None:
    X = np.random.randint(0, output_dim, (N, input_dim))
    X = np.hstack((X, rng.randn(10, 1)))
    kernel1 = Coregion(output_dim=output_dim, rank=rank, active_dims=[0])
    # compute another kernel with additinoal inputs,
    # make sure out kernel is still okay.
    kernel2 = SquaredExponential(active_dims=[1])
    kernel_prod = kernel1 * kernel2
    K1 = kernel_prod(X)
    K2 = kernel1(X) * kernel2(X)  # slicing happens inside kernel
    assert np.allclose(K1, K2)


_dim = 3
kernel_setups_extended: Tuple[Kernel, ...] = (
    kernel_setups
    + (
        SquaredExponential() + Linear(),
        SquaredExponential() * Linear(),
        SquaredExponential() + Linear(variance=rng.rand(_dim)),
    )
    + tuple(ArcCosine(order=order) for order in ArcCosine.implemented_orders)
)


@pytest.mark.parametrize("kernel", kernel_setups_extended)
@pytest.mark.parametrize("N, dim", [[30, _dim]])
def test_diags(kernel: Kernel, N: int, dim: int) -> None:
    X = np.random.randn(N, dim)
    kernel1 = tf.linalg.diag_part(kernel(X, full_cov=True))
    kernel2 = kernel(X, full_cov=False)
    assert np.allclose(kernel1, kernel2)


def test_conv_diag() -> None:
    kernel = Convolutional(SquaredExponential(), [3, 3], [2, 2])
    X = np.random.randn(3, 9)
    kernel_full = np.diagonal(kernel(X, full_cov=True))
    kernel_diag = kernel(X, full_cov=False)
    assert np.allclose(kernel_full, kernel_diag)
    assert 4 == kernel.patch_len
    assert 4 == kernel.num_patches


# Add a rbf and linear kernel, make sure the result is the same as adding the result of
# the kernels separately.
_kernel_setups_add: Tuple[Kernel, ...] = (
    SquaredExponential(),
    Linear(),
    SquaredExponential() + Linear(),
)


@pytest.mark.parametrize("N, D", [[10, 1]])
def test_add_symmetric(N: int, D: int) -> None:
    X = rng.randn(N, D)
    Kffs = [kernel(X) for kernel in _kernel_setups_add]

    assert np.allclose(Kffs[0] + Kffs[1], Kffs[2])


@pytest.mark.parametrize("N, M, D", [[10, 12, 1]])
def test_add_asymmetric(N: int, M: int, D: int) -> None:
    X, Z = rng.randn(N, D), rng.randn(M, D)
    Kfus = [kernel(X, Z) for kernel in _kernel_setups_add]

    assert np.allclose(Kfus[0] + Kfus[1], Kfus[2])


@pytest.mark.parametrize("N, D", [[10, 1]])
def test_white(N: int, D: int) -> None:
    """
    The white kernel should not give the same result when called with k(X) and
    k(X, X)
    """
    X = rng.randn(N, D)
    kernel = White()
    Kff_sym = kernel(X)
    Kff_asym = kernel(X, X)

    assert not np.allclose(Kff_sym, Kff_asym)


_kernel_classes_slice = [
    kernel
    for kernel in gpflow.ci_utils.subclasses(Stationary)
    if kernel not in (IsotropicStationary, AnisotropicStationary)
] + [
    Constant,
    Linear,
    Polynomial,
]

_kernel_triples_slice = [
    (k1(active_dims=[0]), k2(active_dims=[1]), k3(active_dims=slice(0, 1)))
    for k1, k2, k3 in zip(_kernel_classes_slice, _kernel_classes_slice, _kernel_classes_slice)
]


@pytest.mark.parametrize("kernel_triple", _kernel_triples_slice)
@pytest.mark.parametrize("N, D", [[20, 2]])
def test_slice_symmetric(kernel_triple: Tuple[Kernel, Kernel, Kernel], N: int, D: int) -> None:
    X = rng.randn(N, D)
    K1, K3 = kernel_triple[0](X), kernel_triple[2](X[:, :1])
    assert np.allclose(K1, K3)
    K2, K4 = kernel_triple[1](X), kernel_triple[2](X[:, 1:])
    assert np.allclose(K2, K4)


@pytest.mark.parametrize("kernel_triple", _kernel_triples_slice)
@pytest.mark.parametrize("N, M, D", [[10, 12, 2]])
def test_slice_asymmetric(
    kernel_triple: Tuple[Kernel, Kernel, Kernel], N: int, M: int, D: int
) -> None:
    X = rng.randn(N, D)
    Z = rng.randn(M, D)
    K1, K3 = kernel_triple[0](X, Z), kernel_triple[2](X[:, :1], Z[:, :1])
    assert np.allclose(K1, K3)
    K2, K4 = kernel_triple[1](X, Z), kernel_triple[2](X[:, 1:], Z[:, 1:])
    assert np.allclose(K2, K4)


_kernel_setups_prod: Tuple[Kernel, ...] = (
    Matern32(),
    Matern52(lengthscales=0.3),
    Matern32() * Matern52(lengthscales=0.3),
)


@pytest.mark.parametrize("N, D", [[30, 2]])
def test_product(N: int, D: int) -> None:
    X = rng.randn(N, D)
    Kffs = [kernel(X) for kernel in _kernel_setups_prod]

    assert np.allclose(Kffs[0] * Kffs[1], Kffs[2])


@pytest.mark.parametrize("N, D", [[30, 4], [10, 7]])
def test_active_product(N: int, D: int) -> None:
    X = rng.randn(N, D)
    dims, rand_idx, ls = (
        list(range(D)),
        int(rng.randint(0, D)),
        rng.uniform(1.0, 7.0, D),
    )
    active_dims_list = [dims[:rand_idx] + dims[rand_idx + 1 :], [rand_idx], dims]
    lengthscales_list = [
        np.hstack([ls[:rand_idx], ls[rand_idx + 1 :]]),
        ls[rand_idx],
        ls,
    ]
    kernels = [
        SquaredExponential(lengthscales=lengthscales, active_dims=dims)
        for dims, lengthscales in zip(active_dims_list, lengthscales_list)
    ]
    kernel_prod = kernels[0] * kernels[1]

    Kff = kernels[2](X)
    Kff_prod = kernel_prod(X)

    assert np.allclose(Kff, Kff_prod)


@pytest.mark.parametrize("D", [4, 7])
def test_ard_init_scalar(D: int) -> None:
    """
    For ard kernels, make sure that kernels can be instantiated with a single
    scalar lengthscale or a suitable array of lengthscales
    """
    kernel_1 = SquaredExponential(lengthscales=2.3)
    kernel_2 = SquaredExponential(lengthscales=np.ones(D) * 2.3)
    lengthscales_1 = kernel_1.lengthscales.numpy()
    lengthscales_2 = kernel_2.lengthscales.numpy()
    assert np.allclose(lengthscales_1, lengthscales_2, atol=1e-10)


def test_ard_invalid_active_dims() -> None:
    msg = r"Size of `active_dims` \[1\] does not match size of ard parameter \(2\)"
    with pytest.raises(ValueError, match=msg):
        SquaredExponential(lengthscales=np.ones(2), active_dims=[1])


@pytest.mark.parametrize(
    "kernel_class, param_name",
    [
        [SquaredExponential, "lengthscales"],
        [Linear, "variance"],
        [ArcCosine, "weight_variances"],
        [Cosine, "lengthscales"],
    ],
)
@pytest.mark.parametrize(
    "param_value, ard",
    [
        [1.0, False],
        [[1.0], True],
        [[1.0, 1.0], True],
    ],
)
def test_ard_property(
    kernel_class: Type[Kernel], param_name: str, param_value: Any, ard: bool
) -> None:
    kernel = kernel_class(**{param_name: param_value})
    assert kernel.ard is ard


@pytest.mark.parametrize(
    "locations, steepness, error_msg",
    [
        # 1. Kernels locations dimension mismatch
        [
            [1.0],
            1.0,
            r"Number of kernels \(3\) must be one more than the number of changepoint locations \(1\)",
        ],
        # 2. Locations steepness dimension mismatch
        [
            [1.0, 2.0],
            [1.0],
            r"Dimension of steepness \(1\) does not match number of changepoint locations \(2\)",
        ],
    ],
)
def test_changepoints_init_fail(
    locations: Sequence[float], steepness: Union[float, Sequence[float]], error_msg: str
) -> None:
    kernels: Sequence[Kernel] = [
        Matern12(),
        Linear(),
        Matern32(),
    ]
    with pytest.raises(ValueError, match=error_msg):
        ChangePoints(kernels, locations, steepness)


@check_shapes(
    "X: [N, D]",
    "locations: [L]",
    "steepness: [broadcast L]",
)
def _assert_changepoints_kern_err(
    X: TensorType,
    kernels: Sequence[Kernel],
    locations: Sequence[float],
    steepness: Union[float, Sequence[float]],
) -> None:
    kernel = ChangePoints(kernels, locations, steepness=steepness)
    reference_gram_matrix = _ref_changepoints(X, kernels, locations, steepness)

    assert_allclose(kernel(X), reference_gram_matrix)
    assert_allclose(kernel.K_diag(X), np.diag(reference_gram_matrix))


@pytest.mark.parametrize("N", [2, 10])
@pytest.mark.parametrize(
    "kernels, locations, steepness",
    [
        # 1. Single changepoint
        [[Constant(), Constant()], [2.0], 5.0],
        # 2. Two changepoints
        [
            [
                Constant(),
                Constant(),
                Constant(),
            ],
            [1.0, 2.0],
            5.0,
        ],
        # 3. Multiple steepness
        [
            [
                Constant(),
                Constant(),
                Constant(),
            ],
            [1.0, 2.0],
            [5.0, 10.0],
        ],
        # 4. Variety of kernels
        [
            [
                Matern12(),
                Linear(),
                SquaredExponential(),
                Constant(),
            ],
            [1.0, 2.0, 3.0],
            5.0,
        ],
    ],
)
def test_changepoints(
    N: int,
    kernels: Sequence[Kernel],
    locations: Sequence[float],
    steepness: Union[float, Sequence[float]],
) -> None:
    X_data = rng.randn(N, 1)
    _assert_changepoints_kern_err(X_data, kernels, locations, steepness)


@pytest.mark.parametrize(
    "active_dims_1, active_dims_2, is_separate",
    [
        [[1, 2, 3], None, False],
        [None, [1, 2, 3], False],
        [None, None, False],
        [[1, 2, 3], [3, 4, 5], False],
        [[1, 2, 3], [4, 5, 6], True],
    ],
)
def test_on_separate_dims(
    active_dims_1: Optional[Sequence[int]],
    active_dims_2: Optional[Sequence[int]],
    is_separate: bool,
) -> None:
    kernel_1 = Linear(active_dims=active_dims_1)
    kernel_2 = SquaredExponential(active_dims=active_dims_2)
    assert kernel_1.on_separate_dims(kernel_2) == is_separate
    assert kernel_2.on_separate_dims(kernel_1) == is_separate
    assert kernel_1.on_separate_dims(kernel_1) is False
    assert kernel_2.on_separate_dims(kernel_2) is False


@pytest.mark.parametrize("kernel", kernel_setups_extended)
def test_kernel_call_diag_and_X2_errors(kernel: Kernel) -> None:
    X = rng.randn(4, 1)
    X2 = rng.randn(5, 1)

    with pytest.raises(ValueError):
        kernel(X, X2, full_cov=False)


def test_periodic_active_dims_matches() -> None:
    active_dims = [1]
    base_kernel = SquaredExponential(active_dims=active_dims)
    kernel = Periodic(base_kernel=base_kernel)

    assert kernel.active_dims == base_kernel.active_dims

    # type-ignores below, is because mypy doesn't understand that the setter and the getter for
    # `active_dims` have different types.

    kernel.active_dims = [2]  # type: ignore[assignment]
    assert kernel.active_dims == base_kernel.active_dims

    base_kernel.active_dims = [3]  # type: ignore[assignment]
    assert kernel.active_dims == base_kernel.active_dims


def test_latent_kernels() -> None:
    kernel_list: Tuple[Kernel, ...] = (SquaredExponential(), White(), White() + Linear())

    multioutput_kernel_list: Tuple[MultioutputKernel, ...] = (
        SharedIndependent(SquaredExponential(), 3),
        SeparateIndependent(kernel_list),
        LinearCoregionalization(kernel_list, np.random.random((5, 3))),
    )
    assert len(multioutput_kernel_list[0].latent_kernels) == 1
    assert multioutput_kernel_list[1].latent_kernels == tuple(kernel_list)
    assert multioutput_kernel_list[2].latent_kernels == tuple(kernel_list)


def test_combination_LMC_kernels() -> None:
    N, D, P = 100, 3, 2
    kernel_list1: Tuple[Kernel, ...] = (Linear(active_dims=[1]), SquaredExponential())
    L1 = len(kernel_list1)
    kernel_list2: Tuple[Kernel, ...] = (SquaredExponential(), Linear(), Linear())
    L2 = len(kernel_list2)
    k1 = LinearCoregionalization(kernel_list1, np.random.randn(P, L1))
    k2 = LinearCoregionalization(kernel_list2, np.random.randn(P, L2))
    kernel = k1 + k2
    X = np.random.randn(N, D)
    K1 = k1(X, full_cov=True)
    K2 = k2(X, full_cov=True)
    K = kernel(X, full_cov=True)
    assert K.shape == [N, P, N, P]
    np.testing.assert_allclose(K, K1 + K2)
