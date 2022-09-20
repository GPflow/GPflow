# Copyright 2020 the GPflow authors.
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

from typing import Any, Callable, List, Sequence, Tuple, cast

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
import gpflow.ci_utils
from gpflow import kernels
from gpflow.base import AnyNDArray, TensorType
from gpflow.experimental.check_shapes import check_shape as cs
from gpflow.experimental.check_shapes import check_shapes


def create_kernels() -> Sequence[kernels.Kernel]:
    result: List[kernels.Kernel] = [
        # Static kernels:
        kernels.White(),
        kernels.Constant(),
        # Stationary kernels:
        kernels.SquaredExponential(),
        kernels.RationalQuadratic(),
        kernels.Exponential(),
        kernels.Matern12(),
        kernels.Matern32(),
        kernels.Matern52(),
        # sum and product kernels:
        kernels.White() + kernels.Matern12(),
        kernels.White() * kernels.Matern12(),
        # slicing:
        kernels.Matern32(active_dims=slice(None, None, 2)),
        kernels.Matern32(active_dims=[1, 2]),
        # other kernels:
        kernels.Cosine(),
        kernels.Linear(),
        kernels.Polynomial(),
        kernels.Periodic(kernels.Matern32()),
        kernels.ChangePoints(
            [kernels.Matern32(), kernels.Matern32()],
            [0.5],
        ),
        kernels.ArcCosine(),
        kernels.Coregion(output_dim=5, rank=2),
        kernels.Convolutional(kernels.Matern32(), [4, 4], [2, 2]),
        # multioutput:
        kernels.SharedIndependent(kernels.Matern32(), output_dim=5),
        kernels.SeparateIndependent([kernels.Matern32() for _ in range(5)]),
        kernels.LinearCoregionalization([kernels.Matern32() for _ in range(3)], np.ones((5, 3))),
    ]
    return result


def make_id(value: Any) -> str:
    if isinstance(value, tuple):
        return f"[{','.join(repr(x) for x in value)}]"
    assert isinstance(value, kernels.Kernel)
    return value.__class__.__name__


def test_no_kernels_missed() -> None:
    tested_kernel_classes = {
        parent
        for kernel in create_kernels()
        for parent in kernel.__class__.__mro__
        if (parent != kernels.Kernel) and issubclass(parent, kernels.Kernel)
    }
    all_kernel_classes = set(gpflow.ci_utils.subclasses(kernels.Kernel))
    assert tested_kernel_classes == all_kernel_classes


@pytest.mark.parametrize("kernel", create_kernels(), ids=make_id)
@pytest.mark.parametrize(
    "batch_shape",
    [
        (3,),
        (2, 3),
        (1, 2, 3),
    ],
    ids=make_id,
)
@pytest.mark.parametrize(
    "batch2_shape",
    [
        (4,),
        (2, 4),
        (1, 2, 4),
    ],
    ids=make_id,
)
@check_shapes()
def test_broadcasting(
    kernel: kernels.Kernel,
    batch_shape: Tuple[int, ...],
    batch2_shape: Tuple[int, ...],
    rng: np.random.Generator,
) -> None:
    if isinstance(kernel, kernels.Coregion):
        D = 1
        X: AnyNDArray = rng.choice(kernel.rank, batch_shape + (D,))
        X2: AnyNDArray = rng.choice(kernel.rank, batch2_shape + (D,))
    else:
        if isinstance(kernel, kernels.ChangePoints):
            D = 1
        elif isinstance(kernel, kernels.Convolutional):
            D = int(np.prod(kernel.image_shape))
        else:
            D = 4

        X = rng.random(batch_shape + (D,))
        X2 = rng.random(batch2_shape + (D,))

    rank = len(batch_shape) - 1
    rank2 = len(batch2_shape) - 1

    if isinstance(kernel, kernels.MultioutputKernel):
        mo_kernel = cast(kernels.MultioutputKernel, kernel)

        loop = cs(
            unroll_batches(
                lambda x: unroll_batches(
                    lambda x2: mo_kernel(x, x2, full_cov=True, full_output_cov=True), X2, 2
                ),
                X,
                2,
            ),
            "[batch..., batch2..., N, P, N2, P]",
        )
        loop = cs(
            tf.transpose(
                loop,
                tf.concat(
                    [
                        np.arange(rank),
                        [rank + rank2, rank + rank2 + 1],
                        np.arange(rank2) + rank,
                        [rank + rank2 + 2, rank + rank2 + 3],
                    ],
                    0,
                ),
            ),
            "[batch..., N, P, batch2..., N2, P]",
        )
        native = cs(
            mo_kernel(X, X2, full_cov=True, full_output_cov=True),
            "[batch..., N, P, batch2..., N2, P]",
        )
        assert_allclose(loop.numpy(), native.numpy())

        loop = cs(
            unroll_batches(
                lambda x: unroll_batches(
                    lambda x2: mo_kernel(x, x2, full_cov=True, full_output_cov=False), X2, 2
                ),
                X,
                2,
            ),
            "[batch..., batch2..., P, N, N2]",
        )
        loop = cs(
            tf.transpose(
                loop,
                tf.concat(
                    [
                        [rank + rank2],
                        np.arange(rank),
                        [rank + rank2 + 1],
                        np.arange(rank2) + rank,
                        [rank + rank2 + 2],
                    ],
                    0,
                ),
            ),
            "[P, batch..., N, batch2..., N2]",
        )
        native = cs(
            mo_kernel(X, X2, full_cov=True, full_output_cov=False),
            "[P, batch..., N, batch2..., N2]",
        )
        assert_allclose(loop.numpy(), native.numpy())

        loop = cs(
            unroll_batches(lambda x: mo_kernel(x, full_cov=True, full_output_cov=True), X, 2),
            "[batch..., N, P, N, P]",
        )
        native = cs(mo_kernel(X, full_cov=True, full_output_cov=True), "[batch..., N, P, N, P]")
        assert_allclose(loop.numpy(), native.numpy())

        loop = cs(
            unroll_batches(lambda x: mo_kernel(x, full_cov=True, full_output_cov=False), X, 2),
            "[batch..., P, N, N]",
        )
        loop = cs(
            tf.transpose(
                loop,
                tf.concat(
                    [
                        [rank],
                        np.arange(rank),
                        [rank + 1, rank + 2],
                    ],
                    0,
                ),
            ),
            "[P, batch..., N, N]",
        )
        native = cs(mo_kernel(X, full_cov=True, full_output_cov=False), "[P, batch..., N, N]")
        assert_allclose(loop.numpy(), native.numpy())

        loop = cs(
            unroll_batches(lambda x: mo_kernel(x, full_cov=False, full_output_cov=True), X, 2),
            "[batch..., N, P, P]",
        )
        native = cs(mo_kernel(X, full_cov=False, full_output_cov=True), "[batch..., N, P, P]")
        assert_allclose(loop.numpy(), native.numpy())

        loop = cs(
            unroll_batches(lambda x: mo_kernel(x, full_cov=False, full_output_cov=False), X, 2),
            "[batch..., N, P]",
        )
        native = cs(mo_kernel(X, full_cov=False, full_output_cov=False), "[batch..., N, P]")
        assert_allclose(loop.numpy(), native.numpy())

    else:  # Single-output kernel:
        loop = cs(
            unroll_batches(
                lambda x: unroll_batches(lambda x2: kernel(x, x2, full_cov=True), X2, 2), X, 2
            ),
            "[batch..., batch2..., N, N2]",
        )
        loop = cs(
            tf.transpose(
                loop,
                tf.concat(
                    [np.arange(rank), [rank + rank2], np.arange(rank2) + rank, [rank + rank2 + 1]],
                    0,
                ),
            ),
            "[batch..., N, batch2..., N2]",
        )
        native = cs(kernel(X, X2, full_cov=True), "[batch..., N, batch2..., N2]")
        assert_allclose(loop.numpy(), native.numpy())

        loop = cs(unroll_batches(lambda x: kernel(x, full_cov=True), X, 2), "[batch..., N, N]")
        native = cs(kernel(X, full_cov=True), "[batch..., N, N]")
        assert_allclose(loop.numpy(), native.numpy())

        loop = cs(unroll_batches(lambda x: kernel(x, full_cov=False), X, 2), "[batch..., N]")
        native = cs(kernel(X, full_cov=False), "[batch..., N]")
        assert_allclose(loop.numpy(), native.numpy())


@check_shapes(
    "x: [batch..., value_shape...]",
    "return: [batch..., result_shape...]",
)
def unroll_batches(
    f: Callable[[TensorType], TensorType], x: TensorType, value_rank: int
) -> tf.Tensor:
    if len(x.shape) == value_rank:
        return f(x)

    return tf.stack([unroll_batches(f, row, value_rank) for row in x])
