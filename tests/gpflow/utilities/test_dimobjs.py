# Copyright 2022 the GPflow authors.
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

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest
import tensorflow as tf

from gpflow.base import TensorData
from gpflow.utilities.dimobjs import (
    Dim,
    Dimensions,
    Dims,
    Op,
    VariableDimensions,
    dims,
    dimses,
    einsum,
    get_safe_mode,
    none,
    one,
    rearrange,
    reduce_sum,
    set_safe_mode,
    tile,
)

h, w = dims("h,w")
b, b2 = dimses("b,b2")
all_dims = [h, w, b, b2]


@pytest.fixture(autouse=True)
def safe_mode() -> Iterator[None]:
    old_safe_mode = get_safe_mode()
    set_safe_mode(True)
    yield
    set_safe_mode(old_safe_mode)


@pytest.fixture(autouse=True)
def reset_sizes() -> None:
    for dim in all_dims:
        if isinstance(dim, Dim):
            dim.size = None
        else:
            assert isinstance(dim, Dims)
            dim.sizes = None


def set_sizes(shapes: Mapping[VariableDimensions, Any]) -> None:
    for dim, size in shapes.items():
        size = tf.convert_to_tensor(size, dtype=tf.int32)
        if isinstance(dim, Dim):
            dim.size = size
        else:
            assert isinstance(dim, Dims)
            dim.sizes = size


def test_compare_dimensions() -> None:
    dimensions: Sequence[Tuple[Dimensions, Dimensions]] = [
        (h, h),
        (w, w),
        (b, b),
        (b2, b2),
        (none, none),
        (one, one),
        (h * w, h * w),
        (w * h, w * h),
        (h + one, h + one),
        (one + h, one + h),
        (h + w, h + w),
        (w + h, w + h),
    ]

    for d1, d2 in dimensions:
        assert d1 == d2
        assert hash(d1) == hash(d2)

    for i, (di, _di) in enumerate(dimensions):
        for (dj, _dj) in dimensions[i + 1 :]:
            assert di != dj
            assert hash(di) != hash(dj)


@dataclass
class BindTest:
    name: str
    x: TensorData
    frm: Dimensions
    to: Dimensions
    expected: TensorData
    extra_shapes: Mapping[VariableDimensions, Any] = field(default_factory=dict)
    op: Op = rearrange

    @property
    def __name__(self) -> str:
        return self.name


@pytest.mark.parametrize(
    "t",
    [
        BindTest(
            "add_dims",
            [[1, 2, 3], [4, 5, 6]],
            h + w,
            one + h + one + w + one,
            [[[[[1], [2], [3]]], [[[4], [5], [6]]]]],
        ),
        BindTest("add_product", [[1, 2, 3], [4, 5, 6]], h + w, h * w, [1, 2, 3, 4, 5, 6]),
        BindTest(
            "remove_product",
            [1, 2, 3, 4, 5, 6],
            h * w,
            h + w,
            [[1, 2, 3], [4, 5, 6]],
            extra_shapes={w: 3, h: 2},
        ),
        BindTest(
            "remove_dims",
            [[[[[1], [2]]], [[[3], [4]]], [[[5], [6]]]]],
            one + h + one + w + one,
            h + w,
            [[1, 2], [3, 4], [5, 6]],
        ),
        BindTest("transpose", [[1, 2, 3], [4, 5, 6]], w + h, h + w, [[1, 4], [2, 5], [3, 6]]),
        BindTest("transpose_0_batch", tf.ones((3, 4)), b + w + h, h + w + b, tf.ones((4, 3))),
        BindTest("transpose_1_batch", tf.ones((1, 3, 4)), b + w + h, h + w + b, tf.ones((4, 3, 1))),
        BindTest(
            "transpose_2_batch",
            tf.ones((1, 2, 3, 4)),
            b + w + h,
            h + w + b,
            tf.ones((4, 3, 1, 2)),
        ),
        BindTest(
            "multiple_batches_2",
            tf.ones((1, 2, 3, 4)),
            b + b2 + w + h,
            h + b2 + w + b,
            tf.ones((4, 3, 1, 2)),
            extra_shapes={b: (1, 2)},
        ),
        BindTest(
            "multiple_batches_1",
            tf.ones((1, 2, 3, 4)),
            b + b2 + w + h,
            h + b2 + w + b,
            tf.ones((4, 2, 3, 1)),
            extra_shapes={b: (1,)},
        ),
        BindTest(
            "multiple_batches_0",
            tf.ones((1, 2, 3, 4)),
            b + b2 + w + h,
            h + b2 + w + b,
            tf.ones((4, 1, 2, 3)),
            extra_shapes={b: ()},
        ),
        BindTest(
            "add_batch_product",
            tf.ones((1, 2, 3, 4)),
            h + b,
            h + b.prod(),
            tf.ones((1, 24)),
        ),
        BindTest(
            "remove_batch_product",
            tf.ones((1, 24)),
            h + b.prod(),
            h + b,
            tf.ones((1, 2, 3, 4)),
            extra_shapes={b: (2, 3, 4)},
        ),
        BindTest(
            "reduce_1",
            [[1, 2, 3], [4, 5, 6]],
            h + w,
            w,
            [5, 7, 9],
            op=reduce_sum,
        ),
        BindTest(
            "reduce_2",
            [[1, 2, 3], [4, 5, 6]],
            h + w,
            none,
            21,
            op=reduce_sum,
        ),
        BindTest(
            "reduce_batch_0",
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            w + b,
            w,
            [10, 26],
            op=reduce_sum,
        ),
        BindTest(
            "reduce_batch_1",
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            b + w,
            w,
            [16, 20],
            op=reduce_sum,
        ),
        BindTest(
            "expand_1",
            [1, 2, 3],
            w,
            h + w,
            [[1, 2, 3], [1, 2, 3]],
            extra_shapes={h: 2},
            op=tile,
        ),
        BindTest(
            "expand_2",
            1,
            none,
            h + w,
            [[1, 1, 1], [1, 1, 1]],
            extra_shapes={h: 2, w: 3},
            op=tile,
        ),
        BindTest(
            "expand_batch_0",
            [1, 2],
            w,
            w + b,
            [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
            extra_shapes={b: [2, 2]},
            op=tile,
        ),
        BindTest(
            "expand_batch_1",
            [1, 2],
            w,
            b + w,
            [[[1, 2], [1, 2]], [[1, 2], [1, 2]]],
            extra_shapes={b: [2, 2]},
            op=tile,
        ),
        # BindTest(
        #     "big_bang",
        #     [[1, 2, 3], [4, 5, 6]],
        #     h + b,
        #     b + w,
        #     [
        #         [5, 5, 5, 5],
        #         [7, 7, 7, 7],
        #         [9, 9, 9, 9],
        #     ],
        #     extra_shapes={w: 4},
        # ),
    ],
)
def test_bind(t: BindTest) -> None:
    set_sizes(t.extra_shapes)
    v = tf.Variable(t.x, shape=tf.TensorShape(None))

    # @tf.function
    def run() -> tf.Tensor:
        return t.frm >> v >> t.op >> t.to

    y = run()
    np.testing.assert_allclose(t.expected, y)


def test_einsum() -> None:
    # TODO(jesper): Be more thorough here...
    y = (
        (
            b + h + w >> tf.ones((1, 2, 3, 4)),
            w + h >> tf.ones((4, 3)),
        )
        >> einsum
        >> b
    )
    np.testing.assert_allclose([[12.0, 12.0]], y)


def test_stream() -> None:
    t = tf.zeros((1, 2, 3, 4))

    # Implicit rearrange:
    r11 = b + h + w >> t >> b.prod() + w + h
    r12 = (b + h + w >> t) >> b.prod() + w + h
    # r13 = b + h + w >> (t >> b.prod() + w + h)

    # Explicit operator:
    r21 = b + h + w >> t >> rearrange >> b.prod() + w + h
    r22 = (b + h + w >> t) >> rearrange >> b.prod() + w + h
    # r23 = b + h + w >> (t >> rearrange) >> b.prod() + w + h
    r24 = b + h + w >> t >> (rearrange >> b.prod() + w + h)
    r25 = (b + h + w >> t) >> (rearrange >> b.prod() + w + h)

    # Tuple inputs and outputs:
    r31 = (b + h + w >> t,) >> rearrange >> b.prod() + w + h
    (r32,) = b + h + w >> t >> rearrange >> (b.prod() + w + h,)
    (r33,) = (b + h + w >> t,) >> rearrange >> (b.prod() + w + h,)

    np.testing.assert_allclose(r11, r12)
    # np.testing.assert_allclose(r11, r13)

    np.testing.assert_allclose(r11, r21)
    np.testing.assert_allclose(r11, r22)
    # np.testing.assert_allclose(r11, r23)
    np.testing.assert_allclose(r11, r24)
    np.testing.assert_allclose(r11, r25)

    np.testing.assert_allclose(r11, r31)
    np.testing.assert_allclose(r11, r32)
    np.testing.assert_allclose(r11, r33)


def test_bind__short() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        w + h >> tf.zeros((7,))


def test_bind__long() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        w + h >> tf.zeros((1, 2, 3))


def test_bind__size_mismatch() -> None:
    w + h >> tf.zeros((2, 3))
    with pytest.raises(tf.errors.InvalidArgumentError):
        h + w >> tf.zeros((2, 4))


def test_bind__short_batch() -> None:
    b >> tf.zeros((1, 2, 3))
    with pytest.raises(tf.errors.InvalidArgumentError):
        b >> tf.zeros((2, 3))


def test_bind__long_batch() -> None:
    b >> tf.zeros((2, 3))
    with pytest.raises(tf.errors.InvalidArgumentError):
        b >> tf.zeros((1, 2, 3))


def test_bind__size_mismatch_batch() -> None:
    b >> tf.zeros((2, 3))
    with pytest.raises(tf.errors.InvalidArgumentError):
        b >> tf.zeros((2, 4))


def test_bind__repeated_variable() -> None:
    with pytest.raises(AssertionError):
        (h * w) + one + h >> tf.zeros((2, 1, 2))
