# Copyright 2022 The GPflow Contributors. All Rights Reserved.
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

# pylint: disable=unused-argument  # Bunch of fake functions below has unused arguments.
# pylint: disable=no-member  # PyLint struggles with TensorFlow.

from typing import Callable, Tuple

import numpy as np
import pytest
import tensorflow as tf

from gpflow.experimental.check_shapes import check_shapes


def test_check_shapes__numpy() -> None:
    @check_shapes(
        "a: [d1, d2]",
        "b: [d1, d3]",
        "return: [d2, d3]",
    )
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.zeros((3, 4))

    f(np.zeros((2, 3)), np.zeros((2, 4)))  # Don't crash...


def test_check_shapes__tensorflow() -> None:
    @check_shapes(
        "a: [d1, d2]",
        "b: [d1, d3]",
        "return: [d2, d3]",
    )
    def f(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.zeros((3, 4))

    f(tf.zeros((2, 3)), tf.zeros((2, 4)))  # Don't crash...


_F = Callable[[tf.Tensor], tf.Tensor]
_Loss = Callable[[], tf.Tensor]


@pytest.mark.parametrize(
    "f_wrapper,loss_wrapper",
    [
        (
            lambda x: x,
            lambda x: x,
        ),
        (
            tf.function,
            tf.function,
        ),
        (
            tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float64)]),
            tf.function(input_signature=[]),
        ),
        (
            tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)]),
            tf.function(input_signature=[]),
        ),
        (
            tf.function(experimental_relax_shapes=True),
            tf.function(experimental_relax_shapes=True),
        ),
    ],
)
def test_check_shapes__tensorflow_compilation(
    f_wrapper: Callable[[_F], _F], loss_wrapper: Callable[[_Loss], _Loss]
) -> None:
    target = 0.5

    @f_wrapper
    @check_shapes(
        "x: [n]",
        "return: [n]",
    )
    def f(x: tf.Tensor) -> Tuple[tf.Tensor]:
        return (x - target) ** 2

    v = tf.Variable(np.linspace(0.0, 1.0))

    @loss_wrapper
    @check_shapes(
        "return: [1]",
    )
    def loss() -> tf.Tensor:
        # keepdims is just to add an extra dimension to make the check more interesting.
        return tf.reduce_sum(f(v), keepdims=True)

    optimiser = tf.keras.optimizers.SGD(learning_rate=0.25)
    for _ in range(10):
        optimiser.minimize(loss, var_list=[v])

    np.testing.assert_allclose(target, v.numpy(), atol=0.01)
