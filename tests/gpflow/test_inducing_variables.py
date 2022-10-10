# Copyright 2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Callable, Optional, Tuple

import numpy as np
import pytest
import tensorflow as tf

import gpflow
import gpflow.inducing_variables as giv
from gpflow.experimental.check_shapes import get_shape
from tests.gpflow.experimental.check_shapes.utils import TestContext


def test_inducing_points_with_variable_shape() -> None:
    N, M1, D, P = 50, 13, 3, 1
    X, Y = np.random.randn(N, D), np.random.randn(N, P)

    Z1 = np.random.randn(M1, D)

    # use explicit tf.Variable with None shape:
    iv = giv.InducingPoints(
        tf.Variable(Z1, trainable=False, dtype=gpflow.default_float(), shape=(None, D))
    )
    # Note that we cannot have Z be trainable if we want to be able to change its shape;
    # TensorFlow optimizers expect shape to be known at construction time.

    m = gpflow.models.SGPR(data=(X, Y), kernel=gpflow.kernels.Matern32(), inducing_variable=iv)

    # Check 1: that we can still optimize with None shape
    opt = tf.optimizers.Adam()

    @tf.function
    def optimization_step() -> None:
        opt.minimize(m.training_loss, m.trainable_variables)

    optimization_step()

    # Check 2: that we can successfully assign a new Z with different number of inducing points!
    Z2 = np.random.randn(M1 + 1, D)
    m.inducing_variable.Z.assign(Z2)

    # Check 3: that we can also optimize with changed Z tensor
    optimization_step()


@pytest.mark.parametrize(
    "iv_factory,expected_shape",
    [
        (lambda t: giv.InducingPoints(t), (7, 3, 1)),
        (lambda t: giv.Multiscale(t, t), (7, 3, 1)),
        (lambda t: giv.InducingPatches(t), (7, 3, 1)),
        (
            lambda t: giv.FallbackSharedIndependentInducingVariables(giv.InducingPoints(t)),
            (7, 3, None),
        ),
        (
            lambda t: giv.FallbackSeparateIndependentInducingVariables(
                [
                    giv.InducingPoints(t),
                    giv.InducingPoints(t),
                ]
            ),
            (7, 3, 2),
        ),
        (
            lambda t: giv.SharedIndependentInducingVariables(giv.InducingPoints(t)),
            (7, 3, None),
        ),
        (
            lambda t: giv.SeparateIndependentInducingVariables(
                [
                    giv.InducingPoints(t),
                    giv.InducingPoints(t),
                ]
            ),
            (7, 3, 2),
        ),
    ],
)
@pytest.mark.parametrize("none_shape", [False, True])
def test_shape(
    iv_factory: Callable[[tf.Tensor], giv.InducingVariables],
    none_shape: bool,
    expected_shape: Tuple[Optional[int], ...],
) -> None:
    M = expected_shape[0]
    ones = tf.ones((7, 3))
    kwargs = {"shape": tf.TensorShape(None)} if none_shape else {}
    iv = iv_factory(tf.Variable(ones, **kwargs))
    if none_shape:
        assert get_shape(iv, TestContext()) is None
    else:
        assert expected_shape == get_shape(iv, TestContext())
    assert M == iv.num_inducing
    assert M == len(iv)


@pytest.mark.parametrize("none_shape", [False, True])
def test_shape__inconsistent(none_shape: bool) -> None:
    kwargs = {"shape": tf.TensorShape(None)} if none_shape else {}
    iv = giv.SeparateIndependentInducingVariables(
        [
            giv.InducingPoints(tf.Variable(tf.ones((5, 3)), **kwargs)),
            giv.InducingPoints(tf.Variable(tf.ones((7, 3)), **kwargs)),
        ]
    )
    assert get_shape(iv, TestContext()) is None
    with pytest.raises(tf.errors.InvalidArgumentError):
        iv.num_inducing
    with pytest.raises(tf.errors.InvalidArgumentError):
        len(iv)
