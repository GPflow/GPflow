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


import tempfile
from typing import Sequence

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.base import AnyNDArray, PriorOn, TensorData
from gpflow.config import default_float
from gpflow.utilities import positive, triangular


def test_parameter_assign_validation() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        param = gpflow.Parameter(0.0, transform=positive())

    param = gpflow.Parameter(0.1, transform=positive())
    param.assign(0.2)
    with pytest.raises(tf.errors.InvalidArgumentError):
        param.assign(0.0)


def test_cast_to_dtype_precision_issue() -> None:
    """
    TensorFlow's tf.cast(value, dtype) implicitly does a tf.convert_to_tensor(value)
    *before* the cast when the value is not a tensor already. When value is a python float,
    this results in the following behaviour:

    >>> tf.cast(0.2, tf.float64)
    <tf.Tensor: id=37, shape=(), dtype=float64, numpy=0.20000000298023224>

    instead of the expected expansion of 0.2 to float64 precision that you get when
    passing in an object that already carries dtype information, such as a numpy array
    (which has float64 precision by default):

    >>> tf.cast(np.array(0.2), tf.float64)
    <tf.Tensor: id=40, shape=(), dtype=float64, numpy=0.2>

    This affected *all* gpflow.Parameter objects, resulting in numerical discrepancies
    between GPflow 1 and 2, due to the pass through _cast_to_dtype, which is now fixed.
    This is the corresponding regression test.
    """
    p = gpflow.Parameter(0.2, dtype=np.float64)
    actual_value = p.numpy()
    assert actual_value.dtype == np.float64
    expected_value = np.float64(0.2)
    assert actual_value == expected_value


def test_parameter_saved() -> None:
    dtype = tf.float64

    class Model(tf.Module):
        def __init__(self) -> None:
            self.p = gpflow.Parameter(0.1, dtype=dtype, transform=gpflow.utilities.positive())

        @tf.function(input_signature=[tf.TensorSpec([], dtype=dtype)])
        def exec(self, x: tf.Tensor) -> tf.Tensor:
            return tf.square(x * self.p)

    m0 = Model()
    x = tf.convert_to_tensor(2.0, dtype=dtype)
    expected = m0.exec(x)
    with tempfile.TemporaryDirectory() as dirname:
        tf.saved_model.save(m0, dirname)
        m1 = tf.saved_model.load(dirname)
        actual = m1.exec(x)
        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("value", [0.0, [1.2, 1.1]])
def test_construct_parameter_from_existing_parameter_check_value(value: TensorData) -> None:
    initial_parameter = gpflow.Parameter(value)
    new_parameter = gpflow.Parameter(initial_parameter)

    np.testing.assert_equal(new_parameter.numpy(), value)


@pytest.mark.parametrize("value", [0.0, [1.2, 1.1]])
def test_construct_parameter_from_existing_parameter_override_value(value: TensorData) -> None:
    initial_parameter = gpflow.Parameter(value)
    new_parameter = gpflow.Parameter(initial_parameter + 1.0)

    np.testing.assert_equal(new_parameter.numpy(), np.array(value) + 1.0)


def test_construct_parameter_from_existing_parameter_check_transform() -> None:
    transform = tfp.bijectors.Sigmoid(
        tf.constant(0.0, dtype=tf.float64), tf.constant(2.0, dtype=tf.float64)
    )
    initial_parameter = gpflow.Parameter([1.2, 1.1], transform=transform)
    new_parameter = gpflow.Parameter(initial_parameter)

    assert new_parameter.transform == transform


def test_construct_parameter_from_existing_parameter_override_transform() -> None:
    initial_parameter = gpflow.Parameter([1.2, 1.1])

    transform = tfp.bijectors.Sigmoid(
        tf.constant(0.0, dtype=tf.float64), tf.constant(2.0, dtype=tf.float64)
    )
    new_parameter = gpflow.Parameter(initial_parameter, transform=transform)

    assert new_parameter.transform == transform


def test_construct_parameter_from_existing_parameter_check_prior() -> None:
    prior = tfp.distributions.Normal(0.0, 1.0)
    initial_parameter = gpflow.Parameter([1.2, 1.1], prior=prior)
    new_parameter = gpflow.Parameter(initial_parameter)

    assert new_parameter.prior == prior


def test_construct_parameter_from_existing_parameter_override_prior() -> None:
    initial_parameter = gpflow.Parameter([1.2, 1.1])

    prior = tfp.distributions.Normal(0.0, 1.0)
    new_parameter = gpflow.Parameter(initial_parameter, prior=prior)

    assert new_parameter.prior == prior


@pytest.mark.parametrize("prior_on", [PriorOn.CONSTRAINED, PriorOn.UNCONSTRAINED])
def test_construct_parameter_from_existing_parameter_check_prior_on(prior_on: PriorOn) -> None:
    initial_parameter = gpflow.Parameter([1.2, 1.1], prior_on=prior_on)
    new_parameter = gpflow.Parameter(initial_parameter)

    assert new_parameter.prior_on == prior_on


@pytest.mark.parametrize("prior_on", [PriorOn.CONSTRAINED, PriorOn.UNCONSTRAINED])
def test_construct_parameter_from_existing_parameter_override_prior_on(prior_on: PriorOn) -> None:
    initial_parameter = gpflow.Parameter([1.2, 1.1])
    new_parameter = gpflow.Parameter(initial_parameter, prior_on=prior_on)

    assert new_parameter.prior_on == prior_on


@pytest.mark.parametrize("trainable", [True, False])
def test_construct_parameter_from_existing_parameter_check_trainable(trainable: bool) -> None:
    initial_parameter = gpflow.Parameter([1.2, 1.1], trainable=trainable)
    new_parameter = gpflow.Parameter(initial_parameter)

    assert new_parameter.trainable == trainable


@pytest.mark.parametrize("trainable", [True, False])
def test_construct_parameter_from_existing_parameter_override_trainable(trainable: bool) -> None:
    initial_parameter = gpflow.Parameter([1.2, 1.1], trainable=trainable)
    new_parameter = gpflow.Parameter(initial_parameter, trainable=not trainable)

    assert new_parameter.trainable is not trainable


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_construct_parameter_from_existing_parameter_check_dtype(dtype: tf.DType) -> None:
    initial_parameter = gpflow.Parameter([1.1, 2.1], dtype=dtype)
    new_parameter = gpflow.Parameter(initial_parameter)

    assert new_parameter.dtype == dtype


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_construct_parameter_from_existing_parameter_override_dtype(dtype: tf.DType) -> None:
    initial_parameter = gpflow.Parameter([1.1, 2.1])
    new_parameter = gpflow.Parameter(initial_parameter, dtype=dtype)

    assert new_parameter.dtype == dtype


def test_construct_parameter_from_existing_parameter_check_name() -> None:
    transform = tfp.bijectors.Sigmoid(
        tf.constant(0.0, dtype=tf.float64), tf.constant(2.0, dtype=tf.float64)
    )
    initial_parameter = gpflow.Parameter([1.2, 1.1], transform=transform)
    new_parameter = gpflow.Parameter(initial_parameter)

    assert new_parameter.name == transform.name


def test_construct_parameter_from_existing_parameter_override_name() -> None:
    initial_parameter = gpflow.Parameter([1.2, 1.1])
    transform = tfp.bijectors.Sigmoid(
        tf.constant(0.0, dtype=tf.float64), tf.constant(2.0, dtype=tf.float64)
    )
    new_parameter = gpflow.Parameter(initial_parameter, transform=transform)

    assert new_parameter.name == transform.name


def test_construct_parameter_from_existing_parameter_value_becomes_invalid() -> None:
    initial_parameter = gpflow.Parameter(0.0)
    transform = tfp.bijectors.Reciprocal()

    with pytest.raises(tf.errors.InvalidArgumentError) as exc:
        gpflow.Parameter(initial_parameter, transform=transform)

    assert "gpflow.Parameter" in exc.value.message


def test_construct_parameter_with_variable_shape() -> None:
    parameter = gpflow.Parameter([[1, 2, 3]], shape=[None, None])

    values: Sequence[AnyNDArray] = [
        np.ones((0, 0), dtype=default_float()),
        np.ones((1, 3), dtype=default_float()),
        np.ones((3, 1), dtype=default_float()),
        np.ones((3, 4), dtype=default_float()),
    ]

    for value in values:
        parameter.assign(value)
        np.testing.assert_equal(value, parameter.numpy())


def test_construct_parameter_with_variable_shape__different_constrained_shape() -> None:
    parameter = gpflow.Parameter(
        [[1, 0], [2, 3]],
        transform=triangular(),
        unconstrained_shape=[None],
        constrained_shape=[None, None],
    )

    values: Sequence[AnyNDArray] = [
        # The triangular() transform doesn't appear to support 0x0 matrices.
        np.tril(np.ones((1, 1), dtype=default_float())),
        np.tril(np.ones((2, 2), dtype=default_float())),
        np.tril(np.ones((3, 3), dtype=default_float())),
    ]

    for value in values:
        parameter.assign(value)
        np.testing.assert_equal(value, parameter.numpy())
