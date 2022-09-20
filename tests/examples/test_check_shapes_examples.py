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
"""
Code examples used in `check_shapes` docstrings.
"""


def test_example__basic() -> None:
    # This test / example is placed *above* the imports, to ensure that the example has the
    # necessary imports.

    # pylint: disable=import-outside-toplevel, redefined-outer-name, reimported

    # [basic]

    import tensorflow as tf

    from gpflow.experimental.check_shapes import check_shapes

    @tf.function
    @check_shapes(
        "features: [batch..., n_features]",
        "weights: [n_features]",
        "return: [batch...]",
    )
    def linear_model(features: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        return tf.einsum("...i,i -> ...", features, weights)

    # [basic]

    w = tf.ones((3,))
    for batch_shape in [(), (2,), (2, 4)]:
        f = tf.ones(batch_shape + (3,))
        linear_model(f, w)


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import (
    DocstringFormat,
    ErrorContext,
    Shape,
    ShapeChecker,
    ShapeCheckingState,
    check_shape,
    check_shapes,
    disable_check_shapes,
    get_check_shapes,
    get_enable_check_shapes,
    get_enable_function_call_precompute,
    get_rewrite_docstrings,
    get_shape,
    inherit_check_shapes,
    register_get_shape,
    set_enable_check_shapes,
    set_enable_function_call_precompute,
    set_rewrite_docstrings,
)


def test_example__disable__manual() -> None:
    old_value = get_enable_check_shapes()

    try:

        # [disable__manual]

        set_enable_check_shapes(ShapeCheckingState.DISABLED)

        # [disable__manual]

    finally:
        set_enable_check_shapes(old_value)


def test_example__disable__context_manager() -> None:
    def performance_sensitive_function() -> None:
        pass

    # [disable__context_manager]

    with disable_check_shapes():
        performance_sensitive_function()

    # [disable__context_manager]


def test_example__pytest_fixture() -> None:
    # pylint: disable=unused-variable
    # [pytest_fixture]

    @pytest.fixture(autouse=True)
    def enable_shape_checks() -> Iterable[None]:
        old_enable = get_enable_check_shapes()
        old_rewrite_docstrings = get_rewrite_docstrings()
        old_function_call_precompute = get_enable_function_call_precompute()
        set_enable_check_shapes(ShapeCheckingState.ENABLED)
        set_rewrite_docstrings(DocstringFormat.SPHINX)
        set_enable_function_call_precompute(True)
        yield
        set_enable_function_call_precompute(old_function_call_precompute)
        set_rewrite_docstrings(old_rewrite_docstrings)
        set_enable_check_shapes(old_enable)

    # [pytest_fixture]


def test_example__argument_ref_attribute() -> None:
    # [argument_ref_attribute]

    @dataclass
    class Statistics:
        mean: AnyNDArray
        std: AnyNDArray

    @check_shapes(
        "data: [n_rows, n_columns]",
        "return.mean: [n_columns]",
        "return.std: [n_columns]",
    )
    def compute_statistics(data: AnyNDArray) -> Statistics:
        return Statistics(np.mean(data, axis=0), np.std(data, axis=0))

    # [argument_ref_attribute]

    compute_statistics(np.ones((4, 3)))


def test_example__argument_ref_index() -> None:
    # [argument_ref_index]

    @check_shapes(
        "data: [n_rows, n_columns]",
        "return[0]: [n_columns]",
        "return[1]: [n_columns]",
    )
    def compute_mean_and_std(data: AnyNDArray) -> Tuple[AnyNDArray, AnyNDArray]:
        return np.mean(data, axis=0), np.std(data, axis=0)

    # [argument_ref_index]

    compute_mean_and_std(np.ones((4, 3)))


def test_example__argument_ref_all() -> None:
    # [argument_ref_all]

    @check_shapes(
        "data[all]: [., n_columns]",
        "return: [., n_columns]",
    )
    def concat_rows(data: Sequence[AnyNDArray]) -> AnyNDArray:
        return np.concatenate(data, axis=0)

    concat_rows(
        [
            np.ones((1, 3)),
            np.ones((4, 3)),
        ]
    )

    # [argument_ref_all]


def test_example__argument_ref_keys() -> None:
    # [argument_ref_keys]

    @check_shapes(
        "data.keys(): [.]",
        "return: []",
    )
    def sum_key_lengths(data: Mapping[Tuple[int, ...], str]) -> int:
        return sum(len(k) for k in data)

    sum_key_lengths(
        {
            (3,): "foo",
            (1, 2): "bar",
        }
    )

    # [argument_ref_keys]


def test_example__argument_ref_values() -> None:
    # [argument_ref_values]

    @check_shapes(
        "data.values(): [., n_columns]",
        "return: [., n_columns]",
    )
    def concat_rows(data: Mapping[str, AnyNDArray]) -> AnyNDArray:
        return np.concatenate(list(data.values()), axis=0)

    concat_rows(
        {
            "foo": np.ones((1, 3)),
            "bar": np.ones((4, 3)),
        }
    )

    # [argument_ref_values]


def test_example__argument_ref_optional() -> None:
    # [argument_ref_optional]

    @check_shapes(
        "x1: [n_rows_1, n_inputs]",
        "x2: [n_rows_2, n_inputs]",
        "return: [n_rows_1, n_rows_2]",
    )
    def squared_exponential_kernel(
        variance: float, x1: AnyNDArray, x2: Optional[AnyNDArray] = None
    ) -> AnyNDArray:
        if x2 is None:
            x2 = x1
        cov: AnyNDArray = variance * np.exp(
            -0.5 * np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        )
        return cov

    squared_exponential_kernel(1.0, np.ones((3, 2)), np.ones((4, 2)))
    squared_exponential_kernel(3.2, np.ones((3, 2)))

    # [argument_ref_optional]


def test_example__dimension_spec_constant() -> None:
    # [dimension_spec_constant]

    @check_shapes(
        "v1: [2]",
        "v2: [2]",
    )
    def vector_2d_distance(v1: AnyNDArray, v2: AnyNDArray) -> float:
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))

    # [dimension_spec_constant]

    vector_2d_distance(np.ones((2,)), np.ones((2,)))


def test_example__dimension_spec_variable() -> None:
    # [dimension_spec_variable]

    @check_shapes(
        "v1: [d]",
        "v2: [d]",
    )
    def vector_distance(v1: AnyNDArray, v2: AnyNDArray) -> float:
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))

    # [dimension_spec_variable]

    vector_distance(np.ones((3,)), np.ones((3,)))


def test_example__dimension_spec_anonymous__dot() -> None:
    # [dimension_spec_anonymous__dot]

    @check_shapes(
        "v: [.]",
    )
    def vector_length(v: AnyNDArray) -> float:
        return float(np.sqrt(np.sum(v ** 2)))

    # [dimension_spec_anonymous__dot]

    vector_length(np.ones((3,)))


def test_example__dimension_spec_anonymous__none() -> None:
    # [dimension_spec_anonymous__none]

    @check_shapes(
        "v: [None]",
    )
    def vector_length(v: AnyNDArray) -> float:
        return float(np.sqrt(np.sum(v ** 2)))

    # [dimension_spec_anonymous__none]

    vector_length(np.ones((3,)))


def test_example__dimension_spec_variable_rank__star() -> None:
    # [dimension_spec_variable_rank__star]

    @check_shapes(
        "x: [*batch, n_columns]",
        "return: [*batch]",
    )
    def batch_mean(x: AnyNDArray) -> AnyNDArray:
        mean: AnyNDArray = np.mean(x, axis=-1)
        return mean

    # [dimension_spec_variable_rank__star]

    for batch_shape in [(), (2,), (2, 4)]:
        x = np.ones(batch_shape + (3,))
        batch_mean(x)


def test_example__dimension_spec_variable_rank__ellipsis() -> None:
    # [dimension_spec_variable_rank__ellipsis]

    @check_shapes(
        "x: [batch..., n_columns]",
        "return: [batch...]",
    )
    def batch_mean(x: AnyNDArray) -> AnyNDArray:
        mean: AnyNDArray = np.mean(x, axis=-1)
        return mean

    # [dimension_spec_variable_rank__ellipsis]

    for batch_shape in [(), (2,), (2, 4)]:
        x = np.ones(batch_shape + (3,))
        batch_mean(x)


def test_example__dimension_spec_anonymous_variable_rank__star() -> None:
    # [dimension_spec_anonymous_variable_rank__star]

    @check_shapes(
        "x: [*]",
    )
    def rank(x: AnyNDArray) -> int:
        return len(x.shape)

    # [dimension_spec_anonymous_variable_rank__star]

    for batch_shape in [(), (2,), (2, 4)]:
        x = np.ones(batch_shape)
        rank(x)


def test_example__dimension_spec_anonymous_variable_rank__ellipsis() -> None:
    # [dimension_spec_anonymous_variable_rank__ellipsis]

    @check_shapes(
        "x: [...]",
    )
    def rank(x: AnyNDArray) -> int:
        return len(x.shape)

    # [dimension_spec_anonymous_variable_rank__ellipsis]

    for batch_shape in [(), (2,), (2, 4)]:
        x = np.ones(batch_shape)
        rank(x)


def test_example__dimension_spec__scalar() -> None:
    # [dimension_spec__scalar]

    @check_shapes(
        "x: [...]",
        "return: []",
    )
    def mean(x: AnyNDArray) -> AnyNDArray:
        mean: AnyNDArray = np.sum(x) / x.size
        return mean

    # [dimension_spec__scalar]

    for batch_shape in [(), (2,), (2, 4)]:
        x = np.ones(batch_shape)
        mean(x)


def test_example__dimension_spec_broadcast() -> None:

    # [dimension_spec_broadcast]

    @check_shapes(
        "a: [broadcast batch...]",
        "b: [broadcast batch...]",
        "return: [batch...]",
    )
    def add(a: AnyNDArray, b: AnyNDArray) -> AnyNDArray:
        return a + b

    # [dimension_spec_broadcast]

    add(np.ones((3, 1)), np.ones((1, 4)))


def test_example__bool_spec_argument_ref() -> None:
    # pylint: disable=unused-argument

    # [bool_spec_argument_ref]

    @check_shapes(
        "a: [broadcast batch...] if check_a",
        "b: [broadcast batch...] if check_b",
        "return: [batch...]",
    )
    def add(a: AnyNDArray, b: AnyNDArray, check_a: bool = True, check_b: bool = True) -> AnyNDArray:
        return a + b

    add(np.ones((3, 1)), np.ones((1, 4)), check_b=False)

    # [bool_spec_argument_ref]


def test_example__bool_spec_argument_ref_is_none() -> None:
    # pylint: disable=unused-argument

    # [bool_spec_argument_ref_is_none]

    @check_shapes(
        "a: [n_a]",
        "b: [n_b]",
        "return: [n_a, n_a] if b is None",
        "return: [n_a, n_b] if b is not None",
    )
    def square(a: AnyNDArray, b: Optional[AnyNDArray] = None) -> AnyNDArray:
        if b is None:
            b = a
        result: AnyNDArray = a[:, None] * b[None, :]
        return result

    square(np.ones((3,)))
    square(np.ones((3,)), np.ones((4,)))

    # [bool_spec_argument_ref_is_none]


def test_example__bool_spec_or() -> None:
    # pylint: disable=unused-argument

    # [bool_spec_or]

    @check_shapes(
        "a: [broadcast batch...] if check_all or check_a",
        "b: [broadcast batch...] if check_all or check_b",
        "return: [batch...]",
    )
    def add(
        a: AnyNDArray,
        b: AnyNDArray,
        check_all: bool = False,
        check_a: bool = True,
        check_b: bool = True,
    ) -> AnyNDArray:
        return a + b

    add(np.ones((3, 1)), np.ones((1, 4)), check_b=False)

    # [bool_spec_or]


def test_example__bool_spec_and() -> None:
    # pylint: disable=unused-argument

    # [bool_spec_and]

    @check_shapes(
        "a: [broadcast batch...] if enable_checks and check_a",
        "b: [broadcast batch...] if enable_checks and check_b",
        "return: [batch...]",
    )
    def add(
        a: AnyNDArray,
        b: AnyNDArray,
        enable_checks: bool = True,
        check_a: bool = True,
        check_b: bool = True,
    ) -> AnyNDArray:
        return a + b

    add(np.ones((3, 1)), np.ones((1, 4)), check_b=False)

    # [bool_spec_and]


def test_example__bool_spec_not() -> None:
    # pylint: disable=unused-argument

    # [bool_spec_not]

    @check_shapes(
        "a: [broadcast batch...] if not disable_checks",
        "b: [broadcast batch...] if not disable_checks",
        "return: [batch...]",
    )
    def add(a: AnyNDArray, b: AnyNDArray, disable_checks: bool = False) -> AnyNDArray:
        return a + b

    add(np.ones((3, 1)), np.ones((1, 4)))

    # [bool_spec_not]


def test_example__bool_spec__composition() -> None:
    # pylint: disable=unused-argument

    # [bool_spec__composition]

    @check_shapes(
        "a: [j] if a_vector",
        "a: [i, j] if (not a_vector)",
        "b: [j] if b_vector",
        "b: [j, k] if (not b_vector)",
        "return: [1, 1] if a_vector and b_vector",
        "return: [1, k] if a_vector and (not b_vector)",
        "return: [i, 1] if (not a_vector) and b_vector",
        "return: [i, k] if (not a_vector) and (not b_vector)",
    )
    def multiply(a: AnyNDArray, b: AnyNDArray, a_vector: bool, b_vector: bool) -> AnyNDArray:
        if a_vector:
            a = a[None, :]
        if b_vector:
            b = b[:, None]

        return a @ b

    multiply(np.ones((4,)), np.ones((4, 5)), a_vector=True, b_vector=False)

    # [bool_spec__composition]

    multiply(np.ones((4,)), np.ones((4,)), a_vector=True, b_vector=True)
    multiply(np.ones((3, 4)), np.ones((4,)), a_vector=False, b_vector=True)
    multiply(np.ones((3, 4)), np.ones((4, 5)), a_vector=False, b_vector=False)


def test_example__note_spec__global() -> None:
    # [note_spec__global]

    @check_shapes(
        "features: [batch..., n_features]",
        "# linear_model currently only supports a single output.",
        "weights: [n_features]",
        "return: [batch...]",
    )
    def linear_model(features: AnyNDArray, weights: AnyNDArray) -> AnyNDArray:
        prediction: AnyNDArray = np.einsum("...i,i -> ...", features, weights)
        return prediction

    # [note_spec__global]

    linear_model(np.ones((4, 3)), np.ones((3,)))


def test_example__note_spec__local() -> None:
    # [note_spec__local]

    @check_shapes(
        "features: [batch..., n_features]",
        "weights: [n_features] # linear_model currently only supports a single output.",
        "return: [batch...]",
    )
    def linear_model(features: AnyNDArray, weights: AnyNDArray) -> AnyNDArray:
        prediction: AnyNDArray = np.einsum("...i,i -> ...", features, weights)
        return prediction

    # [note_spec__local]

    linear_model(np.ones((4, 3)), np.ones((3,)))


def test_example__reuse__inherit_check_shapes() -> None:
    # [reuse__inherit_check_shapes]

    class Model(ABC):
        @abstractmethod
        @check_shapes(
            "features: [batch..., n_features]",
            "return: [batch...]",
        )
        def predict(self, features: AnyNDArray) -> AnyNDArray:
            pass

    class LinearModel(Model):
        @check_shapes(
            "weights: [n_features]",
        )
        def __init__(self, weights: AnyNDArray) -> None:
            self._weights = weights

        @inherit_check_shapes
        def predict(self, features: AnyNDArray) -> AnyNDArray:
            prediction: AnyNDArray = np.einsum("...i,i -> ...", features, self._weights)
            return prediction

    # [reuse__inherit_check_shapes]

    model = LinearModel(np.ones((3,)))
    model.predict(np.ones((10, 3)))


def test_example__reuse__functional() -> None:
    # [reuse__functional]

    check_metric_shapes = check_shapes(
        "actual: [n_rows, n_labels]",
        "predicted: [n_rows, n_labels]",
        "return: []",
    )

    @check_metric_shapes
    def rmse(actual: AnyNDArray, predicted: AnyNDArray) -> float:
        return float(np.mean(np.sqrt(np.mean((predicted - actual) ** 2, axis=-1))))

    @check_metric_shapes
    def mape(actual: AnyNDArray, predicted: AnyNDArray) -> float:
        return float(np.mean(np.abs((predicted - actual) / actual)))

    # [reuse__functional]

    actual = np.ones((10, 3))
    predicted = np.ones((10, 3))
    rmse(actual, predicted)
    mape(actual, predicted)


def test_example__reuse__get_check_shapes() -> None:
    # [reuse__get_check_shapes]

    class Model(ABC):
        @abstractmethod
        @check_shapes(
            "features: [batch..., n_features]",
            "return: [batch...]",
        )
        def predict(self, features: AnyNDArray) -> AnyNDArray:
            pass

    @check_shapes(
        "test_features: [n_rows, n_features]",
        "test_labels: [n_rows]",
    )
    def evaluate_model(model: Model, test_features: AnyNDArray, test_labels: AnyNDArray) -> float:
        prediction = model.predict(test_features)
        return float(np.mean(np.sqrt(np.mean((prediction - test_labels) ** 2, axis=-1))))

    def test_evaluate_model() -> None:
        fake_features = np.ones((10, 3))
        fake_labels = np.ones((10,))
        fake_predictions = np.ones((10,))

        @get_check_shapes(Model.predict)
        def fake_predict(features: AnyNDArray) -> AnyNDArray:
            assert features is fake_features
            return fake_predictions

        fake_model = MagicMock(spec=Model, predict=fake_predict)

        assert pytest.approx(0.0) == evaluate_model(fake_model, fake_features, fake_labels)

    # [reuse__get_check_shapes]

    test_evaluate_model()


def test_example__intermediate_results() -> None:
    # [intermediate_results]

    @check_shapes(
        "weights: [n_features, n_labels]",
        "test_features: [n_rows, n_features]",
        "test_labels: [n_rows, n_labels]",
        "return: []",
    )
    def loss(weights: AnyNDArray, test_features: AnyNDArray, test_labels: AnyNDArray) -> AnyNDArray:
        prediction = check_shape(test_features @ weights, "[n_rows, n_labels]")
        error: AnyNDArray = check_shape(prediction - test_labels, "[n_rows, n_labels]")
        square_error = check_shape(error ** 2, "[n_rows, n_labels]")
        mean_square_error = check_shape(np.mean(square_error, axis=-1), "[n_rows]")
        root_mean_square_error = check_shape(np.sqrt(mean_square_error), "[n_rows]")
        loss: AnyNDArray = np.mean(root_mean_square_error)
        return loss

    # [intermediate_results]

    loss(
        np.ones((5, 3)),
        np.ones((10, 5)),
        np.ones((10, 3)),
    )


def test_example__shape_checker__raw() -> None:
    # [shape_checker__raw]

    def linear_model(features: AnyNDArray, weights: AnyNDArray) -> AnyNDArray:
        checker = ShapeChecker()
        checker.check_shape(features, "[batch..., n_features]")
        checker.check_shape(weights, "[n_features]")
        prediction: AnyNDArray = checker.check_shape(
            np.einsum("...i,i -> ...", features, weights), "[batch...]"
        )
        return prediction

    # [shape_checker__raw]

    linear_model(np.ones((4, 3)), np.ones((3,)))


def test_example__disable_function_call_precompute() -> None:
    def buggy_function() -> None:
        pass

    old_value = get_enable_function_call_precompute()

    try:

        # [disable_function_call_precompute]

        set_enable_function_call_precompute(True)

        buggy_function()

        # [disable_function_call_precompute]

    finally:
        set_enable_function_call_precompute(old_value)


def test_example__doc_rewrite() -> None:

    # [doc_rewrite__definition]

    @check_shapes(
        "features: [batch..., n_features]",
        "weights: [n_features]",
        "return: [batch...]",
    )
    def linear_model(features: AnyNDArray, weights: AnyNDArray) -> AnyNDArray:
        """
        Computes a prediction from a linear model.

        :param features: Data to make predictions from.
        :param weights: Model weights.
        :returns: Model predictions.
        """
        prediction: AnyNDArray = np.einsum("...i,i -> ...", features, weights)
        return prediction

    # [doc_rewrite__definition]

    expected_doc = (
        # [doc_rewrite__rewritten]
        """
        Computes a prediction from a linear model.

        :param features:
            * **features** has shape [*batch*..., *n_features*].

            Data to make predictions from.
        :param weights:
            * **weights** has shape [*n_features*].

            Model weights.
        :returns:
            * **return** has shape [*batch*...].

            Model predictions.
        """
        # [doc_rewrite__rewritten]
    )

    assert expected_doc == linear_model.__doc__


def test_example__doc_rewrite__disable() -> None:

    old_rewrite_docstrings = get_rewrite_docstrings()

    try:
        # [doc_rewrite__disable]
        set_rewrite_docstrings(None)
        # [doc_rewrite__disable]
    finally:
        set_rewrite_docstrings(old_rewrite_docstrings)


def test_example__custom_type() -> None:
    # pylint: disable=protected-access,unused_variable

    # [custom_type]

    class LinearModel:
        @check_shapes(
            "weights: [n_features]",
        )
        def __init__(self, weights: AnyNDArray) -> None:
            self._weights = weights

        @check_shapes(
            "self: [n_features]",
            "features: [batch..., n_features]",
            "return: [batch...]",
        )
        def predict(self, features: AnyNDArray) -> AnyNDArray:
            prediction: AnyNDArray = np.einsum("...i,i -> ...", features, self._weights)
            return prediction

    @register_get_shape(LinearModel)
    def get_linear_model_shape(model: LinearModel, context: ErrorContext) -> Shape:
        shape: Shape = model._weights.shape
        return shape

    @check_shapes(
        "model: [n_features]",
        "test_features: [n_rows, n_features]",
        "test_labels: [n_rows]",
        "return: []",
    )
    def loss(model: LinearModel, test_features: AnyNDArray, test_labels: AnyNDArray) -> float:
        prediction = model.predict(test_features)
        return float(np.mean(np.sqrt(np.mean((prediction - test_labels) ** 2, axis=-1))))

    # [custom_type]

    model = LinearModel(np.ones((3,)))
    loss(model, np.ones((10, 3)), np.ones((10,)))
