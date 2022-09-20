# Copyright 2017-2020 the GPflow authors.
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

from gpflow.base import AnyNDArray, TensorType
from gpflow.config import default_float, default_int
from gpflow.experimental.check_shapes import inherit_check_shapes
from gpflow.likelihoods import Bernoulli, MultiClass, RobustMax, Softmax
from gpflow.utilities import to_default_float, to_default_int

tf.random.set_seed(99012)


@pytest.mark.parametrize("dimX", [3])
@pytest.mark.parametrize("num, dimF", [[10, 5], [3, 2]])
@pytest.mark.parametrize("dimY", [10, 2, 1])
def test_softmax_y_shape_assert(num: int, dimX: int, dimF: int, dimY: int) -> None:
    """
    SoftMax assumes the class is given as a label (not, e.g., one-hot
    encoded), and hence just uses the first column of Y. To prevent
    silent errors, there is a tf assertion that ensures Y only has one
    dimension. This test checks that this assert works as intended.
    """
    X = tf.random.normal((num, dimX))
    F = tf.random.normal((num, dimF))
    dY: AnyNDArray = np.vstack((np.random.randn(num - 3, dimY), np.ones((3, dimY)))) > 0
    Y = tf.convert_to_tensor(dY, dtype=default_int())
    likelihood = Softmax(dimF)
    try:
        likelihood.log_prob(X, F, Y)
    except tf.errors.InvalidArgumentError as e:
        assert "Condition x == y did not hold." in e.message


@pytest.mark.parametrize("dimX", [3])
@pytest.mark.parametrize("num", [10, 3])
@pytest.mark.parametrize("dimF, dimY", [[2, 1]])
def test_softmax_bernoulli_equivalence(num: int, dimX: int, dimF: int, dimY: int) -> None:
    X = tf.random.normal((num, dimX))
    dF: AnyNDArray = np.vstack(
        (np.random.randn(num - 3, dimF), np.array([[-3.0, 0.0], [3, 0.0], [0.0, 0.0]]))
    )
    dY: AnyNDArray = np.vstack((np.random.randn(num - 3, dimY), np.ones((3, dimY)))) > 0
    F = to_default_float(dF)
    Fvar = tf.exp(tf.stack([F[:, 1], -10.0 + tf.zeros(F.shape[0], dtype=F.dtype)], axis=1))
    F = tf.stack([F[:, 0], tf.zeros(F.shape[0], dtype=F.dtype)], axis=1)
    Y = to_default_int(dY)
    Ylabel = 1 - Y

    softmax_likelihood = Softmax(dimF)
    bernoulli_likelihood = Bernoulli(invlink=tf.sigmoid)
    softmax_likelihood.num_monte_carlo_points = int(
        0.3e7
    )  # Minimum number of points to pass the test on CircleCI
    bernoulli_likelihood.num_gauss_hermite_points = 40

    assert_allclose(
        softmax_likelihood.conditional_mean(X, F)[:, :1],
        bernoulli_likelihood.conditional_mean(X, F[:, :1]),
    )

    assert_allclose(
        softmax_likelihood.conditional_variance(X, F)[:, :1],
        bernoulli_likelihood.conditional_variance(X, F[:, :1]),
    )

    assert_allclose(
        softmax_likelihood.log_prob(X, F, Ylabel),
        bernoulli_likelihood.log_prob(X, F[:, :1], Y.numpy()),
    )

    mean1, var1 = softmax_likelihood.predict_mean_and_var(X, F, Fvar)
    mean2, var2 = bernoulli_likelihood.predict_mean_and_var(X, F[:, :1], Fvar[:, :1])

    assert_allclose(mean1[:, 0, None], mean2, rtol=2e-3)
    assert_allclose(var1[:, 0, None], var2, rtol=2e-3)

    ls_ve = softmax_likelihood.variational_expectations(X, F, Fvar, Ylabel)
    lb_ve = bernoulli_likelihood.variational_expectations(X, F[:, :1], Fvar[:, :1], Y.numpy())
    assert_allclose(ls_ve, lb_ve, rtol=5e-3)


@pytest.mark.parametrize("num_classes, num_points", [[10, 3]])
@pytest.mark.parametrize("tol, epsilon", [[1e-4, 1e-3]])
def test_robust_max_multiclass_symmetric(
    num_classes: int, num_points: int, tol: float, epsilon: float
) -> None:
    """
    This test is based on the observation that for
    symmetric inputs the class predictions must have equal probability.
    """
    rng = np.random.RandomState(1)
    p = 1.0 / num_classes
    X = tf.ones((num_points, 1), dtype=default_float())
    F = tf.ones((num_points, num_classes), dtype=default_float())
    Y = tf.convert_to_tensor(rng.randint(num_classes, size=(num_points, 1)), dtype=default_float())

    likelihood = MultiClass(num_classes)
    likelihood.invlink.epsilon = tf.convert_to_tensor(epsilon, dtype=default_float())

    mu, _ = likelihood.predict_mean_and_var(X, F, F)
    pred = likelihood.predict_log_density(X, F, F, Y)
    variational_expectations = likelihood.variational_expectations(X, F, F, Y)

    expected_mu: AnyNDArray = (
        p * (1.0 - epsilon) + (1.0 - p) * epsilon / (num_classes - 1)
    ) * np.ones((num_points, 1))
    expected_log_density = np.log(expected_mu)

    # assert_allclose() would complain about shape mismatch
    assert np.allclose(mu, expected_mu, tol, tol)
    assert np.allclose(pred, expected_log_density, 1e-3, 1e-3)

    validation_variational_expectation = p * np.log(1.0 - epsilon) + (1.0 - p) * np.log(
        epsilon / (num_classes - 1)
    )
    assert_allclose(
        variational_expectations,
        np.ones((num_points,)) * validation_variational_expectation,
        tol,
        tol,
    )


@pytest.mark.parametrize("num_classes, num_points", [[5, 100]])
@pytest.mark.parametrize(
    "mock_prob, expected_prediction, tol, epsilon",
    [
        [0.73, -0.5499780059, 1e-4, 0.231]
        # Expected prediction evaluated on calculator:
        # log((1 - ε) * 0.73 + (1-0.73) * ε / (num_classes -1))
    ],
)
def test_robust_max_multiclass_predict_log_density(
    num_classes: int,
    num_points: int,
    mock_prob: float,
    expected_prediction: float,
    tol: float,
    epsilon: float,
) -> None:
    class MockRobustMax(RobustMax):
        @inherit_check_shapes
        def prob_is_largest(
            self,
            Y: TensorType,
            mu: TensorType,
            var: TensorType,
            gh_x: TensorType,
            gh_w: TensorType,
        ) -> tf.Tensor:
            return tf.ones((num_points, 1), dtype=default_float()) * mock_prob

    likelihood = MultiClass(num_classes, invlink=MockRobustMax(num_classes, epsilon))
    X = tf.ones((num_points, 2))
    F = tf.ones((num_points, num_classes))
    rng = np.random.RandomState(1)
    Y = to_default_int(rng.randint(num_classes, size=(num_points, 1)))
    prediction = likelihood.predict_log_density(X, F, F, Y)

    assert_allclose(prediction, expected_prediction, tol, tol)


@pytest.mark.parametrize("num_classes", [5, 100])
@pytest.mark.parametrize("initial_epsilon, new_epsilon", [[1e-3, 0.412]])
def test_robust_max_multiclass_eps_k1_changes(
    num_classes: int, initial_epsilon: float, new_epsilon: float
) -> None:
    """
    Checks that eps K1 changes when epsilon changes. This used to not happen and had to be
    manually changed.
    """
    likelihood = RobustMax(num_classes, initial_epsilon)
    expected_eps_k1 = initial_epsilon / (num_classes - 1.0)
    actual_eps_k1 = likelihood.eps_k1
    assert_allclose(expected_eps_k1, actual_eps_k1)

    likelihood.epsilon = tf.convert_to_tensor(new_epsilon, dtype=default_float())
    expected_eps_k2 = new_epsilon / (num_classes - 1.0)
    actual_eps_k2 = likelihood.eps_k1
    assert_allclose(expected_eps_k2, actual_eps_k2)


@pytest.mark.skip(
    "ndiagquad cannot handle MultiClass (see https://github.com/GPflow/GPflow/issues/1091"
)
def test_multiclass_quadrature_variational_expectations() -> None:
    pass


@pytest.mark.skip(
    "ndiagquad cannot handle MultiClass (see https://github.com/GPflow/GPflow/issues/1091"
)
def test_multiclass_quadrature_predict_log_density() -> None:
    pass


@pytest.mark.skip(
    "ndiagquad cannot handle MultiClass (see https://github.com/GPflow/GPflow/issues/1091"
)
def test_multiclass_quadrature_predict_mean_and_var() -> None:
    pass
