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


import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from gpflow.conditionals.util import leading_transpose, rollaxis_left, rollaxis_right


def test_leading_transpose():
    dims = [1, 2, 3, 4]
    a = tf.zeros(dims)
    b = leading_transpose(a, [..., -1, -2])
    c = leading_transpose(a, [-1, ..., -2])
    d = leading_transpose(a, [-1, -2, ...])
    e = leading_transpose(a, [3, 2, ...])
    f = leading_transpose(a, [3, -2, ...])

    assert len(a.shape) == len(b.shape) == len(c.shape) == len(d.shape)
    assert len(a.shape) == len(e.shape) == len(f.shape)
    assert b.shape[-2:] == [4, 3]
    assert c.shape[0] == 4 and c.shape[-1] == 3
    assert d.shape[:2] == [4, 3]
    assert d.shape == e.shape == f.shape


def test_leading_transpose_fails():
    """ Check that error is thrown if `perm` is not compatible with `a` """
    dims = [1, 2, 3, 4]
    a = tf.zeros(dims)

    with pytest.raises(ValueError):
        leading_transpose(a, [-1, -2])


# rollaxis
@pytest.mark.parametrize("rolls", [1, 2])
@pytest.mark.parametrize("direction", ["left", "right"])
def test_rollaxis(rolls, direction):
    A = np.random.randn(10, 5, 3)
    A_tf = tf.convert_to_tensor(A)

    if direction == "left":
        perm = [1, 2, 0] if rolls == 1 else [2, 0, 1]
    elif direction == "right":
        perm = [2, 0, 1] if rolls == 1 else [1, 2, 0]
    else:
        raise NotImplementedError

    A_rolled_ref = np.transpose(A, perm)

    if direction == "left":
        A_rolled_tf = rollaxis_left(A_tf, rolls)
    elif direction == "right":
        A_rolled_tf = rollaxis_right(A_tf, rolls)
    else:
        raise NotImplementedError

    assert_allclose(A_rolled_ref, A_rolled_tf)


@pytest.mark.parametrize("rolls", [1, 2])
def test_rollaxis_idempotent(rolls):
    A = np.random.randn(10, 5, 3, 20, 1)
    A_tf = tf.convert_to_tensor(A)
    A_left_right = rollaxis_left(rollaxis_right(A_tf, 2), 2)
    A_right_left = rollaxis_right(rollaxis_left(A_tf, 2), 2)

    assert_allclose(A, A_left_right)
    assert_allclose(A, A_right_left)
