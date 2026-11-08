# Copyright 2024 the GPflow authors.
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
import tensorflow as tf

from gpflow.config import default_float
from gpflow.conditionals.util import base_conditional_with_lm


def test_base_conditional_with_lm_clips_negative_variance() -> None:
    """
    `fvar = Knn - A^T @ A` is mathematically a variance (the Schur
    complement of a valid joint Gaussian) and can never truly be negative,
    but floating-point cancellation in that subtraction can push it
    slightly below zero when `Knn` and the reduction term are very close,
    e.g. when a test point coincides with (or is very near) a training
    point (see issue #1813). This directly constructs that case -- `Knn`
    deliberately set just under `A^T @ A` -- and checks the result is
    clipped to zero rather than returned negative.
    """
    Kmn = tf.constant([[1.0]], dtype=default_float())  # [M=1, N=1]
    Lm = tf.constant([[1.0]], dtype=default_float())  # [M=1, M=1] -> A = Kmn
    f = tf.constant([[0.0]], dtype=default_float())  # [M=1, R=1]
    # A^T @ A == 1.0 here; Knn is set just under that.
    Knn = tf.constant([0.999], dtype=default_float())  # [N=1]

    _, fvar = base_conditional_with_lm(Kmn=Kmn, Lm=Lm, Knn=Knn, f=f, full_cov=False)

    assert np.all(fvar.numpy() >= 0.0)
