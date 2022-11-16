# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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


def inv_probit(x: tf.Tensor, jitter: float = 1e-3) -> tf.Tensor:
    """
    Compute the inverse probit, i.e. the N(0,1) CDF, at x.

    :param x: argument
    :param jitter: small jitter that ensures output is strictly between 0 and 1
    """
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter
