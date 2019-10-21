# Copyright 2019 the GPflow authors.
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

import gpflow
from gpflow.models import GPR
from gpflow.utilities import set_trainable

rng = np.random.RandomState(0)

class Data:
    N = 10
    D = 1
    X = rng.rand(N, D)
    Y = rng.rand(N, 1)
    ls = 2.0
    var = 1.0


# ------------------------------------------
# Fixtures
# ------------------------------------------

@pytest.fixture
def model():
    return gpflow.models.GPR((Data.X, Data.Y),
        kernel=gpflow.kernels.SquaredExponential(lengthscale=Data.ls, variance=Data.var),
    )


def test_empty_model_objective(model):
    set_trainable(model, False)
    assert model.log_prior() == 0.0
