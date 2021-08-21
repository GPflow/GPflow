# Copyright 2021 the GPflow authors.
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

from dataclasses import dataclass

import numpy as np

from gpflow.kernels import SquaredExponential
from gpflow.models import CGLB, SGPR
from gpflow.utilities import to_default_float as tdf


def data(rng: np.random.RandomState):
    n: int = 100
    d: int = 2

    x = rng.randn(n, d)
    c = np.array([[-1.4], [0.5]])
    y = np.sin(x @ c + 0.5 * rng.randn(n, 1))
    z = rng.randn(20, 2)

    return (tdf(x), tdf(y)), tdf(z)


def test_cglb_check_basics():
    """
    * Quadratic term of the CGLB with v=0 equivalent to the quadratic term of the SGPR.
    * Log determinant term of the CGLB is less or equal to SGPR log determinant.
    """

    rng: np.random.RandomState = np.random.RandomState(999)
    train, z = data(rng)

    sgpr = SGPR(train, kernel=SquaredExponential(), inducing_variable=z)
    # `v_grad_optimization=True` turns off the CG in the quadratic term
    cglb = CGLB(train, kernel=SquaredExponential(), inducing_variable=z, v_grad_optimization=True)

    sgpr_common = sgpr._common_calculation()
    cglb_common = cglb._common_calculation()

    sgpr_quad_term = sgpr.quad_term(sgpr_common)
    cglb_quad_term = cglb.quad_term(cglb_common)
    np.testing.assert_almost_equal(sgpr_quad_term, cglb_quad_term)

    sgpr_logdet = sgpr.logdet_term(sgpr_common)
    cglb_logdet = cglb.logdet_term(cglb_common)
    assert cglb_logdet >= sgpr_logdet
