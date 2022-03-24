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
"""
Likelihoods are another core component of GPflow. This describes how likely the
data is under the assumptions made about the underlying latent functions
p(Y|F). Different likelihoods make different
assumptions about the distribution of the data, as such different data-types
(continuous, binary, ordinal, count) are better modelled with different
likelihood assumptions.

Use of any likelihood other than Gaussian typically introduces the need to use
an approximation to perform inference, if one isn't already needed.
Variational inference and MCMC models are included in GPflow and allow
approximate inference with non-Gaussian likelihoods. An introduction to these
models can be found :ref:`here <implemented_models>`. Specific notebooks
illustrating non-Gaussian likelihood regressions are available for
`classification <notebooks/classification.html>`_ (binary data), `ordinal
<notebooks/ordinal.html>`_ and `multiclass <notebooks/multiclass.html>`_.

Creating new likelihoods
------------------------

Likelihoods are defined by their
log-likelihood. When creating new likelihoods, the
:func:`logp <gpflow.likelihoods.Likelihood.logp>` method (log p(Y|F)), the
:func:`conditional_mean <gpflow.likelihoods.Likelihood.conditional_mean>`,
:func:`conditional_variance
<gpflow.likelihoods.Likelihood.conditional_variance>`.

In order to perform variational inference with non-Gaussian likelihoods a term
called ``variational expectations``, ∫ q(F) log p(Y|F) dF, needs to
be computed under a Gaussian distribution q(F) ~ N(μ, Σ).

The :func:`variational_expectations <gpflow.likelihoods.Likelihood.variational_expectations>`
method can be overriden if this can be computed in closed form, otherwise; if
the new likelihood inherits
:class:`Likelihood <gpflow.likelihoods.Likelihood>` the default will use
Gauss-Hermite numerical integration (works well when F is 1D
or 2D), if the new likelihood inherits from
:class:`MonteCarloLikelihood <gpflow.likelihoods.MonteCarloLikelihood>` the
integration is done by sampling (can be more suitable when F is higher dimensional).
"""

from .base import (
    Likelihood,
    MonteCarloLikelihood,
    QuadratureLikelihood,
    ScalarLikelihood,
    SwitchedLikelihood,
)
from .misc import GaussianMC
from .multiclass import MultiClass, RobustMax, Softmax
from .multilatent import (
    HeteroskedasticTFPConditional,
    MultiLatentLikelihood,
    MultiLatentTFPConditional,
)
from .scalar_continuous import Beta, Exponential, Gamma, Gaussian, StudentT
from .scalar_discrete import Bernoulli, Ordinal, Poisson

__all__ = [
    "Bernoulli",
    "Beta",
    "Exponential",
    "Gamma",
    "Gaussian",
    "GaussianMC",
    "HeteroskedasticTFPConditional",
    "Likelihood",
    "MonteCarloLikelihood",
    "MultiClass",
    "MultiLatentLikelihood",
    "MultiLatentTFPConditional",
    "Ordinal",
    "Poisson",
    "QuadratureLikelihood",
    "RobustMax",
    "ScalarLikelihood",
    "Softmax",
    "StudentT",
    "SwitchedLikelihood",
    "base",
    "misc",
    "multiclass",
    "multilatent",
    "scalar_continuous",
    "scalar_discrete",
    "utils",
]
