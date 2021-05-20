#  Copyright 2021 The GPflow Contributors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Union

from .. import kernels
from ..inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from ..utilities import Dispatcher
from .posterior import PrecomputeCacheType
from .svgp import (
    FallbackIndependentLatentPosterior,
    FullyCorrelatedPosterior,
    IndependentPosteriorMultiOutput,
    IndependentPosteriorSingleOutput,
    LinearCoregionalizationPosterior,
)

get_posterior_class = Dispatcher("get_posterior_class")


@get_posterior_class.register(kernels.Kernel, InducingVariables)
def _get_posterior_base_case(kernel, inducing_variable):
    # independent single output
    return IndependentPosteriorSingleOutput


@get_posterior_class.register(kernels.MultioutputKernel, InducingPoints)
def _get_posterior_fully_correlated_mo(kernel, inducing_variable):
    return FullyCorrelatedPosterior


@get_posterior_class.register(
    (kernels.SharedIndependent, kernels.SeparateIndependent),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_independent_mo(kernel, inducing_variable):
    # independent multi-output
    return IndependentPosteriorMultiOutput


@get_posterior_class.register(
    kernels.IndependentLatent,
    (FallbackSeparateIndependentInducingVariables, FallbackSharedIndependentInducingVariables),
)
def _get_posterior_independentlatent_mo_fallback(kernel, inducing_variable):
    return FallbackIndependentLatentPosterior


@get_posterior_class.register(
    kernels.LinearCoregionalization,
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_linearcoregionalization_mo_efficient(kernel, inducing_variable):
    # Linear mixing---efficient multi-output
    return LinearCoregionalizationPosterior


def _validate_precompute_cache_type(value) -> PrecomputeCacheType:
    if value is None:
        return PrecomputeCacheType.NOCACHE
    elif isinstance(value, PrecomputeCacheType):
        return value
    elif isinstance(value, str):
        return PrecomputeCacheType(value.lower())
    else:
        raise ValueError(
            f"{value} is not a valid PrecomputeCacheType. Valid options: 'tensor', 'variable', 'nocache' (or None)."
        )


def create_posterior(
    kernel,
    inducing_variable,
    q_mu,
    q_sqrt,
    whiten,
    mean_function=None,
    precompute_cache: Union[PrecomputeCacheType, str, None] = PrecomputeCacheType.TENSOR,
):
    posterior_class = get_posterior_class(kernel, inducing_variable)
    precompute_cache = _validate_precompute_cache_type(precompute_cache)
    return posterior_class(
        kernel,
        inducing_variable,
        q_mu,
        q_sqrt,
        whiten,
        mean_function,
        precompute_cache=precompute_cache,
    )
