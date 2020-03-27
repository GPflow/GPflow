# Copyright 2018 GPflow authors
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

from ..inducing_variables import InducingVariables


class MultioutputInducingVariables(InducingVariables):
    """
    Multioutput Inducing Variables
    Base class for methods which define a collection of inducing variables which
    in some way can be grouped. The main example is where the inducing variables
    consist of outputs of various independent GPs. This can be because our model
    uses multiple independent GPs (SharedIndependent, SeparateIndependent) or
    because it is constructed from independent GPs (eg IndependentLatent,
    LinearCoregionalization).
    """

    @property
    def inducing_variables(self):
        raise NotImplementedError


class FallbackSharedIndependentInducingVariables(MultioutputInducingVariables):
    """
    Shared definition of inducing variables for each independent latent process.

    This class is designated to be used to
     - provide a general interface for multioutput kernels
       constructed from independent latent processes,
     - only require the specification of Kuu and Kuf.
    All multioutput kernels constructed from independent latent processes allow
    the inducing variables to be specified in the latent processes, and a
    reasonably efficient method (i.e. one that takes advantage of the
    independence in the latent processes) can be specified quite generally by
    only requiring the following covariances:
     - Kuu: [L, M, M],
     - Kuf: [L, M, N, P].
    In `gpflow/conditionals/multioutput/conditionals.py` we define a conditional() implementation for this
    combination. We specify this code path for all kernels which inherit from
    `IndependentLatentBase`. This set-up allows inference with any such kernel
    to be implemented by specifying only `Kuu()` and `Kuf()`.

    We call this the base class, since many multioutput GPs that are constructed
    from independent latent processes acutally allow even more efficient
    approximations. However, we include this code path, as it does not require
    specifying a new `conditional()` implementation.

    Here, we share the definition of inducing variables between all latent
    processes.
    """

    def __init__(self, inducing_variable):
        super().__init__()
        self.inducing_variable = inducing_variable

    def __len__(self):
        return len(self.inducing_variable)

    @property
    def inducing_variables(self):
        return (self.inducing_variable,)


class FallbackSeparateIndependentInducingVariables(MultioutputInducingVariables):
    """
    Separate set of inducing variables for each independent latent process.

    This class is designated to be used to
     - provide a general interface for multioutput kernels
       constructed from independent latent processes,
     - only require the specification of Kuu and Kuf.
    All multioutput kernels constructed from independent latent processes allow
    the inducing variables to be specified in the latent processes, and a
    reasonably efficient method (i.e. one that takes advantage of the
    independence in the latent processes) can be specified quite generally by
    only requiring the following covariances:
     - Kuu: [L, M, M],
     - Kuf: [L, M, N, P].
    In `gpflow/multioutput/conditionals.py` we define a conditional() implementation for this
    combination. We specify this code path for all kernels which inherit from
    `IndependentLatentBase`. This set-up allows inference with any such kernel
    to be implemented by specifying only `Kuu()` and `Kuf()`.

    We call this the base class, since many multioutput GPs that are constructed
    from independent latent processes acutally allow even more efficient
    approximations. However, we include this code path, as it does not require
    specifying a new `conditional()` implementation.

    We use a different definition of inducing variables for each latent process.
    Note: each object should have the same number of inducing variables, M.
    """

    def __init__(self, inducing_variable_list):
        super().__init__()
        self.inducing_variable_list = inducing_variable_list

    def __len__(self):
        return len(self.inducing_variable_list[0])

    @property
    def inducing_variables(self):
        return tuple(self.inducing_variable_list)


class SharedIndependentInducingVariables(FallbackSharedIndependentInducingVariables):
    """
    Here, we define the same inducing variables as in the base class. However,
    this class is intended to be used without the constraints on the shapes that
    `Kuu()` and `Kuf()` return. This allows a custom `conditional()` to provide
    the most efficient implementation.
    """

    pass


class SeparateIndependentInducingVariables(FallbackSeparateIndependentInducingVariables):
    """
    Here, we define the same inducing variables as in the base class. However,
    this class is intended to be used without the constraints on the shapes that
    `Kuu()` and `Kuf()` return. This allows a custom `conditional()` to provide
    the most efficient implementation.
    """

    pass
