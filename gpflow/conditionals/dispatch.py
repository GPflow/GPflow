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

from deprecated import deprecated

from ..utilities import Dispatcher

conditional = Dispatcher("conditional")
conditional._gpflow_internal_register = conditional.register

# type-ignore below is because mypy doesn't like it when we assign to a function.
conditional.register = deprecated(  # type: ignore[assignment]
    reason="Registering new implementations of conditional() is deprecated. "
    "Instead, create your own subclass of gpflow.posteriors.AbstractPosterior "
    "and register an implementation of gpflow.posteriors.get_posterior_class "
    "that returns your class."
)(conditional._gpflow_internal_register)

sample_conditional = Dispatcher("sample_conditional")
