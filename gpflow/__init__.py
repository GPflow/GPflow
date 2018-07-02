# Copyright 2016 alexggmatthews, James Hensman
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


# flake8: noqa

from ._version import __version__
from ._settings import SETTINGS as settings

from .session_manager import get_session
from .session_manager import get_default_session
from .session_manager import reset_default_session
from .session_manager import reset_default_graph_and_session

from . import misc
from . import transforms
from . import conditionals
from . import logdensities
from . import likelihoods
from . import kernels
from . import priors
from . import core
from . import models
from . import test_util
from . import training as train
from . import features
from . import expectations
from . import probability_distributions

from .decors import autoflow
from .decors import defer_build
from .decors import name_scope
from .decors import params_as_tensors
from .decors import params_as_tensors_for

from .core.errors import GPflowError
from .core.compilable import Build

from .params import Parameter as Param
from .params import ParamList
from .params import DataHolder
from .params import Minibatch
from .params import Parameterized

from .saver import Saver
from .saver import SaverContext

from . import multioutput
