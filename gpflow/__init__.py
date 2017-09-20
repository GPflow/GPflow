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
from __future__ import absolute_import

from gpflow._version import __version__
from gpflow._settings import SETTINGS as settings

from gpflow import misc
from gpflow import transforms
from gpflow import conditionals
from gpflow import densities
from gpflow import likelihoods
from gpflow import kernels
from gpflow import ekernels
from gpflow import priors

from gpflow.models import models as model
from gpflow.training import training as train

from gpflow.base import Build

from gpflow.misc import GPflowError

from gpflow.params import Param
from gpflow.params import ParamList
from gpflow.params import DataHolder

from gpflow.decors import autoflow
from gpflow.decors import name_scope
from gpflow.decors import params_as_tensors
