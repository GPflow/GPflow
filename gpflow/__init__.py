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
from . import (likelihoods, kernels, ekernels, param,
               model, gpmc, sgpmc, priors, gpr, svgp,
               vgp, sgpr, gplvm, tf_wraps, tf_hacks)
from ._version import __version__
from ._settings import settings
