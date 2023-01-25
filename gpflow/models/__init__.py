# Copyright 2022 The GPflow Contributors. All Rights Reserved.
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
Sub-package containing the GPflow model implementations.

All our models derive from :class:`GPModel`, which itself derives from :class:`BayesianModel`.

For an overview of the implemented model see :ref:`implemented_models`, and for a basic example of
how to use one see :doc:`../../../../notebooks/getting_started/basic_usage`.
"""
from .cglb import CGLB
from .gplvm import GPLVM, BayesianGPLVM
from .gpmc import GPMC
from .gpr import GPR
from .model import BayesianModel, GPModel
from .sgpmc import SGPMC
from .sgpr import GPRFITC, SGPR
from .svgp import SVGP
from .training_mixins import ExternalDataTrainingLossMixin, InternalDataTrainingLossMixin
from .util import maximum_log_likelihood_objective, training_loss, training_loss_closure
from .vgp import VGP, VGPOpperArchambeau

__all__ = [
    "BayesianGPLVM",
    "BayesianModel",
    "CGLB",
    "ExternalDataTrainingLossMixin",
    "GPLVM",
    "GPMC",
    "GPModel",
    "GPR",
    "GPRFITC",
    "InternalDataTrainingLossMixin",
    "SGPMC",
    "SGPR",
    "SVGP",
    "VGP",
    "VGPOpperArchambeau",
    "cglb",
    "gplvm",
    "gpmc",
    "gpr",
    "maximum_log_likelihood_objective",
    "model",
    "sgpmc",
    "sgpr",
    "svgp",
    "training_loss",
    "training_loss_closure",
    "training_mixins",
    "util",
    "vgp",
]
