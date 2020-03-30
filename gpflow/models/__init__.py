# Copyright 2017 Artem Artemev @awav
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

from .gplvm import GPLVM, BayesianGPLVM
from .gpmc import GPMC
from .gpr import GPR
from .model import BayesianModel, GPModel
from .training_mixins import (
    ExternalDataTrainingLossMixin,
    InternalDataTrainingLossMixin,
)

# from .gplvm import PCA_reduce
from .sgpmc import SGPMC
from .sgpr import GPRFITC, SGPR
from .svgp import SVGP
from .vgp import VGP, VGPOpperArchambeau
from .util import (
    training_loss,
    training_loss_closure,
    maximum_log_likelihood_objective,
)
