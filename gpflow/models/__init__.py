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
r"""
Models are typically the highest level component that are used in GPflow;
usually they comprise of at least one `kernel <kernel_options.html>`_ and at
least one `likelihood <likelihood_options.html>`_. They additionally sometimes
contain a `mean function <mean_function_options.html>`_, and `priors
<prior_options.html>`_ over model parameters.

The following table summarizes the six core model options in GPflow.

+----------------------+----------------------------+----------------------------+-----------------------------+
|                      | Gaussian                   | Non-Gaussian (variational) | Non-Gaussian                |
|                      | Likelihood                 |                            | (MCMC)                      |
+======================+============================+============================+=============================+
| Full covariance      | :class:`gpflow.models.GPR` | :class:`gpflow.models.VGP` | :class:`gpflow.models.GPMC` |
+----------------------+----------------------------+----------------------------+-----------------------------+
| Sparse approximation | :class:`gpflow.models.SGPR`| :class:`gpflow.models.SVGP`| :class:`gpflow.models.SGPMC`|
+----------------------+----------------------------+----------------------------+-----------------------------+

The GPLVM which adds latent variables is also included
(`notebook <notebooks/GPLVM.html>`_).
"""

from .model import Model
from .model import GPModel
from .gpr import GPR
from .gpmc import GPMC
from .gplvm import GPLVM
from .gplvm import BayesianGPLVM
from .gplvm import PCA_reduce
from .sgpmc import SGPMC
from .sgpr import SGPRUpperMixin
from .sgpr import SGPR
from .sgpr import GPRFITC
from .svgp import SVGP
from .vgp import VGP
from .vgp import VGP_opper_archambeau
