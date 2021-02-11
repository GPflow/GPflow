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
