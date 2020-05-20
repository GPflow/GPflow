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
