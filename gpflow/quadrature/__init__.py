from .base import GaussianQuadrature
from .gauss_hermite import NDDiagGHQuadrature
from .quadrature import hermgauss, mvhermgauss, mvnquad, ndiagquad, ndiag_mc

from .deprecated import ndiagquad as ndiagquad_old
