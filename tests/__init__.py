import warnings

import numpy as np
import tensorflow as tf

import gpflow

warnings.filterwarnings("ignore")
gpflow.config.set_default_float(np.float64)
