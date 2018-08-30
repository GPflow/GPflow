from typing import Union
import numpy as np
import tensorflow as tf


# this should be a Union[] but multipledispatch module requires unions as tuples
TensorArray = (np.ndarray, tf.Tensor)

