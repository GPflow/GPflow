# Copyright 2016 James Hensman, alexggmatthews
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

import tensorflow as tf
import numpy as np
import pandas as pd
from collections import OrderedDict

from .util import default_float
from . import settings
from ._version import __version__


def pretty_pandas_table(row_names, column_names, column_values):
    return pd.DataFrame(
        OrderedDict(zip(column_names, column_values)),
        index=row_names)


def normalize_num_type(num_type):
    """
    Work out what a sensible type for the array is. if the default type
    is float32, downcast 64bit float to float32. For ints, assume int32
    """
    if isinstance(num_type, tf.DType):
        num_type = num_type.as_numpy_dtype.type

    if num_type in [np.float32, np.float64]:  # pylint: disable=E1101
        num_type = default_float()
    elif num_type in [np.int16, np.int32, np.int64]:
        num_type = settings.int_type
    else:
        raise ValueError('Unknown dtype "{0}" passed to normalizer.'.format(num_type))

    return num_type


def version():
    return __version__
