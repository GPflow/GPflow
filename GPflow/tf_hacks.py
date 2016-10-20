# Copyright 2016 James Hensman, alexggmatthews
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
A collection of hacks for tensorflow.

this has been renamed tf_wraps, and is deprecated
"""

from . import tf_wraps
import numpy as np
import warnings


def eye(N):  # pragma: no cover
    warnings.warn('tf_hacks is deprecated: use tf_wraps instead', np.VisibleDeprecationWarning)
    return tf_wraps.eye(N)


def vec_to_tri(N):  # pragma: no cover
    warnings.warn('tf_hacks is deprecated: use tf_wraps instead', np.VisibleDeprecationWarning)
    return tf_wraps.vec_to_tri(N)


def tri_to_vec(N):  # pragma: no cover
    warnings.warn('tf_hacks is deprecated: use tf_wraps instead', np.VisibleDeprecationWarning)
    return tf_wraps.tri_to_vec(N)
