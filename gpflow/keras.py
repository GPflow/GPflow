# Copyright 2024 The GPflow Contributors. All Rights Reserved.
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

# GPflow currently uses Keras 2. Depending on the version of Tensorflow installed,
# this is available either at tf_keras or at tensorflow.keras. This module provides
# a helpful version-agnostic shortcut for importing this. Note though that importing specific
# identifiers can only be done via attribute lookups:
#
# >> from gpflow.keras import tf_keras
# >> Adam = tf_keras.optimizers.Adam
# >> # The following does NOT work
# >> from tf_keras.optimizers import Adam

try:
    import tf_keras
except ModuleNotFoundError:
    import tensorflow.keras as tf_keras

tf_keras = tf_keras

__all__ = ["tf_keras"]
