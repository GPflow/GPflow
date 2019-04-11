# Copyright 2018 GPflow authors
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


from ..util import create_logger
from .features import InducingFeature

logger = create_logger()


class Mof(InducingFeature):
    """
    Class used to indicate that we are dealing with
    features that are used for multiple outputs.
    """
    pass


class SharedIndependentMof(Mof):
    """
    Same feature is used for each output.
    """
    def __init__(self, feature):
        Mof.__init__(self)
        self.feature = feature

    def __len__(self):
        return len(self.feature)


class SeparateIndependentMof(Mof):
    """
    A different feature is used for each output.
    Note: each feature should have the same number of points, M.
    """
    def __init__(self, features):
        Mof.__init__(self)
        self.features = features

    def __len__(self):
        return len(self.features[0])


class MixedKernelSharedMof(SharedIndependentMof):
    """
    This Mof is used in combination with the `SeparateMixedMok`.
    Using this feature with the `SeparateMixedMok` leads to the most efficient code.
    """
    pass

class MixedKernelSeparateMof(SeparateIndependentMof):
    """
    This Mof is used in combination with the `SeparateMixedMok`.
    Using this feature with the `SeparateMixedMok` leads to the most efficient code.
    """
    pass