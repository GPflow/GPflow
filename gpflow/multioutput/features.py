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

from ..features import InducingPoints, InducingFeature


class Mof(InducingFeature):
    pass


class SharedIndependentMof(Mof):
    def __init__(self, feat):
        Mof.__init__(self)
        self.feat = feat

    def __len__(self):
        return len(self.feat)


class SeparateIndependentMof(Mof):
    def __init__(self, feat_list):
        Mof.__init__(self)
        self.feat_list = feat_list

    def __len__(self):
        return len(self.feat_list[0])


class MixedKernelSharedMof(SharedIndependentMof):
    pass