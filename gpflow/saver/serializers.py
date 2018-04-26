# Copyright 2018 Artem Artemev @awav
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


import abc
from datetime import datetime

import h5py
import numpy as np

from .. import misc
from .context import Contexture


class BaseSerializer(Contexture, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def dump(self, pathname, data):
        pass

    @abc.abstractmethod
    def load(self, pathname):
        pass


class HDF5Serializer(BaseSerializer):
    def dump(self, pathname, data):
        with h5py.File(pathname) as h5file:
            meta = h5file.create_group('meta')
            date = datetime.now().isoformat() #TODO(@awav): py3.6 timespec='seconds'.
            version = misc.version()
            meta.create_dataset(name='date', data=date)
            meta.create_dataset(name='version', data=version)
            h5file.create_dataset(name='data', data=data)

    def load(self, pathname):
        with h5py.File(pathname) as h5file:
            return h5file['data'].value