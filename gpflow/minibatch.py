# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
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

import numpy as np
from .param import DataHolder


class IndexManager(object):
    """
    Base clase for methods of batch indexing data.
    rng is an instance of np.random.RandomState, defaults to seed 0.
    """
    def __init__(self, minibatch_size, total_points, rng=None):
        self.minibatch_size = minibatch_size
        self.total_points = total_points
        self.rng = rng or np.random.RandomState(0)

    def nextIndices(self):
        raise NotImplementedError


class ReplacementSampling(IndexManager):
    def nextIndices(self):
        return self.rng.randint(self.total_points,
                                size=self.minibatch_size)


class NoReplacementSampling(IndexManager):
    def __init__(self, minibatch_size, total_points, rng=None):
        # Can't sample without replacement is minibatch_size is larger
        # than total_points
        assert(minibatch_size <= total_points)
        IndexManager.__init__(self, minibatch_size, total_points, rng)

    def nextIndices(self):
        permutation = self.rng.permutation(self.total_points)
        return permutation[:self.minibatch_size]


class SequenceIndices(IndexManager):
    """
    A class that maintains the state necessary to manage
    sequential indexing of data holders.
    """
    def __init__(self, minibatch_size, total_points, rng=None):
        self.counter = 0
        IndexManager.__init__(self, minibatch_size, total_points, rng)

    def nextIndices(self):
        """
        Written so that if total_points
        changes this will still work
        """
        firstIndex = self.counter
        lastIndex = self.counter + self.minibatch_size
        self.counter = lastIndex % self.total_points
        return np.arange(firstIndex, lastIndex) % self.total_points


class MinibatchData(DataHolder):
    """
    A special DataHolder class which feeds a minibatch
    to tensorflow via update_feed_dict().
    """

    # List of valid specifiers for generation methods.
    _generation_methods = ['replace', 'noreplace', 'sequential']

    def __init__(self, array, minibatch_size, rng=None, batch_manager=None):
        """
        array is a numpy array of data.
        minibatch_size (int) is the size of the minibatch

        batch_manager specified data sampling scheme and is a subclass
        of IndexManager.

        Note: you may want to randomize the order of the data
        if using sequential generation.
        """
        DataHolder.__init__(self, array, on_shape_change='pass')
        total_points = self._array.shape[0]
        self.parseGenerationMethod(batch_manager,
                                   minibatch_size,
                                   total_points,
                                   rng)

    def parseGenerationMethod(self,
                              input_batch_manager,
                              minibatch_size,
                              total_points,
                              rng):
        # Logic for default behaviour.
        # When minibatch_size is a small fraction of total_point
        # ReplacementSampling should give similar results to
        # NoReplacementSampling and the former can be much faster.
        if input_batch_manager is None:
            fraction = float(minibatch_size) / float(total_points)
            if fraction < 0.5:
                self.index_manager = ReplacementSampling(minibatch_size,
                                                         total_points,
                                                         rng)
            else:
                self.index_manager = NoReplacementSampling(minibatch_size,
                                                           total_points,
                                                           rng)
        else:  # Explicitly specified behaviour.
            if input_batch_manager.__class__ not in IndexManager.__subclasses__():
                raise NotImplementedError
            self.index_manager = input_batch_manager

    def update_feed_dict(self, key_dict, feed_dict):
        next_indices = self.index_manager.nextIndices()
        feed_dict[key_dict[self]] = self._array[next_indices]
