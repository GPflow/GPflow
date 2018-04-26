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

import abc


class IPrior(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def logp(self, x):
        """
        The log density of the prior as x

        All priors (for the moment) are univariate, so if x is a vector or an
        array, this is the sum of the log densities.
        """
        pass

    @abc.abstractmethod
    def sample(self, shape=(1,)):
        """
        A sample utility function for the prior.
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        """
        A short string to describe the prior at print time
        """
        pass


class ITransform(metaclass=abc.ABCMeta):
    """
    x is the unconstrained, free-space parameter, which can take any value.
    y is the constrained, "real units" parameter corresponding to the data.

    y = forward(x)
    x = backward(y)
    """

    @abc.abstractmethod
    def forward(self, x):
        """
        Map from the free-space to the variable space, using numpy
        """
        pass

    @abc.abstractmethod
    def backward(self, y):
        """
        Map from the variable-space to the free space, using numpy
        """
        pass


    @abc.abstractmethod
    def forward_tensor(self, x):
        """
        Map from the free-space to the variable space, using tensorflow
        """
        pass

    @abc.abstractmethod
    def backward_tensor(self, x):
        """
        Map from the variable-space to the free space, using tensorflow
        """
        pass

    @abc.abstractmethod
    def log_jacobian_tensor(self, x):
        """
        Return the log Jacobian of the forward_tensor mapping.

        Note that we *could* do this using a tf manipulation of
        self.forward_tensor, but tensorflow may have difficulty: it doesn't have a
        Jacobian at time of writing. We do this in the tests to make sure the
        implementation is correct.
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        """
        A short string describing the nature of the constraint
        """
        pass
