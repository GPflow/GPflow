# Copyright 2016 alexggmatthews
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
import unittest

from testing.gpflow_testcase import GPflowTestCase
from gpflow.minibatch import SequenceIndices, MinibatchData
from gpflow.minibatch import ReplacementSampling, NoReplacementSampling


class TestSequentialManager(GPflowTestCase):
    def testA(self):
        minibatch_size = 3
        total_points = 5
        sequenceManager = SequenceIndices(minibatch_size,total_points)

        indecesA = sequenceManager.nextIndices()
        self.assertTrue((indecesA==np.arange(0,minibatch_size)).all())

        indecesB = sequenceManager.nextIndices()
        self.assertTrue((indecesB==np.array([3,4,0 ])).all())

    def testB(self):
        minibatch_size = 5
        total_points = 2
        sequenceManager = SequenceIndices(minibatch_size,total_points)

        indecesA = sequenceManager.nextIndices()
        targetIndicesA = np.array([0,1,0,1,0])

        self.assertTrue((indecesA==targetIndicesA).all())

        indecesB = sequenceManager.nextIndices()
        targetIndicesB = np.array([1,0,1,0,1])

        self.assertTrue((indecesB==targetIndicesB).all())

class TestRandomIndexManagers(GPflowTestCase):
    def checkUniformDist(self, indeces, nChoices):
        fTotalPoints = float(len(indeces))
        tolerance = 1e-2
        for possibleIndex in range(nChoices):
            empirical = indeces.count(possibleIndex) / fTotalPoints
            error = np.abs(empirical - 1./nChoices)
            if error > tolerance:
                return False
        return True

    def testReplacement(self):
        minibatch_size = 1000
        data_size = 3
        tolerance = 1e-2
        rs = ReplacementSampling(minibatch_size,data_size)
        indeces = rs.nextIndices().tolist()
        self.assertTrue(self.checkUniformDist(indeces,data_size))

    def testNoReplacement(self):
        mini_size_err = 5
        data_size_err = 3
        constructor = lambda : NoReplacementSampling(mini_size_err,
                                                     data_size_err)
        self.assertRaises(AssertionError,constructor)

        mini_size = 3
        data_size = 3
        nrs = NoReplacementSampling(mini_size,data_size)
        indeces = np.sort(nrs.nextIndices()).tolist()
        self.assertEqual(indeces,[0,1,2])

        one = 1
        nrsb = NoReplacementSampling(one,data_size)

        indecesOverall = []
        for repeatIndex in range(3000):
            indeces = nrsb.nextIndices().tolist()
            indecesOverall = indecesOverall + indeces
        self.assertTrue(self.checkUniformDist(indecesOverall,data_size))

class TestMinibatchData(GPflowTestCase):
    def setUp(self):
        self.nDataPoints = 10
        self.minibatch_size = 4
        self.dummyArray = np.atleast_2d(np.arange(self.nDataPoints)).T

    def testA(self):
        constructor = lambda : MinibatchData(self.dummyArray,
                                         self.minibatch_size,
                                         rng=None,
                                         batch_manager=[])
        self.assertRaises(NotImplementedError,constructor)

    def testB(self):
        sm = SequenceIndices(self.minibatch_size, self.nDataPoints)
        md = MinibatchData(self.dummyArray,
                           self.minibatch_size,
                           rng=None,
                           batch_manager=sm)
        output_string = 'test_out'
        key_dict = {md: output_string}
        feed_dict = {}
        md.update_feed_dict( key_dict, feed_dict )
        test_out_array = np.atleast_2d(np.arange(self.minibatch_size)).T
        test_dict = {output_string : test_out_array }
        self.assertEqual(feed_dict.keys(),test_dict.keys())
        self.assertTrue((feed_dict[output_string]==test_out_array).all())

    def testC(self):
        #Test the manager selection logic.
        md = MinibatchData(self.dummyArray,
                           self.minibatch_size,
                           rng=None)
        self.assertEqual(md.index_manager.__class__, ReplacementSampling)


        md = MinibatchData(self.dummyArray,
                           self.minibatch_size*2,
                           rng=None)
        self.assertEqual(md.index_manager.__class__, NoReplacementSampling)

if __name__ == "__main__":
    unittest.main()
