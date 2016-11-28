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
from GPflow.svgp import SequenceIndeces, MinibatchData

class TestSequentialManager(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
     
    def testA(self):
		minibatch_size = 3
		total_points = 5
		sequenceManager = SequenceIndeces(minibatch_size,total_points)
		
		indecesA = sequenceManager.nextIndeces()
		self.assertTrue((indecesA==np.arange(0,minibatch_size)).all())

		indecesB = sequenceManager.nextIndeces()
		self.assertTrue((indecesB==np.array([3,4,0 ])).all())
	
    def testB(self):
		minibatch_size = 5
		total_points = 2
		sequenceManager = SequenceIndeces(minibatch_size,total_points)
		
		indecesA = sequenceManager.nextIndeces()
		targetIndecesA = np.array([0,1,0,1,0])
			
		self.assertTrue((indecesA==targetIndecesA).all())

		indecesB = sequenceManager.nextIndeces()
		targetIndecesB = np.array([1,0,1,0,1])
			
		self.assertTrue((indecesB==targetIndecesB).all())

class TestMinibatchData(unittest.TestCase):
	def setUp(self):
		tf.reset_default_graph()
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
		sm = SequenceIndeces(self.minibatch_size, self.nDataPoints)
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
		
if __name__ == "__main__":
    unittest.main()
