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


import contextlib
import tensorflow as tf


class GPflowTestCase(tf.test.TestCase):
    """
    Wrapper for TestCase to avoid massive duplication of resetting
    Tensorflow Graph.
    """

    _multiprocess_can_split_ = True

    def __init__(self, *args, **kwargs):
        super(GPflowTestCase, self).__init__(*args, **kwargs)
        self.test_graph = None

    @contextlib.contextmanager
    def test_context(self, graph=None):
        graph = graph if graph else self.test_graph
        if graph is None:
            graph = tf.Graph()
            with graph.as_default(), self.test_session(graph=graph) as session:
                yield session
        else:
            with self.test_session(graph=graph) as session:
                yield session

    def tearDown(self):
        tf.reset_default_graph()
        super(GPflowTestCase, self).tearDown()
