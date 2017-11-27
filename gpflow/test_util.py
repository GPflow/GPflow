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


class session_context(contextlib.ContextDecorator):
    def __init__(self, graph=None, close_on_exit=True, **kwargs):
        self.graph = graph
        self.close_on_exit = close_on_exit
        self.session = None
        self.session_args = kwargs

    def __enter__(self):
        graph = tf.Graph() if self.graph is None else self.graph
        session = tf.Session(graph=graph, **self.session_args)
        self.session = session
        session.__enter__()
        return session

    def __exit__(self, *exc):
        session = self.session
        session.__exit__(*exc)
        if self.close_on_exit:
            session.close()
        return False


class GPflowTestCase(tf.test.TestCase):
    """
    Wrapper for TensorFlow TestCase to avoid massive duplication of resetting
    Tensorflow Graph.
    """

    _multiprocess_can_split_ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_graph = tf.Graph()

    @contextlib.contextmanager
    def test_context(self, graph=None):
        graph = self.test_graph if graph is None else graph
        with graph.as_default(), self.test_session(graph=graph) as session:
            yield session
