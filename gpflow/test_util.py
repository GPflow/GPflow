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


# pragma: no cover
# pylint: skip-file


import functools
import contextlib
import tensorflow as tf
import pytest
import os


@pytest.fixture
def session_tf():
    """
    Session creation pytest fixture.

    ```
    def test_simple(session_tf):
        tensor = tf.constant(1.0)
        result = session_tf.run(tensor)
        # ...
    ```

    In example above the test_simple is wrapped within graph and session created
    at `session_tf()` fixture. Session and graph are created per each pytest
    function where `session_tf` argument is used.
    """
    with session_context() as session:
        yield session


def cache_tensor(method):
    """
    Caches result for wrapped function wrt default TensorFlow graph.
    Whenever function is called under another default graph, execution will be
    performed. It does make sense cache tensors: build once, use multiple times
    per TensorFlow graph.

    Example:
    ```
    @cache_tensor
    def create_const():
        return tf.constant(1.0, name='wow')

    > const1 = create_const()
    > const2 = create_const()
    > const1 == const2
    True
    ```
    """
    cache = {}
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        graph = tf.get_default_graph()
        if graph not in cache:
            cache[graph] = method(*args, **kwargs)
        return cache[graph]
    return wrapper


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


def is_continuous_integration():
    ci = os.environ.get('CI', '').lower()
    return (ci == 'true') or (ci == '1')


def notebook_niter(n, test_n=2):
    return test_n if is_continuous_integration() else n

def notebook_range(n, test_n=2):
    return range(notebook_niter(n, test_n))

def notebook_list(lst, test_n=2):
    return lst[:test_n] if is_continuous_integration() else lst
