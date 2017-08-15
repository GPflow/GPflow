# Copyright 2016 James Hensman
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
from functools import wraps


class NameScoped(object):
    """
    A decorator for functions, so that they can be executed within tensorflow
    scopes. Usage:

    >>> @NameScoped('foo_scope'):
    >>> def foobar(x, y=3):
    >>>     return x + y

    or

    >>> def foobaz(x, y=4):
        return x + y
    >>> myfunc = NameScoped('my_fave_scope')(foobaz)
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, f):
        @wraps(f)
        def runnable(*args, **kwargs):
            with tf.name_scope(self.name):
                return f(*args, **kwargs)
        return runnable
