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

import tensorflow as tf

from .session_tracer import TracerSession
from ._settings import settings

def get_session(*args, **kwargs):
    """
    Pass session configuration options
    """
    if 'config' not in kwargs:
        kwargs['config'] = tf.ConfigProto(**settings.session)
    if settings.profiling.dump_timeline:
        def fill_kwargs(key, value):
            """
            Internal function for filling default None values with meaningful
            values from gpflow settings.
            """
            if kwargs.get(key) is None:
                kwargs[key] = value
        fill_kwargs('output_file_name', settings.profiling.output_file_name)
        fill_kwargs('output_directory', settings.profiling.output_directory)
        fill_kwargs('each_time', settings.profiling.each_time)
        return TracerSession(*args, **kwargs)
    kwargs.pop("output_file_name", None)
    kwargs.pop("output_directory", None)
    kwargs.pop("each_time", None)
    return tf.Session(*args, **kwargs)
