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

import os
import warnings

import tensorflow as tf
from tensorflow.python.client import timeline

from . import settings


logger = settings.logger()


class _DefaultSessionKeeper:
    session = None


class TracerSession(tf.Session):
    def __init__(self, output_file_name=None, output_directory=None,
                 each_time=None, **kwargs):
        self.output_file_name = output_file_name
        self.output_directory = output_directory
        self.each_time = each_time
        self.local_run_metadata = None
        if self.each_time:
            logger.warn("Outputting a trace for each run. May result in large disk usage.")

        super(TracerSession, self).__init__(**kwargs)
        self.counter = 0
        self.profiler_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        if self.output_directory is not None:
            if os.path.isfile(self.output_directory):
                raise IOError("In tracer: given directory name is a file.")
            if not os.path.isdir(self.output_directory):
                os.mkdir(self.output_directory)

    def _trace_filename(self):
        """
        Creates trace filename.
        """
        dir_stub = ''
        if self.output_directory is not None:
            dir_stub = self.output_directory
        if self.each_time:
            filename = '{0}_{1}.json'.format(
                self.output_file_name, self.counter)
        else:
            filename = '{0}.json'.format(self.output_file_name)
        return os.path.join(dir_stub, filename)

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        # Make sure there is no disagreement doing this.
        if options is not None:
            if options.trace_level != self.profiler_options.trace_level:  # pragma: no cover
                raise ValueError(
                    'In profiler session. Inconsistent trace '
                    'level from run call')  # pragma: no cover
            self.profiler_options.update(options)  # pragma: no cover

        self.local_run_metadata = tf.RunMetadata()
        output = super(TracerSession, self).run(
            fetches, feed_dict=feed_dict,
            options=self.profiler_options,
            run_metadata=self.local_run_metadata)

        trace_time = timeline.Timeline(self.local_run_metadata.step_stats)
        ctf = trace_time.generate_chrome_trace_format()
        with open(self._trace_filename(), 'w') as trace_file:
            trace_file.write(ctf)

        if self.each_time:
            self.counter += 1

        return output


def reset_default_session(*args, **kwargs):
    _DefaultSessionKeeper.session = get_session(*args, **kwargs)


def reset_default_graph_and_session(*args, **kwargs):
    tf.reset_default_graph()
    reset_default_session(*args, **kwargs)


def get_default_session(*args, **kwargs):
    reset = kwargs.pop('reset', False)
    if reset or _DefaultSessionKeeper.session is None:
        _DefaultSessionKeeper.session = get_session(*args, **kwargs)
    return _DefaultSessionKeeper.session


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
