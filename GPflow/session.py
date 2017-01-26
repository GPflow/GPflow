import os
import warnings
import tensorflow as tf
from tensorflow.python.client import timeline
from ._settings import settings


class TracerSession(tf.Session):
    def __init__(self, output_file_name, output_directory, each_time, **kwargs):
        self.output_file_name = output_file_name
        self.output_directory = output_directory
        self.eachTime = each_time
        self.local_run_metadata = None
        if self.eachTime:
            warnings.warn("Outputting a trace for each run. May result in large disk usage.")

        super(TracerSession, self).__init__(**kwargs)
        self.counter = 0
        self.profiler_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        if self.output_directory is not None:
            if os.path.isfile(self.output_directory):
                raise IOError("In tracer: given directory name is a file.")
            if not (os.path.isdir(self.output_directory)):
                os.mkdir(self.output_directory)

    def get_filename(self):
        if self.output_directory is not None:
            dir_stub = self.output_directory
        else:
            dir_stub = ''
        if self.eachTime:
            return os.path.join(dir_stub, self.output_file_name + '_' + str(self.counter) + '.json')
        else:
            return os.path.join(dir_stub, self.output_file_name + '.json')

    def run(self, fetches, feed_dict=None, options=None):
        # Make sure there is no disagreement doing this.
        if options is not None:
            if options.trace_level != self.profiler_options.trace_level:
                raise ValueError('In profiler session. Inconsistent trace level from run call')
            self.profiler_options.update(options)

        self.local_run_metadata = tf.RunMetadata()
        output = super(TracerSession, self).run(fetches, feed_dict=feed_dict, options=self.profiler_options,
                                                run_metadata=self.local_run_metadata)

        tl = timeline.Timeline(self.local_run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(self.get_filename(), 'w') as f:
            f.write(ctf)

        if self.eachTime:
            self.counter += 1

        return output


def get_session(*args, **kwargs):
    if settings.profiling.dump_timeline:
        return TracerSession(*args, **kwargs)
    else:
        kwargs.pop("output_file_name", None)
        kwargs.pop("output_directory", None)
        kwargs.pop("each_time", None)
        return tf.Session(*args, **kwargs)
