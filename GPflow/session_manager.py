import tensorflow as tf

from .session_tracer import TracerSession
from ._settings import settings

def get_session(*args, **kwargs):
    # Pass session configuration options
    if 'config' not in kwargs:
        kwargs['config'] = tf.ConfigProto(**settings.session)
    if settings.profiling.dump_timeline:
        return TracerSession(*args, **kwargs)
    kwargs.pop("output_file_name", None)
    kwargs.pop("output_directory", None)
    kwargs.pop("each_time", None)
    return tf.Session(*args, **kwargs)
