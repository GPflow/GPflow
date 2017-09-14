import functools
import tensorflow as tf

from .misc import GPflowError
from .misc import TF_FLOAT_TYPE
from .base import Build
from .params import Parameterized


class AutoFlow:

    def __init__(self, *args):
        """
        :param tf_arg_tuples: list of tuples, each of which can be
                              used to construct a tensorflow placeholder.
        """
        self._args = args

    def _setup_storage(self, store):
        store['arguments'] = [tf.placeholder(*a) for a in self._args]

    @staticmethod
    def storage_name(method):
        return ''.join(['_autoflow_store_', method.__name__])

    @staticmethod
    def name_scope_name(obj, method):
        return ''.join(['autoflow_', obj.name, '_', method.__name__])

    @staticmethod
    def get_storage(obj, name):
        return getattr(obj, name, default=None)

    @staticmethod
    def session_run(session, store, *args):
        feed_dict = dict(zip(store['arguments'], args))
        return session.run(store['result'], feed_dict=feed_dict)

    @staticmethod
    def build_method(method, store, obj):
        store['result'] = method(obj, *store['arguments'])

    def __call__(self, method):
        @functools.wraps(method)
        def runnable(obj, *args, **kwargs):
            if not isinstance(obj, Parameterized):
                raise ValueError('AutoFlow can decorate only Parameterized methods.')
            if obj.is_built_coherence(obj.graph) is Build.NO:
                raise GPflowError('Parameterized object must be built.')

            store_name = AutoFlow.storage_name(method)
            store = AutoFlow.get_storage(obj, store_name)
            session = kwargs.pop('session', None)
            session = obj.enquire_session(session=session)
            if not store:
                scope_name = AutoFlow.name_scope_name(obj, method)
                with session.graph.as_default(), tf.name_scope(scope_name):
                    self._setup_storage(store)
                    AutoFlow.build_method(method, store, obj)
            return self.session_run(session, store, *args)
        return runnable
