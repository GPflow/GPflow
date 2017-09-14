import functools
import tensorflow as tf

from .misc import FLOAT_TYPE


class AutoFlow:
    """
    This decorator-class is designed for use on methods of the Parameterized class
    (below). It extends wrapped method argument list with `session` and `graph`
    parameters and allows you to integrate GPflow computation into your existing
    graph.

    The idea is that methods that compute relevant quantities (such as
    predictions) can define a tf graph which we automatically run when the
    (decorated) function is called.

    The syntax looks like:

    >>> class MyClass(Parameterized):
    >>>
    >>>   @AutoFlow((tf.float64), (tf.float64))
    >>>   def my_method(self, x, y):
    >>>       #compute something, returning a tf graph.
    >>>       return tf.foo(self.baz, x, y)

    >>> m = MyClass()
    >>> x = np.random.randn(3,1)
    >>> y = np.random.randn(3,1)
    >>> result = my_method(x, y)

    Now the output of the method call is the _result_ of the computation,
    equivalent to

    >>> m = MyModel()
    >>> x = np.random.randn(3,1)
    >>> y = np.random.randn(3,1)
    >>> x_tf = tf.placeholder(tf.float64)
    >>> y_tf = tf.placeholder(tf.float64)
    >>> with m.tf_mode():
    >>>     graph = tf.foo(m.baz, x_tf, y_tf)
    >>> result = m.session.run(graph, feed_dict={
                     x_tf:x, y_tf:y, m._free_vars:m.get_free_state()})
    Not only is the syntax cleaner, but multiple calls to the method will
    result in the graph being constructed only once.

    """

    def __init__(self, *tf_arg_tuples):
        """
        :param tf_arg_tuples: list of tuples, each of which can be
                              used to construct a tensorflow placeholder.
        """
        self.tf_arg_tuples = tf_arg_tuples

    def __call__(self, tf_method):
        @functools.wraps(tf_method)
        def runnable(instance, *np_args, **kwargs):

            if not set(kwargs.keys()).issubset(['session', 'graph']):
                raise ValueError('Unknown arguments passed.')

            session = kwargs.get('session')
            graph = kwargs.get('graph')

            if (session and graph) and (session.graph is not graph):
                raise ValueError(
                    'Ambiguous session and graph parameters passed')

            storage_name = ''.join(['_', tf_method.__name__, '_AF_storage'])
            storage, storage_session, storage_graph = None, None, None
            if hasattr(instance, storage_name):
                storage = getattr(instance, storage_name)
                storage_session = storage['session']
                storage_graph = storage_session.graph

            def get_session(session_, graph_):
                default_session = tf.get_default_session()
                if session_ is not None:
                    return session_
                elif default_session is not None and (
                        graph_ is None or default_session.graph is graph_):
                    return default_session
                filename = ''.join([settings.profiling.output_file_name, '_', tf_method.__name__])
                return session_mngr.get_session(graph=graph_, output_file_name=filename)

            def tf_init_feed_dict(instance_, storage_, session_):
                feed_dict = {}
                feed_dict_keys = storage_['feed_dict_keys']
                instance_.update_feed_dict(feed_dict_keys, feed_dict)
                init = tf.global_variables_initializer()
                session_.run(init, feed_dict=feed_dict)

            if storage is not None and (
                    (session is None and graph is None) or (storage_session is session)):
                # the method has been compiled already, get things out of storage
                session = storage_session
                graph = session.graph
            elif storage is not None and (session is None and storage_graph is graph):
                # 1. either use new session or default one
                # 2. initialize variables for chosen session
                session = get_session(None, graph)
                with graph.as_default():
                    tf_init_feed_dict(instance, storage, session)
            else:
                # the method needs to be compiled.
                storage = {}  # an empty dict to keep things in
                setattr(instance, storage_name, storage)
                storage = getattr(instance, storage_name)
                session = get_session(session, graph)
                graph = session.graph
                with graph.as_default():
                    tf_args = [tf.placeholder(*a) for a in self.tf_arg_tuples]
                    storage['tf_args'] = tf_args
                    storage['free_vars'] = tf.placeholder(FLOAT_TYPE, [None])
                    instance.make_tf_array(storage['free_vars'])
                    with instance.tf_mode():
                        storage['tf_result'] = tf_method(instance, *tf_args)
                    storage['feed_dict_keys'] = instance.get_feed_dict_keys()
                    tf_init_feed_dict(instance, storage, session)

            storage['session'] = session
            feed_dict = dict(zip(storage['tf_args'], np_args))
            feed_dict[storage['free_vars']] = instance.get_free_state()
            instance.update_feed_dict(storage['feed_dict_keys'], feed_dict)
            return session.run(storage['tf_result'], feed_dict=feed_dict)

        return runnable
