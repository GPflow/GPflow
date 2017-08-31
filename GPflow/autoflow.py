from functools import wraps
import tensorflow as tf


class AutoFlow:
    """
    This decorator-class is designed for use on methods of the Parameterized class
    (below).

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
    >>> result = m._session.run(graph,
                                feed_dict={x_tf:x,
                                           y_tf:y,
                                           m._free_vars:m.get_free_state()})

    Not only is the syntax cleaner, but multiple calls to the method will
    result in the graph being constructed only once.

    """

    def __init__(self, *tf_arg_tuples):
        # NB. TF arg_tuples is a list of tuples, each of which can be used to
        # construct a tf placeholder.
        self.tf_arg_tuples = tf_arg_tuples

    def __call__(self, tf_method):
        @wraps(tf_method)
        def runnable(instance, *np_args):
            storage_name = '_' + tf_method.__name__ + '_AF_storage'
            if hasattr(instance, storage_name):
                # the method has been compiled already, get things out of storage
                storage = getattr(instance, storage_name)
            else:
                # the method needs to be compiled.
                storage = {}  # an empty dict to keep things in
                setattr(instance, storage_name, storage)
                storage['graph'] = tf.Graph()
                # storage['session'] = tf.Session(graph=storage['graph'])
                storage['session'] = session.get_session(
                    graph=storage['graph'],
                    output_file_name=settings.profiling.output_file_name + "_" + tf_method.__name__,
                    output_directory=settings.profiling.output_directory,
                    each_time=settings.profiling.each_time
                )
                with storage['graph'].as_default():
                    storage['tf_args'] = [tf.placeholder(*a) for a in self.tf_arg_tuples]
                    storage['free_vars'] = tf.placeholder(float_type, [None])
                    instance.make_tf_array(storage['free_vars'])
                    with instance.tf_mode():
                        storage['tf_result'] = tf_method(instance, *storage['tf_args'])
                    storage['feed_dict_keys'] = instance.get_feed_dict_keys()
                    feed_dict = {}
                    instance.update_feed_dict(storage['feed_dict_keys'], feed_dict)
                    storage['session'].run(tf.global_variables_initializer(), feed_dict=feed_dict)
            feed_dict = dict(zip(storage['tf_args'], np_args))
            feed_dict[storage['free_vars']] = instance.get_free_state()
            instance.update_feed_dict(storage['feed_dict_keys'], feed_dict)
            return storage['session'].run(storage['tf_result'], feed_dict=feed_dict)

        return runnable
