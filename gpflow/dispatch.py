from multipledispatch import dispatch, Dispatcher
from functools import partial

# By default multipledispatch uses a global namespace in multipledispatch.core.global_namespace
# We define our own GPflow namespace to avoid any conflict which may arise
gpflow_md_namespace = dict()
dispatch = partial(dispatch, namespace=gpflow_md_namespace)

conditional = Dispatcher('conditional')
sample_conditional = Dispatcher('sample_conditional')