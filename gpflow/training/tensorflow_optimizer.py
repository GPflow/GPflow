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

import sys

import tensorflow as tf

from . import optimizer
from .. import misc
from ..actions import Optimization
from ..models.model import Model

_REGISTERED_TENSORFLOW_OPTIMIZERS = {}


class _TensorFlowOptimizer(optimizer.Optimizer):
    def __init__(self, *args, **kwargs):
        name = self.__class__.__name__
        tf_optimizer = _get_registered_optimizer(name)
        self._model = None
        super().__init__()
        self._optimizer = tf_optimizer(*args, **kwargs)
        self._minimize_operation = None
    
    def make_optimize_tensor(self, model, session=None, var_list=None, **kwargs):
        """
        Make Tensorflow optimization tensor.
        This method builds optimization tensor and initializes all necessary variables
        created by optimizer.

            :param model: GPflow model.
            :param session: Tensorflow session.
            :param var_list: List of variables for training.
            :param kwargs: Dictionary of extra parameters passed to Tensorflow
                optimizer's minimize method.
            :return: Tensorflow optimization tensor or operation.
        """
        session = model.enquire_session(session)
        objective = model.objective
        full_var_list = self._gen_var_list(model, var_list)
        # Create optimizer variables before initialization.
        with session.as_default():
            minimize = self.optimizer.minimize(objective, var_list=full_var_list, **kwargs)
            model.initialize(session=session)
            self._initialize_optimizer(session)
            return minimize
    
    def make_optimize_action(self, model, session=None, var_list=None, **kwargs):
        """
        Build Optimization action task with Tensorflow optimizer.

            :param model: GPflow model.
            :param session: Tensorflow session.
            :param var_list: List of Tensorflow variables to train.
            :param feed_dict: Tensorflow feed_dict dictionary.
            :param kwargs: Extra parameters passed to `make_optimize_tensor`.
            :return: Optimization action.
        """
        if model is None or not isinstance(model, Model):
            raise ValueError('Unknown type passed for optimization.')
        session = model.enquire_session(session)
        feed_dict = kwargs.pop('feed_dict', None)
        feed_dict_update = self._gen_feed_dict(model, feed_dict)
        run_kwargs = {} if feed_dict_update is None else {'feed_dict': feed_dict_update}
        optimizer_tensor = self.make_optimize_tensor(model, session, var_list=var_list, **kwargs)
        opt = Optimization()
        opt.with_optimizer(self)
        opt.with_model(model)
        opt.with_optimizer_tensor(optimizer_tensor)
        opt.with_run_kwargs(**run_kwargs)
        return opt
    
    def minimize(self, model, session=None, var_list=None, feed_dict=None,
                 maxiter=1000, initialize=False, anchor=True, step_callback=None, **kwargs):
        """
        Minimizes objective function of the model.

        :param model: GPflow model with objective tensor.
        :param session: Session where optimization will be run.
        :param var_list: List of extra variables which should be trained during optimization.
        :param feed_dict: Feed dictionary of tensors passed to session run method.
        :param maxiter: Number of run interation.
        :param initialize: If `True` model parameters will be re-initialized even if they were
            initialized before for gotten session.
        :param anchor: If `True` trained variable values computed during optimization at
            particular session will be synchronized with internal parameter values.
        :param step_callback: A callback function to execute at each optimization step.
            The callback should accept variable argument list, where first argument is
            optimization step number.
        :type step_callback: Callable[[], None]
        :param kwargs: This is a dictionary of extra parameters for session run method.
        """

        if model is None or not isinstance(model, Model):
            raise ValueError('The `model` argument must be a GPflow model.')

        opt = self.make_optimize_action(model,
            session=session,
            var_list=var_list,
            feed_dict=feed_dict, **kwargs)

        self._model = opt.model
        self._minimize_operation = opt.optimizer_tensor

        session = model.enquire_session(session)
        with session.as_default():
            for step in range(maxiter):
                opt()
                if step_callback is not None:
                    step_callback(step)

        if anchor:
            opt.model.anchor(session)

    def _initialize_optimizer(self, session: tf.Session):
        var_list = self.optimizer.variables()
        misc.initialize_variables(var_list, session=session, force=False)

    @property
    def minimize_operation(self):
        return self._minimize_operation

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @model.setter
    def model(self, value):
        self._model = value
        self._optimizer = None
        self._minimize_operation = None


def _get_registered_optimizer(name):
    tf_optimizer = _REGISTERED_TENSORFLOW_OPTIMIZERS.get(name, None)
    if tf_optimizer is None:
        raise TypeError('Optimizer not found.')
    return tf_optimizer


def _register_optimizer(name, optimizer_type):
    if optimizer_type.__base__ is not tf.train.Optimizer:
        raise ValueError('Wrong TensorFlow optimizer type passed: "{0}".'
                         .format(optimizer_type))
    gp_optimizer = type(name, (_TensorFlowOptimizer, ), {})
    _REGISTERED_TENSORFLOW_OPTIMIZERS[name] = optimizer_type
    module = sys.modules[__name__]
    setattr(module, name, gp_optimizer)


# Create GPflow optimizer classes with same names as TensorFlow optimizers
for key, train_type in tf.train.__dict__.items():
    suffix = 'Optimizer'
    if key != suffix and key.endswith(suffix):
        _register_optimizer(key, train_type)


__all__ = list(_REGISTERED_TENSORFLOW_OPTIMIZERS.keys())
