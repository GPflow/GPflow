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

# pylint: disable=no-self-use
# pylint: disable=too-few-public-methods

import abc


class Optimizer:
    @abc.abstractmethod
    def make_optimize_tensor(self, model, session=None, var_list=None, **kwargs):
        """
        Make optimization tensor.
        The `make_optimize_tensor` method builds optimization tensor and initializes
        all necessary variables created by optimizer.

            :param model: GPflow model.
            :param session: Tensorflow session.
            :param var_list: List of variables for training.
            :param kwargs: Dictionary of extra parameters necessary for building
                optimizer tensor.
            :return: Tensorflow optimization tensor or operation.
        """
        pass

    @abc.abstractmethod
    def minimize(self, model, session=None, var_list=None, feed_dict=None,
                 maxiter=1000, initialize=True, anchor=True, step_callback=None, **kwargs):
        """
        :param model: GPflow model with objective tensor.
        :param session: Session where optimization will be run.
        :param var_list: List of extra variables which should be trained during optimization.
        :param feed_dict: Feed dictionary of tensors passed to session run method.
        :param maxiter: Number of run interation.
        :param initialize: If `True` model parameters will be re-initialized even if they were
            initialized before in the specified session.
        :param anchor: If `True` trained variable values computed during optimization at
            particular session will be synchronized with internal parameter values.
        :param step_callback: A callback function to execute at each optimization step.
            Callback takes an arbitrary list of arguments. Input arguments depend on
            interface implementation.
        :param kwargs: This is a dictionary of extra parameters for session run method.

        """
        raise NotImplementedError()

    @staticmethod
    def _gen_var_list(model, var_list):
        var_list = var_list or []
        all_vars = list(set(model.trainable_tensors).union(var_list))
        return sorted(all_vars, key=lambda x: x.name)

    @staticmethod
    def _gen_feed_dict(model, feed_dict):
        feed_dict = feed_dict or {}
        model_feeds = {} if model.feeds is None else model.feeds
        feed_dict.update(model_feeds)
        if feed_dict == {}:
            return None
        return feed_dict
