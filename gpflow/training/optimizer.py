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
    def minimize(self, model, session=None, var_list=None, feed_dict=None,
                 maxiter=1000, initialize=True, anchor=True, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def _gen_var_list(model, var_list):
        var_list = var_list or []
        return list(set(model.trainable_tensors).union(var_list))

    @staticmethod
    def _gen_feed_dict(model, feed_dict):
        feed_dict = feed_dict or {}
        model_feeds = {} if model.feeds is None else model.feeds
        feed_dict.update(model_feeds)
        if feed_dict == {}:
            return None
        return feed_dict
