# Copyright 2022 The GPflow Contributors. All Rights Reserved.
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
"""
Concrete definitions of our metrics.
"""
from benchmark.metric_api import MetricOrientation, make_metric

n_training_iterations = make_metric(
    name="n_training_iterations",
    pretty_name="Training iterations",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit=None,
)


training_time = make_metric(
    name="training_time",
    pretty_name="Time to train",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit="s",
)


training_iteration_time = make_metric(
    name="training_iteration_time",
    pretty_name="Time run one iteration",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit="s",
)


prediction_time = make_metric(
    name="prediction_time",
    pretty_name="Time to predict Y",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit="s",
)


nlpd = make_metric(
    name="nlpd",
    pretty_name="Negative Log Predictive Density",
    lower_bound=None,
    upper_bound=None,
    orientation=MetricOrientation.GREATER_IS_BETTER,
    unit=None,
)


mae = make_metric(
    name="mae",
    pretty_name="Mean Absolute Error",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit=None,
)


rmse = make_metric(
    name="rmse",
    pretty_name="Root Mean Squared Error",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit=None,
)


posterior_build_time = make_metric(
    name="posterior_build_time",
    pretty_name="Time to build posterior",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit="s",
)


posterior_prediction_time = make_metric(
    name="posterior_prediction_time",
    pretty_name="Time to predict test Y (posterior)",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit="s",
)


posterior_nlpd = make_metric(
    name="posterior_nlpd",
    pretty_name="Negative Log Predictive Density (posterior)",
    lower_bound=None,
    upper_bound=None,
    orientation=MetricOrientation.GREATER_IS_BETTER,
    unit=None,
)


posterior_mae = make_metric(
    name="posterior_mae",
    pretty_name="Mean Absolute Error (posterior)",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit=None,
)


posterior_rmse = make_metric(
    name="posterior_rmse",
    pretty_name="Root Mean Squared Error (posterior)",
    lower_bound=0,
    upper_bound=None,
    orientation=MetricOrientation.LOWER_IS_BETTER,
    unit=None,
)
