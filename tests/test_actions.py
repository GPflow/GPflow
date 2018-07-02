# Copyright 2018 the GPflow authors.
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


import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow as gp

@pytest.fixture
def loop():
    return gp.actions.Loop([]).with_settings(stop=10)

def test_loop_simple(loop):
    iters = range(loop.stop)
    def fun(ctx):
        assert iters[ctx.iteration] == ctx.iteration

    # Single actions
    loop.with_action(fun)()
    assert loop.iteration == loop.stop

    # Multiple actions
    loop.with_action([fun, fun])()
    assert loop.iteration == loop.stop


def test_loop_condition_continue(loop):
    def continue_loop(ctx):
        raise gp.actions.Loop.Continue
    number = [0]
    def count(ctx):
        number[0] += 1
    cond = gp.actions.Condition(lambda ctx: ctx.iteration % 2, continue_loop, count)
    loop.with_action(cond)()
    assert loop.iteration == loop.stop
    assert number[0] == (loop.stop / 2)


def test_loop_condition_break(loop):
    def break_loop(ctx):
        raise gp.actions.Loop.Break
    stop = 5
    number = [0]
    def count(ctx):
        number[0] += 1
    cond = gp.actions.Condition(lambda ctx: ctx.iteration < stop, count, break_loop)
    loop.with_action(cond)()
    assert loop.iteration == stop
    assert number[0] == stop
    
    