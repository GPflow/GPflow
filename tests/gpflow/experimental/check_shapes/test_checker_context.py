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
Unit test for utilities for accessing the ShapeChecker from a wrapping `check_shapes` decorator.
"""
import asyncio
import threading
from random import random
from time import sleep

import pytest

from gpflow.experimental.check_shapes import ShapeChecker, check_shape, get_shape_checker
from gpflow.experimental.check_shapes.checker_context import set_shape_checker
from gpflow.experimental.check_shapes.exceptions import ShapeMismatchError

from .utils import current_line, t


def test_set_get_shape_checker() -> None:
    checker0 = ShapeChecker()
    with set_shape_checker(checker0):
        assert checker0 is get_shape_checker()

        checker1 = ShapeChecker()
        with set_shape_checker(checker1):
            assert checker1 is get_shape_checker()

        assert checker0 is get_shape_checker()

        with pytest.raises(ValueError):
            with set_shape_checker(checker1):
                assert checker1 is get_shape_checker()
                raise ValueError("test error")

        assert checker0 is get_shape_checker()


def test_set_get_shape_checker__multithreaded() -> None:
    errors = []

    def perform_test() -> None:
        try:
            for _ in range(25):
                checker0 = ShapeChecker()
                with set_shape_checker(checker0):

                    sleep(random() / 1000)
                    assert checker0 is get_shape_checker()

                    checker1 = ShapeChecker()
                    with set_shape_checker(checker1):

                        sleep(random() / 1000)
                        assert checker1 is get_shape_checker()

                    sleep(random() / 1000)
                    assert checker0 is get_shape_checker()

                    with pytest.raises(ValueError):
                        with set_shape_checker(checker1):

                            sleep(random() / 1000)
                            assert checker1 is get_shape_checker()
                            raise ValueError("test error")

                    sleep(random() / 1000)
                    assert checker0 is get_shape_checker()
        except Exception as e:  # pylint: disable=broad-except
            errors.append(e)

    threads = [threading.Thread(target=perform_test) for _ in range(25)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors


def test_set_get_shape_checker__async() -> None:
    errors = []

    async def perform_test() -> None:
        try:
            for _ in range(25):
                checker0 = ShapeChecker()
                with set_shape_checker(checker0):

                    await asyncio.sleep(random() / 1000)
                    assert checker0 is get_shape_checker()

                    checker1 = ShapeChecker()
                    with set_shape_checker(checker1):

                        await asyncio.sleep(random() / 1000)
                        assert checker1 is get_shape_checker()

                    await asyncio.sleep(random() / 1000)
                    assert checker0 is get_shape_checker()

                    with pytest.raises(ValueError):
                        with set_shape_checker(checker1):

                            await asyncio.sleep(random() / 1000)
                            assert checker1 is get_shape_checker()
                            raise ValueError("test error")

                    await asyncio.sleep(random() / 1000)
                    assert checker0 is get_shape_checker()
        except Exception as e:  # pylint: disable=broad-except
            errors.append(e)

    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(perform_test()) for _ in range(25)]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
    asyncio.set_event_loop(None)

    assert not errors


def test_check_shape() -> None:
    checker = ShapeChecker()

    with set_shape_checker(checker):
        check_shape(t(1, 2), "[d1, d2]")
        check_shape(t(2, 3), "[d2, d3]")

    with set_shape_checker(checker):
        check_shape(t(3, 4), "[d3, d4]")

        with pytest.raises(ShapeMismatchError):
            check_shape(t(1, 4), "[d3, d4]")


def test_check_shape__error_message() -> None:
    checker = ShapeChecker()

    with set_shape_checker(checker):
        call_line_1 = current_line() + 1
        check_shape(t(1, 2), "[d1, d2]")

        with pytest.raises(ShapeMismatchError) as e:
            call_line_2 = current_line() + 1
            check_shape(t(3, 3), "[d2, d3]")

    (message,) = e.value.args
    assert (
        f"""
Tensor shape mismatch.
  check_shape called at: {__file__}:{call_line_1}
    Expected: [d1, d2]
    Actual:   [1, 2]
  check_shape called at: {__file__}:{call_line_2}
    Expected: [d2, d3]
    Actual:   [3, 3]
"""
        == message
    )
