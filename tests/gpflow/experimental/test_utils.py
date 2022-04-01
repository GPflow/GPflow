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
from _pytest.recwarn import WarningsRecorder

from gpflow.experimental.utils import experimental


def test_experimental(recwarn: WarningsRecorder) -> None:
    @experimental
    def f() -> None:
        pass

    assert len(recwarn) == 0  # pylint: disable=len-as-condition
    f()
    assert len(recwarn) == 1
    f()
    assert len(recwarn) == 1
