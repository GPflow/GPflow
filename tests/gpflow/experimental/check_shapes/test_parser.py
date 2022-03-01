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

# pylint: disable=unused-argument  # Bunch of fake functions below has unused arguments.

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest

from gpflow.experimental.check_shapes.parser import parse_and_rewrite_docstring, parse_argument_spec
from gpflow.experimental.check_shapes.specs import ParsedArgumentSpec

from .utils import make_argument_ref, make_shape_spec, varrank


@dataclass
class TestData:
    test_id: str
    argument_spec_strs: Tuple[str, ...]
    expected_specs: Tuple[ParsedArgumentSpec, ...]
    doc: Optional[str]
    expected_doc: Optional[str]

    def __str__(self) -> str:
        return self.test_id


_TEST_DATA = [
    TestData(
        "constant_dimensions",
        (
            "a: [2, 3]",
            "b: [2, 4]",
            "return: [3, 4]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("a"),
                make_shape_spec(2, 3),
            ),
            ParsedArgumentSpec(
                make_argument_ref("b"),
                make_shape_spec(2, 4),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return"),
                make_shape_spec(3, 4),
            ),
        ),
        """
        :param a: Parameter a.
        :param b: Parameter b.
        :returns: Return value.
        """,
        """
        :param a:
            * **a** has shape [2, 3].

            Parameter a.
        :param b:
            * **b** has shape [2, 4].

            Parameter b.
        :returns:
            * **return** has shape [3, 4].

            Return value.
        """,
    ),
    TestData(
        "variable_dimensions",
        (
            "a: [d1, d2]",
            "b: [d1, d3]",
            "return: [d2, d3]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("a"),
                make_shape_spec("d1", "d2"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("b"),
                make_shape_spec("d1", "d3"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return"),
                make_shape_spec("d2", "d3"),
            ),
        ),
        """
        :param a: Parameter a.
        :param b: Parameter b.
        :returns: Return value.
        """,
        """
        :param a:
            * **a** has shape [*d1*, *d2*].

            Parameter a.
        :param b:
            * **b** has shape [*d1*, *d3*].

            Parameter b.
        :returns:
            * **return** has shape [*d2*, *d3*].

            Return value.
        """,
    ),
    TestData(
        "variable_rank",
        (
            "a: [*ds]",
            "b: [ds..., d1]",
            "c: [d1, ds..., d2]",
            "d: [d1, ds...]",
            "return: [*ds, d1, d2]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("a"),
                make_shape_spec(varrank("ds")),
            ),
            ParsedArgumentSpec(
                make_argument_ref("b"),
                make_shape_spec(varrank("ds"), "d1"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("c"),
                make_shape_spec("d1", varrank("ds"), "d2"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("d"),
                make_shape_spec("d1", varrank("ds")),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return"),
                make_shape_spec(varrank("ds"), "d1", "d2"),
            ),
        ),
        """
        :param a: Parameter a.
        :param b: Parameter b.
        :param c: Parameter c.
        :param d: Parameter d.
        :returns: Return value.
        """,
        """
        :param a:
            * **a** has shape [*ds*...].

            Parameter a.
        :param b:
            * **b** has shape [*ds*..., *d1*].

            Parameter b.
        :param c:
            * **c** has shape [*d1*, *ds*..., *d2*].

            Parameter c.
        :param d:
            * **d** has shape [*d1*, *ds*...].

            Parameter d.
        :returns:
            * **return** has shape [*ds*..., *d1*, *d2*].

            Return value.
        """,
    ),
    TestData(
        "scalars",
        (
            "a: []",
            "b: []",
            "return: []",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("a"),
                make_shape_spec(),
            ),
            ParsedArgumentSpec(
                make_argument_ref("b"),
                make_shape_spec(),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return"),
                make_shape_spec(),
            ),
        ),
        """
        :param a: Parameter a.
        :param b: Parameter b.
        :returns: Return value.
        """,
        """
        :param a:
            * **a** has shape [].

            Parameter a.
        :param b:
            * **b** has shape [].

            Parameter b.
        :returns:
            * **return** has shape [].

            Return value.
        """,
    ),
    TestData(
        "argument_refs",
        (
            "x.ins[0]: [a_batch..., 1]",
            "x.ins[1]: [b_batch..., 2]",
            "return[0].out: [a_batch..., 3]",
            "return[1].out: [b_batch..., 4]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("x", "ins", 0),
                make_shape_spec(varrank("a_batch"), 1),
            ),
            ParsedArgumentSpec(
                make_argument_ref("x", "ins", 1),
                make_shape_spec(varrank("b_batch"), 2),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 0, "out"),
                make_shape_spec(varrank("a_batch"), 3),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 1, "out"),
                make_shape_spec(varrank("b_batch"), 4),
            ),
        ),
        """
        :param x: Parameter x.
        :returns: Return value.
        """,
        """
        :param x:
            * **x.ins[0]** has shape [*a_batch*..., 1].
            * **x.ins[1]** has shape [*b_batch*..., 2].

            Parameter x.
        :returns:
            * **return[0].out** has shape [*a_batch*..., 3].
            * **return[1].out** has shape [*b_batch*..., 4].

            Return value.
        """,
    ),
    TestData(
        "no_docstring",
        (
            "a: [d1, d2]",
            "b: [d1, d3]",
            "return: [d2, d3]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("a"),
                make_shape_spec("d1", "d2"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("b"),
                make_shape_spec("d1", "d3"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return"),
                make_shape_spec("d2", "d3"),
            ),
        ),
        None,
        None,
    ),
    TestData(
        "partial_docstring",
        (
            "a: [d1, d2]",
            "b: [d1, d3]",
            "return: [d2, d3]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("a"),
                make_shape_spec("d1", "d2"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("b"),
                make_shape_spec("d1", "d3"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return"),
                make_shape_spec("d2", "d3"),
            ),
        ),
        """
        :param b: Parameter b.
        """,
        """
        :param b:
            * **b** has shape [*d1*, *d3*].

            Parameter b.
        """,
    ),
    TestData(
        "no_indent",
        (
            "a: [d1, d2]",
            "b: [d1, d3]",
            "return: [d2, d3]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("a"),
                make_shape_spec("d1", "d2"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("b"),
                make_shape_spec("d1", "d3"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return"),
                make_shape_spec("d2", "d3"),
            ),
        ),
        """:param b: Parameter b.""",
        """:param b:
    * **b** has shape [*d1*, *d3*].

    Parameter b.""",
    ),
    TestData(
        "other_info_fields",
        (
            "a: [batch..., n_features]",
            "return: [batch..., 1]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("a"),
                make_shape_spec(varrank("batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return"),
                make_shape_spec(varrank("batch"), 1),
            ),
        ),
        """
        This is a boring docstring.

        :meta: Blah blah.
        :param a: Some stuff about argument `a`.
        :type a: TensorType
        :meta: More chaff.
        :returns: Some description of the return type.
        :rtype: TensorType
        """,
        """
        This is a boring docstring.

        :meta: Blah blah.
        :param a:
            * **a** has shape [*batch*..., *n_features*].

            Some stuff about argument `a`.
        :type a: TensorType
        :meta: More chaff.
        :returns:
            * **return** has shape [*batch*..., 1].

            Some description of the return type.
        :rtype: TensorType
        """,
    ),
    TestData(
        "other_colons",
        (
            "a: [batch..., n_features]",
            "return: [batch..., 1]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("a"),
                make_shape_spec(varrank("batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return"),
                make_shape_spec(varrank("batch"), 1),
            ),
        ),
        """
        This: is a docstring, :: with some extra :s in it.

        Here: are: some more:.

        :param a: Some stuff about: argument `a`.
        :returns: Some description of the:: return type.
        """,
        """
        This: is a docstring, :: with some extra :s in it.

        Here: are: some more:.

        :param a:
            * **a** has shape [*batch*..., *n_features*].

            Some stuff about: argument `a`.
        :returns:
            * **return** has shape [*batch*..., 1].

            Some description of the:: return type.
        """,
    ),
    TestData(
        "funny_formatting_1",
        (
            "train.features: [train_batch..., n_features]",
            "train.labels: [train_batch..., n_labels]",
            "test_features: [test_batch..., n_features]",
            "return[0]: [test_batch..., n_labels]",
            "return[1]: [test_batch..., n_labels]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("train", "features"),
                make_shape_spec(varrank("train_batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("train", "labels"),
                make_shape_spec(varrank("train_batch"), "n_labels"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("test_features"),
                make_shape_spec(varrank("test_batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 0),
                make_shape_spec(varrank("test_batch"), "n_labels"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 1),
                make_shape_spec(varrank("test_batch"), "n_labels"),
            ),
        ),
        """
        Predict mean and variance from some test features.

        First trains a model an `train`, then makes a prediction from the model and `test`.

        :param train:
            Data to train on.

        :param test_features:
            Features to make test prediction from.

        :returns:
            Model mean and variance prediction.
        """,
        """
        Predict mean and variance from some test features.

        First trains a model an `train`, then makes a prediction from the model and `test`.

        :param train:
            * **train.features** has shape [*train_batch*..., *n_features*].
            * **train.labels** has shape [*train_batch*..., *n_labels*].

            Data to train on.

        :param test_features:
            * **test_features** has shape [*test_batch*..., *n_features*].

            Features to make test prediction from.

        :returns:
            * **return[0]** has shape [*test_batch*..., *n_labels*].
            * **return[1]** has shape [*test_batch*..., *n_labels*].

            Model mean and variance prediction.
        """,
    ),
    TestData(
        "funny_formatting_2",
        (
            "train.features: [train_batch..., n_features]",
            "train.labels: [train_batch..., n_labels]",
            "test_features: [test_batch..., n_features]",
            "return[0]: [test_batch..., n_labels]",
            "return[1]: [test_batch..., n_labels]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("train", "features"),
                make_shape_spec(varrank("train_batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("train", "labels"),
                make_shape_spec(varrank("train_batch"), "n_labels"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("test_features"),
                make_shape_spec(varrank("test_batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 0),
                make_shape_spec(varrank("test_batch"), "n_labels"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 1),
                make_shape_spec(varrank("test_batch"), "n_labels"),
            ),
        ),
        """
        Predict mean and variance from some test features.
        First trains a model an `train`, then makes a prediction from the model and `test`.
        :param train: Data to train on.
        :param test_features: Features to make test prediction from.
        :returns: Model mean and variance prediction.
        """,
        """
        Predict mean and variance from some test features.
        First trains a model an `train`, then makes a prediction from the model and `test`.
        :param train:
            * **train.features** has shape [*train_batch*..., *n_features*].
            * **train.labels** has shape [*train_batch*..., *n_labels*].

            Data to train on.
        :param test_features:
            * **test_features** has shape [*test_batch*..., *n_features*].

            Features to make test prediction from.
        :returns:
            * **return[0]** has shape [*test_batch*..., *n_labels*].
            * **return[1]** has shape [*test_batch*..., *n_labels*].

            Model mean and variance prediction.
        """,
    ),
    TestData(
        "funny_formatting_3",
        (
            "train.features: [train_batch..., n_features]",
            "train.labels: [train_batch..., n_labels]",
            "test_features: [test_batch..., n_features]",
            "return[0]: [test_batch..., n_labels]",
            "return[1]: [test_batch..., n_labels]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("train", "features"),
                make_shape_spec(varrank("train_batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("train", "labels"),
                make_shape_spec(varrank("train_batch"), "n_labels"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("test_features"),
                make_shape_spec(varrank("test_batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 0),
                make_shape_spec(varrank("test_batch"), "n_labels"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 1),
                make_shape_spec(varrank("test_batch"), "n_labels"),
            ),
        ),
        """Predict mean and variance from some test features.

        First trains a model an `train`, then makes a prediction from the model and `test`.
        :param train:
        Data to train on.
        And another line.
        :param test_features:
        Features to make test prediction from.
        And some more comment.
        :returns:
        Model mean and variance prediction.""",
        """Predict mean and variance from some test features.

        First trains a model an `train`, then makes a prediction from the model and `test`.
        :param train:
        * **train.features** has shape [*train_batch*..., *n_features*].
        * **train.labels** has shape [*train_batch*..., *n_labels*].

        Data to train on.
        And another line.
        :param test_features:
        * **test_features** has shape [*test_batch*..., *n_features*].

        Features to make test prediction from.
        And some more comment.
        :returns:
        * **return[0]** has shape [*test_batch*..., *n_labels*].
        * **return[1]** has shape [*test_batch*..., *n_labels*].

        Model mean and variance prediction.""",
    ),
    TestData(
        "funny_formatting_4",
        (
            "train.features: [train_batch..., n_features]",
            "train.labels: [train_batch..., n_labels]",
            "test_features: [test_batch..., n_features]",
            "return[0]: [test_batch..., n_labels]",
            "return[1]: [test_batch..., n_labels]",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("train", "features"),
                make_shape_spec(varrank("train_batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("train", "labels"),
                make_shape_spec(varrank("train_batch"), "n_labels"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("test_features"),
                make_shape_spec(varrank("test_batch"), "n_features"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 0),
                make_shape_spec(varrank("test_batch"), "n_labels"),
            ),
            ParsedArgumentSpec(
                make_argument_ref("return", 1),
                make_shape_spec(varrank("test_batch"), "n_labels"),
            ),
        ),
        """Predict mean and variance from some test features.

First trains a model an `train`, then makes a prediction from the model and `test`.
:param train:
Data to train on.
And another line.
:param test_features:
Features to make test prediction from.
And some more comment.
:returns:
Model mean and variance prediction.""",
        """Predict mean and variance from some test features.

First trains a model an `train`, then makes a prediction from the model and `test`.
:param train:
* **train.features** has shape [*train_batch*..., *n_features*].
* **train.labels** has shape [*train_batch*..., *n_labels*].

Data to train on.
And another line.
:param test_features:
* **test_features** has shape [*test_batch*..., *n_features*].

Features to make test prediction from.
And some more comment.
:returns:
* **return[0]** has shape [*test_batch*..., *n_labels*].
* **return[1]** has shape [*test_batch*..., *n_labels*].

Model mean and variance prediction.""",
    ),
]


@pytest.mark.parametrize("data", _TEST_DATA, ids=str)
def test_parse_argument_spec(data: TestData) -> None:
    actual_specs = tuple(parse_argument_spec(s) for s in data.argument_spec_strs)
    assert data.expected_specs == actual_specs


@pytest.mark.parametrize("data", _TEST_DATA, ids=str)
def test_parse_and_rewrite_docstring(data: TestData) -> None:
    rewritten_docstring = parse_and_rewrite_docstring(data.doc, data.expected_specs)
    assert data.expected_doc == rewritten_docstring
