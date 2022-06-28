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
from functools import wraps
from unittest.mock import MagicMock

import pytest
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken

from gpflow.experimental.check_shapes import set_enable_function_call_precompute
from gpflow.experimental.check_shapes.error_contexts import (
    ArgumentContext,
    AttributeContext,
    ConditionContext,
    ErrorContext,
    FunctionCallContext,
    FunctionDefinitionContext,
    IndexContext,
    LarkUnexpectedInputContext,
    MappingKeyContext,
    MappingValueContext,
    MessageBuilder,
    MultipleElementBoolContext,
    NoteContext,
    ObjectTypeContext,
    ObjectValueContext,
    ParallelContext,
    ShapeContext,
    StackContext,
    TensorSpecContext,
    VariableContext,
)
from gpflow.experimental.check_shapes.specs import ParsedNoteSpec

from .utils import TestContext, barg, bnot, bor, current_line, make_shape_spec, make_tensor_spec


def to_str(context: ErrorContext) -> str:
    builder = MessageBuilder()
    builder.add_line("")  # Makes multi-line strings easier to read.
    context.print(builder)
    return builder.build()


def check_eq(context: ErrorContext) -> None:
    # Sanity checking of __eq__ and __hash__
    assert context == context  # pylint: disable=comparison-with-itself
    assert context != TestContext()
    assert context != None  # pylint: disable=singleton-comparison
    assert context in {context}


def test_message_builder__add_line() -> None:
    builder = MessageBuilder()
    builder.add_line("foo")
    assert "foo\n" == builder.build()


def test_message_builder__add_line__convert() -> None:
    builder = MessageBuilder()
    builder.add_line(42)
    assert "42\n" == builder.build()


def test_message_builder__add_lines() -> None:
    builder = MessageBuilder()
    builder.add_line("foo")
    builder.add_line("bar")
    builder.add_line("baz")
    assert (
        """
foo
bar
baz
"""[
            1:
        ]
        == builder.build()
    )


def test_message_builder__columns() -> None:
    builder = MessageBuilder()
    builder.add_columned_line(1, 11, 11111)
    builder.add_columned_line(222, 22, 2)
    builder.add_columned_line(33, 3333, 3)
    assert (
        """
1   11   11111
222 22   2
33  3333 3
"""[
            1:
        ]
        == builder.build()
    )


def test_message_builder__indent() -> None:
    builder = MessageBuilder()
    builder.add_line(11)
    with builder.indent() as b1:
        b1.add_line(21)
        with b1.indent() as b2:
            b2.add_line(31)
        b1.add_line(22)
    builder.add_line(12)
    assert (
        """
11
  21
    31
  22
12
"""[
            1:
        ]
        == builder.build()
    )


def test_message_builder__indent_columns() -> None:
    builder = MessageBuilder()
    builder.add_columned_line(11, 11111)
    with builder.indent() as b1:
        b1.add_columned_line(2, 22)
        with b1.indent() as b2:
            b2.add_columned_line(3, 333)
        with b1.indent() as b2:
            b2.add_columned_line(444, 4)
        b1.add_columned_line(2222, 2)
    builder.add_columned_line(1, 11)
    assert (
        """
11 11111
  2    22
    3 333
    444 4
  2222 2
1  11
"""[
            1:
        ]
        == builder.build()
    )


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            StackContext(TestContext("A"), TestContext("B")),
            """
A
  B
""",
        ),
        (
            StackContext(
                TestContext("A"),
                StackContext(
                    TestContext("B"),
                    StackContext(
                        TestContext("C"),
                        TestContext("D"),
                    ),
                ),
            ),
            """
A
  B
    C
      D
""",
        ),
        (
            StackContext(
                StackContext(
                    TestContext("A"),
                    TestContext("B"),
                ),
                StackContext(
                    TestContext("C"),
                    TestContext("D"),
                ),
            ),
            """
A
  B
    C
      D
""",
        ),
        (
            StackContext(
                StackContext(
                    StackContext(
                        TestContext("A"),
                        TestContext("B"),
                    ),
                    TestContext("C"),
                ),
                TestContext("D"),
            ),
            """
A
  B
    C
      D
""",
        ),
    ],
)
def test_stack_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            ParallelContext(()),
            """
""",
        ),
        (
            ParallelContext(
                (
                    TestContext("A"),
                    TestContext("B"),
                )
            ),
            """
A
B
""",
        ),
        (
            ParallelContext(
                (
                    TestContext("A"),
                    ParallelContext(
                        (
                            TestContext("B"),
                            TestContext("C"),
                        )
                    ),
                    TestContext("D"),
                    ParallelContext(
                        (
                            TestContext("E"),
                            TestContext("F"),
                        )
                    ),
                    TestContext("G"),
                )
            ),
            """
A
B
C
D
E
F
G
""",
        ),
        (
            ParallelContext(
                (
                    StackContext(
                        TestContext("A"),
                        StackContext(
                            TestContext("B1"),
                            TestContext("C1"),
                        ),
                    ),
                    StackContext(
                        StackContext(
                            TestContext("A"),
                            TestContext("B1"),
                        ),
                        TestContext("C2"),
                    ),
                    StackContext(
                        TestContext("A"),
                        StackContext(
                            TestContext("B2"),
                            TestContext("C1"),
                        ),
                    ),
                )
            ),
            """
A
  B1
    C1
    C2
  B2
    C1
""",
        ),
    ],
)
def test_parallel_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


def test_function_call_context() -> None:
    def f() -> str:
        context = FunctionCallContext(f)
        check_eq(context)
        return to_str(context)

    call_line = current_line() + 1
    assert (
        f"""
f called at: {__file__}:{call_line}
"""
        == f()
    )


def test_function_call_context__wrapping() -> None:
    def f() -> str:
        context = FunctionCallContext(f3)
        check_eq(context)
        return to_str(context)

    @wraps(f)
    def f2() -> str:
        return f()

    @wraps(f2)
    def f3() -> str:
        return f2()

    call_line = current_line() + 1
    assert (
        f"""
f called at: {__file__}:{call_line}
"""
        == f3()
    )


def test_function_call_context__precompute__disabled() -> None:
    set_enable_function_call_precompute(False)

    def f() -> ErrorContext:
        return FunctionCallContext(f).precompute()

    context = f()

    assert f"""
f called at: <Unknown file>:<Unknown line> (Disabled. Call gpflow.experimental.check_shapes.set_enable_function_call_precompute(True) to see this. (Slow.))
""" == to_str(
        context
    )
    check_eq(context)


def test_function_call_context__precompute__enabled() -> None:
    def f() -> ErrorContext:
        return FunctionCallContext(f).precompute()

    call_line = current_line() + 1
    context = f()

    assert f"""
f called at: {__file__}:{call_line}
""" == to_str(
        context
    )
    check_eq(context)


def test_function_definition_context() -> None:
    def_line = current_line() + 2

    def f() -> None:
        ...

    context = FunctionDefinitionContext(f)
    assert f"""
Function: test_function_definition_context.<locals>.f
  Declared: {__file__}:{def_line}
""" == to_str(
        context
    )
    check_eq(context)


def test_function_definition_context__builtin() -> None:
    context = FunctionDefinitionContext(str)
    assert """
Function: str
  Declared: <Unknown file>:<Unknown line>
""" == to_str(
        context
    )
    check_eq(context)


def test_variable_context() -> None:
    context = VariableContext("foo")
    assert """
Variable: foo
""" == to_str(
        context
    )
    check_eq(context)


def test_tensor_spec_context() -> None:
    context = TensorSpecContext(make_tensor_spec(make_shape_spec("a", 1), None))
    assert """
Specification: [a, 1]
""" == to_str(
        context
    )
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            ArgumentContext("arg"),
            """
Argument: arg
""",
        ),
        (
            ArgumentContext(3),
            """
Argument number (0-indexed): 3
""",
        ),
        (
            ArgumentContext("arg", value=7),
            """
Argument: arg
  Value: 7
""",
        ),
        (
            ArgumentContext("arg", value={}),
            """
Argument: arg
  Value: {}
""",
        ),
        (
            ArgumentContext(3, value=None),
            """
Argument number (0-indexed): 3
  Value: None
""",
        ),
    ],
)
def test_argument_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            AttributeContext("attr"),
            """
Attribute: .attr
""",
        ),
        (
            AttributeContext("attr", value=3),
            """
Attribute: .attr
  Value: 3
""",
        ),
        (
            AttributeContext("attr", value={}),
            """
Attribute: .attr
  Value: {}
""",
        ),
        (
            AttributeContext("attr", value=None),
            """
Attribute: .attr
  Value: None
""",
        ),
    ],
)
def test_attribute_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            IndexContext(2),
            """
Index: [2]
""",
        ),
        (
            IndexContext(2, value=3),
            """
Index: [2]
  Value: 3
""",
        ),
        (
            IndexContext(2, value={}),
            """
Index: [2]
  Value: {}
""",
        ),
        (
            IndexContext(2, value=None),
            """
Index: [2]
  Value: None
""",
        ),
    ],
)
def test_index_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            MappingKeyContext(2),
            """
Map key: [2]
""",
        ),
        (
            MappingKeyContext("foo"),
            """
Map key: ['foo']
""",
        ),
        (
            MappingKeyContext({}),
            """
Map key: [{}]
""",
        ),
        (
            MappingKeyContext(None),
            """
Map key: [None]
""",
        ),
    ],
)
def test_mapping_key_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            MappingValueContext(2),
            """
Map value, of key: [2]
""",
        ),
        (
            MappingValueContext("foo", value=3),
            """
Map value, of key: ['foo']
  Value: 3
""",
        ),
        (
            MappingValueContext({}, value={}),
            """
Map value, of key: [{}]
  Value: {}
""",
        ),
        (
            MappingValueContext(None, value=None),
            """
Map value, of key: [None]
  Value: None
""",
        ),
    ],
)
def test_mapping_value_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


def test_condition_context() -> None:
    context = ConditionContext(bor(barg("foo"), bnot(barg("bar"))))
    assert """
Condition: foo or (not bar)
""" == to_str(
        context
    )
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            ShapeContext(make_shape_spec("x", 3), (2, 3)),
            """
Expected: [x, 3]
Actual:   [2, 3]
""",
        ),
        (
            ShapeContext(make_shape_spec("x", 3), (2, None)),
            """
Expected: [x, 3]
Actual:   [2, None]
""",
        ),
        (
            ShapeContext(make_shape_spec("x", 3), None),
            """
Expected: [x, 3]
Actual:   <Tensor is None or has unknown shape>
""",
        ),
    ],
)
def test_shape_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


def test_note_context() -> None:
    context = NoteContext(ParsedNoteSpec("Some note."))
    assert """
Note: Some note.
""" == to_str(
        context
    )
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            ObjectValueContext(None),
            """
Value: None
""",
        ),
        (
            ObjectValueContext(7),
            """
Value: 7
""",
        ),
        (
            ObjectValueContext({}),
            """
Value: {}
""",
        ),
        (
            ObjectValueContext("foo"),
            """
Value: 'foo'
""",
        ),
    ],
)
def test_object_value_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            ObjectTypeContext(None),
            """
Object type: builtins.NoneType
""",
        ),
        (
            ObjectTypeContext(7),
            """
Object type: builtins.int
""",
        ),
        (
            ObjectTypeContext({}),
            """
Object type: builtins.dict
""",
        ),
        (
            ObjectTypeContext(ObjectTypeContext(7)),
            """
Object type: gpflow.experimental.check_shapes.error_contexts.ObjectTypeContext
""",
        ),
    ],
)
def test_object_type_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)


def test_object_type_context__nested_type() -> None:
    class A:
        class B:

            pass

    context = ObjectTypeContext(A.B())
    assert """
Object type: tests.gpflow.experimental.check_shapes.test_error_contexts.test_object_type_context__nested_type.<locals>.A.B
""" == to_str(
        context
    )
    check_eq(context)


def test_lark_unexpected_input_context__unexpected_eof() -> None:
    text = ""
    error = MagicMock(UnexpectedEOF, expected=["A", "B", "C"], line=-1, column=-1)
    terminal_descriptions = {
        "A": "Some a's",
        "B": "Some b's",
        "C": "Some c's",
    }

    context = LarkUnexpectedInputContext(text, error, terminal_descriptions)
    assert """
Expected one of: Some a's
                 Some b's
                 Some c's
Found unexpected end of input.
""" == to_str(
        context
    )
    check_eq(context)


def test_lark_unexpected_input_context__unexpected_characters() -> None:
    text = """This is line 1
This is line 2
This is line 3
"""
    error = MagicMock(UnexpectedCharacters, line=2, column=9)
    terminal_descriptions = {
        "A": "Some a's",
        "B": "Some b's",
        "C": "Some c's",
    }

    context = LarkUnexpectedInputContext(text, error, terminal_descriptions)
    assert """
Line: "This is line 2"
               ^
Found unexpected character.
""" == to_str(
        context
    )
    check_eq(context)


def test_lark_unexpected_input_context__unexpected_token() -> None:
    text = """This is line 1
This is line 2
This is line 3
"""
    error = MagicMock(UnexpectedToken, accepts=["B"], line=1, column=14)
    terminal_descriptions = {
        "A": "Some a's",
        "B": "Some b's",
        "C": "Some c's",
    }

    context = LarkUnexpectedInputContext(text, error, terminal_descriptions)
    assert """
Line:     "This is line 1"
                        ^
Expected: Some b's
""" == to_str(
        context
    )
    check_eq(context)


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            MultipleElementBoolContext("foo bar", 1, 5),
            """
Line: "foo bar"
           ^
Argument references that evaluate to multiple values are not supported for boolean expressions.
""",
        ),
        (
            MultipleElementBoolContext("foo bar", 0, 5),
            """
Argument references that evaluate to multiple values are not supported for boolean expressions.
""",
        ),
        (
            MultipleElementBoolContext("foo bar", 1, 0),
            """
Line: "foo bar"
Argument references that evaluate to multiple values are not supported for boolean expressions.
""",
        ),
        (
            MultipleElementBoolContext("foo bar", 0, 0),
            """
Argument references that evaluate to multiple values are not supported for boolean expressions.
""",
        ),
    ],
)
def test_multiple_elements_bool_context(context: ErrorContext, expected: str) -> None:
    assert expected == to_str(context)
    check_eq(context)
