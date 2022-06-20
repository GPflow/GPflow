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

# pylint: disable=broad-except

"""
Infrastructure for describing the context of an error.

The :class:`MessageBuilder` is used to format / indent messages nicely.

The :class:`ErrorContext` is a reusable bit of information about where/why an error occurred that
can be written to a :class:`MessageBuilder`.

:class:`ErrorContext`\ s can be composed using the :class:`StackContext` and
:class:`ParallelContext`.

This allows reusable error messages in a consistent format.
"""
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedInput

from .base_types import Shape
from .config import get_enable_function_call_precompute

if TYPE_CHECKING:  # pragma: no cover
    from .argument_ref import ArgumentRef
    from .bool_specs import ParsedBoolSpec
    from .specs import ParsedNoteSpec, ParsedShapeSpec, ParsedTensorSpec

_UNKNOWN_FILE = "<Unknown file>"
_UNKNOWN_LINE = "<Unknown line>"
_DISABLED_FILE_AND_LINE = (
    f"{_UNKNOWN_FILE}:{_UNKNOWN_LINE}"
    " (Disabled. Call gpflow.experimental.check_shapes.set_enable_function_call_precompute(True) to"
    " see this. (Slow.))"
)
_NONE_SHAPE = "<Tensor is None or has unknown shape>"

_NO_VALUE = object()
"""
Sentinel to represent "no value" in places where ``None`` is a valid value.
"""


class MessageBuilder:
    """
    Utility for formatting nested text.
    """

    def __init__(self, indent_str: str = "") -> None:
        self._indent_str = indent_str
        self._lines: List[Union[str, Tuple[str, ...], "MessageBuilder"]] = []
        self._column_widths: List[int] = []

    def add_line(self, text: Any) -> None:
        """
        Write a line, indented at the current level.

        Input is converted to a `str` using `str(text)`.
        """
        text_str = str(text)
        self._lines.append(text_str)

    def add_columned_line(self, *texts: Any) -> None:
        """
        Write a line of several values, left-aligned within the current indentation level.

        Inputs are converted to `str`s using `str(text)`.
        """
        widths = self._column_widths
        text_strs = tuple(str(t) for t in texts)
        for i, t in enumerate(text_strs):
            w = len(t)
            if i < len(widths):
                widths[i] = max(widths[i], w)
            else:
                widths.append(w)
        self._lines.append(text_strs)

    @contextmanager
    def indent(self) -> Iterator["MessageBuilder"]:
        """
        Context manager for indenting text.
        """
        child = MessageBuilder(self._indent_str + "  ")
        self._lines.append(child)
        yield child

    def build(self) -> str:
        """
        Compile all collected text into a single string.
        """
        fragments: List[str] = []

        def _collect(builder: "MessageBuilder") -> None:
            # pylint: disable=protected-access
            for line in builder._lines:
                if isinstance(line, str):
                    fragments.extend((builder._indent_str, line, "\n"))
                elif isinstance(line, tuple):
                    fragments.append(builder._indent_str)
                    for i, t in enumerate(line):
                        fragments.append(t)
                        if i + 1 < len(line):
                            fragments.append(" " * (builder._column_widths[i] - len(t) + 1))
                    fragments.append("\n")
                else:
                    assert isinstance(line, MessageBuilder)
                    _collect(line)

        _collect(self)
        return "".join(fragments)


class ErrorContext(ABC):
    """
    A context in which an error can occur.

    Contexts should be immutable, and implement :func:`__eq__` - so that they can be composed using
    :class:`StackContext` and :class:`ParallelContext`.

    The contexts are often created even if an error doesn't actually occur, so they should be cheap
    to create - prefer to do any slow computation in :func:`print`, rather than in :func:`__init__`.

    Maybe think of an :class:`ErrorContext` as a factory of error messages.
    """

    @abstractmethod
    def print(self, builder: MessageBuilder) -> None:
        """
        Print this context to the given `MessageBuilder`.
        """


@dataclass(frozen=True)
class StackContext(ErrorContext):
    """
    Error context where one context is "inside" another one.
    """

    parent: ErrorContext
    child: ErrorContext

    def print(self, builder: MessageBuilder) -> None:
        def _print(context: ErrorContext) -> None:
            nonlocal builder

            if isinstance(context, StackContext):
                _print(context.parent)
                _print(context.child)
            else:
                context.print(builder)

                with builder.indent() as b:
                    builder = b

        _print(self)


@dataclass(frozen=True)
class ParallelContext(ErrorContext):
    """
    Error context with many contexts in parallel.
    """

    children: Tuple[ErrorContext, ...]

    def print(self, builder: MessageBuilder) -> None:
        # Merge any nested parallel contexts.
        flat: List[ErrorContext] = []

        def _flatten(context: ErrorContext) -> None:
            if isinstance(context, ParallelContext):
                for child in context.children:
                    _flatten(child)
            else:
                flat.append(context)

        _flatten(self)

        # If any `StackContexts`s have the same (grand-) parent, merge them, so that we don't repeat
        # stuff in the output.
        def _split_head(context: StackContext) -> Tuple[ErrorContext, ErrorContext]:
            parent = context.parent
            child = context.child
            if isinstance(parent, StackContext):
                # Transform:
                #   /\         /\
                #  /\ C  -->  A /\
                # A  B         B  C
                # So we get:                     /\
                #             head = A,  body = B  C
                return _split_head(StackContext(parent.parent, StackContext(parent.child, child)))
            else:
                return parent, child

        by_head: Dict[ErrorContext, List[ErrorContext]] = {}
        for child in flat:
            if isinstance(child, StackContext):
                head, body = _split_head(child)
            else:
                head = child
                body = None
            bodies = by_head.setdefault(head, [])
            if body is not None:
                bodies.append(body)

        for head, bodies in by_head.items():
            head.print(builder)
            with builder.indent() as b:
                ParallelContext(tuple(bodies)).print(b)


@dataclass(frozen=True)
class FunctionCallContext(ErrorContext):
    """
    An error occured inside a function that was called.

    Normally `print` should be called from within the called function: `func`.
    If that is impossible, first call `precompute` from within `func`.
    """

    func: Callable[..., Any]
    path_and_line: Optional[str] = None

    def _get_target(self) -> Optional[Tuple[str, str]]:
        func = inspect.unwrap(self.func)
        search_name: str = func.__name__
        search_path = inspect.getsourcefile(func)
        if search_path is None:
            return None
        return search_name, search_path

    def _count_wrappers(self) -> int:
        wrappers = -1
        f: Optional[Callable[..., Any]] = self.func
        while f is not None:
            wrappers += 1
            f = getattr(f, "__wrapped__", None)
        return wrappers

    def _get_calling_frame(self) -> Optional[inspect.FrameInfo]:
        search_name_path = self._get_target()
        if search_name_path is None:
            return None

        stack = inspect.stack(0)
        for i, frame in enumerate(stack):
            name_path = (frame.function, frame.filename)
            if search_name_path == name_path:
                return stack[i + self._count_wrappers() + 1]

        return None

    def _compute(self) -> str:
        if self.path_and_line is not None:
            return self.path_and_line

        frame = self._get_calling_frame()
        if frame is None:
            path = _UNKNOWN_FILE
            line = _UNKNOWN_LINE
        else:
            path = frame.filename
            line = str(frame.lineno)

        return f"{path}:{line}"

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line(f"{self.func.__name__} called at:", self._compute())

    def precompute(self) -> "FunctionCallContext":
        """
        Precompute the values to print.

        This is useful to capture the position of the stack of the relevant call, if this object is
        saved for later use.

        :returns: A new instance with precomptued values.
        """
        if get_enable_function_call_precompute():
            return FunctionCallContext(self.func, self._compute())
        else:
            return FunctionCallContext(self.func, _DISABLED_FILE_AND_LINE)


@dataclass(frozen=True)
class FunctionDefinitionContext(ErrorContext):
    """
    An error occured in the context of a function definition.
    """

    func: Callable[..., Any]

    def print(self, builder: MessageBuilder) -> None:
        name = self.func.__qualname__
        try:
            path = inspect.getsourcefile(self.func)
        except Exception:  # pragma: no cover
            path = _UNKNOWN_FILE
        try:
            _, line_int = inspect.getsourcelines(self.func)
            line = str(line_int)
        except Exception:  # pragma: no cover
            line = _UNKNOWN_LINE
        path_and_line = f"{path}:{line}"

        builder.add_columned_line("Function:", name)
        with builder.indent() as b:
            b.add_columned_line("Declared:", path_and_line)


@dataclass(frozen=True)
class VariableContext(ErrorContext):
    """
    An error occurred in the context of a shape specification variable.
    """

    variable_name: str

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line("Variable:", self.variable_name)


@dataclass(frozen=True)
class TensorSpecContext(ErrorContext):
    """
    An error occurred in the context of a tensor specification.
    """

    spec: "ParsedTensorSpec"

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line("Specification:", self.spec)


@dataclass(frozen=True)
class ArgumentContext(ErrorContext):
    """
    An error occurred in the context of an argument to a function.
    """

    name_or_index: Union[str, int]
    value: Any = _NO_VALUE

    def print(self, builder: MessageBuilder) -> None:
        if isinstance(self.name_or_index, str):
            builder.add_columned_line("Argument:", self.name_or_index)
        else:
            builder.add_columned_line("Argument number (0-indexed):", self.name_or_index)
        if self.value is not _NO_VALUE:
            with builder.indent() as b:
                b.add_columned_line("Value:", self.value)

    def __eq__(self, other: Any) -> bool:
        return (
            type(self) == type(other)
            and self.name_or_index == other.name_or_index
            and self.value is other.value
        )

    def __hash__(self) -> int:
        return hash((self.name_or_index, id(self.value)))


@dataclass(frozen=True)
class AttributeContext(ErrorContext):
    """
    An error occurred in the context of an attribute on an object.
    """

    name: str
    value: Any = _NO_VALUE

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line("Attribute:", f".{self.name}")
        if self.value is not _NO_VALUE:
            with builder.indent() as b:
                b.add_columned_line("Value:", self.value)

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.name == other.name and self.value is other.value

    def __hash__(self) -> int:
        return hash((self.name, id(self.value)))


@dataclass(frozen=True)
class IndexContext(ErrorContext):
    """
    An error occurred in the context of an index in a sequence.
    """

    index: int
    value: Any = _NO_VALUE

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line("Index:", f"[{self.index}]")
        if self.value is not _NO_VALUE:
            with builder.indent() as b:
                b.add_columned_line("Value:", self.value)

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.index == other.index and self.value is other.value

    def __hash__(self) -> int:
        return hash((self.index, id(self.value)))


@dataclass(frozen=True)
class MappingKeyContext(ErrorContext):
    """
    An error occurent in the context of a key in a map.
    """

    key: Any

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line("Map key:", f"[{self.key!r}]")

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.key is other.key

    def __hash__(self) -> int:
        return id(self.key)


@dataclass(frozen=True)
class MappingValueContext(ErrorContext):
    """
    An error occurent in the context of a value in a map.
    """

    key: Any
    value: Any = _NO_VALUE

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line("Map value, of key:", f"[{self.key!r}]")
        if self.value is not _NO_VALUE:
            with builder.indent() as b:
                b.add_columned_line("Value:", self.value)

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.key is other.key and self.value is other.value

    def __hash__(self) -> int:
        return hash((id(self.key), id(self.value)))


@dataclass(frozen=True)
class ConditionContext(ErrorContext):
    """
    An error occurred in a conditional context.
    """

    condition: "ParsedBoolSpec"

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line("Condition:", self.condition)


@dataclass(frozen=True)
class ShapeContext(ErrorContext):
    """
    An error occurred in the context of the shapes of function arguments.
    """

    expected: "ParsedShapeSpec"
    actual: Shape

    def print(self, builder: MessageBuilder) -> None:
        if self.actual is None:
            actual_str = _NONE_SHAPE
        else:
            actual_str = f"[{', '.join(str(dim) for dim in self.actual)}]"
        builder.add_columned_line("Expected:", self.expected)
        builder.add_columned_line("Actual:", actual_str)


@dataclass(frozen=True)
class NoteContext(ErrorContext):
    """
    An error occurred in a context where a user has added a note.
    """

    note: "ParsedNoteSpec"

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line("Note:", self.note.note)


@dataclass(frozen=True)
class ObjectValueContext(ErrorContext):
    """
    An error was caused by the value of an object.
    """

    obj: Any

    def print(self, builder: MessageBuilder) -> None:
        builder.add_columned_line("Value:", repr(self.obj))

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.obj is other.obj

    def __hash__(self) -> int:
        return id(self.obj)


@dataclass(frozen=True)
class ObjectTypeContext(ErrorContext):
    """
    An error was caused by the type of an object.
    """

    obj: Any

    def print(self, builder: MessageBuilder) -> None:
        t = type(self.obj)
        builder.add_columned_line("Object type:", f"{t.__module__}.{t.__qualname__}")

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.obj is other.obj

    def __hash__(self) -> int:
        return id(self.obj)


@dataclass(frozen=True)  # type: ignore[misc]  # mypy doesn't like abstract `dataclasses`.
class ParserInputContext(ErrorContext, ABC):
    """
    Abstract base class for contexts that relate to parser input.
    """

    text: str

    def print_line(self, builder: MessageBuilder, line: int, column: int) -> None:
        if line > 0:
            line_content = self.text.split("\n")[line - 1]
            builder.add_columned_line("Line:", f'"{line_content}"')
            if column > 0:
                builder.add_columned_line("", " " * column + "^")


@dataclass(frozen=True)
class LarkUnexpectedInputContext(ParserInputContext):
    """
    An error was caused by an `UnexpectedInput` error from `Lark`.
    """

    error: UnexpectedInput
    terminal_descriptions: Mapping[str, str]

    def print(self, builder: MessageBuilder) -> None:
        self.print_line(
            builder,
            getattr(self.error, "line", -1),
            getattr(self.error, "column", -1),
        )

        expected: Sequence[str] = getattr(self.error, "accepts", [])
        if not expected:
            expected = getattr(self.error, "expected", [])
        expected = sorted(expected)
        expected_key = "Expected:" if len(expected) <= 1 else "Expected one of:"
        for expected_name in expected:
            expected_value = self.terminal_descriptions.get(expected_name, expected_name)
            builder.add_columned_line(expected_key, expected_value)
            expected_key = ""

        if isinstance(self.error, UnexpectedCharacters):
            builder.add_line("Found unexpected character.")
        if isinstance(self.error, UnexpectedEOF):
            builder.add_line("Found unexpected end of input.")

    def __hash__(self) -> int:
        return hash((self.error, *sorted(self.terminal_descriptions.items())))


@dataclass(frozen=True)
class MultipleElementBoolContext(ParserInputContext):
    """
    An error was caused by trying to use a multi-element argument specification as a bool.
    """

    line: int
    column: int

    def print(self, builder: MessageBuilder) -> None:
        self.print_line(builder, self.line, self.column)
        builder.add_line(
            "Argument references that evaluate to multiple values are not supported for boolean"
            " expressions."
        )
