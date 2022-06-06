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

# pylint: disable=unused-argument

from abc import ABC
from dataclasses import replace
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

from lark.exceptions import UnexpectedInput
from lark.lark import Lark
from lark.lexer import PatternRE, PatternStr, Token
from lark.tree import Tree

from .argument_ref import (
    RESULT_TOKEN,
    AllElementsRef,
    ArgumentRef,
    AttributeArgumentRef,
    IndexArgumentRef,
    KeysRef,
    RootArgumentRef,
    ValuesRef,
)
from .bool_specs import (
    BoolTest,
    ParsedAndBoolSpec,
    ParsedArgumentRefBoolSpec,
    ParsedBoolSpec,
    ParsedNotBoolSpec,
    ParsedOrBoolSpec,
)
from .config import DocstringFormat, get_rewrite_docstrings
from .error_contexts import (
    ArgumentContext,
    ErrorContext,
    LarkUnexpectedInputContext,
    MultipleElementBoolContext,
    StackContext,
)
from .exceptions import CheckShapesError, DocstringParseError, SpecificationParseError
from .specs import (
    ParsedArgumentSpec,
    ParsedDimensionSpec,
    ParsedFunctionSpec,
    ParsedNoteSpec,
    ParsedShapeSpec,
    ParsedTensorSpec,
)

_VARIABLE_RANK_LEADING_TOKEN = "*"
_VARIABLE_RANK_TRAILING_TOKEN = "..."


def _tree_children(tree: Tree[Token]) -> Iterable[Tree[Token]]:
    """ Return all the children of `tree` that are trees themselves. """
    return (child for child in tree.children if isinstance(child, Tree))


def _token_children(tree: Tree[Token]) -> Iterable[str]:
    """ Return the values of all the children of `tree` that are tokens. """
    return (child.value for child in tree.children if isinstance(child, Token))


class _TreeVisitor(ABC):
    """
    Functionality for visiting the nodes of parse-trees.

    This differs from the classes built-in in Lark, in that it allows passing `*args` and
    `**kwargs`.

    Subclasses should add methods with the same name as Lark rules. Those methods should take the
    parse tree of the rule, followed by any other `*args` and `**kwargs` you want. They may return
    anything.
    """

    def visit(self, tree: Tree[Token], *args: Any, **kwargs: Any) -> Any:
        name = tree.data
        visit = getattr(self, name, None)
        assert visit, f"No method found with name {name}."
        return visit(tree, *args, **kwargs)


class _ParseSpec(_TreeVisitor):
    def __init__(self, source: str) -> None:
        self._source = source

    def argument_spec(self, tree: Tree[Token]) -> ParsedArgumentSpec:
        argument_ref, shape_spec, *other_specs = _tree_children(tree)
        argument = self.visit(argument_ref, False)
        shape = self.visit(shape_spec)
        condition = None
        note = None
        for other_spec in other_specs:
            other = self.visit(other_spec)
            if isinstance(other, ParsedBoolSpec):
                assert condition is None
                condition = other
            else:
                assert isinstance(other, ParsedNoteSpec)
                assert note is None
                note = other
        tensor = ParsedTensorSpec(shape, note)
        return ParsedArgumentSpec(argument, tensor, condition)

    def bool_spec_or(self, tree: Tree[Token]) -> ParsedBoolSpec:
        left, right = _tree_children(tree)
        return ParsedOrBoolSpec(self.visit(left), self.visit(right))

    def bool_spec_and(self, tree: Tree[Token]) -> ParsedBoolSpec:
        left, right = _tree_children(tree)
        return ParsedAndBoolSpec(self.visit(left), self.visit(right))

    def bool_spec_not(self, tree: Tree[Token]) -> ParsedBoolSpec:
        (right,) = _tree_children(tree)
        return ParsedNotBoolSpec(self.visit(right))

    def bool_spec_argument_ref_is_none(self, tree: Tree[Token]) -> ParsedBoolSpec:
        (argument_ref,) = _tree_children(tree)
        return ParsedArgumentRefBoolSpec(self.visit(argument_ref, True), BoolTest.IS_NONE)

    def bool_spec_argument_ref_is_not_none(self, tree: Tree[Token]) -> ParsedBoolSpec:
        (argument_ref,) = _tree_children(tree)
        return ParsedArgumentRefBoolSpec(self.visit(argument_ref, True), BoolTest.IS_NOT_NONE)

    def bool_spec_argument_ref(self, tree: Tree[Token]) -> ParsedBoolSpec:
        (argument_ref,) = _tree_children(tree)
        return ParsedArgumentRefBoolSpec(self.visit(argument_ref, True), BoolTest.BOOL)

    def argument_ref_root(self, tree: Tree[Token], is_for_bool_spec: bool) -> ArgumentRef:
        (token,) = _token_children(tree)
        return RootArgumentRef(token)

    def argument_ref_attribute(self, tree: Tree[Token], is_for_bool_spec: bool) -> ArgumentRef:
        (source,) = _tree_children(tree)
        (token,) = _token_children(tree)
        return AttributeArgumentRef(self.visit(source, is_for_bool_spec), token)

    def argument_ref_index(self, tree: Tree[Token], is_for_bool_spec: bool) -> ArgumentRef:
        (source,) = _tree_children(tree)
        (token,) = _token_children(tree)
        return IndexArgumentRef(self.visit(source, is_for_bool_spec), int(token))

    def argument_ref_all(self, tree: Tree[Token], is_for_bool_spec: bool) -> ArgumentRef:
        (source,) = _tree_children(tree)
        self._disallow_multiple_element_bool_spec(source, is_for_bool_spec)
        return AllElementsRef(self.visit(source, is_for_bool_spec))

    def argument_ref_keys(self, tree: Tree[Token], is_for_bool_spec: bool) -> ArgumentRef:
        (source,) = _tree_children(tree)
        self._disallow_multiple_element_bool_spec(source, is_for_bool_spec)
        return KeysRef(self.visit(source, is_for_bool_spec))

    def argument_ref_values(self, tree: Tree[Token], is_for_bool_spec: bool) -> ArgumentRef:
        (source,) = _tree_children(tree)
        self._disallow_multiple_element_bool_spec(source, is_for_bool_spec)
        return ValuesRef(self.visit(source, is_for_bool_spec))

    def _disallow_multiple_element_bool_spec(
        self, source: Tree[Token], is_for_bool_spec: bool
    ) -> None:
        if is_for_bool_spec:
            meta = source.meta
            raise SpecificationParseError(
                MultipleElementBoolContext(self._source, meta.end_line, meta.end_column)
            )

    def tensor_spec(self, tree: Tree[Token]) -> ParsedTensorSpec:
        shape_spec, *note_specs = _tree_children(tree)
        shape = self.visit(shape_spec)
        if note_specs:
            (note_spec,) = note_specs
            note = self.visit(note_spec)
        else:
            note = None
        return ParsedTensorSpec(shape, note)

    def shape_spec(self, tree: Tree[Token]) -> ParsedShapeSpec:
        (dimension_specs,) = _tree_children(tree)
        return ParsedShapeSpec(self.visit(dimension_specs))

    def dimension_specs(self, tree: Tree[Token]) -> Tuple[ParsedDimensionSpec, ...]:
        return tuple(
            self.visit(dimension_spec, i) for i, dimension_spec in enumerate(_tree_children(tree))
        )

    def dimension_spec_broadcast(self, tree: Tree[Token], i: int) -> ParsedDimensionSpec:
        (dimension_spec,) = _tree_children(tree)
        child = self.visit(dimension_spec, i)
        assert isinstance(child, ParsedDimensionSpec)
        return replace(child, broadcastable=True)

    def dimension_spec_constant(self, tree: Tree[Token], i: int) -> ParsedDimensionSpec:
        (token,) = _token_children(tree)
        return ParsedDimensionSpec(
            constant=int(token), variable_name=None, variable_rank=False, broadcastable=False
        )

    def dimension_spec_variable(self, tree: Tree[Token], i: int) -> ParsedDimensionSpec:
        (token,) = _token_children(tree)
        return ParsedDimensionSpec(
            constant=None, variable_name=token, variable_rank=False, broadcastable=False
        )

    def dimension_spec_anonymous(self, tree: Tree[Token], i: int) -> ParsedDimensionSpec:
        return ParsedDimensionSpec(
            constant=None, variable_name=None, variable_rank=False, broadcastable=False
        )

    def dimension_spec_variable_rank(self, tree: Tree[Token], i: int) -> ParsedDimensionSpec:
        (token1, token2) = _token_children(tree)
        if token1 == _VARIABLE_RANK_LEADING_TOKEN:
            variable_name = token2
        else:
            assert token2 == _VARIABLE_RANK_TRAILING_TOKEN
            variable_name = token1
        return ParsedDimensionSpec(
            constant=None, variable_name=variable_name, variable_rank=True, broadcastable=False
        )

    def dimension_spec_anonymous_variable_rank(
        self, tree: Tree[Token], i: int
    ) -> ParsedDimensionSpec:
        return ParsedDimensionSpec(
            constant=None, variable_name=None, variable_rank=True, broadcastable=False
        )

    def note_spec(self, tree: Tree[Token]) -> ParsedNoteSpec:
        _hash_token, *note_tokens = _token_children(tree)
        return ParsedNoteSpec(" ".join(token.strip() for token in note_tokens))


class _RewritedocString(_TreeVisitor):
    def __init__(self, source: str, function_spec: ParsedFunctionSpec) -> None:
        self._source = source
        self._spec_lines = self._argument_specs_to_sphinx(function_spec.arguments)
        self._notes = tuple(note.note for note in function_spec.notes)
        self._indent = self._guess_indent(source)

    def _argument_specs_to_sphinx(
        self,
        argument_specs: Collection[ParsedArgumentSpec],
    ) -> Mapping[str, Sequence[str]]:
        result: Dict[str, List[str]] = {}
        for spec in argument_specs:
            result.setdefault(spec.argument_ref.root_argument_name, []).append(
                self._argument_spec_to_sphinx(spec)
            )
        for lines in result.values():
            lines.sort()
        return result

    def _argument_spec_to_sphinx(self, argument_spec: ParsedArgumentSpec) -> str:
        tensor_spec = argument_spec.tensor
        shape_spec = tensor_spec.shape
        out = []
        out.append(f"* **{repr(argument_spec.argument_ref)}**")
        out.append(" has shape [")
        out.append(self._shape_spec_to_sphinx(shape_spec))
        out.append("]")

        if argument_spec.condition is not None:
            out.append(" if ")
            out.append(self._bool_spec_to_sphinx(argument_spec.condition, False))

        out.append(".")

        if tensor_spec.note is not None:
            note_spec = tensor_spec.note
            out.append(" ")
            out.append(note_spec.note)
        return "".join(out)

    def _bool_spec_to_sphinx(self, bool_spec: ParsedBoolSpec, paren_wrap: bool) -> str:
        if isinstance(bool_spec, ParsedOrBoolSpec):
            result = (
                self._bool_spec_to_sphinx(bool_spec.left, True)
                + " or "
                + self._bool_spec_to_sphinx(bool_spec.right, True)
            )
        elif isinstance(bool_spec, ParsedAndBoolSpec):
            result = (
                self._bool_spec_to_sphinx(bool_spec.left, True)
                + " and "
                + self._bool_spec_to_sphinx(bool_spec.right, True)
            )
        elif isinstance(bool_spec, ParsedNotBoolSpec):
            result = "not " + self._bool_spec_to_sphinx(bool_spec.right, True)
        else:
            assert isinstance(bool_spec, ParsedArgumentRefBoolSpec)
            if bool_spec.bool_test == BoolTest.BOOL:
                paren_wrap = False  # Never wrap a stand-alone argument.
                result = f"*{bool_spec.argument_ref!r}*"
            elif bool_spec.bool_test == BoolTest.IS_NONE:
                result = f"*{bool_spec.argument_ref!r}* is *None*"
            else:
                assert bool_spec.bool_test == BoolTest.IS_NOT_NONE
                result = f"*{bool_spec.argument_ref!r}* is not *None*"

        if paren_wrap:
            result = f"({result})"

        return result

    def _shape_spec_to_sphinx(self, shape_spec: ParsedShapeSpec) -> str:
        return ", ".join(self._dim_spec_to_sphinx(dim) for dim in shape_spec.dims)

    def _dim_spec_to_sphinx(self, dim_spec: ParsedDimensionSpec) -> str:
        tokens = []

        if dim_spec.broadcastable:
            tokens.append("broadcast ")

        if dim_spec.constant is not None:
            tokens.append(str(dim_spec.constant))
        elif dim_spec.variable_name:
            tokens.append(f"*{dim_spec.variable_name}*")
        else:
            if not dim_spec.variable_rank:
                tokens.append(".")

        if dim_spec.variable_rank:
            tokens.append("...")

        return "".join(tokens)

    def _guess_indent(self, docstring: str) -> Optional[int]:
        """
        Infer the level of indentation of a docstring.

        Returns `None` if the indentation could not be inferred.
        """
        # Algorithm adapted from:
        #     https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation

        # Convert tabs to spaces (following the normal Python rules)
        # and split into a list of lines:
        lines = docstring.expandtabs().splitlines()
        # Determine minimum indentation (first line doesn't count):
        no_indent = -1
        indent = no_indent
        for line in lines[1:]:
            stripped = line.lstrip()
            if not stripped:
                continue
            line_indent = len(line) - len(stripped)
            if indent == no_indent or line_indent < indent:
                indent = line_indent
        return indent if indent != no_indent else None

    def _insert_spec_lines(
        self, out: List[str], pos: int, spec_lines: Sequence[str], docs: Tree[Token]
    ) -> int:
        leading_str = self._source[pos : docs.meta.start_pos].rstrip()
        docs_start = pos + len(leading_str)
        docs_str = self._source[docs_start : docs.meta.end_pos]
        trailing_str = docs_str.lstrip()

        docs_indent = self._guess_indent(docs_str)
        if docs_indent is None:
            if self._indent is None:
                docs_indent = 4
            else:
                docs_indent = self._indent + 4
        indent_str = "\n" + docs_indent * " "

        out.append(leading_str)
        for spec_line in spec_lines:
            out.append(indent_str)
            out.append(spec_line)
        out.append("\n")
        out.append(indent_str)
        out.append(trailing_str)
        return docs.meta.end_pos

    def _insert_param_info_fields(
        self,
        is_first_info_field: bool,
        spec_lines: Mapping[str, Sequence[str]],
        out: List[str],
        pos: int,
    ) -> int:
        leading_str = self._source[pos:].rstrip()
        out.append(leading_str)
        pos += len(leading_str)

        if not self._source:
            # Case where nothing preceeds these fields. Just write them.
            needed_newlines = 0
        elif is_first_info_field:
            # Free-form documentation preceeds these fields. Have 2 newlines to separate them.
            needed_newlines = 2
        else:
            # Another info-field preceeds these fields.
            needed_newlines = 1

        indent = self._indent or 0
        indent_str = indent * " "
        indent_one_str = 4 * " "

        for arg_name, arg_lines in spec_lines.items():
            out.append(needed_newlines * "\n")
            needed_newlines = 1

            out.append(indent_str)
            if arg_name == RESULT_TOKEN:
                out.append(":returns:")
            else:
                out.append(f":param {arg_name}:")
            for arg_line in arg_lines:
                out.append("\n")
                out.append(indent_str)
                out.append(indent_one_str)
                out.append(arg_line)

        return pos

    def docstring(self, tree: Tree[Token]) -> str:
        # The strategy here is:
        # * `out` contains a list of strings that will be concatenated and form the final result.
        # * `pos` is the position such that `self._source[:pos]` has already been added to `out`,
        #   and `self._source[pos:]` still needs to be added.
        # * When visiting children we pass `out` and `pos`, and the children add content to `out`
        #   and return a new `pos`.
        docs, info_fields = _tree_children(tree)
        out: List[str] = []
        pos = 0

        if self._notes:
            if not docs.meta.empty:
                out.append(self._source[pos : docs.meta.end_pos])
                pos = docs.meta.end_pos
            indent = self._indent or 0
            indent_str = indent * " "
            for note in self._notes:
                if out:
                    out.append("\n\n")
                out.append(indent_str)
                out.append(note)

        pos = self.visit(info_fields, out, pos)
        out.append(self._source[pos:])

        return "".join(out)

    def info_fields(self, tree: Tree[Token], out: List[str], pos: int) -> int:
        spec_lines = dict(self._spec_lines)
        is_first_info_field = True
        for child in _tree_children(tree):
            # This will remove the self._spec_lines corresponding to found `:param:`'s.
            pos = self.visit(child, spec_lines, out, pos)
            is_first_info_field = False

        # Add any remaining `:param:`s:
        pos = self._insert_param_info_fields(is_first_info_field, spec_lines, out, pos)

        # Make sure info fields are terminated by a new-line:
        if self._spec_lines:
            if (pos >= len(self._source)) or (self._source[pos] != "\n"):
                out.append("\n")

        return pos

    def info_field_param(
        self, tree: Tree[Token], spec_lines: Dict[str, Sequence[str]], out: List[str], pos: int
    ) -> int:
        info_field_args, docs = _tree_children(tree)
        arg_name = self.visit(info_field_args)
        arg_lines = spec_lines.pop(arg_name, None)
        if arg_lines:
            pos = self._insert_spec_lines(out, pos, arg_lines, docs)
        return pos

    def info_field_returns(
        self, tree: Tree[Token], spec_lines: Dict[str, Sequence[str]], out: List[str], pos: int
    ) -> int:
        (docs,) = _tree_children(tree)
        return_lines = spec_lines.pop(RESULT_TOKEN, None)
        if return_lines:
            pos = self._insert_spec_lines(out, pos, return_lines, docs)
        return pos

    def info_field_other(
        self, tree: Tree[Token], spec_lines: Dict[str, Sequence[str]], out: List[str], pos: int
    ) -> int:
        return pos

    def info_field_args(self, tree: Tree[Token]) -> str:
        tokens = list(_token_children(tree))
        if not tokens:
            return ""
        return tokens[-1]


class _CachedParser:
    """
    Small wrapper around Lark so that we can reuse as much code as possible between the different
    things we parse.
    """

    def __init__(
        self,
        grammar_filename: str,
        start_symbol: str,
        parser_name: str,
        re_terminal_descriptions: Mapping[str, str],
        transformer_class: Type[_TreeVisitor],
        exception_class: Type[CheckShapesError],
    ) -> None:
        self._cache: Dict[Tuple[str, Tuple[Any, ...]], Any] = {}
        self._parser = Lark.open(
            grammar_filename,
            rel_to=__file__,
            propagate_positions=True,
            start=start_symbol,
            parser=parser_name,
        )
        self._terminal_descriptions = {}
        self._transformer_class = transformer_class
        self._exception_class = exception_class

        # Pre-compute nice terminal descriptions for our error messages:
        missing = set()
        unused = dict(re_terminal_descriptions)
        for terminal in self._parser.terminals:
            name = terminal.name
            pattern = terminal.pattern
            if isinstance(pattern, PatternStr):
                description = f'"{pattern.value}"'
            else:
                assert isinstance(pattern, PatternRE)
                unused_description = unused.pop(name, None)
                # If we enter this `if` then the parser is misconfigured, so we never get here, even
                # during tests.
                if unused_description is None:  # pragma: no cover
                    missing.add(name)
                    description = "ERROR"
                else:
                    description = f"{re_terminal_descriptions[name]} (re={pattern.value})"
            self._terminal_descriptions[name] = description
        assert not unused, f"Redundant terminal descriptions were provided: {sorted(unused)}"
        assert not missing, f"Some RE terminals did not have a description: {sorted(missing)}"

    def parse(self, text: str, transformer_args: Tuple[Any, ...], context: ErrorContext) -> Any:
        sentinel = object()
        cache_key = (text, transformer_args)
        result = self._cache.get(cache_key, sentinel)
        if result is sentinel:
            try:
                tree = self._parser.parse(text)
            except UnexpectedInput as ui:
                raise self._exception_class(
                    StackContext(
                        context, LarkUnexpectedInputContext(text, ui, self._terminal_descriptions)
                    )
                ) from ui

            try:
                result = self._transformer_class(*transformer_args).visit(tree)
            except CheckShapesError as cse:
                raise self._exception_class(StackContext(context, cse.context)) from cse

            self._cache[cache_key] = result
        return result


_TENSOR_SPEC_PARSER = _CachedParser(
    grammar_filename="check_shapes.lark",
    start_symbol="tensor_spec",
    parser_name="lalr",
    re_terminal_descriptions={
        "NOTE_TEXT": "note / comment text",
        "CNAME": "variable name",
        "INT": "integer",
        "WS": "whitespace",
    },
    transformer_class=_ParseSpec,
    exception_class=SpecificationParseError,
)
_ARGUMENT_SPEC_PARSER = _CachedParser(
    grammar_filename="check_shapes.lark",
    start_symbol="argument_or_note_spec",
    parser_name="lalr",
    re_terminal_descriptions={
        "NOTE_TEXT": "note / comment text",
        "CNAME": "variable name",
        "INT": "integer",
        "WS": "whitespace",
    },
    transformer_class=_ParseSpec,
    exception_class=SpecificationParseError,
)
_SPHINX_DOCSTRING_PARSER = _CachedParser(
    grammar_filename="docstring.lark",
    start_symbol="docstring",
    parser_name="earley",
    re_terminal_descriptions={
        "ANY": "any text",
        "CNAME": "variable name",
        "INFO_FIELD_OTHER": "Sphinx info field",
        "PARAM": "Sphinx parameter field",
        "RETURNS": "Sphinx `return` field",
        "WS": "whitespace",
    },
    transformer_class=_RewritedocString,
    exception_class=DocstringParseError,
)


def parse_tensor_spec(tensor_spec: str, context: ErrorContext) -> ParsedTensorSpec:

    """
    Parse a `check_shapes` tensor specification.
    """
    result = _TENSOR_SPEC_PARSER.parse(tensor_spec, (tensor_spec,), context)
    assert isinstance(result, ParsedTensorSpec)
    return result


def parse_function_spec(function_spec: Sequence[str], context: ErrorContext) -> ParsedFunctionSpec:
    """
    Parse all `check_shapes` argument or note specification for a single function.
    """
    arguments = []
    notes = []
    for i, spec in enumerate(function_spec):
        argument_context = StackContext(context, ArgumentContext(i))
        parsed_spec = _ARGUMENT_SPEC_PARSER.parse(spec, (spec,), argument_context)
        if isinstance(parsed_spec, ParsedArgumentSpec):
            arguments.append(parsed_spec)
        else:
            assert isinstance(parsed_spec, ParsedNoteSpec)
            notes.append(parsed_spec)
    return ParsedFunctionSpec(tuple(arguments), tuple(notes))


def parse_and_rewrite_docstring(
    docstring: Optional[str], function_spec: ParsedFunctionSpec, context: ErrorContext
) -> Optional[str]:
    """
    Rewrite `docstring` to include the shapes specified by the `argument_specs`.
    """
    if docstring is None:
        return None

    docstring_format = get_rewrite_docstrings()
    if docstring_format == DocstringFormat.NONE:
        return docstring

    assert docstring_format == DocstringFormat.SPHINX, (
        f"Current docstring format is {docstring_format}, but I don't know how to rewrite that."
        " See `gpflow.experimental.check_shapes.config.set_rewrite_docstrings`."
    )
    result = _SPHINX_DOCSTRING_PARSER.parse(docstring, (docstring, function_spec), context)
    assert isinstance(result, str)
    return result
