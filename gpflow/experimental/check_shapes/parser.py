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
from functools import lru_cache
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lark import Lark, Token, Tree

from .argument_ref import (
    RESULT_TOKEN,
    ArgumentRef,
    AttributeArgumentRef,
    IndexArgumentRef,
    RootArgumentRef,
)
from .specs import ParsedArgumentSpec, ParsedDimensionSpec, ParsedShapeSpec


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


class _ParseArgumentSpec(_TreeVisitor):
    def argument_spec(self, tree: Tree[Token]) -> ParsedArgumentSpec:
        argument_name, argument_refs, shape_spec = _tree_children(tree)
        root_argument_ref = self.visit(argument_name)
        argument_ref = self.visit(argument_refs, root_argument_ref)
        shape = self.visit(shape_spec, argument_ref)
        return ParsedArgumentSpec(argument_ref, shape)

    def argument_name(self, tree: Tree[Token]) -> ArgumentRef:
        (token,) = _token_children(tree)
        return RootArgumentRef(token)

    def argument_refs(self, tree: Tree[Token], result: ArgumentRef) -> ArgumentRef:
        for argument_ref in _tree_children(tree):
            result = self.visit(argument_ref, result)
        return result

    def argument_ref_attribute(self, tree: Tree[Token], source: ArgumentRef) -> ArgumentRef:
        (token,) = _token_children(tree)
        return AttributeArgumentRef(source, token)

    def argument_ref_index(self, tree: Tree[Token], source: ArgumentRef) -> ArgumentRef:
        (token,) = _token_children(tree)
        return IndexArgumentRef(source, int(token))

    def shape_spec(self, tree: Tree[Token], argument_ref: ArgumentRef) -> ParsedShapeSpec:
        (dimension_specs,) = _tree_children(tree)
        return ParsedShapeSpec(self.visit(dimension_specs))

    def dimension_specs(self, tree: Tree[Token]) -> Tuple[ParsedDimensionSpec, ...]:
        return tuple(self.visit(dimension_spec) for dimension_spec in _tree_children(tree))

    def dimension_spec_constant(self, tree: Tree[Token]) -> ParsedDimensionSpec:
        (token,) = _token_children(tree)
        return ParsedDimensionSpec(constant=int(token), variable_name=None, variable_rank=False)

    def dimension_spec_variable(self, tree: Tree[Token]) -> ParsedDimensionSpec:
        (token,) = _token_children(tree)
        return ParsedDimensionSpec(constant=None, variable_name=token, variable_rank=False)

    def dimension_spec_variable_rank(self, tree: Tree[Token]) -> ParsedDimensionSpec:
        (token,) = _token_children(tree)
        return ParsedDimensionSpec(constant=None, variable_name=token, variable_rank=True)


class _RewriteDocString(_TreeVisitor):
    def __init__(self, source: str, argument_specs: Collection[ParsedArgumentSpec]) -> None:
        self._source = source
        self._spec_lines = self._argument_specs_to_sphinx(argument_specs)
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
        out = []
        out.append(f"* **{repr(argument_spec.argument_ref)}**")
        out.append(" has shape [")
        out.append(self._shape_spec_to_sphinx(argument_spec.shape))
        out.append("].")
        return "".join(out)

    def _shape_spec_to_sphinx(self, shape_spec: ParsedShapeSpec) -> str:
        out = []
        for dim in shape_spec.dims:
            if dim.constant is not None:
                out.append(str(dim.constant))
            if dim.variable_name is not None:
                suffix = "..." if dim.variable_rank else ""
                out.append(f"*{dim.variable_name}*{suffix}")
        return ", ".join(out)

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

    def docstring(self, tree: Tree[Token]) -> str:
        # The strategy here is:
        # * `out` contains a list of strings that will be concatenated and form the final result.
        # * `pos` is the position such that `self._source[:pos]` has already been added to `out`,
        #   and `self._source[pos:]` still needs to be added.
        # * When visiting children we pass `out` and `pos`, and the children add content to `out`
        #   and return a new `pos`.
        _docs, info_fields = _tree_children(tree)
        out: List[str] = []
        pos = 0
        pos = self.visit(info_fields, out, pos)
        out.append(self._source[pos:])
        return "".join(out)

    def info_fields(self, tree: Tree[Token], out: List[str], pos: int) -> int:
        for child in _tree_children(tree):
            pos = self.visit(child, out, pos)
        return pos

    def info_field_param(self, tree: Tree[Token], out: List[str], pos: int) -> int:
        info_field_args, docs = _tree_children(tree)
        arg_name = self.visit(info_field_args)
        spec_lines = self._spec_lines.get(arg_name, None)
        if spec_lines:
            pos = self._insert_spec_lines(out, pos, spec_lines, docs)
        return pos

    def info_field_returns(self, tree: Tree[Token], out: List[str], pos: int) -> int:
        (docs,) = _tree_children(tree)
        spec_lines = self._spec_lines.get(RESULT_TOKEN, None)
        if spec_lines:
            pos = self._insert_spec_lines(out, pos, spec_lines, docs)
        return pos

    def info_field_other(self, tree: Tree[Token], out: List[str], pos: int) -> int:
        return pos

    def info_field_args(self, tree: Tree[Token]) -> str:
        tokens = list(_token_children(tree))
        if not tokens:
            return ""
        return tokens[-1]


_DOCSTRING_PARSER = Lark.open(
    "docstring.lark", rel_to=__file__, propagate_positions=True, start="docstring"
)


_ARGUMENT_SPEC_PARSER = Lark.open(
    "check_shapes.lark",
    rel_to=__file__,
    propagate_positions=True,
    start="argument_spec",
    parser="lalr",
)


@lru_cache(maxsize=None)
def parse_argument_spec(argument_spec: str) -> ParsedArgumentSpec:
    """
    Parse a `check_shapes` argument specification.
    """
    tree = _ARGUMENT_SPEC_PARSER.parse(argument_spec)
    specs = _ParseArgumentSpec().visit(tree)
    return specs


@lru_cache(maxsize=None)
def parse_and_rewrite_docstring(
    docstring: Optional[str], argument_specs: Tuple[ParsedArgumentSpec, ...]
) -> Optional[str]:
    """
    Rewrite `docstring` to include the shapes specified by the `argument_specs`.
    """
    if docstring is None:
        return None

    tree = _DOCSTRING_PARSER.parse(docstring)
    return _RewriteDocString(docstring, argument_specs).visit(tree)
