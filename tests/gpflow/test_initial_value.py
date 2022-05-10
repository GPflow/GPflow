# Copyright 2022 the GPflow authors.
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
import ast
from pathlib import Path
from typing import List, Sequence, Union

import pytest

import gpflow


def find_py_files() -> Sequence[Path]:
    """
    Find all ``.py`` files of the GPflow user-facing code.
    """
    root_module = gpflow
    root_path_str = root_module.__file__
    assert root_path_str is not None
    root_path = Path(root_path_str).parent
    return tuple(root_path.glob("**/*.py"))


_PY_FILES = find_py_files()


@pytest.mark.parametrize("path", _PY_FILES, ids=str)
def test_assignment_value(path: Path) -> None:
    """
    Python allows assignments without a value, generally used for typing. For example::

        @tf.function
        def foo(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
            result: tf.Tensor  # <-- Assignment without a value. Only for typing.
            result = a + b
            return result

    However, ``TensorFlow < 2.9.0`` cannot compile these.

    This test parses all our ``.py`` files and look for assignments without a value, to ensure all
    our code can be compiled by TensorFlow.

    See also: https://github.com/tensorflow/tensorflow/issues/56024
    """

    tree = ast.parse(path.read_text(), str(path))

    class RequireAssignmentValue(ast.NodeVisitor):
        """
        When ``.visit(tree)`` is called this will recursively visit all nodes of ``tree``.

        Visitor methods are called based on the types of the nodes of the tree. If a node has type
        ``Foo``, then ``visit_Foo(...)`` below will be called. If no such method exists
        ``generic_visit`` is called insted, which just visits the children of the node recursively.
        """

        def __init__(self) -> None:
            super().__init__()

            self.in_function_stack: List[bool] = []
            """
            Keeps track of whether we're inside a function.  We only raise errors for assignments in
            functions - I think variables on classes are handled differently, and doesn't cause
            these problems.
            """

            self.bad_assigns: List[ast.AST] = []
            """
            List of bad assignments, so that we can report multiple errors at the same time.
            """

        def set_in_function(self, node: ast.AST, in_function: bool) -> None:
            """
            Set whether we're inside a function, and visit children recursively.
            """
            self.in_function_stack.append(in_function)
            self.generic_visit(node)
            self.in_function_stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """
            Called for class definitions.
            """
            self.set_in_function(node, False)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """
            Called for function (and method) definitions.
            """
            self.set_in_function(node, True)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """
            Called for async function definitions.
            """
            self.set_in_function(node, True)

        def check_assign(self, node: Union[ast.Assign, ast.AugAssign, ast.AnnAssign]) -> None:
            """
            Checks whether this is a valid assignment, and visit children recursively.
            """
            if self.in_function_stack and self.in_function_stack[-1]:
                if not node.value:
                    self.bad_assigns.append(node)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            """
            Called for a regular assignment.
            """
            self.check_assign(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            """
            Called for +=, -=, etc.
            """
            self.check_assign(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            """
            Called for an assigment with a type annotation.
            """
            self.check_assign(node)

    visitor = RequireAssignmentValue()
    visitor.visit(tree)

    if visitor.bad_assigns:
        lines = [""]
        lines.extend(f"{path}:{n.lineno}" for n in visitor.bad_assigns)
        lines.append(
            "TensorFlow cannot compile assignments without a value,"
            " so for performance reasons we require all assgnments to have a value."
            " See: https://github.com/tensorflow/tensorflow/issues/56024 "
        )
        raise AssertionError("\n".join(lines))
