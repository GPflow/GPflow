# Copyright 2019 GPflow Authors
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
"""Script to autogenerate .rst files for autodocumentation of classes and modules in GPflow.
To be run by the CI system to update docs.
"""
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Deque, Dict, List, Mapping, Set, TextIO, Type, Union

from gpflow.utilities import Dispatcher

RST_LEVEL_SYMBOLS = ["=", "-", "~", '"', "'", "^"]

IGNORE_MODULES = {
    "gpflow.covariances.dispatch",
    "gpflow.conditionals.dispatch",
    "gpflow.expectations.dispatch",
    "gpflow.kullback_leiblers.dispatch",
    "gpflow.versions",
}


def _header(header: str, level: int) -> str:
    return f"{header}\n{RST_LEVEL_SYMBOLS[level] * len(header)}"


@dataclass
class DocumentableDispatcher:

    name: str
    obj: Dispatcher

    def implementations(self) -> Mapping[Callable[..., Any], List[Type[Any]]]:
        implementations: Dict[Callable[..., Any], List[Type[Any]]] = {}
        for args, impl in self.obj.funcs.items():
            implementations.setdefault(impl, []).append(args)
        return implementations

    def write(self, out: TextIO) -> None:
        out.write(
            f"""
{_header(self.name, 2)}

This function uses multiple dispatch, which will depend on the type of argument passed in:
"""
        )
        for impl, argss in self.implementations().items():
            impl_name = f"{impl.__module__}.{impl.__name__}"

            out.write(
                """
.. code-block:: python

"""
            )
            for args in argss:
                arg_names = ", ".join([a.__name__ for a in args])
                out.write(f"    {self.name}( {arg_names} )\n")
            out.write(f"    # dispatch to -> {impl_name}(...)\n")
            out.write(
                f"""
.. autofunction:: {impl_name}
"""
            )


@dataclass
class DocumentableClass:

    name: str
    obj: Type[Any]

    def write(self, out: TextIO) -> None:
        out.write(
            f"""
{_header(self.name, 2)}

.. autoclass:: {self.name}
   :show-inheritance:
   :members:
"""
        )


@dataclass
class DocumentableFunction:

    name: str
    obj: Callable[..., Any]

    def write(self, out: TextIO) -> None:
        out.write(
            f"""
{_header(self.name, 2)}

.. autofunction:: {self.name}
"""
        )


@dataclass
class DocumentableModule:

    name: str
    obj: ModuleType
    modules: List["DocumentableModule"]
    classes: List[DocumentableClass]
    functions: List[Union[DocumentableDispatcher, DocumentableFunction]]

    @staticmethod
    def collect(
        root: ModuleType,
    ) -> "DocumentableModule":
        root_name = root.__name__
        exported_names = set(getattr(root, "__all__", []))

        modules: List["DocumentableModule"] = []
        classes: List[DocumentableClass] = []
        functions: List[Union[DocumentableDispatcher, DocumentableFunction]] = []

        for key in dir(root):
            if key.startswith("_"):
                continue

            child = getattr(root, key)
            child_name = root_name + "." + key
            if child_name in IGNORE_MODULES:
                continue

            # pylint: disable=cell-var-from-loop
            def _should_ignore(child: Union[Callable[..., Any], Type[Any]]) -> bool:
                declared_in_root = child.__module__ == root_name
                explicitly_exported = key in exported_names
                return not (declared_in_root or explicitly_exported)

            # pylint: enable=cell-var-from-loop

            if isinstance(child, Dispatcher):
                functions.append(DocumentableDispatcher(child_name, child))
            elif inspect.ismodule(child):
                if child.__name__ != child_name:  # Ignore imports of modules.
                    continue
                modules.append(DocumentableModule.collect(child))
            elif inspect.isclass(child):
                if _should_ignore(child):
                    continue
                classes.append(DocumentableClass(child_name, child))
            elif inspect.isfunction(child):
                if _should_ignore(child):
                    continue
                functions.append(DocumentableFunction(child_name, child))

        return DocumentableModule(root_name, root, modules, classes, functions)

    def seen_in_dispatchers(self, seen: Set[int]) -> None:
        for module in self.modules:
            module.seen_in_dispatchers(seen)
        for function in self.functions:
            if isinstance(function, DocumentableDispatcher):
                impls = function.obj.funcs.values()
                for impl in impls:
                    seen.add(id(impl))

    def prune_duplicates(self) -> None:
        seen: Set[int] = set()
        self.seen_in_dispatchers(seen)

        # Breadth-first search so that we prefer objects with shorter names.
        todo = Deque([self])
        while todo:
            module = todo.popleft()

            new_classes = []
            for c in module.classes:
                if id(c.obj) not in seen:
                    seen.add(id(c.obj))
                    new_classes.append(c)
            module.classes = new_classes

            new_functions = []
            for f in module.functions:
                if id(f.obj) not in seen:
                    seen.add(id(f.obj))
                    new_functions.append(f)
            module.functions = new_functions

            todo.extend(module.modules)

    def prune_empty_modules(self) -> None:
        new_modules = []
        for m in self.modules:
            m.prune_empty_modules()

            if m.modules or m.classes or m.functions:
                new_modules.append(m)
        self.modules = new_modules

    def prune(self) -> None:
        self.prune_duplicates()
        self.prune_empty_modules()

    def write_modules(self, out: TextIO) -> None:
        if not self.modules:
            return

        out.write(
            f"""
{_header('Modules', 1)}

.. toctree::
   :maxdepth: 1

"""
        )
        for module in self.modules:
            out.write(f"   {module.name} <{module.name.split('.')[-1]}/index>\n")

    def write_classes(self, out: TextIO) -> None:
        if not self.classes:
            return

        out.write(
            f"""
{_header('Classes', 1)}
"""
        )
        for cls in self.classes:
            cls.write(out)

    def write_functions(self, out: TextIO) -> None:
        if not self.functions:
            return

        out.write(
            f"""
{_header('Functions', 1)}
"""
        )
        for function in self.functions:
            function.write(out)

    def write(self, path: Path) -> None:
        dir_path = path / f"{self.name.replace('.', '/')}"
        dir_path.mkdir(parents=True, exist_ok=True)
        index_path = dir_path / "index.rst"
        with index_path.open("wt") as out:
            print("Writing", index_path)
            out.write(
                f"""{_header(self.name, 0)}

.. THIS IS AN AUTOGENERATED RST FILE

.. automodule:: {self.name}
"""
            )
            self.write_modules(out)
            self.write_classes(out)
            self.write_functions(out)

        for module in self.modules:
            module.write(path)

    def str_into(self, indent: int, lines: List[str]) -> None:
        lines.append(2 * indent * " " + "Module: " + self.name)
        for module in self.modules:
            module.str_into(indent + 1, lines)
        for cls in self.classes:
            lines.append(2 * (indent + 1) * " " + "Class: " + cls.name)
        for function in self.functions:
            lines.append(2 * (indent + 1) * " " + "Function: " + function.name)

    def __str__(self) -> str:
        lines: List[str] = []
        self.str_into(0, lines)
        return "\n".join(lines)


def generate_module_rst(module: ModuleType, dest: Path) -> None:
    """
    Traverses the given `module` and generates `.rst` files for Sphinx.
    """
    docs = DocumentableModule.collect(module)
    docs.prune()
    docs.write(dest)
