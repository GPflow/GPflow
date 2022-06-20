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
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional, Sequence

import pytest

import gpflow


def is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def get_module_path(module: ModuleType) -> Path:
    path_str = module.__file__
    assert path_str is not None
    return Path(path_str)


def is_package(module: ModuleType) -> bool:
    return get_module_path(module).name == "__init__.py"


def find_modules(root: ModuleType) -> Sequence[ModuleType]:
    root_path = get_module_path(root).parent
    parent = root_path.parent
    result: List[ModuleType] = []
    for path in root_path.glob("**/*.py"):
        relative_path = path.relative_to(parent)
        if relative_path.name == "__init__.py":
            module_name__slashes = str(relative_path.parent)
        else:
            module_name__slashes = str(relative_path)[:-3]
        module_name = module_name__slashes.replace("/", ".")
        result.append(importlib.import_module(module_name))
    return result


_MODULES = find_modules(gpflow)
_PACKAGES = [m for m in _MODULES if is_package(m)]
_MODULES_WITH_ALL = [m for m in _MODULES if hasattr(m, "__all__")]


@pytest.mark.parametrize("package", _PACKAGES)
def test_all_present_and_up_to_date(package: ModuleType) -> None:
    imported_sorted = sorted(attr for attr in dir(package) if not is_dunder(attr))
    all_list = getattr(package, "__all__", None)
    assert all_list is not None, f"Package {package} is missing an explicit __all__."
    all_sorted = sorted(attr for attr in all_list if not is_dunder(attr))
    assert imported_sorted == all_sorted, (
        f"{package}.__all__ is outdated."
        f" Imported values are {imported_sorted}, but exported values are {all_list}."
    )


@pytest.mark.parametrize("module", _MODULES_WITH_ALL)
def test_all_sorted(module: ModuleType) -> None:
    all_list = getattr(module, "__all__", None)
    assert all_list is not None  # Hint for mypy.
    all_sorted = sorted(all_list)
    assert (
        all_sorted == all_list
    ), f"{module}.__all__ is not sorted. Expected {all_sorted}, found {all_list}."


@pytest.mark.parametrize("module", _MODULES_WITH_ALL)
def test_all_static(module: ModuleType) -> None:
    module_path = get_module_path(module)
    tree = ast.parse(module_path.read_text(), str(module_path))
    assert isinstance(tree, ast.Module)

    def is_all_assignment(t: Any) -> bool:
        return (
            isinstance(t, ast.Assign)
            and len(t.targets) == 1
            and isinstance(t.targets[0], ast.Name)
            and t.targets[0].id == "__all__"
        )

    all_assignment: Optional[ast.Assign] = None
    for t in tree.body:
        if is_all_assignment(t):
            all_assignment = t  # type: ignore[assignment]
    assert all_assignment is not None, f"No `__all__` found in module {module}."

    error_message = "__all__ must be a static list of constant strings. Some tools expect this."

    all_value = all_assignment.value
    assert isinstance(all_value, ast.List), error_message

    assert all(
        # Actual type depends on Python version:
        (isinstance(t, ast.Constant) or isinstance(t, ast.Str))
        for t in all_value.elts
    ), error_message
