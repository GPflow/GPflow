# Copyright 2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

from multipledispatch import Dispatcher as GeneratorDispatcher
from multipledispatch.dispatcher import str_signature, variadic_signature_matches
from multipledispatch.variadic import isvariadic

__all__ = ["Dispatcher"]


AnyCallable = Callable[..., Any]
_C = TypeVar("_C", bound=AnyCallable)
Types = Union[Type[Any], Tuple[Type[Any], ...]]


class Dispatcher(GeneratorDispatcher):
    """
    multipledispatch.Dispatcher uses a generator to yield the
    desired function implementation, which is problematic as TensorFlow's
    autograph is not able to compile code that passes through generators.

    This class overwrites the problematic method in the original
    Dispatcher and solely makes use of simple for-loops, which are
    compilable by AutoGraph.
    """

    def register(self, *types: Types, **kwargs: Any) -> Callable[[_C], _C]:
        # Override to add type hints...
        result: Callable[[_C], _C] = super().register(*types, **kwargs)
        return result

    def dispatch(self, *types: Types) -> Optional[AnyCallable]:
        """
        Returns matching function for `types`; if not existing returns None.
        """
        if types in self.funcs:
            result: AnyCallable = self.funcs[types]
            return result

        return self.get_first_occurrence(*types)

    def dispatch_or_raise(self, *types: Types) -> AnyCallable:
        """
        Returns matching function for `types`; if not existing raises an error.
        """
        f = self.dispatch(*types)
        if f is None:
            raise NotImplementedError(
                f"Could not find signature for {self.name}: <{str_signature(types)}>"
            )
        return f

    def get_first_occurrence(self, *types: Types) -> Optional[AnyCallable]:
        """
        Returns the first occurrence of a matching function

        Based on `multipledispatch.Dispatcher.dispatch_iter`, which
        returns an iterator of matching functions. This method uses
        the same logic to select functions, but simply returns the first
        element of the iterator. If no matching functions are found,
        `None` is returned.
        """
        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(map(issubclass, types, signature)):  # type: ignore[arg-type]
                result: AnyCallable = self.funcs[signature]
                return result
            elif len(signature) and isvariadic(signature[-1]):
                if variadic_signature_matches(types, signature):
                    result = self.funcs[signature]
                    return result
        return None
