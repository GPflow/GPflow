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
"""
Class responsible for remembering and checking shapes.
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union

from ..utils import experimental
from .base_types import Shape
from .config import get_enable_check_shapes
from .error_contexts import (
    ErrorContext,
    FunctionCallContext,
    IndexContext,
    NoteContext,
    ParallelContext,
    ShapeContext,
    StackContext,
    TensorSpecContext,
    VariableContext,
)
from .exceptions import ShapeMismatchError, VariableTypeError
from .parser import parse_tensor_spec
from .shapes import get_shape
from .specs import ParsedDimensionSpec, ParsedShapeSpec, ParsedTensorSpec

S = TypeVar("S")


TensorSpecLike = Union[ParsedTensorSpec, str, Shape]


def _as_parsed_tensor_spec(tensor_spec: TensorSpecLike, context: ErrorContext) -> ParsedTensorSpec:
    if isinstance(tensor_spec, ParsedTensorSpec):
        return tensor_spec
    elif isinstance(tensor_spec, str):
        return parse_tensor_spec(tensor_spec, context)
    else:
        dimension_specs = []
        if isinstance(tensor_spec, tuple):
            for dim in tensor_spec:
                if isinstance(dim, int):
                    dimension_specs.append(ParsedDimensionSpec(dim, None, False, False))
                else:
                    assert dim is None
                    dimension_specs.append(ParsedDimensionSpec(None, None, False, False))
        else:
            assert tensor_spec is None
            dimension_specs.append(ParsedDimensionSpec(None, None, True, False))
        shape = ParsedShapeSpec(tuple(dimension_specs))
        return ParsedTensorSpec(shape, None)


@dataclass
class _ObservedDim:
    """
    Storage of observed size of a single dimension variable.
    """

    size: Optional[int] = None

    def check_and_update(self, actual: Optional[int], broadcast: bool) -> bool:
        """
        Attempt to merge new data into this observation.

        Returns whether the new data is compatible with existing observations.  If this returns
        `False` it may have been left in an invalid state and should not be used again.
        """
        if (actual is None) or (broadcast and actual == 1):
            # Update contains no information. Nothing to do.
            return True

        if self.size is None:
            self.size = actual

        return self.size == actual


@dataclass
class _ObservedDims:
    """
    Storage of observed sizes of a var-rank / batch variable.
    """

    sizes: Optional[List[_ObservedDim]] = None
    known_rank: bool = False

    def check_and_update(
        self, actual: Optional[Tuple[Optional[int], ...]], broadcast: bool
    ) -> bool:
        """
        Attempt to merge new data into this observation.

        Returns whether the new data is compatible with existing observations.  If this returns
        `False` it may have been left in an invalid state and should not be used again.
        """
        if actual is None:
            # Update contains no information. Nothing to do.
            return True

        if self.sizes is None:
            self.sizes = []
            assert not self.known_rank

        # First make sure lengths are set up and matches.
        longer = len(self.sizes) - len(actual)
        if self.known_rank:
            if broadcast:
                if longer < 0:
                    return False
            else:
                if longer != 0:
                    return False
        else:
            if longer < 0:
                self.sizes = [_ObservedDim() for _ in range(-longer)] + self.sizes
                longer = 0
            if broadcast:
                pass  # We don't know anything about total rank.
            else:
                if longer > 0:
                    return False
                self.known_rank = True
        assert longer >= 0

        # Then match individual dimensions.
        for i, actual_dim in enumerate(actual):
            if not self.sizes[i + longer].check_and_update(actual_dim, broadcast):
                return False

        return True


class ShapeChecker:
    """
    Mechanism for checking the shapes of tensors.

    This remembers observed shapes and specifications, so that tensors can be checked for
    compatibility across multiple calls, and so that we can provide good error messages.

    Example:

    .. literalinclude:: /examples/test_check_shapes_examples.py
       :start-after: [shape_checker__raw]
       :end-before: [shape_checker__raw]
       :dedent:
    """

    @experimental
    def __init__(self) -> None:
        self._seen_shapes: List[Tuple[Shape, ParsedTensorSpec, ErrorContext]] = []

        # Here we're (ab-)using `Dict[ , None]` instead of `set`, because `dict`s retain ordering.
        self._specs_by_variable: Dict[str, Dict[Tuple[ParsedTensorSpec, ErrorContext], None]] = {}
        self._rank1_variables: Set[str] = set()
        self._varrank_variables: Set[str] = set()
        self._additional_context: List[ErrorContext] = []
        self._seen_dims: Dict[str, Union[_ObservedDim, _ObservedDims]] = {}

    def add_context(self, context: ErrorContext) -> None:
        """
        Add arbirtary context to the shape checker.

        This context will be included in any error messages.

        :param context: Context to add to this shape checker.
        """
        self._additional_context.append(context)

    def check_shape(
        self, shaped: S, tensor_spec: TensorSpecLike, context: Optional[ErrorContext] = None
    ) -> S:
        """
        Raise an error if a tensor has the wrong shape.

        This remembers observed shapes and specifications, so that tensors can be checked for
        compatibility across multiple calls, and so that we can provide good error messages.

        :param shaped: The object whose shape to check.
        :param tensor_spec: Specification to check the tensor against.
            Usually this is a ``str`` in the format described in `Shape specification`_.
            Alternatively this can be a pre-parsed :class:`ParsedTensorSpec`, or an actual
            :class:`Shape`.
        :param context: Information about where ``shaped`` is coming from, for improved error
            messages.
        :returns: ``shaped``, for convenience.
        """
        if not get_enable_check_shapes():
            return shaped

        if context is None:
            context = FunctionCallContext(self.check_shape).precompute()

        self.check_shapes([(shaped, tensor_spec, context)])

        return shaped

    def check_shapes(
        self,
        checks: Iterable[
            Union[Tuple[Any, TensorSpecLike], Tuple[Any, TensorSpecLike, ErrorContext]]
        ],
    ) -> None:
        """
        Raise an error if any tensor has the wrong shape.

        This remembers observed shapes and specifications, so that tensors can be checked for
        compatibility across multiple calls, and so that we can provide good error messages.

        :param checks: Checks to perform. The elements can either be ``(shaped, tensor_spec)`` or
            ``(shaped, tensor_spec, context)`` tuples. Where: ``shaped`` is the tensor whose shape
            to check; ``tensor_spec`` is the specification to check it against (see `Shape
            specification`_); and ``context`` contains (optional) information about where ``shaped``
            came from - for better error messages.
        """
        if not get_enable_check_shapes():
            return

        new_shapes = []
        new_variables = set()
        call_context: Optional[ErrorContext] = None
        for i, check in enumerate(checks):
            shaped, tensor_spec, *contexts = check
            if contexts:
                (context,) = contexts
            else:
                if call_context is None:
                    call_context = FunctionCallContext(self.check_shapes).precompute()
                context = StackContext(call_context, IndexContext(i))
            parsed_tensor_check = _as_parsed_tensor_spec(tensor_spec, context)
            shape: Shape
            if shaped is None:
                shape = None
            else:
                shape = get_shape(shaped, context)
            self._seen_shapes.append((shape, parsed_tensor_check, context))
            if shape is not None:
                new_shapes.append((shape, parsed_tensor_check))

            for dim_spec in parsed_tensor_check.shape.dims:
                if dim_spec.variable_name is None:
                    continue
                self._specs_by_variable.setdefault(dim_spec.variable_name, {})[
                    (parsed_tensor_check, context)
                ] = None
                if dim_spec.variable_rank:
                    self._varrank_variables.add(dim_spec.variable_name)
                else:
                    self._rank1_variables.add(dim_spec.variable_name)
                new_variables.add(dim_spec.variable_name)

        new_variable_error_contexts = []
        for variable in sorted(new_variables):
            if (variable in self._rank1_variables) and (variable in self._varrank_variables):
                new_variable_error_contexts.append(
                    StackContext(
                        VariableContext(variable),
                        ParallelContext(
                            tuple(
                                StackContext(c, TensorSpecContext(s))
                                for s, c in self._specs_by_variable[variable]
                            )
                        ),
                    )
                )
        if new_variable_error_contexts:
            raise VariableTypeError(ParallelContext(tuple(new_variable_error_contexts)))

        def _assert(condition: bool) -> None:
            if not condition:
                contexts: List[ErrorContext] = []
                contexts.extend(self._additional_context)

                for shape, tensor_spec, context in self._seen_shapes:
                    shape_error_context: ErrorContext = ShapeContext(tensor_spec.shape, shape)
                    if tensor_spec.note is not None:
                        shape_error_context = ParallelContext(
                            (NoteContext(tensor_spec.note), shape_error_context)
                        )
                    contexts.append(StackContext(context, shape_error_context))
                raise ShapeMismatchError(ParallelContext(tuple(contexts)))

        for actual, tensor_spec in new_shapes:
            actual_len = len(actual)
            actual_i = 0

            shape_spec = tensor_spec.shape
            expected = shape_spec.dims
            expected_len = len(expected)

            n_variable_rank = sum(dim_spec.variable_rank for dim_spec in expected)
            assert n_variable_rank <= 1, "At most one variable-rank ParsedDimensionSpec allowed."
            if n_variable_rank == 0:
                _assert(expected_len == actual_len)
            else:
                _assert(expected_len - n_variable_rank <= actual_len)

            for dim_spec in expected:

                if dim_spec.variable_rank:
                    variable_rank_len = actual_len - (expected_len - n_variable_rank)
                    actual_dims = actual[actual_i : actual_i + variable_rank_len]
                    actual_i += variable_rank_len

                    expected_name = dim_spec.variable_name
                    if expected_name is None:
                        # Anonymous dimension spec - we don't care about the actual values.
                        continue

                    expected_dims = self._seen_dims.setdefault(expected_name, _ObservedDims())
                    assert isinstance(expected_dims, _ObservedDims)
                    _assert(expected_dims.check_and_update(actual_dims, dim_spec.broadcastable))

                else:
                    actual_dim = actual[actual_i]
                    actual_i += 1
                    if actual_dim is None:
                        continue

                    expected_dim: Union[_ObservedDim, _ObservedDims]
                    if dim_spec.constant is not None:
                        expected_dim = _ObservedDim()
                        assert expected_dim.check_and_update(dim_spec.constant, broadcast=False)
                    elif dim_spec.variable_name is not None:
                        expected_name = dim_spec.variable_name
                        expected_dim = self._seen_dims.setdefault(expected_name, _ObservedDim())
                    else:
                        # Anonymous dimension - we don't care about the actual value.
                        continue
                    assert isinstance(expected_dim, _ObservedDim)
                    _assert(expected_dim.check_and_update(actual_dim, dim_spec.broadcastable))
