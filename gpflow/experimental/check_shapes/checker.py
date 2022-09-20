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
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union

from ..utils import experimental
from .base_types import Dimension, Shape
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
_Shape = Tuple[Dimension, ...]


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

        Returns whether the new data is compatible with existing observations. If this method
        returns `False` this object may have been left in an invalid state and should not be used
        again.
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
        self,
        actual: Optional[Tuple[Optional[int], ...]],
        broadcast: bool,
        shape_possibly_truncated: bool,
    ) -> bool:
        """
        Attempt to merge new data into this observation.

        Returns whether the new data is compatible with existing observations. If this method
        returns `False` this object may have been left in an invalid state and should not be used
        again.
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
            if shape_possibly_truncated:
                if longer < 0:
                    return False
            else:
                if longer != 0:
                    return False
        else:
            if longer < 0:
                self.sizes = [_ObservedDim() for _ in range(-longer)] + self.sizes
                longer = 0
            if shape_possibly_truncated:
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


@dataclass(eq=False)
class _ShapeCheck:
    """
    A shape check that is waiting to be performed.
    """

    actual: _Shape
    """
    Actual observed shape.

    Only `actual[actual_begin:actual_end]` is still waiting to be checked. The beginning and end may
    already have been checked.
    """

    actual_begin: int
    actual_end: int

    expected: ParsedShapeSpec
    """
    Specification to check against.

    Only `expected[expected_begin:expected_end]` is still waiting to be checked. The beginning and
    end may already have been checked.
    """

    expected_begin: int
    expected_end: int

    @property
    def finished(self) -> bool:
        """
        Whether this entire check has been performed.
        """
        return self.expected_begin >= self.expected_end


@dataclass
class _VariableState:
    """
    Structure of stuff we need to know about each variable.
    """

    uses: List[Tuple[ParsedTensorSpec, ErrorContext]] = field(default_factory=list)
    """ List of specs where this variable is used. """

    observed_dim: Optional[_ObservedDim] = None
    """
    Observed size of this variable, if the variable is rank-1.

    Set this `None` if this variable is varrank.
    """

    observed_dims: Optional[_ObservedDims] = None
    """
    Observed shape of this variable, if the variable is varrank.

    Set this `None` if this variable is rank-1.
    """

    waiting_for_varrank: Set[_ShapeCheck] = field(default_factory=set)
    """ Checks that are waiting for the rank of this variable to be determined. """


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
        self._variables: DefaultDict[str, _VariableState] = DefaultDict(_VariableState)
        self._observed_shapes: List[Tuple[Shape, ParsedTensorSpec, ErrorContext]] = []
        self._additional_context: List[ErrorContext] = []

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

        shape_check_queue = self._parse_checks(checks)
        while shape_check_queue:
            shape_check = shape_check_queue.pop()

            # Note that self._match_dims may not be able to match all dims. If that is the case it
            # adds the check to `_VariableState.waiting_for_varrank`:
            dim_checks = self._match_dims(shape_check)
            for dim_spec, actual_dims, shape_possibly_truncated in dim_checks:
                # Note that self._check_dim may revive checks from
                # `_VariableState.waiting_for_varrank` and add them back to `shape_check_queue`:
                self._check_dim(dim_spec, actual_dims, shape_possibly_truncated, shape_check_queue)

    def _parse_checks(
        self,
        checks: Iterable[
            Union[Tuple[Any, TensorSpecLike], Tuple[Any, TensorSpecLike, ErrorContext]]
        ],
    ) -> List[_ShapeCheck]:
        """
        Sanity check, register and parse the given ``checks`` into :class:`_ShapeCheck` objects.
        """
        shape_checks = []
        variable_type_errors = []
        call_context: Optional[ErrorContext] = None
        for i, check in enumerate(checks):
            shaped, tensor_spec, *contexts = check

            # Determine error context:
            if contexts:
                (context,) = contexts
            else:
                if call_context is None:
                    call_context = FunctionCallContext(self.check_shapes).precompute()
                context = StackContext(call_context, IndexContext(i))

            parsed_tensor_check = _as_parsed_tensor_spec(tensor_spec, context)
            parsed_shape_check = parsed_tensor_check.shape

            # Determine shape:
            if shaped is None:
                shape = None
            else:
                shape = get_shape(shaped, context)
            self._observed_shapes.append((shape, parsed_tensor_check, context))

            if shape is not None:
                shape_checks.append(
                    _ShapeCheck(
                        actual=shape,
                        actual_begin=0,
                        actual_end=len(shape),
                        expected=parsed_shape_check,
                        expected_begin=0,
                        expected_end=len(parsed_shape_check.dims),
                    )
                )

            # Record used variables:
            for dim_spec in parsed_shape_check.dims:
                if dim_spec.variable_name is None:
                    continue
                variable = self._variables[dim_spec.variable_name]
                variable.uses.append((parsed_tensor_check, context))
                if dim_spec.variable_rank:
                    if variable.observed_dims is None:
                        variable.observed_dims = _ObservedDims()
                        if variable.observed_dim is not None:
                            variable_type_errors.append(dim_spec.variable_name)
                else:
                    if variable.observed_dim is None:
                        variable.observed_dim = _ObservedDim()
                        if variable.observed_dims is not None:
                            variable_type_errors.append(dim_spec.variable_name)

        if variable_type_errors:
            error_contexts = [
                StackContext(
                    VariableContext(variable),
                    ParallelContext(
                        tuple(
                            StackContext(c, TensorSpecContext(s))
                            for s, c in self._variables[variable].uses
                        )
                    ),
                )
                for variable in sorted(variable_type_errors)
            ]
            raise VariableTypeError(ParallelContext(tuple(error_contexts)))

        return shape_checks

    def _match_dims(
        self, shape_check: _ShapeCheck
    ) -> Iterable[Tuple[ParsedDimensionSpec, _Shape, bool]]:
        """
        Match expected dimensions against actual dimensions.

        If some dimensions cannot be determined, the remaining dimensions are added to
        `_VariableState.waiting_for_varrank`.
        """
        # Determine rank of expected dimensions:

        # Length of the current shape, which we have not yet matched against a dimension spec:
        unallocated_len = shape_check.actual_end - shape_check.actual_begin

        # Length of dimension specs, `None` if it cannot be determined:
        allocated_sizes: List[Optional[int]] = []

        # Count of `None`s in `allocated_sizes`:
        n_allocated_none = 0

        # Mapping from variables, that we don't know the rank of, to whether they're used with
        # `broadcast`.
        unknown_len_variables: Dict[Optional[str], bool] = {}

        # The length of a shape is allowed to be short if the dimension, and all leading dimensions
        # are `broadcast`, so pre-compute where this applies.
        leading_broadcast = True
        leading_broadcasts = []
        for i in range(0, shape_check.expected_end):
            expected_dim = shape_check.expected.dims[i]
            leading_broadcast &= expected_dim.broadcastable
            leading_broadcasts.append(leading_broadcast)

        for i in range(shape_check.expected_begin, shape_check.expected_end)[::-1]:
            leading_broadcast = leading_broadcasts[i]
            expected_dim = shape_check.expected.dims[i]
            expected_name = expected_dim.variable_name

            rank: Optional[int] = None
            if expected_dim.variable_rank:
                if expected_name is not None:
                    observed_dims = self._variables[expected_name].observed_dims
                    assert observed_dims is not None
                    if observed_dims.known_rank and (observed_dims.sizes is not None):
                        rank = len(observed_dims.sizes)
            else:
                rank = 1

            if rank is not None and leading_broadcast:
                all_trailing_shapes_known = n_allocated_none == 0
                if all_trailing_shapes_known:
                    # We know where the dimension ends, because everything after this has known
                    # size, so we can infer the start from the `unallocated_len`.
                    rank = min(rank, unallocated_len)
                else:
                    # We don't know where this dimension starts, because of broadcasting, and we
                    # don't know where it ends, because something after this has an unknown size
                    # - all bets are off...
                    rank = None

            if rank is None:
                n_allocated_none += 1
                unknown_len_variables[expected_name] = leading_broadcast
            else:
                unallocated_len -= rank

            allocated_sizes.append(rank)

        allocated_sizes = allocated_sizes[::-1]  # This list was built in reverse order.

        # Determine length of variables with outknown length:

        # Number of variables with unknown length.
        n_unknown = len(unknown_len_variables)

        if n_unknown == 0:
            self._assert(unallocated_len == 0)
        else:
            self._assert(unallocated_len >= 0)
            if n_unknown == 1:
                (expected_name,) = unknown_len_variables
                if expected_name is not None:
                    broadcastable = unknown_len_variables[expected_name]
                    if n_allocated_none <= 1 or (not broadcastable):
                        self._assert(unallocated_len % n_allocated_none == 0)
                        unknown_len = unallocated_len // n_allocated_none
                        allocated_sizes = [unknown_len if s is None else s for s in allocated_sizes]

        dim_checks = []

        # Take as much as we can from the left side:

        allocated_i = 0
        while not shape_check.finished:
            expected_rank = allocated_sizes[allocated_i]
            if expected_rank is None:
                break

            dim_checks.append(
                (
                    shape_check.expected.dims[shape_check.expected_begin],
                    shape_check.actual[
                        shape_check.actual_begin : shape_check.actual_begin + expected_rank
                    ],
                    leading_broadcasts[shape_check.expected_begin],
                )
            )
            allocated_i += 1
            shape_check.actual_begin += expected_rank
            shape_check.expected_begin += 1

        # Take as much as we can from the right side:

        allocated_i = len(allocated_sizes) - 1
        while not shape_check.finished:
            expected_rank = allocated_sizes[allocated_i]
            if expected_rank is None:
                break

            dim_checks.append(
                (
                    shape_check.expected.dims[shape_check.expected_end - 1],
                    shape_check.actual[
                        shape_check.actual_end - expected_rank : shape_check.actual_end
                    ],
                    leading_broadcasts[shape_check.expected_begin],
                )
            )
            allocated_i -= 1
            shape_check.actual_end -= expected_rank
            shape_check.expected_end -= 1

        if not shape_check.finished:
            waiting_for: Set[str] = set(unknown_len_variables) - {None}  # type: ignore[assignment]
            for name in waiting_for:
                self._variables[name].waiting_for_varrank.add(shape_check)

        return dim_checks

    def _check_dim(
        self,
        expected: ParsedDimensionSpec,
        actual_dims: _Shape,
        shape_possibly_truncated: bool,
        shape_checks: List[_ShapeCheck],
    ) -> None:
        """
        Checks that ``actual_dim`` matches ``expected``.

        Newly learned information may enable the evaluation of deferred shape checks - any such will
        be added to ``shape_checks``.
        """
        if expected.variable_rank:
            expected_name = expected.variable_name
            if expected_name is None:
                # Anonymous dimension spec - we don't care about the actual values.
                return

            variable = self._variables[expected_name]
            observed_dims = variable.observed_dims
            assert observed_dims is not None

            self._assert(
                observed_dims.check_and_update(
                    actual_dims, expected.broadcastable, shape_possibly_truncated
                )
            )
            if observed_dims.known_rank and variable.waiting_for_varrank:
                shape_checks.extend(variable.waiting_for_varrank)
                variable.waiting_for_varrank.clear()

        else:
            if not actual_dims:
                # Broadcastable dimension was not matched against anything.
                assert shape_possibly_truncated
                return

            (actual_dim,) = actual_dims
            if actual_dim is None:
                # Acutal dimension is unknown - nothing to check.
                return

            if expected.constant is not None:
                observed_dim = _ObservedDim()
                assert observed_dim.check_and_update(expected.constant, broadcast=False)
            elif expected.variable_name is not None:
                maybe_observed_dim = self._variables[expected.variable_name].observed_dim
                assert maybe_observed_dim is not None
                observed_dim = maybe_observed_dim
            else:
                # Anonymous dimension - we don't care about the actual value.
                return
            self._assert(observed_dim.check_and_update(actual_dim, expected.broadcastable))

    def _assert(self, condition: bool) -> None:
        """
        Raise a nicely formatted :class:`ShapeMismatchError` if ``condition`` is not ``True``.
        """
        if not condition:
            contexts: List[ErrorContext] = []
            contexts.extend(self._additional_context)

            for shape, tensor_spec, context in self._observed_shapes:
                shape_error_context: ErrorContext = ShapeContext(tensor_spec.shape, shape)
                if tensor_spec.note is not None:
                    shape_error_context = ParallelContext(
                        (NoteContext(tensor_spec.note), shape_error_context)
                    )
                contexts.append(StackContext(context, shape_error_context))
            raise ShapeMismatchError(ParallelContext(tuple(contexts)))
