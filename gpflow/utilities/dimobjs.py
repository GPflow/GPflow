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
"""
Inspired by / related work:
einops: https://github.com/arogozhnikov/einops
torchdim: https://github.com/facebookresearch/torchdim
"""
from abc import ABC, abstractmethod
from typing import Any, Counter, Dict, List, Mapping, Optional, Sequence, Tuple, Union, overload

import tensorflow as tf
from typing_extensions import Protocol

from gpflow.experimental.check_shapes import check_shapes, get_check_shapes, inherit_check_shapes

_safe_mode: bool = False


def set_safe_mode(safe_mode: bool) -> None:
    global _safe_mode
    _safe_mode = safe_mode


def get_safe_mode() -> bool:
    global _safe_mode
    return _safe_mode


class Reducer(Protocol):
    # Compatible with tf.reduce_sum.

    @check_shapes(
        "input_tensor: [...]",
        "axis: [n]",
    )
    def __call__(self, input_tensor: tf.Tensor, axis: tf.Tensor) -> tf.Tensor:
        ...


class Expander(Protocol):
    @check_shapes(
        "input_tensor: [...]",
        "axis: [n]",
        "sizes: [n]",
    )
    def __call__(self, input_tensor: tf.Tensor, axis: tf.Tensor, sizes: tf.Tensor) -> tf.Tensor:
        ...


@get_check_shapes(Expander.__call__)
def tile_fn(input_tensor: tf.Tensor, axis: tf.Tensor, sizes: tf.Tensor) -> tf.Tensor:
    # This is terrible. Is there no easier way to do this?
    old_shape = tf.shape(input_tensor)
    n_old = tf.size(old_shape)
    n_new = tf.size(axis)
    n_result = n_old + n_new

    result_indices = tf.range(n_result)
    is_new_index = tf.reduce_any(result_indices[:, None] == axis, axis=-1)
    old_indices = tf.cumsum(tf.cast(~is_new_index, tf.int32), exclusive=True)
    new_indices = tf.cumsum(tf.cast(is_new_index, tf.int32), exclusive=True)

    # To avoid index-out-of-bounds in gather below:
    sentinel = 9999
    old_shape = tf.concat([old_shape, [sentinel]], 0)
    new_multiples = tf.concat([sizes, [sentinel]], 0)

    result_shape = tf.where(is_new_index, 1, tf.gather(old_shape, old_indices))
    result_multiples = tf.where(is_new_index, tf.gather(new_multiples, new_indices), 1)
    return tf.tile(tf.reshape(input_tensor, result_shape), result_multiples)


class Dimensions(ABC):
    """
    A sequence of dimensions of a tensor.
    """

    @abstractmethod
    @check_shapes(
        "return: []",
    )
    def rank(self) -> Optional[tf.Tensor]:
        """
        Return the number of dimensions.

        Returns ``None`` if this is unknown, which can happen for variables that have not yet been
        bound.
        """

    @abstractmethod
    @check_shapes(
        "return: [rank]",
    )
    def shape(self) -> tf.Tensor:
        """
        Return the shape of this dimension.
        """

    @abstractmethod
    def is_flat(self) -> bool:
        """
        Returns whether this dimension does not require any reshaping.
        """

    @abstractmethod
    @check_shapes(
        "return: [rank]",
    )
    def flat_shape(self) -> tf.Tensor:
        """
        Return the shapes of the variables used in this dimension.
        """

    @abstractmethod
    def flat_vars(self) -> Tuple["VariableDimensions", ...]:
        """
        Return a list of the variables used in this _DimSpec, in the order they are used.
        """

    @abstractmethod
    @check_shapes(
        "begin: []",
        "return[0].values(): [.]",
        "return[1]: []",
    )
    def enumerate_vars(
        self, begin: tf.Tensor
    ) -> Tuple[Mapping["VariableDimensions", tf.Tensor], tf.Tensor]:
        """
        Blah blah...
        """

    def __add__(self, other: "Dimensions") -> "Sum":
        return Sum(Sum.as_terms_tuple(self) + Sum.as_terms_tuple(other))

    def __radd__(self, other: "Dimensions") -> "Sum":
        return Sum(Sum.as_terms_tuple(other) + Sum.as_terms_tuple(self))

    @overload
    def __rshift__(self, other: "DataOp") -> "InOp":  # type: ignore[misc]
        ...

    @overload
    def __rshift__(self, other: Union["DataOut", "DataOpOut"]) -> tf.Tensor:
        ...

    @overload
    def __rshift__(self, other: "DataOpOuts") -> Tuple[tf.Tensor, ...]:  # type: ignore[misc]
        ...

    @overload
    def __rshift__(self, other: tf.Tensor) -> "DimensionsData":
        ...

    def __rshift__(
        self, other: Union["DataOp", "DataOut", "DataOpOut", "DataOpOuts", tf.Tensor]
    ) -> Union["DimensionsData", "InOp", tf.Tensor, Tuple[tf.Tensor, ...]]:
        if isinstance(other, DataOp):
            _infer(other.in_data, self)
            return InOp(self, other.in_data, other.op)
        if isinstance(other, DataOut):
            ((out_data, out_dim),) = rearrange.run(((other.in_data, self),), (other.out_dim,))
            return out_data
        if isinstance(other, DataOpOut):
            ((out_data, out_dim),) = other.op.run(((other.in_data, self),), (other.out_dim,))
            return out_data
        if isinstance(other, DataOpOuts):
            outs = other.op.run(((other.in_data, self),), other.outs)
            return tuple(out_data for out_data, out_dim in outs)
        _infer(other, self)
        return DimensionsData(self, other)

    @overload
    def __rrshift__(self, other: "Op") -> "OpOut":  # type: ignore[misc]
        ...

    @overload
    def __rrshift__(self, other: "DataOp") -> "DataOpOut":  # type: ignore[misc]
        ...

    @overload
    def __rrshift__(self, other: Union["InOp", "InsOp"]) -> tf.Tensor:
        ...

    @overload
    def __rrshift__(self, other: tf.Tensor) -> "DataOut":
        ...

    def __rrshift__(
        self, other: Union["Op", "DataOp", "InOp", "InsOp", tf.Tensor]
    ) -> Union["OpOut", "DataOpOut", "DataOut", tf.Tensor]:
        if isinstance(other, Op):
            return OpOut(other, self)
        if isinstance(other, DataOp):
            return DataOpOut(other.in_data, other.op, self)
        if isinstance(other, InOp):
            ((out_data, out_dim),) = other.op.run(((other.in_data, other.in_dim),), (self,))
            return out_data
        if isinstance(other, InsOp):
            ((out_data, out_dim),) = other.op.run(other.ins, (self,))
            return out_data
        return DataOut(self, other)


class FactorDimensions(Dimensions, ABC):
    """
    A sequence of dimensions that can be multiplied.
    """

    def __mul__(self, other: "FactorDimensions") -> "Product":
        return Product(Product.as_factor_tuple(self) + Product.as_factor_tuple(other))


class VariableDimensions(FactorDimensions, ABC):
    """
    A sequence of dimensions that has a, possibly unknown, variable shape.
    """

    def is_flat(self) -> bool:
        return True

    @inherit_check_shapes
    def flat_shape(self) -> tf.Tensor:
        return self.shape()

    def flat_vars(self) -> Tuple["VariableDimensions", ...]:
        return (self,)


def assert_variables_unique(dimensions: Dimensions) -> None:
    counts = Counter(dimensions.flat_vars())
    tokens = []
    for dim, count in counts.items():
        if count > 1:
            tokens.append(repr(dim))
    if tokens:
        raise AssertionError(
            "Variables cannot be reused in the same shape - they must uniquely identify dimensions."
            f" Duplicated variables: {', '.join(tokens)}."
        )


class Dim(VariableDimensions):
    def __init__(self, name: str, size: Optional[tf.Tensor] = None) -> None:
        self.name = name
        self.size = size

    @inherit_check_shapes
    def rank(self) -> Optional[tf.Tensor]:
        return tf.ones((), dtype=tf.int32)

    @inherit_check_shapes
    def shape(self) -> tf.Tensor:
        return tf.reshape(self.size, [1])

    @inherit_check_shapes
    def enumerate_vars(
        self, begin: tf.Tensor
    ) -> Tuple[Mapping["VariableDimensions", tf.Tensor], tf.Tensor]:
        return {self: tf.reshape(begin, [1])}, begin + 1

    def __repr__(self) -> str:
        tokens = [self.name, "="]
        tokens.append("?" if self.size is None else str(int(self.size.numpy())))
        return "".join(tokens)


class Dims(VariableDimensions):
    def __init__(self, name: str, sizes: Optional[tf.Tensor] = None) -> None:
        self.name = name
        self.sizes = sizes

    @inherit_check_shapes
    def rank(self) -> Optional[tf.Tensor]:
        return None if self.sizes is None else tf.size(self.sizes)

    @inherit_check_shapes
    def shape(self) -> tf.Tensor:
        return self.sizes

    @inherit_check_shapes
    def enumerate_vars(
        self, begin: tf.Tensor
    ) -> Tuple[Mapping["VariableDimensions", tf.Tensor], tf.Tensor]:
        size = tf.size(self.sizes)
        end = begin + size
        return {self: tf.range(begin, end)}, end

    def prod(self) -> "Product":
        return Product(Product.as_factor_tuple(self))

    def __repr__(self) -> str:
        tokens = [self.name, "="]
        if self.sizes is None:
            tokens.append("?...")
        else:
            tokens.append("[")
            tokens.append(",".join(str(int(d)) for d in self.sizes.numpy()))
            tokens.append("]")
        return "".join(tokens)


class Product(FactorDimensions):
    def __init__(self, factors: Tuple[VariableDimensions, ...]) -> None:
        self.factors = factors

        if _safe_mode:
            assert_variables_unique(self)
            for dim in self.flat_vars():
                assert isinstance(dim, VariableDimensions), type(dim)

    @staticmethod
    def as_factor_tuple(factor: FactorDimensions) -> Tuple[VariableDimensions, ...]:
        if isinstance(factor, Product):
            return factor.factors
        return (factor,)  # type: ignore[return-value]

    @inherit_check_shapes
    def rank(self) -> Optional[tf.Tensor]:
        return tf.ones((), dtype=tf.int32)

    @inherit_check_shapes
    def shape(self) -> tf.Tensor:
        input_shape = tf.concat([f.shape() for f in self.factors], 0)
        return tf.reduce_prod(input_shape)[None]

    def is_flat(self) -> bool:
        return False

    @inherit_check_shapes
    def flat_shape(self) -> tf.Tensor:
        return tf.concat([f.flat_shape() for f in self.factors], axis=0)

    def flat_vars(self) -> Tuple["VariableDimensions", ...]:
        return tuple(v for f in self.factors for v in f.flat_vars())

    @inherit_check_shapes
    def enumerate_vars(
        self, begin: tf.Tensor
    ) -> Tuple[Mapping["VariableDimensions", tf.Tensor], tf.Tensor]:
        result_vars: Dict["VariableDimensions", tf.Tensor] = {}
        for f in self.factors:
            factor_vars, end = f.enumerate_vars(begin)
            result_vars.update(factor_vars)
            begin = end
        return result_vars, begin

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.factors == other.factors

    def __hash__(self) -> int:
        return hash(type(self)) + hash(self.factors)

    def __repr__(self) -> str:
        return "(" + " * ".join(repr(f) for f in self.factors) + ")"


class Ones(Dimensions):
    def __init__(self, rank: int) -> None:
        self._rank = rank

    @inherit_check_shapes
    def rank(self) -> Optional[tf.Tensor]:
        return tf.constant(self._rank, dtype=tf.int32)

    @inherit_check_shapes
    def shape(self) -> tf.Tensor:
        return tf.ones((self._rank,), dtype=tf.int32)

    def is_flat(self) -> bool:
        return False

    @inherit_check_shapes
    def flat_shape(self) -> tf.Tensor:
        return tf.constant([], dtype=tf.int32)

    def flat_vars(self) -> Tuple["VariableDimensions", ...]:
        return ()

    @inherit_check_shapes
    def enumerate_vars(
        self, begin: tf.Tensor
    ) -> Tuple[Mapping["VariableDimensions", tf.Tensor], tf.Tensor]:
        return {}, begin

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self._rank == other._rank

    def __hash__(self) -> int:
        return hash(type(self)) + hash(self._rank)

    def __repr__(self) -> str:
        return "1"


none = Ones(0)
one = Ones(1)


class Sum(Dimensions):
    def __init__(self, terms: Tuple[Dimensions, ...]) -> None:
        self.terms = terms

        if _safe_mode:
            assert_variables_unique(self)

    @staticmethod
    def as_terms_tuple(term: Dimensions) -> Tuple[Dimensions, ...]:
        if isinstance(term, Sum):
            return term.terms
        return (term,)

    @inherit_check_shapes
    def rank(self) -> tf.Tensor:
        return tf.add_n([t.rank() for t in self.terms])

    @inherit_check_shapes
    def shape(self) -> tf.Tensor:
        return tf.concat([t.shape() for t in self.terms], axis=0)

    def is_flat(self) -> bool:
        return all(t.is_flat() for t in self.terms)

    @inherit_check_shapes
    def flat_shape(self) -> tf.Tensor:
        return tf.concat([t.flat_shape() for t in self.terms], axis=0)

    def flat_vars(self) -> Tuple["VariableDimensions", ...]:
        return tuple(v for t in self.terms for v in t.flat_vars())

    @inherit_check_shapes
    def enumerate_vars(
        self, begin: tf.Tensor
    ) -> Tuple[Mapping["VariableDimensions", tf.Tensor], tf.Tensor]:
        result_vars: Dict["VariableDimensions", tf.Tensor] = {}
        for t in self.terms:
            term_vars, end = t.enumerate_vars(begin)
            result_vars.update(term_vars)
            begin = end
        return result_vars, begin

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.terms == other.terms

    def __hash__(self) -> int:
        return hash(type(self)) + hash(self.terms)

    def __repr__(self) -> str:
        return " + ".join(repr(t) for t in self.terms)


def dims(names: Optional[str] = None, **kwdims: Any) -> Sequence[Dim]:
    result: List[Dim] = []
    if names is not None:
        result.extend(Dim(name.strip()) for name in names.split(","))
    result.extend(
        Dim(name, tf.convert_to_tensor(size, dtype=tf.int32)) for name, size in kwdims.items()
    )
    return result


def dimses(names: Optional[str] = None, **kwdims: Any) -> Sequence[Dims]:
    result: List[Dims] = []
    if names is not None:
        result.extend(Dims(name.strip()) for name in names.split(","))
    result.extend(
        Dims(name, tf.convert_to_tensor(sizes, dtype=tf.int32)) for name, sizes in kwdims.items()
    )
    return result


def _get_indices(
    haystack: Sequence[VariableDimensions], needles: Sequence[VariableDimensions]
) -> tf.Tensor:
    i = tf.constant(0, dtype=tf.int32)
    indices = {}
    for straw in haystack:
        rank = straw.rank()
        indices[straw] = tf.range(i, i + rank)
        i += rank
    return tf.concat([indices[needle] for needle in needles], 0)


def _apply(
    t: tf.Tensor,
    frm: Dimensions,
    to: Dimensions,
    *,
    allow_transpose: bool,
    reduce_fn: Optional[Reducer],
    expand_fn: Optional[Expander],
) -> tf.Tensor:
    if frm == to:
        # Optimisation: No action necessary.
        return t

    frm_flat = frm.flat_vars()
    to_flat = to.flat_vars()

    if frm_flat == to_flat:
        # Optimisation: Only reshape required.
        return tf.reshape(t, to.shape())

    if not frm.is_flat():
        t = tf.reshape(t, frm.flat_shape())

    frm_set = set(frm_flat)
    to_set = set(to_flat)

    removed = [f for f in frm_flat if f not in to_set]
    if removed:
        assert reduce_fn is not None
        t = reduce_fn(t, axis=_get_indices(frm_flat, removed))

    frm_inner = [f for f in frm_flat if f in to_set]
    to_inner = [t for t in to_flat if t in frm_set]
    if frm_inner != to_inner:
        assert allow_transpose
        t = tf.transpose(t, _get_indices(frm_inner, to_inner))

    added = [t for t in to_flat if t not in frm_set]
    if added:
        assert expand_fn is not None
        multiples = tf.concat([t.shape() for t in added], 0)
        t = expand_fn(t, _get_indices(to_flat, added), multiples)

    if not to.is_flat():
        t = tf.reshape(t, to.shape())

    return t


def _infer(t: tf.Tensor, dimensions: Dimensions) -> None:
    dimensions_tuple = Sum.as_terms_tuple(dimensions)
    t_rank = tf.rank(t)
    t_shape = tf.shape(t)
    t_begin = tf.constant(0, dtype=tf.int32)

    rank_sum = tf.zeros((), dtype=tf.int32)
    unknown_rank_dim = None
    for dim in dimensions_tuple:
        rank = dim.rank()
        if rank is None:
            assert unknown_rank_dim is None, (
                f"Unable to infer shapes of {unknown_rank_dim} and {dim}."
                " Try setting `bind`ing them somewhere else, or setting their ``sizes` manually."
            )
            unknown_rank_dim = dim
        else:
            rank_sum += rank

        if _safe_mode and isinstance(dim, Product):
            for d in dim.factors:
                if isinstance(d, Dim):
                    assert d.size is not None
                elif isinstance(d, Dims):
                    assert d.sizes is not None
                else:
                    assert False, type(d)

    if _safe_mode:
        tf.debugging.assert_less_equal(
            rank_sum, t_rank, message="Bound tensor has lower rank that suggested dimensions."
        )

    for dim in dimensions_tuple:
        if isinstance(dim, Dim):
            if _safe_mode:
                tf.debugging.assert_less(
                    t_begin,
                    t_rank,
                    message="Bound tensor has lower rank that suggested dimensions.",
                )
            if dim.size is None:
                dim.size = t_shape[t_begin]
            elif _safe_mode:
                tf.debugging.assert_equal(
                    dim.size, t_shape[t_begin], message=f"Tensor shape mismatch for dimension {dim}"
                )
        elif isinstance(dim, Dims):
            rank = t_rank - rank_sum if dim.sizes is None else tf.size(dim.sizes)
            t_end = t_begin + rank
            if _safe_mode:
                tf.debugging.assert_less_equal(
                    t_end, t_rank, message="Bound tensor has lower rank that suggested dimensions."
                )
            if dim.sizes is None:
                dim.sizes = t_shape[t_begin:t_end]
            elif _safe_mode:
                tf.debugging.assert_equal(
                    dim.sizes,
                    t_shape[t_begin:t_end],
                    message=f"Tensor shape mismatch for dimension {dim}",
                )
        t_begin += dim.rank()

    if _safe_mode:
        tf.debugging.assert_equal(
            t_begin, t_rank, message="Bound tensor does not have same rank as suggested dimensions."
        )


class Op(ABC):
    @abstractmethod
    def run(
        self, inputs: Tuple[Tuple[tf.Tensor, Dimensions], ...], outputs: Tuple[Dimensions, ...]
    ) -> Tuple[Tuple[tf.Tensor, Dimensions], ...]:
        ...

    @overload
    def __rshift__(self, other: Dimensions) -> "OpOut":
        ...

    @overload
    def __rshift__(self, other: Tuple[Dimensions, ...]) -> "OpOuts":
        ...

    def __rshift__(
        self, other: Union[Dimensions, Tuple[Dimensions, ...]]
    ) -> Union["OpOut", "OpOuts"]:
        if isinstance(other, tuple):
            return OpOuts(self, other)
        if isinstance(other, Dimensions):
            return OpOut(self, other)
        assert False, type(other)

    @overload
    def __rrshift__(self, other: Tuple["DimensionsData", ...]) -> "InsOp":  # type: ignore[misc]
        ...

    @overload
    def __rrshift__(self, other: "DimensionsData") -> "InOp":  # type: ignore[misc]
        ...

    @overload
    def __rrshift__(self, other: tf.Tensor) -> "DataOp":  # type: ignore[misc]
        ...

    def __rrshift__(
        self, other: Union[tf.Tensor, "DimensionsData", Tuple["DimensionsData", ...]]
    ) -> Union["DataOp", "InOp", "InsOp"]:
        if isinstance(other, DimensionsData):
            return InOp(other.in_dim, other.in_data, self)
        if isinstance(other, tuple):
            for o in other:
                _infer(o.in_data, o.in_dim)
            return InsOp(tuple((o.in_data, o.in_dim) for o in other), self)
        return DataOp(other, self)


class SingleIOOp(Op, ABC):
    def run(
        self, inputs: Tuple[Tuple[tf.Tensor, Dimensions], ...], outputs: Tuple[Dimensions, ...]
    ) -> Tuple[Tuple[tf.Tensor, Dimensions], ...]:
        ((in_data, in_dim),) = inputs
        (out_dim,) = outputs
        out_data = self.single_io_run(in_data, in_dim, out_dim)
        return ((out_data, out_dim),)

    @abstractmethod
    def single_io_run(
        self, in_data: tf.Tensor, in_dim: Dimensions, out_dim: Dimensions
    ) -> tf.Tensor:
        ...


class FunctionSingleIOOp(SingleIOOp):
    def __init__(
        self,
        allow_transpose: bool,
        reduce_fn: Optional[Reducer],
        expand_fn: Optional[Expander],
    ) -> None:
        self._allow_transpose = allow_transpose
        self._reduce_fn = reduce_fn
        self._expand_fn = expand_fn

    def single_io_run(
        self, in_data: tf.Tensor, in_dim: Dimensions, out_dim: Dimensions
    ) -> tf.Tensor:
        return _apply(
            in_data,
            in_dim,
            out_dim,
            allow_transpose=self._allow_transpose,
            reduce_fn=self._reduce_fn,
            expand_fn=self._expand_fn,
        )


reshape = FunctionSingleIOOp(allow_transpose=False, reduce_fn=None, expand_fn=None)
rearrange = FunctionSingleIOOp(allow_transpose=True, reduce_fn=None, expand_fn=None)
reduce_sum = FunctionSingleIOOp(allow_transpose=True, reduce_fn=tf.reduce_sum, expand_fn=None)
tile = FunctionSingleIOOp(allow_transpose=True, reduce_fn=None, expand_fn=tile_fn)


class EinSum(Op):
    def run(
        self, inputs: Tuple[Tuple[tf.Tensor, Dimensions], ...], outputs: Tuple[Dimensions, ...]
    ) -> Tuple[Tuple[tf.Tensor, Dimensions], ...]:
        batch_dim: Optional[Dims] = None
        dim_map: Dict[VariableDimensions, str] = {}
        spec_tokens: List[str] = []

        def add_spec(dim: Dimensions) -> None:
            nonlocal batch_dim
            nonlocal dim_map
            nonlocal spec_tokens

            for d in dim.flat_vars():
                if isinstance(d, Dims):
                    if batch_dim is None:
                        batch_dim = d
                    else:
                        assert batch_dim == d, "EinSum only supports a single batch dimension."
                    spec_tokens.append("...")
                else:
                    dim_char = dim_map.get(d)
                    if dim_char is None:
                        dim_char = chr(ord("a") + len(dim_map))
                        dim_map[d] = dim_char
                    spec_tokens.append(dim_char)

        in_datas = []
        sep = ""
        for in_data, in_dim in inputs:
            if not in_dim.is_flat():
                in_data = tf.reshape(in_data, in_dim.flat_shape())
            in_datas.append(in_data)

            spec_tokens.append(sep)
            sep = ","
            add_spec(in_dim)

        (out_dim,) = outputs
        spec_tokens.append("->")
        add_spec(out_dim)

        spec = "".join(spec_tokens)
        out_data = tf.einsum(spec, *in_datas)

        if not out_dim.is_flat():
            out_data = tf.reshape(out_data, out_dim.flat_shape())

        return ((out_data, out_dim),)


einsum = EinSum()


class DimensionsData:
    def __init__(self, in_dim: Dimensions, in_data: tf.Tensor) -> None:
        self.in_dim = in_dim
        self.in_data = in_data

    @overload
    def __rshift__(self, other: Op) -> "InOp":
        ...

    @overload
    def __rshift__(self, other: "OpOut") -> tf.Tensor:
        ...

    @overload
    def __rshift__(self, other: "OpOuts") -> Tuple[tf.Tensor, ...]:
        ...

    @overload
    def __rshift__(self, other: Dimensions) -> tf.Tensor:
        ...

    def __rshift__(
        self, other: Union[Op, "OpOut", "OpOuts", Dimensions]
    ) -> Union["InOp", tf.Tensor, Tuple[tf.Tensor, ...]]:
        if isinstance(other, Op):
            return InOp(self.in_dim, self.in_data, other)
        if isinstance(other, OpOut):
            ((out_data, out_dim),) = other.op.run(((self.in_data, self.in_dim),), (other.out_dim,))
            return out_data
        if isinstance(other, OpOuts):
            outs = other.op.run(((self.in_data, self.in_dim),), other.outs)
            return tuple(out_data for out_data, out_dim in outs)
        if isinstance(other, Dimensions):
            ((out_data, out_dim),) = rearrange.run(((self.in_data, self.in_dim),), (other,))
            return out_data
        assert False, type(other)

    def __repr__(self) -> str:
        return f"DimensionsData({self.in_dim}, {self.in_data})"


class DataOut:
    def __init__(self, in_data: tf.Tensor, out_dim: Dimensions) -> None:
        self.in_data = in_data
        self.out_dim = out_dim

    def __rrshift__(self, other: Dimensions) -> tf.Tensor:
        ((out_data, out_dim),) = rearrange.run(((self.in_data, other),), (self.out_dim,))
        return out_data

    def __repr__(self) -> str:
        return f"DataOut({self.in_data}, {self.out_dim})"


class InOp:
    def __init__(self, in_dim: Dimensions, in_data: tf.Tensor, op: Op) -> None:
        self.in_dim = in_dim
        self.in_data = in_data
        self.op = op

    @overload
    def __rshift__(self, other: Dimensions) -> tf.Tensor:
        ...

    @overload
    def __rshift__(self, other: Tuple[Dimensions, ...]) -> Tuple[tf.Tensor, ...]:
        ...

    def __rshift__(
        self, other: Union[Dimensions, Tuple[Dimensions, ...]]
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, ...]]:
        if isinstance(other, Dimensions):
            ((out_data, out_dim),) = self.op.run(((self.in_data, self.in_dim),), (other,))
            return out_data
        if isinstance(other, tuple):
            outs = self.op.run(((self.in_data, self.in_dim),), other)
            return tuple(out_data for out_data, out_dim in outs)
        assert False, type(other)

    def __repr__(self) -> str:
        return f"InOp({self.in_dim}, {self.in_data}, {self.op})"


class OpOut:
    def __init__(self, op: Op, out_dim: Dimensions) -> None:
        self.op = op
        self.out_dim = out_dim

    @overload
    def __rrshift__(
        self, other: Union["DimensionsData", Tuple["DimensionsData", ...]]
    ) -> tf.Tensor:
        ...

    @overload
    def __rrshift__(self, other: tf.Tensor) -> "DataOpOut":
        ...

    def __rrshift__(
        self, other: Union[tf.Tensor, "DimensionsData", Tuple["DimensionsData", ...]]
    ) -> Union[tf.Tensor, "DataOpOut"]:
        if isinstance(other, DimensionsData):
            ((out_data, out_dim),) = self.op.run(((other.in_data, other.in_dim),), (self.out_dim,))
            return out_data
        if isinstance(other, tuple):
            ins = tuple((i.in_data, i.in_dim) for i in other)
            ((out_data, out_dim),) = self.op.run(ins, (self.out_dim,))
            return out_data
        return DataOpOut(other, self.op, self.out_dim)

    def __repr__(self) -> str:
        return f"OpOut({self.op}, {self.out_dim})"


class DataOp:
    def __init__(self, in_data: tf.Tensor, op: Op) -> None:
        self.in_data = in_data
        self.op = op

    @overload
    def __rshift__(self, other: Dimensions) -> "DataOpOut":
        ...

    @overload
    def __rshift__(self, other: Tuple[Dimensions, ...]) -> "DataOpOuts":
        ...

    def __rshift__(
        self, other: Union[Dimensions, Tuple[Dimensions, ...]]
    ) -> Union["DataOpOut", "DataOpOuts"]:
        if isinstance(other, Dimensions):
            return DataOpOut(self.in_data, self.op, other)
        if isinstance(other, tuple):
            return DataOpOuts(self.in_data, self.op, other)
        assert False, type(other)

    def __rrshift__(self, other: Dimensions) -> "InOp":
        _infer(self.in_data, other)
        return InOp(other, self.in_data, self.op)

    def __repr__(self) -> str:
        return f"DataOp({self.in_data}, {self.op})"


class DataOpOut:
    def __init__(self, in_data: tf.Tensor, op: Op, out_dim: Dimensions) -> None:
        self.in_data = in_data
        self.op = op
        self.out_dim = out_dim

    def __rrshift__(self, other: Dimensions) -> tf.Tensor:
        ((out_data, out_dim),) = self.op.run(((self.in_data, other),), (self.out_dim,))
        return out_data

    def __repr__(self) -> str:
        return f"DataOpOut({self.in_data}, {self.op}, {self.out_dim})"


class InsOp:
    def __init__(self, ins: Tuple[Tuple[tf.Tensor, Dimensions], ...], op: Op) -> None:
        self.ins = ins
        self.op = op

    @overload
    def __rshift__(self, other: Dimensions) -> tf.Tensor:
        ...

    @overload
    def __rshift__(self, other: Tuple[Dimensions, ...]) -> Tuple[tf.Tensor, ...]:
        ...

    def __rshift__(
        self, other: Union[Dimensions, Tuple[Dimensions, ...]]
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, ...]]:
        if isinstance(other, Dimensions):
            ((out_data, out_dim),) = self.op.run(self.ins, (other,))
            return out_data
        if isinstance(other, tuple):
            outs = self.op.run(self.ins, other)
            return tuple(out_data for out_data, out_dim in outs)
        assert False, type(other)

    def __repr__(self) -> str:
        return f"InsOp({self.ins}, {self.op})"


class OpOuts:
    def __init__(self, op: Op, outs: Tuple[Dimensions, ...]) -> None:
        self.op = op
        self.outs = outs

    @overload
    def __rrshift__(
        self, other: Union["DimensionsData", Tuple["DimensionsData", ...]]
    ) -> tf.Tensor:
        ...

    @overload
    def __rrshift__(self, other: tf.Tensor) -> "DataOpOuts":
        ...

    def __rrshift__(
        self, other: Union[tf.Tensor, "DimensionsData", Tuple["DimensionsData", ...]]
    ) -> Union[tf.Tensor, "DataOpOuts"]:
        if isinstance(other, DimensionsData):
            outs = self.op.run(((other.in_data, other.in_dim),), self.outs)
            return tuple(out_data for out_data, out_dim in outs)
        if isinstance(other, tuple):
            ins = tuple((i.in_data, i.in_dim) for i in other)
            outs = self.op.run(ins, self.outs)
            return tuple(out_data for out_data, out_dim in outs)
        return DataOpOuts(other, self.op, self.outs)

    def __repr__(self) -> str:
        return f"OpOuts({self.op}, {self.outs})"


class DataOpOuts:
    def __init__(self, in_data: tf.Tensor, op: Op, outs: Tuple[Dimensions, ...]) -> None:
        self.in_data = in_data
        self.op = op
        self.outs = outs

    def __rrshift__(self, other: Dimensions) -> tf.Tensor:
        outs = self.op.run(((self.in_data, other),), self.outs)
        return tuple(out_data for out_data, out_dim in outs)

    def __repr__(self) -> str:
        return f"DataOpOuts({self.in_data}, {self.op}, {self.outs})"
