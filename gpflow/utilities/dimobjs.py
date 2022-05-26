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
from typing import Any, Counter, Dict, Mapping, Optional, Sequence, Tuple, Union

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
def tile(input_tensor: tf.Tensor, axis: tf.Tensor, sizes: tf.Tensor) -> tf.Tensor:
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


def dims(names: str) -> Sequence[Dim]:
    return [Dim(name.strip()) for name in names.split(",")]


def dimses(names: str) -> Sequence[Dims]:
    return [Dims(name.strip()) for name in names.split(",")]


class BoundTensor:
    def __init__(self, t: tf.Tensor, dimensions: Dimensions) -> None:
        self.t = t
        self.dimensions = dimensions

    def _get_indices(
        self, haystack: Sequence[VariableDimensions], needles: Sequence[VariableDimensions]
    ) -> tf.Tensor:
        i = tf.constant(0, dtype=tf.int32)
        indices = {}
        for straw in haystack:
            rank = straw.rank()
            indices[straw] = tf.range(i, i + rank)
            i += rank
        return tf.concat([indices[needle] for needle in needles], 0)

    def apply(
        self,
        to: Dimensions,
        *,
        reduce_fn: Reducer = tf.reduce_sum,
        expand_fn: Expander = tile,
    ) -> tf.Tensor:
        t = self.t
        frm = self.dimensions

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
            t = reduce_fn(t, axis=self._get_indices(frm_flat, removed))

        frm_inner = [f for f in frm_flat if f in to_set]
        to_inner = [t for t in to_flat if t in frm_set]
        if frm_inner != to_inner:
            t = tf.transpose(t, self._get_indices(frm_inner, to_inner))

        added = [t for t in to_flat if t not in frm_set]
        if added:
            multiples = tf.concat([t.shape() for t in added], 0)
            t = expand_fn(t, self._get_indices(to_flat, added), multiples)

        if not to.is_flat():
            t = tf.reshape(t, to.shape())

        return t


def bind(t: tf.Tensor, dimensions: Dimensions) -> BoundTensor:
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

    return BoundTensor(t, dimensions)


if __name__ == "__main__":
    x, ys, z = dims("x,ys...,z")

    t = tf.ones(tf.range(6) + 1)
    print(x + ys + z)
    bind(t, x + ys + z)
    print(x + ys + z)
    print(bind(t, x + ys + z).apply(z * x + ys + one).shape)
