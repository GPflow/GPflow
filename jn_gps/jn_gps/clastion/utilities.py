import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, List, Mapping, Tuple, TypeVar, cast

from .clastion import Clastion

C = TypeVar("C", bound="Clastion")
CompareResult = Any  # Union[bool, NotImplementedType]


class _PathElement(ABC):
    @abstractmethod
    def get(self, source: Any) -> Any:
        ...

    def __lt__(self, other: "_PathElement") -> CompareResult:
        return compare_path_element(operator.lt, self, other)

    def __le__(self, other: "_PathElement") -> CompareResult:
        return compare_path_element(operator.le, self, other)

    def __gt__(self, other: "_PathElement") -> CompareResult:
        return compare_path_element(operator.gt, self, other)

    def __ge__(self, other: "_PathElement") -> CompareResult:
        return compare_path_element(operator.ge, self, other)


def compare_path_element(
    op: Callable[[Any, Any], bool], lhs: _PathElement, rhs: _PathElement
) -> CompareResult:
    t_lhs = type(lhs)
    t_rhs = type(rhs)
    if t_lhs != t_rhs:
        return op(t_lhs.__name__, t_rhs.__name__)
    if t_lhs is _AttributePathElement:
        return op(lhs.name, rhs.name)  # type: ignore[attr-defined]
    if t_lhs is _ItemPathElement:
        return op(lhs.key, rhs.key)  # type: ignore[attr-defined]
    return NotImplemented


@dataclass(frozen=True)
class _AttributePathElement(_PathElement):
    name: str

    def get(self, source: Any) -> Any:
        return getattr(source, self.name)

    def __repr__(self) -> str:
        return "." + self.name


@dataclass(frozen=True)
class _ItemPathElement(_PathElement):
    key: Any

    def get(self, source: Any) -> Any:
        return source[self.key]

    def __repr__(self) -> str:
        return "[" + repr(self.key) + "]"


class ClastionPath:
    def __init__(self, elements: Tuple[_PathElement, ...] = ()) -> None:
        self.__elements__ = elements

    def __getattr__(self, name: str) -> "ClastionPath":
        return ClastionPath(self.__elements__ + (_AttributePathElement(name),))

    def __getitem__(self, key: Any) -> "ClastionPath":
        return ClastionPath(self.__elements__ + (_ItemPathElement(key),))

    def __repr__(self) -> str:
        return "".join(repr(e) for e in self.__elements__)

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.__elements__ == other.__elements__

    def __lt__(self, other: "ClastionPath") -> CompareResult:
        return self.__elements__ < other.__elements__

    def __le__(self, other: "ClastionPath") -> CompareResult:
        return self.__elements__ <= other.__elements__

    def __gt__(self, other: "ClastionPath") -> CompareResult:
        return self.__elements__ > other.__elements__

    def __ge__(self, other: "ClastionPath") -> CompareResult:
        return self.__elements__ >= other.__elements__

    def __hash__(self) -> int:
        return hash(self.__elements__)


root = ClastionPath()


def _multi_get(
    instance: Any, paths: Collection[ClastionPath], depth: int
) -> Mapping[ClastionPath, Any]:
    result = {}
    by_element: Dict[_PathElement, List[ClastionPath]] = {}
    for path in paths:
        if depth < len(path.__elements__):
            element = path.__elements__[depth]
            by_element.setdefault(element, []).append(path)
        else:
            result[path] = instance

    next_depth = depth + 1
    for element, next_paths in by_element.items():
        next_instance = element.get(instance)
        result.update(_multi_get(next_instance, next_paths, next_depth))

    return result


def multi_get(instance: Any, paths: Collection[ClastionPath]) -> Mapping[ClastionPath, Any]:
    return _multi_get(instance, paths, 0)


def _multi_set(instance: Any, values: Mapping[ClastionPath, Any], depth: int) -> Any:
    updates = {}
    by_element: Dict[_PathElement, Dict[ClastionPath, Any]] = {}
    for path, value in values.items():
        if depth < len(path.__elements__):
            element = path.__elements__[depth]
            by_element.setdefault(element, {})[path] = value
        else:
            assert len(values) == 1, values
            return value

    next_depth = depth + 1
    for element, next_values in by_element.items():
        next_instance = element.get(instance)
        updates[element] = _multi_set(next_instance, next_values, next_depth)

    if isinstance(instance, Clastion):
        kwargs = {}
        for e, v in updates.items():
            assert isinstance(e, _AttributePathElement)
            kwargs[e.name] = v
        return instance(**kwargs)
    elif isinstance(instance, (tuple, list)):
        modifiable = list(instance)
        for e, v in updates.items():
            assert isinstance(e, _ItemPathElement)
            modifiable[e.key] = v
        return instance.__class__(modifiable)
    else:
        raise NotImplementedError(f"Don't know how to update objects of type {type(instance)}.")


def multi_set(instance: C, values: Mapping[ClastionPath, Any]) -> C:
    return cast(C, _multi_set(instance, values, 0))


def to_loss_function(
    instance: C, trainable_paths: Collection[ClastionPath], loss_path: ClastionPath
) -> Tuple[Callable[[Mapping[ClastionPath, Any]], Any], Mapping[ClastionPath, Any]]:
    def loss(values: Mapping[ClastionPath, Any]) -> Any:
        next_instance = multi_set(instance, values)
        return multi_get(next_instance, [loss_path])[loss_path]

    return loss, multi_get(instance, trainable_paths)
