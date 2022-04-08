import inspect
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    NewType,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

_Sentinel = NewType("_Sentinel", object)
_SENTINEL = _Sentinel(object())
_EMPTY_SET: Set["Put[Any]"] = set()


T = TypeVar("T")
U = TypeVar("U")
C = TypeVar("C", bound="Clastion")
P = TypeVar("P", bound="Put[Any]")


class Clastion:
    def __init__(self: C, *parents: C, **assignments: Any) -> None:
        self._values: Dict["Put[Any]", Any] = {}
        self._dependencies: Dict["Put[Any]", Set["Put[Any]"]] = {}
        self._current: Optional["Put[Any]"] = None

        if parents:
            assert len(parents) == 1, "Only a single parent supported."
            (parent,) = parents
            self._values.update(parent._values)
            for key, dependencies in parent._dependencies.items():
                self._dependencies[key] = set(dependencies)

        cls = self.__class__
        new_values = {getattr(cls, name): value for name, value in assignments.items()}
        assert all((isinstance(key, Put) and key.settable) for key in new_values)

        clear_queue = {k for k in new_values if k in self._values}
        while clear_queue:
            i = clear_queue.pop()
            for dependency in self._dependencies.get(i, _EMPTY_SET):
                if (dependency in self._values) and (dependency not in clear_queue):
                    clear_queue.add(dependency)
            del self._values[i]
            del self._dependencies[i]

        for key, value in new_values.items():
            self._values[key] = value
            self._dependencies[key] = set()

        assert set(self._values) == set(self._dependencies)

        for key, value in self._values.items():
            self._values[key] = key.preprocess(self, value)

    def __call__(self: C, **assignments: Any) -> C:
        cls = self.__class__
        return cls(self, **assignments)

    def __repr__(self) -> str:
        ignore = {"_values", "_dependencies", "_current"}
        tokens = []
        tokens.append(self.__class__.__name__)
        tokens.append("(")
        any_fields = False
        for name in dir(self):
            if name.startswith("__") or name in ignore:
                continue
            any_fields = True
            tokens.append("\n    ")
            tokens.append(name)
            tokens.append("=")
            class_value = getattr(self.__class__, name, None)
            is_put = isinstance(class_value, Put)
            has_value = class_value in self._values
            if is_put and not has_value:
                tokens.append("** Not computed / cached **")
            else:
                value = getattr(self, name)
                if inspect.ismethod(value) and value.__self__ == self:
                    # Don't repr() own methods - that causes an infinte recursion...
                    tokens.append(f"<bound method {value.__qualname__} of self>")
                else:
                    value_str = repr(value)
                    head, *tail = value_str.split("\n")
                    tokens.append(head)
                    for t in tail:
                        tokens.append("\n    ")
                        tokens.append(t)
            tokens.append(",")
        if any_fields:
            tokens.append("\n")
        tokens.append(")")

        return "".join(tokens)


class Preprocessor(ABC):
    @abstractmethod
    def process(self, instance: Clastion, key: "Put[Any]", value: Any) -> Any:
        ...


class Put(Generic[T], ABC):
    def __init__(self, settable: bool, preprocessors: Tuple[Preprocessor, ...]) -> None:
        self.settable = settable
        self.preprocessors = preprocessors

        self._name: Optional[str] = None

    @overload
    def __get__(self: P, instance: None, owner: Type[Clastion]) -> P:
        ...

    @overload
    def __get__(self, instance: Clastion, owner: Type[Clastion]) -> T:
        ...

    def __get__(self: P, instance: Optional[Clastion], owner: Type[Clastion]) -> Union[T, P]:
        if instance is None:
            return self

        value = instance._values.get(self, _SENTINEL)
        if value is _SENTINEL:
            assert isinstance(self, FactoryPut)
            value = self.compute(instance)
            value = self.preprocess(instance, value)
            instance._values[self] = value
            instance._dependencies[self] = set()
            assert set(instance._values) == set(instance._dependencies)

        if instance._current is not None:
            instance._dependencies[self].add(instance._current)

        return cast(T, value)

    def __set_name__(self, owner: Type[Clastion], name: str) -> None:
        self._name = name

    def preprocess(self, instance: Clastion, value: T) -> T:
        for preprocessor in self.preprocessors:
            value = preprocessor.process(instance, self, value)
        return value

    def __repr__(self) -> str:
        token_list = []
        if self._name is not None:
            token_list.append(f"name={self._name}")
        token_list.append(f"settable={self.settable}")
        for preprocessor in self.preprocessors:
            token_list.append(repr(preprocessor))
        tokens = ",".join(token_list)
        return f"{self.__class__.__name__}[{tokens}]"


class InPut(Put[T], Generic[T]):
    def __call__(self, factory: Callable[[Any], U]) -> "FactoryPut[U]":
        return FactoryPut(factory, self.settable, self.preprocessors)


class FactoryPut(Put[T], Generic[T]):
    def __init__(
        self,
        factory: Callable[[Any], T],
        settable: bool,
        preprocessors: Tuple[Preprocessor, ...],
    ) -> None:
        super().__init__(settable, preprocessors)
        self._factory = factory

    def compute(self, instance: Clastion) -> T:
        # pylint: disable=protected-access
        prev = instance._current
        instance._current = self
        try:
            return self._factory(instance)
        finally:
            instance._current = prev


@overload
def put(t: Type[T], *preprocessors: Preprocessor) -> InPut[T]:
    ...


@overload
def put(*preprocessors: Preprocessor) -> InPut[Any]:
    ...


def put(*type_and_preprocessors: Union[Type[T], Preprocessor]) -> InPut[Any]:  # type: ignore
    only_preprocessors = tuple(p for p in type_and_preprocessors if isinstance(p, Preprocessor))
    return InPut(settable=True, preprocessors=only_preprocessors)


def derived(*preprocessors: Preprocessor) -> Callable[[Callable[[Any], T]], FactoryPut[T]]:
    def wrap(factory: Callable[[Any], T]) -> FactoryPut[T]:
        return FactoryPut(factory, settable=False, preprocessors=preprocessors)

    return wrap
