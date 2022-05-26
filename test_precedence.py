import numpy as np
from typing import Tuple, Any, Union
import tensorflow as tf

class A:
    def __init__(self, *args) -> None:
        self.args = args

    def __repr__(self) -> str:
        tokens = []
        tokens.append("(")
        for arg in self.args:
            for arg_line in repr(arg).split("\n"):
                tokens.append("\n")
                tokens.append("  ")
                tokens.append(arg_line)
            tokens.append(",")
        if self.args:
            tokens.append("\n")
        tokens.append(")")
        return "".join(tokens)

    def __add__(self, other) -> None:
        return A(self, "+", other)

    def __radd__(self, other) -> None:
        return A(other, "+", self)

    def __lshift__(self, other) -> None:
        return A(self, "<<", other)

    def __rlshift__(self, other) -> None:
        return A(other, "<<", self)

    def __rshift__(self, other) -> None:
        return A(self, ">>", other)

    def __rrshift__(self, other) -> None:
        return A(other, ">>", self)

    def __and__(self, other) -> None:
        return A(self, "&", other)

    def __rand__(self, other) -> None:
        return A(other, "&", self)

    def __xor__(self, other) -> None:
        return A(self, "^", other)

    def __rxor__(self, other) -> None:
        return A(other, "^", self)

    def __or__(self, other) -> None:
        return A(self, "|", other)

    def __ror__(self, other) -> None:
        return A(other, "|", self)

    def __lt__(self, other) -> None:
        return A(self, "<", other)

    def __le__(self, other) -> None:
        return A(self, "<=", other)

    def __ge__(self, other) -> None:
        return A(self, ">=", other)

    def __gt__(self, other) -> None:
        return A(self, ">", other)

    def __getitem__(self, other) -> None:
        return A(self, "[]", other)



X = A("X")
s1 = A("s1")
s2 = A("s2")
s3 = A("s3")
t1 = A("tensor1")
t2 = A("tensor2")
o1 = A("operator")
print("----")
print(s1 + s2 + s3 | tf.constant([5, 2]) >> s1 + s2)
print("----")
print(s1 + s2 + s3 >> tf.constant([5, 2]) >> s1 + s2)

class Dim:

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Dim({self.name})"

    def __add__(self, other):
        return Dim(self.name + other.name)

    def __rrshift__(self, other):
        return PartialInputOutputBoundOp(other, rearrange, self)

    def __or__(self, other) -> "DimData":
        return DimData(self, other)


b = Dim("b")
h = Dim("h")
w = Dim("w")


class Op:

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, inputs: Tuple[Tuple[np.ndarray, Dim], ...], outputs: Tuple[Dim, ...]) -> Tuple[Tuple[np.ndarray, Dim], ...]:
        print("Running", self.name)
        for i in inputs:
            print("  in", i)
        for o in outputs:
            print("  out", o)
        return tuple((np.zeros((1, 2)), o) for o in outputs)

    def __rrshift__(self, other) -> Any:
        if isinstance(other, tuple):
            return InputBoundOp(other, self)
        else:
            return PartialInputBoundOp(other, self)

    def __repr__(self) -> str:
        return f"Op({self.name})"


class DimData:
    def __init__(self, dim: Dim, tensor: np.ndarray) -> None:
        self.dim = dim
        self.tensor = tensor

    def __or__(self, other) -> "DimData":
        return other.op.run(((other.tensor, self),), self.outputs)

    def __repr__(self) -> str:
        return f"DimData({self.dim},{self.tensor})"


class PartialInputBoundOp:

    def __init__(self, tensor: np.ndarray, op: Op) -> None:
        self.tensor = tensor
        self.op = op

    def __rshift__(self, other) -> "PartialInputOutputBoundOp":
        return PartialInputOutputBoundOp(self.tensor, self.op, other)

    def __repr__(self) -> str:
        return f"PartialInputBoundOp({self.tensor},{self.op})"


class PartialInputOutputBoundOp:

    def __init__(self, tensor: np.ndarray, op: Op, outputs: Union[Dim, Tuple[Dim, ...]]) -> None:
        if isinstance(outputs, Dim):
            outputs = (outputs,)
        self.tensor = tensor
        self.op = op
        self.outputs = outputs

    def __ror__(self, dim: Dim) -> np.ndarray:
        return self.op.run(((self.tensor, dim),), self.outputs)

    def __repr__(self) -> str:
        return f"PartialInputOutputBoundOp({self.tensor},{self.op},{self.outputs})"


class InputBoundOp:

    def __init__(self, inputs: Tuple[Tuple[np.ndarray, Dim], ...], op: Op) -> None:
        self.inputs = inputs
        self.op = op

    def __repr__(self) -> str:
        return f"InputBoundOp({self.inputs},{self.op})"

    def __rshift__(self, outputs) -> np.ndarray:
        return self.op.run(tuple((i.dim, i.tensor) for i in self.inputs), outputs)

reduce_sum = Op("reduce_sum")
tile = Op("tile")
einsum = Op("einsum")
rearrange = Op("rearrange")

print("rearrange")
print(b + h + w | np.array(7) >> h + b)
print("single input single output")
print(b + h + w | np.array(7) >> reduce_sum >> b + h)
print("single input multiple output")
print(b + h + w | np.array(7) >> reduce_sum >> (b + h, w))
print("multiple input single output")
print((b + h + w | np.array(1), w + h | 4) >> reduce_sum >> b + h)
print("multiple input multiple output")
print((b + h + w | np.array(1), w + h | 4) >> reduce_sum >> (b + h, w))


# Combinations:
# Dim
#   | data    (DimData)
#   + Dim     (Dim)
#   Op >>     (OpOut)
# Op
#   >> Dim    (OpOut)
#   >> tuple  (OpOuts)
#   data >>   (DataOp)
#   tuple >>  (InsOp)
# DimData
#   >> Op     (InOp)
#   >> OpOut  (data)
#   >> OpOuts (data)
# DataOut
#   Dim |
# InOp
# OpOut
# DataOp
# DataOpOut
# InsOp
# OpOuts
# DataOpOuts
