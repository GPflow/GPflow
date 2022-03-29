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

# flake8: noqa
"""
A library for annotating and checking the shapes of tensors.

This library is compatible with both TensorFlow and NumPy.

The main entry point is :func:`check_shapes.check_shapes`.

For example::

    @tf.function
    @check_shapes(
        "features: [batch_shape..., n_features]",
        "weights: [n_features]",
        "return: [batch_shape...]",
    )
    def linear_model(
        features: tf.Tensor, weights: tf.Tensor
    ) -> tf.Tensor:
        ...


Check specification
+++++++++++++++++++

The shapes to check are specified by the arguments to :func:`check_shapes`. Each argument is a
string of the format::

    <argument specifier>: <shape specifier>


Argument specification
----------------------

The ``<argument specifier>`` must start with either the name of an argument to the decorated
function, or the special name ``return``. The value ``return`` refers to the value returned by the
function.

The ``<argument specifier>`` can then be modified to refer to elements of the object in two ways:

    * Use ``.<name>`` to refer to attributes of the object.
    * Use ``[<index>]`` to refer to elements of a sequence. This is particularly useful if your
      function returns a tuple of values.

We do not support looking up values in a  ``dict``.

If the argument, or any of the looked-up values, are `None` the check is skipped.

For example::

    @check_shapes(
        "weights: ...",
        "data.training_data: ...",
        "return: ...",
        "return[0]: ...",
        "something[0].foo.bar[23]: ...",
    )
    def f(...):
        ...


Shape specification
-------------------

Shapes are specified by the syntax
``[<dimension specifier 1>, <dimension specifer 2>, ..., <dimension specifier n>]``, where
``<dimension specifier i>`` is one of:

    * ``<integer>``, to require that dimension to have that exact size.
    * ``<name>``, to bind that dimension to a variable. Dimensions bound to the same variable must
      have the same size, though that size can be anything.
    * ``None`` or ``.`` to allow exactly one single dimension without constraints.
    * ``*<name>`` or ``<name>...``, to bind *any* number of dimensions to a variable. Again,
      multiple uses of the same variable name must match the same dimension sizes.
    * ``*`` or ``...``, to allow *any* number of dimensions without constraints.

A scalar shape is specified by ``[]``.

For example::

    @check_shapes(
        "...: []",
        "...: [3, 4]",
        "...: [width, height]",
        "...: [., height]",
        "...: [width, None],
        "...: [n_samples, *batch]",
        "...: [batch..., 2]",
        "...: [n_samples, *]",
        "...: [..., 2]",
    )
    def f(...):
        ...


Shape reuse
+++++++++++

Just like with other code it is useful to be able to define a shape in one place and reuse it.
In particular this ensures that your code keep having consistent shapes, even if it is refactored.


Class inheritance
-----------------

If you have a class hiererchy, you probably want to ensure that derived classes handle tensors with
the same shapes as the base classes. You can use the :func:`inherit_check_shapes` decorator to
inherit shapes from overridden methods.

Example::

    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            ("a", ["batch...", 4]),
            ("return", ["batch...", 1]),
        )
        def f(self, a: tf.Tensor) -> tf.Tensor:
            ...

    class SubClass(SuperClass):
        @inherit_check_shapes
        def f(self, a: tf.Tensor) -> tf.Tensor:
            ...


Functional programming
----------------------

If you prefer functional- over object oriented programming, you may have functions that you require
to handle the same shapes. To do this, remember that in Python a decorator is just a function, and
functions are objects that can be stored::

    check_my_shapes = check_shapes(
        ("a", ["batch...", 4]),
        ("return", ["batch...", 1]),
    )

    @check_my_shapes
    def f(a: tf.Tensor) -> tf.Tensor:
        ...

    @check_my_shapes
    def g(a: tf.Tensor) -> tf.Tensor:
        ...


Other reuse of shapes
---------------------

You can use :func:`get_check_shapes` to get, and reuse, the shape definitions from a previously
declared function. This is particularly useful to ensure fakes in tests use the same shapes as the
production implementation::

    @check_shapes(
        ("a", ["batch...", 4]),
        ("return", ["batch...", 1]),
    )
    def f(a: tf.Tensor) -> tf.Tensor:
        ...

    def test_something() -> None:
        @get_check_shapes(f)
        def fake_f(a: tf.Tensor) -> tf.Tensor:
            ...

        # Test that patches `f` with `fake_f` goes here...


Speed, and interactions with `tf.function`
++++++++++++++++++++++++++++++++++++++++++

If you want to wrap your function in both :func:`tf.function` and :func:`check_shapes` it is
recommended you put the :func:`tf.function` outermost so that the shape checks are inside
:func:`tf.function`.  Shape checks are performed while tracing graphs, but *not* compiled into the
actual graphs.  This is considered a feature as that means that :func:`check_shapes` doesn't impact
the execution speed of compiled functions. However, it also means that tensor dimensions of dynamic
size are not verified in compiled mode.

If your code is very performance sensitive and :mod:`check_shapes` is causing an unacceptable
slowdown it can be disabled. Preferably use :func:`disable_check_shapes`::

    with disable_check_shapes():
        function_that_is_performance_sensitive()

Alternatively :mod:`check_shapes` can also be disable globally with
:func:`set_enable_check_shapes`::

    set_enable_check_shapes(False)

Beware that any function declared while shape checking is disabled, will continue not to check
shapes, even if shape checking is otherwise enabled again.


Documenting shapes
++++++++++++++++++

The :func:`check_shapes` decorator rewrites the docstring (`.__doc__`) of the decorated function to
add information about shapes, in a format compatible with
`Sphinx <https://www.sphinx-doc.org/en/master/>`_. Only parameters that already have a `:param ...:`
section will be modified.

For example::

    @tf.function
    @check_shapes(
        "features: [batch_shape..., n_features]",
        "weights: [n_features]",
        "return: [batch_shape...]",
    )
    def linear_model(
        features: tf.Tensor, weights: tf.Tensor
    ) -> tf.Tensor:
        \"\"\"
        Computes a prediction from a linear model.
        :param features: Data to make predictions from.
        :param weights: Model weights.
        :returns: Model predictions.
        \"\"\"
        ...

will have `.__doc__`::

    \"\"\"
    Computes a prediction from a linear model.

    :param a:
        * **features** has shape [*batch_shape*..., *n_features*].

        Data to make predictions from.
    :param b:
        * **weights** has shape [*n_features*].

        Model weights.
    :returns:
        * **return** has shape [*batch_shape*...].

        Model predictions.
    \"\"\"

if you do not wish to have your docstrings rewritten, you can disable it with
:func:`set_rewrite_docstrings`::

    set_rewrite_docstrings(None)


Shapes of custom objects
++++++++++++++++++++++++

:mod:`check_shapes` uses the function :func:`get_shape` to extract the shape of an object.

:func:`get_shape` uses :func:`functools.singledispatch` to branch on the type of object to the shape
from, and you can extend this to extract shapes for you own custom types.

For example::

    import numpy as np

    from gpflow.experimental.check_shapes import Shape, check_shapes, get_shape


    class LinearModel:
        def __init__(self, weights: np.ndarray) -> None:
            self._weights = weights

        @check_shapes(
            "self: [n_features, n_labels]",
            "features: [n_rows, n_features]",
            "return: [n_rows, n_labels]",
        )
        def predict(self, features: np.ndarray) -> None:
            return features @ self._weights


    @get_shape.register(LinearModel)
    def get_linear_model_shape(model: LinearModel) -> Shape:
        return model._weights.shape


    @check_shapes(
        "model: [n_features, n_labels]",
        "test_features: [n_rows, n_features]",
        "test_labels: [n_rows, n_labels]",
    )
    def loss(
        model: LinearModel, test_features: np.ndarray, test_labels: np.ndarray
    ) -> None:
        prediction = model.predict(test_features)
        return np.mean(np.sqrt(np.mean((prediction - test_labels) ** 2, axis=-1)))

"""

from .accessors import get_check_shapes, maybe_get_check_shapes
from .base_types import Dimension, Shape
from .check_shapes import check_shapes
from .config import (
    DocstringFormat,
    disable_check_shapes,
    get_enable_check_shapes,
    get_rewrite_docstrings,
    set_enable_check_shapes,
    set_rewrite_docstrings,
)
from .error_contexts import (
    ArgumentContext,
    AttributeContext,
    ErrorContext,
    FunctionCallContext,
    FunctionDefinitionContext,
    IndexContext,
    LarkUnexpectedInputContext,
    MessageBuilder,
    ObjectTypeContext,
    ParallelContext,
    ShapeContext,
    StackContext,
)
from .exceptions import (
    ArgumentReferenceError,
    CheckShapesError,
    DocstringParseError,
    NoShapeError,
    ShapeMismatchError,
    SpecificationParseError,
)
from .inheritance import inherit_check_shapes
from .shapes import get_shape

__all__ = [
    "ArgumentContext",
    "ArgumentReferenceError",
    "AttributeContext",
    "CheckShapesError",
    "Dimension",
    "DocstringFormat",
    "DocstringParseError",
    "ErrorContext",
    "FunctionCallContext",
    "FunctionDefinitionContext",
    "IndexContext",
    "LarkUnexpectedInputContext",
    "MessageBuilder",
    "NoShapeError",
    "ObjectTypeContext",
    "ParallelContext",
    "Shape",
    "ShapeContext",
    "ShapeMismatchError",
    "SpecificationParseError",
    "StackContext",
    "accessors",
    "argument_ref",
    "base_types",
    "check_shapes",
    "config",
    "disable_check_shapes",
    "error_contexts",
    "exceptions",
    "get_check_shapes",
    "get_enable_check_shapes",
    "get_rewrite_docstrings",
    "get_shape",
    "inherit_check_shapes",
    "inheritance",
    "maybe_get_check_shapes",
    "parser",
    "set_enable_check_shapes",
    "set_rewrite_docstrings",
    "shapes",
    "specs",
]
