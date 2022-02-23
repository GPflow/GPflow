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
============
check_shapes
============

A library for annotating and checking the shapes of tensors.

This library is compatible with both TensorFlow and NumPy.

The main entry point is :func:`shape_checking_study.numpy_example_2.check_shapes`.

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
``<dimension specifier i>`` is
one of:

    * ``<integer>``, to require that dimension to have that exact size.
    * ``<name>``, to bind that dimension to a variable. Dimensions bound to the same variable must
      have the same size, though that size can be anything.
    * ``*<name>`` or ``<name>...``, to bind *any* number of dimensions to a variable. Again,
      multiple uses of the same variable name must match the same dimension sizes.

A scalar shape is specified by ``[]``.

For example::

    @check_shapes(
        "...: []",
        "...: [3, 4]",
        "...: [width, height]",
        "...: [n_samples, *batch]",
        "...: [batch..., 2]",
    )
    def f(...):
        ...


Class inheritance
+++++++++++++++++

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


Speed, and interactions with `tf.function`
++++++++++++++++++++++++++++++++++++++++++

If you want to wrap your function in both :func:`tf.function` and :func:`check_shapes` it is
recommended you put the :func:`tf.function` outermost so that the shape checks are inside
:func:`tf.function`.  Shape checks are performed while tracing graphs, but *not* compiled into the
actual graphs.  This is considered a feature as that means that :func:`check_shapes` doesn't impact
the execution speed of compiled functions. However, it also means that tensor dimensions of dynamic
size are not verified in compiled mode.


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
"""

from .check_shapes import check_shapes
from .errors import ArgumentReferenceError, ShapeMismatchError
from .inheritance import inherit_check_shapes

__all__ = [export for export in dir()]
