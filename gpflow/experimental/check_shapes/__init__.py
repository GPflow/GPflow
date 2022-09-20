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
A library for annotating and checking the shapes of tensors.

The main entry point is :func:`check_shapes`.

Example:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [basic]
   :end-before: [basic]
   :dedent:


Speed, and interactions with `tf.function`
++++++++++++++++++++++++++++++++++++++++++

Shape checking has some performance impact. Shape checking can be disabled to help alleviate this.
Shape checking can be set to one of three different states:

* ``ENABLED``. Shapes are checked wherever they can be.
* ``EAGER_MODE_ONLY``. Shapes are not checked within anything wrapped in :func:`tf.function`.
* ``DISABLED``. Shapes are never checked.

The state can be set with :func:`set_enable_check_shapes`:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [disable__manual]
   :end-before: [disable__manual]
   :dedent:

Alternatively you can use :func:`disable_check_shapes` to disable shape checking in smaller scopes:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [disable__context_manager]
   :end-before: [disable__context_manager]
   :dedent:

Beware that any function declared while shape checking is disabled, will continue not to check
shapes, even if shape checking is otherwise enabled again.

The default state is ``EAGER_MODE_ONLY``; which is appropriate for smaller project, experiments, and
notebooks. Write and debug your code in eager mode, and add :func:`tf.function` when you believe
your code is correct and you want it to run fast. For larger project you probably want to modify
this setting. In particular you may want to enable all shape checks in your unit tests. If you use
`pytest <https://docs.pytest.org/>`_ you can do this by updating your root ``conftest.py`` with:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [pytest_fixture]
   :end-before: [pytest_fixture]
   :dedent:


If shape checking is set to ``ENABLED`` and your code is wrapped in :func:`tf.function` shape checks
are performed while tracing graphs, but *not* compiled into the actual graphs.  This is considered a
feature as that means that :func:`check_shapes` doesn't impact the execution speed of your functions
after they have been compiled.


Best-effort checking
++++++++++++++++++++

This library will perform shape checks on a best-effort basis. Many things can prevent this library
from being able to check shapes. For example:

* Unknown shapes. Sometimes the library is not able to determine the shape of an object, and thus
  cannot check that object. For example ``Optional`` arguments with value ``None`` cannot be
  checked, and compiled TensorFlow code can have variables with an unknown shape.

* Use of variable-rank dimensions (see below). In general we cannot infer the size of variable-rank
  dimensions if there are multiple variable-rank specifications within the same shape specification
  (e.g. ``cov: [m..., n...]``). This library will try to learn the size of these variable-rank
  dimensions from neighbouring shape specifications, but this is not always possible. Use of
  ``broadcast`` with variable-rank dimensions makes it even harder to infer these values.


Check specification
+++++++++++++++++++

The shapes to check are specified by the arguments to :func:`check_shapes`. Each argument is a
string of the format:

.. code-block:: bnf

   <argument specifier> ":" <shape specifier> ["if" <condition>] ["#" <note>]


Argument specification
----------------------

The ``<argument specifier>`` must start with either the name of an argument to the decorated
function, or the special name ``return``. The value ``return`` refers to the value returned by the
function.

The ``<argument specifier>`` can then be modified to refer to elements of the object in several
ways:

* Use ``.<name>`` to refer to an attribute of an object:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [argument_ref_attribute]
     :end-before: [argument_ref_attribute]
     :dedent:

* Use ``[<index>]`` to refer to a specific element of a sequence. This is particularly useful if
  your function returns a tuple of values:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [argument_ref_index]
     :end-before: [argument_ref_index]
     :dedent:

* Use ``[all]`` to select all elements of a collection:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [argument_ref_all]
     :end-before: [argument_ref_all]
     :dedent:

* Use ``.keys()`` to select all keys of a mapping:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [argument_ref_keys]
     :end-before: [argument_ref_keys]
     :dedent:

* Use ``.values()`` to select all values of a mapping:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [argument_ref_values]
     :end-before: [argument_ref_values]
     :dedent:

.. note::

   We do not support looking up a specific key or value in a  ``dict``.

If the argument, or any of the looked-up values, are ``None`` the check is skipped. This is useful
for optional values:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [argument_ref_optional]
   :end-before: [argument_ref_optional]
   :dedent:


Shape specification
-------------------

Shapes are specified by the syntax:


.. code-block:: bnf

   "[" <dimension specifier 1> "," <dimension specifer 2> "," ... "," <dimension specifier n> "]"

where ``<dimension specifier i>`` is one of:

* ``<integer>``, to require that dimension to have that exact size:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [dimension_spec_constant]
     :end-before: [dimension_spec_constant]
     :dedent:

* ``<name>``, to bind that dimension to a variable. Dimensions bound to the same variable must
  have the same size, though that size can be anything:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [dimension_spec_variable]
     :end-before: [dimension_spec_variable]
     :dedent:

* ``None`` or ``.`` to allow exactly one single dimension without constraints:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [dimension_spec_anonymous__none]
     :end-before: [dimension_spec_anonymous__none]
     :dedent:

  or:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [dimension_spec_anonymous__dot]
     :end-before: [dimension_spec_anonymous__dot]
     :dedent:

* ``*<name>`` or ``<name>...``, to bind *any* number of dimensions to a variable. Again,
  multiple uses of the same variable name must match the same dimension sizes:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [dimension_spec_variable_rank__star]
     :end-before: [dimension_spec_variable_rank__star]
     :dedent:

  or:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [dimension_spec_variable_rank__ellipsis]
     :end-before: [dimension_spec_variable_rank__ellipsis]
     :dedent:

* ``*`` or ``...``, to allow *any* number of dimensions without constraints:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [dimension_spec_anonymous_variable_rank__star]
     :end-before: [dimension_spec_anonymous_variable_rank__star]
     :dedent:

  or:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [dimension_spec_anonymous_variable_rank__ellipsis]
     :end-before: [dimension_spec_anonymous_variable_rank__ellipsis]
     :dedent:

A scalar shape is specified by ``[]``:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [dimension_spec__scalar]
   :end-before: [dimension_spec__scalar]
   :dedent:

Any of the above can be prefixed with the keyword ``broadcast`` to allow any value that broadcasts
to the specification. For example:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [dimension_spec_broadcast]
   :end-before: [dimension_spec_broadcast]
   :dedent:

Specifically, to mark a dimension as ``broadcast`` means:

* If the specification is that the dimension should have size ``n``, then the actual dimension must
  have value ``1`` or ``n``.
* If all leading dimension specifications are also marked ``broadcast``, then the actual shape is
  allowed to be shorter than the specification — the dimension is allowed to be missing.


Condition specification
-----------------------

You can use the optional ``if <condition>`` syntax to conditionally evaluate shapes. If an ``if
<condition>`` is used, the specification is only appplied if ``<condition>`` evaluates to ``True``.
This is useful if shapes depend on other input parameters. Valid conditions are:

* ``<argument specifier>``, with the same syntax and rules as above, except that constructions that
  evaluates to multiple elements are disallowed. Uses the ``bool`` built-in to convert the value of
  the argument to a ``bool``:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [bool_spec_argument_ref]
     :end-before: [bool_spec_argument_ref]
     :dedent:

* ``<argument specifier> is None``, and ``<argument specifier> is not None``, with the usual rules
  for an ``<argument specifier>``, to test whether an argument is, or is not, ``None``. We currently
  only allow tests against ``None``, not general Python equality tests:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [bool_spec_argument_ref_is_none]
     :end-before: [bool_spec_argument_ref_is_none]
     :dedent:

* ``<left> or <right>``, evaluates to ``True`` if any of ``<left>`` or ``<right>`` evaluates to
  ``True`` and to ``False`` otherwise:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [bool_spec_or]
     :end-before: [bool_spec_or]
     :dedent:

* ``<left> and <right>``, evaluates to ``False`` if any of ``<left>`` or ``<right>`` evaluates to
  ``False`` and to ``True`` otherwise:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [bool_spec_and]
     :end-before: [bool_spec_and]
     :dedent:

* ``not <right>``, evaluates to the opposite value of ``<right>``:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [bool_spec_not]
     :end-before: [bool_spec_not]
     :dedent:

* ``(<exp>)``, uses parenthesis to change operator precedence, as usual.

Conditions can be composed to apply different specs, depending on function arguments:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [bool_spec__composition]
   :end-before: [bool_spec__composition]
   :dedent:

.. note::

   All specifications with either no ``if`` syntax or a ``<condition>`` that evaluates to ``True``
   will be applied. It is possible for multiple specifications to apply to the same value.


Note specification
------------------

You can add notes to your specifications using a ``#`` followed by the note. These notes will be
appended to relevant error messages and appear in rewritten docstrings. You can add notes in two
places:

* On a single line by itself, to add a note to the entire function:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [note_spec__global]
     :end-before: [note_spec__global]
     :dedent:

* After the specification of a single argument, to add a note to that argument only:

  .. literalinclude:: /examples/test_check_shapes_examples.py
     :start-after: [note_spec__local]
     :end-before: [note_spec__local]
     :dedent:


Shape reuse
+++++++++++

Just like with other code it is useful to be able to specify a shape in one place and reuse the
specification. In particular this ensures that your code keep having internally consistent shapes,
even if it is refactored.


Class inheritance
-----------------

If you have a class hiererchy, you probably want to ensure that derived classes handle tensors with
the same shapes as the base classes. You can use the :func:`inherit_check_shapes` decorator to
inherit shapes from overridden methods:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [reuse__inherit_check_shapes]
   :end-before: [reuse__inherit_check_shapes]
   :dedent:


Functional programming
----------------------

If you prefer functional- over object oriented programming, you may have functions that you require
to handle the same shapes. To do this, remember that in Python a decorator is just a function, and
functions are objects that can be stored:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [reuse__functional]
   :end-before: [reuse__functional]
   :dedent:


Other reuse of shapes
---------------------

You can use :func:`get_check_shapes` to get, and reuse, the shape definitions from a previously
declared function. This is particularly useful to ensure fakes in tests use the same shapes as the
production implementation:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [reuse__get_check_shapes]
   :end-before: [reuse__get_check_shapes]
   :dedent:


Checking intermediate results
+++++++++++++++++++++++++++++

You can use the function :func:`check_shape` to check the shape of an intermediate result. This
function will use the same namespace as the immediately surrounding :func:`check_shapes` decorator,
and should only be called within functions that has such a decorator. For example:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [intermediate_results]
   :end-before: [intermediate_results]
   :dedent:


Checking shapes without a decorator
+++++++++++++++++++++++++++++++++++

While the :func:`check_shapes` decorator is the recommend way to use this library, it is possible to
use it without the decorator. In fact the decorator is just a wrapper around the class
:class:`ShapeChecker`, which can be used to check shapes directly:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [shape_checker__raw]
   :end-before: [shape_checker__raw]
   :dedent:

You can use the function :func:`get_shape_checker` to get the :class:`ShapeChecker` used by any
immediately surrounding :func:`check_shapes` decorator.


Documenting shapes
++++++++++++++++++

The :func:`check_shapes` decorator rewrites the docstring (``.__doc__``) of the decorated function
to add information about shapes, in a format compatible with
`Sphinx <https://www.sphinx-doc.org/en/master/>`_.

Only functions that already have a docstring will be updated. Functions that have no docstring at
all will not have one added, this is so that we do not override a docstring that would have been
inherited from a super class.

For example:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [doc_rewrite__definition]
   :end-before: [doc_rewrite__definition]
   :dedent:

will have ``.__doc__``:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [doc_rewrite__rewritten]
   :end-before: [doc_rewrite__rewritten]
   :dedent:

if you do not wish to have your docstrings rewritten, you can disable it with
:func:`set_rewrite_docstrings`:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [doc_rewrite__disable]
   :end-before: [doc_rewrite__disable]
   :dedent:


Supported types
+++++++++++++++

This library has built-in support for checking the shapes of:

* Python built-in scalars: ``bool``, ``int``, ``float`` and ``str``.
* Python built-in sequences: ``tuple`` and ``list``.
* NumPy ``ndarray``\ s.
* TensorFlow ``Tensor``\ s and ``Variable``\ s.
* TensorFlow Probability ``DeferredTensor``\ s, including ``TransformedVariable`` and
  :class:`gpflow.Parameter`.


Shapes of custom types
----------------------

:mod:`check_shapes` uses the function :func:`get_shape` to extract the shape of an object.
You can use :func:`register_get_shape` to extend :func:`get_shape` to extract shapes for you own
custom types:

.. literalinclude:: /examples/test_check_shapes_examples.py
   :start-after: [custom_type]
   :end-before: [custom_type]
   :dedent:
"""

from .accessors import get_check_shapes
from .base_types import Dimension, Shape
from .checker import ShapeChecker
from .checker_context import check_shape, get_shape_checker
from .config import (
    DocstringFormat,
    ShapeCheckingState,
    disable_check_shapes,
    get_enable_check_shapes,
    get_enable_function_call_precompute,
    get_rewrite_docstrings,
    set_enable_check_shapes,
    set_enable_function_call_precompute,
    set_rewrite_docstrings,
)
from .decorator import check_shapes
from .error_contexts import ErrorContext
from .inheritance import inherit_check_shapes
from .shapes import get_shape, register_get_shape

__all__ = [
    "Dimension",
    "DocstringFormat",
    "ErrorContext",
    "Shape",
    "ShapeChecker",
    "ShapeCheckingState",
    "accessors",
    "argument_ref",
    "base_types",
    "bool_specs",
    "check_shape",
    "check_shapes",
    "checker",
    "checker_context",
    "config",
    "decorator",
    "disable_check_shapes",
    "error_contexts",
    "exceptions",
    "get_check_shapes",
    "get_enable_check_shapes",
    "get_enable_function_call_precompute",
    "get_rewrite_docstrings",
    "get_shape",
    "get_shape_checker",
    "inherit_check_shapes",
    "inheritance",
    "parser",
    "register_get_shape",
    "set_enable_check_shapes",
    "set_enable_function_call_precompute",
    "set_rewrite_docstrings",
    "shapes",
    "specs",
]
