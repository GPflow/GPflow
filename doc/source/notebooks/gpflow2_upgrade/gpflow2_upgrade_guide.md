---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# GPflow 2 Upgrade Guide

This is a basic guide for people who have GPflow 1 code that needs to be upgraded to GPflow 2.


## Kernel Input Dims

The `input_dim` parameter has been removed from the `Kernel` class’s initialiser. Therefore all calls to create a kernel must be changed to remove the `input_dim` parameter.

For example:

<img src="files/kernel_input_dims_1_new.png">
<img src="files/kernel_input_dims_2_new.png">

**Note**: old code may still run without obvious errors against GPflow, since many kernels take an optional numerical value as their first parameter. You may not get the result you expect though!


## Parameter and tf.Variable

The `Parameter` class in GPflow 1 was a separate class from `tf.Variable`. The `params_as_tensors` decorator or the `params_as_tensors_for` context manager were required to turn them into a tensor that could be consumed by TensorFlow operations.

In GPflow 2, `Parameter` is a subclass of `tf.Module` that wraps a `tf.Variable`, and can directly be used in place of a tensor, so no such conversion is necessary.

References to `params_as_tensors` and `params_as_tensors_for` can simply be removed.



## Parameter Assignment

In GPflow 2 the semantics of assigning values to parameters has changed. It is now necessary to use the Parameter.assign method rather than assigning values directly to parameters. For example:

<img src="files/constant.png">

In the above example, the old (GPflow 1) code would have assigned the value of `likelihood.scale` to 0.1 (assuming that likelihood is a `Parameterized` object and scale is a `Parameter`), rather than replacing the `scale` attribute with a Python float (which would be the “normal” Python behaviour). This maintains the properties of the parameter. For example, it remains trainable etc.

In GPflow 2, it is necessary to use the `Parameter.assign` method explicitly to maintain the same behaviour, otherwise the parameter attribute will be replaced by an (untrainable) constant value.

To change other properties of the parameter (for example, to change transforms etc) you may need to replace the entire parameter object. See [this notebook](../understanding/models.ipynb#Constraints-and-trainable-variables) for further details.


## SciPy Optimizer

Usage of GPflow’s Scipy optimizer has changed. It has been renamed from `gpflow.train.ScipyOptimizer` to `gpflow.optimizers.Scipy` and its `minimize` method has changed in the following ways:

 * Instead of a GPflow model the method now takes a zero-argument function that returns the loss to be minimised (for example, the negative log marginal likelihood), as well as the variables to be optimised (typically `model.trainable_variables`).
 * The options (`disp`, `maxiter`) must now be passed in a dictionary.

For example:

<img src="files/scipy_optimizer.png">

Any additional keyword arguments that are passed to the `minimize` method are passed directly through to the [SciPy optimizer's minimize method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).


# Model Initialisers

In many cases the initialiser for the model will have changed. Typical changes include:

 * Instead of separate parameters for `X` and `Y`, some models now require a single `data` parameter containing a tuple of the X and Y data.
 * The `kern` parameter has been renamed to `kernel`.

For example, for the `GPR` model:

<img src="model_1.png">
<img src="model_2_new.png">


# SVGP Initialiser

The SVGP model’s initialiser no longer accepts X and Y data. Instead this data must be passed to the various computation methods of the model (`elbo`, `log_likelihood` etc).

In the [Introduction to GPflow 2 notebook](../intro_to_gpflow2.ipynb) there is an example of how to use SVGP with optimisation using mini-batches of data.

In addition, SVGP’s `Z` parameter has been removed. To pass-in inducing points use the `inducing_variable` parameter. Also `SVGP`'s `feature` attribute has been renamed to `inducing_variable`.


# Autoflow

The `@autoflow` decorator has been removed. Since eager execution is the default in TensorFlow 2 this is no longer necessary.

You may wish to consider wrapping functions that were previously wrapped in the `@autoflow` decorator in the `tf.function` decorator instead, to improve performance (but this is not necessary from a functionality point of view).


# Use of tf.function

Wrapping compute-heavy operations such as calculating a model objective or even the optimizer steps (such as `tf.optimizers.Adam().minimize()`) with `tf.function` is crucial for efficient computation.

**Note**: you should ensure that functions wrapped in `tf.function` are only passed **tensors** (not numpy arrays or other data structures, with the exception of a small number of bool or enum-style flags), or the decorator will re-compile the graph each time the function is passed new objects as its arguments. See the [TensorFlow documentation on re-tracing](https://www.tensorflow.org/guide/function#re-tracing) for further details.

You can convert a numpy array to a tensor by using `tf.constant`. For example: `compiled_function(tf.constant(numpy_array))`.


# Model Compilation

Models no longer need to be compiled before use. Remove all calls to the `compile` method.


# Sessions and Graphs

GPflow only supports eager execution, which is the default in TensorFlow 2. It does not support graph mode, which was the default execution mode in TensorFlow 1. Therefore all references to Sessions and Graphs should be removed. You should also remove references to the `gpflow.reset_default_graph_and_session` function.

**Warning**: code that creates graphs (for example `tf.Graph().as_default()`) will disable eager execution, which will not work well with GPflow 2. If you get errors like “'Tensor' object has no attribute 'numpy'” then you may not have removed all references to graphs and sessions in your code.


# Defer Build

The `defer_build` context manager has been removed. References to it can simply be removed.


# Return Types from Auto-flowed Methods

GPflow methods that used the `@autoflow` decorator, like for example `predict_f` and `predict_y`, will previously have returned NumPy Arrays. These now return TensorFlow tensors. In many cases these can be used like NumPy arrays (they can be passed directly to many of NumPy’s functions and even be plotted by matplotlib), but to actually turn them into numpy arrays you will need to call `.numpy()` on them.

For example:

<img src="numpy_new.png">


# Parameter Values

GPflow’s `Parameter.value` has changed from a property to a method.

For example:

<img src="param.png">

However, in many cases it is not necessary to call `value` anymore, since `Parameter` just behaves like a TensorFlow tensor.



# Model Class

The `Model` class has been removed. A suitable replacement, for those models that do not wish to inherit from `GPModel`, may be `BayesianModel`.


# Periodic Base Kernel

The base kernel for the `Periodic` kernel must now be specified explicitly. Previously the default was  `SquaredExponential`, so to maintain the same behaviour as before this must be passed-in to the `Periodic` kernel’s initialiser (note that `active_dims` is specified in the base kernel).

For example:

<img src="periodic_1.png">
<img src="periodic_2.png">


# Predict Full Covariance

The `predict_f_full_cov` method has been removed from `GPModel`. Instead, pass `full_cov=True` to the `predict_f` method.

For example:

<img src="full_cov.png">


# Data Types

In some cases TensorFlow will try to figure out an appropriate data type for certain variables. If Python floats have been used, TensorFlow may default these variables to `float32`, which can cause incompatibilities with GPflow, which defaults to using `float64`.

To resolve this you can use `tf.constant` instead of a Python float, and explicitly specify the data type. For example:

<img src="constant.png">


# Float and Int Types

In GPflow 2 `gpflow.settings.float_type` has changed to `gpflow.default_float()` and `gpflow.settings.int_type` has changed to `gpflow.default_int()`.


# Transforms

These have been removed in favour of the tools in `tensorflow_probability.bijectors`. See for example [this Stackoverflow post](https://stackoverflow.com/q/58903446/5986907).

GPflow 2 still provides the `gpflow.utilities.triangular` alias for `tfp.bijectors.FillTriangular`, and `gpflow.utilities.positive` which is configurable to be either softplus or exp, with an optional shift to ensure a lower bound that is larger than zero.


# Name Scoping

The `name_scope` decorator does not exist in GPflow 2 anymore. Use TensorFlow’s [name_scope](https://www.tensorflow.org/api_docs/python/tf/name_scope?version=stable) context manager instead.


# Model Persistence

Model persistence with `gpflow.saver` has been removed in GPflow 2, in favour of TensorFlow 2’s [checkpointing](https://www.tensorflow.org/guide/checkpoint) and [model persistence using the SavedModel format](https://www.tensorflow.org/guide/saved_model).

There is currently a bug in saving GPflow models with TensorFlow's model persistence (SavedModels). See https://github.com/GPflow/GPflow/issues/1127 for more details. However, checkpointing works fine.

```python

```
