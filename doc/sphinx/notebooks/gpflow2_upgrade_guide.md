---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# GPflow 2 Upgrade Guide

This is a basic guide for people who have GPflow 1 code that needs to be upgraded to GPflow 2.
Also see the [Intro to GPflow with TensorFlow 2 notebook](intro_to_gpflow2.ipynb).


## Kernel Input Dims

The `input_dim` parameter has been removed from the `Kernel` class’s initialiser. Therefore all calls to create a kernel must be changed to remove the `input_dim` parameter.

For example:

```diff
-gpflow.kernels.SquaredExponential(1, variance=1.0, lengthscales=0.5)
+gpflow.kernels.SquaredExponential(variance=1.0, lengthscales=0.5)
```

**Note**: old code may still run without obvious errors against GPflow, since many kernels take an optional numerical value as their first parameter. You may not get the result you expect though!


## Parameter and tf.Variable

The `Parameter` class in GPflow 1 was a separate class from `tf.Variable`. The `params_as_tensors` decorator or the `params_as_tensors_for` context manager were required to turn them into a tensor that could be consumed by TensorFlow operations.

In GPflow 2, `Parameter` inherits from `gpflow.Module` (a `tf.Module` subclass) that wraps a `tf.Variable`, and can directly be used in place of a tensor, so no such conversion is necessary.

References to `params_as_tensors` and `params_as_tensors_for` can simply be removed.



## Parameter Assignment

In GPflow 2 the semantics of assigning values to parameters has changed. It is now necessary to use the Parameter.assign method rather than assigning values directly to parameters. For example:

```diff
 # Initializations:
-likelihood.scale = 0.1
+likelihood.scale.assign(0.1)
```

In the above example, the old (GPflow 1) code would have assigned the value of `likelihood.scale` to 0.1 (assuming that likelihood is a `Parameterized` object and scale is a `Parameter`), rather than replacing the `scale` attribute with a Python float (which would be the “normal” Python behaviour). This maintains the properties of the parameter. For example, it remains trainable etc.

In GPflow 2, it is necessary to use the `Parameter.assign` method explicitly to maintain the same behaviour, otherwise the parameter attribute will be replaced by an (untrainable) constant value.

To change other properties of the parameter (for example, to change transforms etc) you may need to replace the entire parameter object. See [this notebook](understanding/models.ipynb#Constraints-and-trainable-variables) for further details.


## Parameter trainable status

A parameter's `trainable` attribute cannot be set. Instead, use `gpflow.set_trainable()`. E.g.:
```diff
-likelihood.trainable = False
+gpflow.set_trainable(likelihood, False)
```


## SciPy Optimizer

Usage of GPflow’s Scipy optimizer has changed. It has been renamed from `gpflow.train.ScipyOptimizer` to `gpflow.optimizers.Scipy` and its `minimize` method has changed in the following ways:

 * Instead of a GPflow model, the method now takes a zero-argument function that returns the loss to be minimised (most GPflow models provide a `model.training_loss` method for this use-case; gpflow.models.SVGP does not encapsulate data and provides a `model.training_loss_closure(data)` closure generating method instead), as well as the variables to be optimised (typically `model.trainable_variables`).
 * The options (`disp`, `maxiter`) must now be passed in a dictionary.

For example:
```diff
-optimizer = gpflow.train.ScipyOptimizer()
-optimizer.minimize(model, disp=True, maxiter=100)
+optimizer = gpflow.optimizers.Scipy()
+optimizer.minimize(
+    model.training_loss,
+    variables=model.trainable_variables,
+    options=dict(disp=True, maxiter=100),
+)
```

Any additional keyword arguments that are passed to the `minimize` method are passed directly through to the [SciPy optimizer's minimize method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).


## Model Initialisers

In many cases the initialiser for the model will have changed. Typical changes include:

 * Instead of separate parameters for `X` and `Y`, some models now require a single `data` parameter containing a tuple of the X and Y data.
 * The `kern` parameter has been renamed to `kernel`.

For example, for the `GPR` model:
```diff
-model = GPR(X, Y, kern=kernel)
+model = GPR(data=(X, Y), kernel=kernel)
```

Models that do not take a `likelihood` argument because they hard-code a Gaussian likelihood (GPR, SGPR) now take a `noise_variance` argument that sets the initial value of the likelihood variance.


## SVGP Initialiser

The SVGP model’s initialiser no longer accepts X and Y data. Instead this data must be passed to the various computation methods of the model (`elbo`, `training_loss` etc).

In the [Introduction to GPflow 2 notebook](intro_to_gpflow2.ipynb) there is an example of how to use SVGP with optimisation using mini-batches of data.

In addition, SVGP’s `Z` parameter has been removed. To pass-in inducing points use the `inducing_variable` parameter. Also `SVGP`'s `feature` attribute has been renamed to `inducing_variable`.


## Autoflow

The `@autoflow` decorator has been removed. Since eager execution is the default in TensorFlow 2 this is no longer necessary.

You may wish to consider wrapping functions that were previously wrapped in the `@autoflow` decorator in the `tf.function` decorator instead, to improve performance (but this is not necessary from a functionality point of view).


## Use of tf.function

Wrapping compute-heavy operations such as calculating a model objective or even the optimizer steps (such as `tf.optimizers.Adam().minimize()`) with `tf.function` is crucial for efficient computation.

**Note**: you should ensure that functions wrapped in `tf.function` are only passed **tensors** (not numpy arrays or other data structures, with the exception of a small number of bool or enum-style flags), or the decorator will re-compile the graph each time the function is passed new objects as its arguments. See the [TensorFlow documentation on re-tracing](https://www.tensorflow.org/guide/function#re-tracing) for further details.

You can convert a numpy array to a tensor by using `tf.constant`. For example: `compiled_function(tf.constant(numpy_array))`.


## Model Compilation

Models no longer need to be compiled before use. Remove all calls to the `compile` method.


## Sessions and Graphs

GPflow only supports eager execution, which is the default in TensorFlow 2. It does not support graph mode, which was the default execution mode in TensorFlow 1. Therefore all references to Sessions and Graphs should be removed. You should also remove references to the `gpflow.reset_default_graph_and_session` function.

**Warning**: code that creates graphs (for example `tf.Graph().as_default()`) will disable eager execution, which will not work well with GPflow 2. If you get errors like “'Tensor' object has no attribute 'numpy'” then you may not have removed all references to graphs and sessions in your code.


## Defer Build

The `defer_build` context manager has been removed. References to it can simply be removed.


## Return Types from Auto-flowed Methods

GPflow methods that used the `@autoflow` decorator, like for example `predict_f` and `predict_y`, will previously have returned NumPy Arrays. These now return TensorFlow tensors. In many cases these can be used like NumPy arrays (they can be passed directly to many of NumPy’s functions and even be plotted by matplotlib), but to actually turn them into numpy arrays you will need to call `.numpy()` on them.

For example:
```diff
 def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
     mean, variance = self._model.predict_f(features)
-    return mean, variance
+    return mean.numpy(), variance.numpy()
```


## Parameter Values

In GPflow 1, `Parameter.value` was a property that returned the numpy (`np.ndarray`) representation of the value of the Parameter.

In GPflow 2, `Parameter` behaves similar to TensorFlow's `tf.Variable`: `Parameter.value()` is a method that returns a constant tf.Tensor with the current (constrained) value of the Parameter. To obtain the *numpy* representation, use the `Parameter.numpy()` method:

For example:
```diff
-std_dev = np.sqrt(model.likelihood.variance.value)
+std_dev = np.sqrt(model.likelihood.variance.numpy())
```


## Model Class

The `Model` class has been removed. A suitable replacement, for those models that do not wish to inherit from `GPModel`, may be `BayesianModel`.


## Periodic Base Kernel

The base kernel for the `Periodic` kernel must now be specified explicitly. Previously the default was  `SquaredExponential`, so to maintain the same behaviour as before this must be passed-in to the `Periodic` kernel’s initialiser (note that `active_dims` is specified in the base kernel).

For example:
```diff
-Periodic(1, active_dims=[2])
+Periodic(SquaredExponential(active_dims=[2]))
```


## Predict Full Covariance

The `predict_f_full_cov` method has been removed from `GPModel`. Instead, pass `full_cov=True` to the `predict_f` method.

For example:
```diff
-f_mean, f_cov = model.predict_f_full_cov(X)
+f_mean, f_cov = model.predict_f(X, full_cov=True)
```


## Predictive (log)density

The `predict_density` method of GPModels and Likelihoods has been renamed to `predict_log_density`. (It always returned the predictive *log*-density, so no change in behaviour.)


## Settings / Configuration

In GPflow 2, the `gpflow.settings` module and the `gpflowrc` file have been removed. Instead, there is `gpflow.config`.

`gpflow.settings.float_type` has changed to `gpflow.default_float()` and `gpflow.settings.int_type` has changed to `gpflow.default_int()`.
`gpflow.settings.jitter`/`gpflow.settings.numerics.jitter_level` has changed to `gpflow.default_jitter()`.

These default settings can be changed using environment variables (`GPFLOW_FLOAT`, `GPFLOW_INT`, `GPFLOW_JITTER`, etc.) or function calls (`gpflow.config.set_default_float()` etc.). There is also a `gpflow.config.as_context()` context manager for temporarily changing settings for only part of the code.

See the `gpflow.config` API documentation for more details.


<!-- #region -->
## Data Types

In some cases TensorFlow will try to figure out an appropriate data type for certain variables. If Python floats have been used, TensorFlow may default these variables to `float32`, which can cause incompatibilities with GPflow, which defaults to using `float64`.

To resolve this you can use `tf.constant` instead of a Python float, and explicitly specify the data type, e.g.
```python
tf.constant(0.1, dtype=gpflow.default_float())
```
<!-- #endregion -->


## Transforms

These have been removed in favour of the tools in `tensorflow_probability.bijectors`. See for example [this Stackoverflow post](https://stackoverflow.com/q/58903446/5986907).

GPflow 2 still provides the `gpflow.utilities.triangular` alias for `tfp.bijectors.FillTriangular`.

To constrain parameters to be positive, there is `gpflow.utilities.positive` which is configurable to be either softplus or exp, with an optional shift to ensure a lower bound that is larger than zero.
Note that the default lower bound used to be `1e-6`; by default, the lower bound if not specified explicitly is now `0.0`. Revert the previous behaviour using `gpflow.config.set_default_positive_minimum(1e-6)`.


## Stationary kernel subclasses

Most stationary kernels are actually *isotropic*-stationary kernels, and should now subclass from `gpflow.kernels.IsotropicStationary` instead of `gpflow.kernels.Stationary`. (The `Cosine` kernel is an example of a non-isotropic stationary kernel that depends on the direction, not just the norm, of $\mathbf{x} - \mathbf{x}'$.)


## Likelihoods

We cleaned up the likelihood API. Likelihoods now explicitly define the expected number of outputs (`observation_dim`) and latent functions (`latent_dim`), and shape-checking is in place by default.

Most of the likelihoods simply broadcasted over outputs; these have now been grouped to subclass from `gpflow.likelihoods.ScalarLikelihood`, and implementations have been moved to leading-underscore functions. `ScalarLikelihood` subclasses need to implement at least `_scalar_log_prob` (previously `logp`), `_conditional_mean`, and `_conditional_variance`.

The likelihood `log_prob`, `predict_log_density`, and `variational_expectations` methods now return a single value per data row; for `ScalarLikelihood` subclasses this means these methods effectively sum over the observation dimension (multiple outputs for the same input).


## Priors

Priors used to be defined on the *unconstrained* variable. The default has changed to the prior to be defined on the *constrained* parameter value; this can be changed by passing the `prior_on` argument to `gpflow.Parameter()`. See the [MCMC notebook](advanced/mcmc.ipynb) for more details.


## Name Scoping

The `name_scope` decorator does not exist in GPflow 2 anymore. Use TensorFlow’s [`name_scope`](https://www.tensorflow.org/api_docs/python/tf/name_scope?version=stable) context manager instead.


## Model Persistence

Model persistence with `gpflow.saver` has been removed in GPflow 2, in favour of TensorFlow 2’s [checkpointing](https://www.tensorflow.org/guide/checkpoint) and [model persistence using the SavedModel format](https://www.tensorflow.org/guide/saved_model).

There is currently a bug in saving GPflow models with TensorFlow's model persistence (SavedModels). See https://github.com/GPflow/GPflow/issues/1127 for more details; a workaround is to replace all trainable parameters with constants using `gpflow.utilities.freeze(model)`.

Checkpointing works fine.
