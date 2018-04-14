# Release 1.1
 - Added inter-domain inducing features. Inducing points are used by default and are now set with `model.feature.Z`.

# Release 1.0
* Clear and aligned with tree-like structure of GPflow models design.
* GPflow trainable parameters are no longer packed into one TensorFlow variable.
* Integration of bare TensorFlow and Keras models with GPflow became very simple.
* GPflow parameter wraps multiple tensors: unconstained variable, constrained tensor and prior tensor.
* Instantaneous parameter's building into the TensorFlow graph. Once you created an instance of parameter, it creates necessary tensors at default graph immediately.
* New implementation for AutoFlow. `autoflow` decorator is a replacement.
* GPflow optimizers match TensorFlow optimizer names. For e.g. `gpflow.train.GradientDescentOptimizer` mimics `tf.train.GradientDescentOptimizer`. They even has the same instantialization signature.
* GPflow has native support for Scipy optimizers - `gpflow.train.ScipyOptimizer`.
* GPflow has advanced HMC implementation - `gpflow.train.HMC`. It works only within TensorFlow memory scope.
* Tensor conversion decorator and context manager designed for cases when user needs to implicitly convert parameters to TensorFlow tensors: `gpflow.params_as_tensors` and `gpflow.params_as_tensors_for`.
* GPflow parameters and parameterized objects provide convenient methods and properties for building, intializing their tensors. Check `initializables`, `initializable_feeds`, `feeds` and other properties and methods.
* Floating shapes of parameters and dataholders without re-building TensorFlow graph.

# Release 0.5
 - bugfix for log_jacobian in transforms

# Release 0.4.1
 - Different variants of `gauss_kl_*` are now deprecated in favour of a unified `gauss_kl` implementation

# Release 0.4.0
 - Rename python package name to `gpflow`.
 - Compile function has external session and graph arguments.
 - Tests use Tensorflow TestCase class for proper session managing.

# Release 0.3.8
 - Change to LowerTriangular transform interface.
 - LowerTriangular transform now used by default in VGP and SVGP
 - LowerTriangular transform now used native TensorFlow
 - No longer use bespoke GPflow user ops.

# Release 0.3.7
 - Improvements to VGP class allow more straightforward optimization

# Release 0.3.6
 - Changed ordering of parameters to be alphabetical, to ensure consistency

# Release 0.3.5
 - Update to work with TensorFlow 0.12.1.

# Release 0.3.4
 - Changes to stop computations all being done on the default graph.
 - Update list of GPflow contributors and other small changes to front page.
 - Better deduction of `input_dim` for `kernels.Combination`
 - Some kernels did not properly respect active dims, now fixed.
 - Make sure log jacobian is computed even for fixed variables

# Release 0.3.3
 - House keeping changes for paper submission.

# Release 0.3.2
 - updated to work with tensorflow 0.11 (release candidate 1 available at time of writing)
 - bugfixes in vgp._compile

# Release 0.3.1
 - Added configuration file, which controls verbosity and level of numerical jitter
 - tf_hacks is deprecated, became tf_wraps (tf_hacks will raise visible deprecation warnings)
 - Documentation now at gpflow.readthedocs.io
 - Many functions are now contained in tensorflow scopes for easier tensorboad visualisation and profiling

# Release 0.3
 - Improvements to the way that parameters for triangular matrices are stored and optimised.
 - Automatically generated Apache license headers.
 - Ability to track log probabilities.

# Release 0.2
 - Significant improvements to the way that data and fixed parameters are handled.

Previously, data and fixed parameters were treated as tensorflow constants.
Now, a new mechanism called `get_feed_dict()` can gather up data and and fixed
parameters and pass them into the graph as placeholders.

 - To enable the above, data are now stored in objects called `DataHolder`. To
   access values of the data, use the same syntax as parameters:
   `print(m.X.value)`
 - Models do not need to be recompiled when the data changes.
 - Two models, VGP and GPMC, do need to be recompiled if the *shape* of the data changes

 - A multi-class likelihood is implemented



# Release 0.1.4
 - Updated to work with tensorflow 0.9
 - Added a Logistic transform to enable contraining a parameter between two bounds
 - Added a Laplace distribution to use as a prior
 - Added a periodic kernel
 - Several improvements to the AutoFlow mechanism
 - added FITC approximation (see comparison notebook)
 - improved readability of code according to pep8
 - significantly improved the speed of the test suite
 - allowed passing of the 'tol' argument to scipy.minimize routine
 - added ability to add and multiply MeanFunction objects
 - Several new contributors (see README.md)

# Release 0.1.3
 - Removed the need for a fork of TensorFlow. Some of our bespoke ops are replaced by equivalent versions.

# Release 0.1.2
 - Included the ability to compute the full covaraince matrix at predict time. See `GPModel.predict_f`
 - Included the ability to sample from the posterior function values. See `GPModel.predict_f_samples`
 - Unified code in conditionals.py: see deprecations in `gp_predict`, etc.
 - Added SGPR method (Sparse GP Regression)

# Release 0.1.1
 -  included the ability to use tensorflow's optimizers as well as the scipy ones

# Release 0.1.0
The initial release of GPflow.
