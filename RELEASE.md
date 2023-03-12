Release notes for all past releases are available in the ['Releases' section](https://github.com/GPflow/GPflow/releases) of the GPflow GitHub Repo. [HOWTO_RELEASE.md](HOWTO_RELEASE.md) explains just that.

# Release x.y.z (template for future releases)

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

## Breaking Changes

* <DOCUMENT BREAKING CHANGES HERE>
* <THIS SECTION SHOULD CONTAIN API AND BEHAVIORAL BREAKING CHANGES>

## Known Caveats

* <CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
* <ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
* <KNOWN LACK OF SUPPORT ON SOME PLATFORM SHOULD GO HERE>

## Major Features and Improvements

* <INSERT MAJOR FEATURE HERE, USING MARKDOWN SYNTAX>
* <IF RELEASE CONTAINS MULTIPLE FEATURES FROM SAME AREA, GROUP THEM TOGETHER>

## Bug Fixes and Other Changes

* <SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
* <IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
* <NOTES SHOULD BE GROUPED PER AREA>

## Thanks to our Contributors

This release contains contributions from:

<INSERT>, <NAME>, <HERE>, <USING>, <GITHUB>, <HANDLE>


# Release 2.8.0 (next release)

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

## Breaking Changes

* <DOCUMENT BREAKING CHANGES HERE>
* <THIS SECTION SHOULD CONTAIN API AND BEHAVIORAL BREAKING CHANGES>

## Known Caveats

* <CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
* <ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
* <KNOWN LACK OF SUPPORT ON SOME PLATFORM SHOULD GO HERE>

## Major Features and Improvements

* Major rework of documentation landing page and "getting started" section.

## Bug Fixes and Other Changes

* Fixed bug related to `tf.saved_model` and methods wrapped in `@check_shapes`.

## Thanks to our Contributors

This release contains contributions from:

<INSERT>, <NAME>, <HERE>, <USING>, <GITHUB>, <HANDLE>


# Release 2.7.0

The main theme of this release is documentation, with a new suite of tutorials, several upgrades to notebooks and the removal of a rather annoying bug in the documentation site.

Perhaps more notably, `check_shapes` has been removed, and can now be found [here](https://github.com/GPflow/check_shapes).  This change is breaking for those who are still getting `check_shapes` from `gpflow`, although being in experimental this change does not require a new version number.

## Breaking Changes

* `gpflow.experimental.check_shapes` has been removed, in favour of an independent release. Use
  `pip install check_shapes` and `import check_shapes` instead.

## Major Features and Improvements

* Major rework of documentation landing page and "getting started" section.

## Bug Fixes and Other Changes

* Fixed bug related to `tf.saved_model` and methods wrapped in `@check_shapes`.
* Documented monitoring with `Adam` optimizer.
* Fixed bug related to switching versions in documentation site
* Fixed several issues relating to mypy


## Thanks to our Contributors

This release contains contributions from:

sc336, st--, sethaxen, jesnie


# Release 2.6.4

This is yet another bug-fix release.

## Bug Fixes and Other Changes

* Fix to `to_default_float` to avoid losing precision when called with python floats.

## Thanks to our Contributors

This release contains contributions from:

ChrisMorter

# Release 2.6.3

This is yet another bug-fix release.

## Bug Fixes and Other Changes

* Fix to `check_shapes` handling of `tfp..._TensorCoercible`.

## Thanks to our Contributors

This release contains contributions from:

jesnie


# Release 2.6.2

This is a bug-fix release, for compatibility with GPflux.

## Bug Fixes and Other Changes

* Extract shapes of `tfp.python.layers.internal.distribution_tensor_coercible._TensorCoercible`.
* Allow `FallbackSeparateIndependentInducingVariables` to have children with different shapes.
* Allow input and output batches on `GaussianQuadrature` to be different.

## Thanks to our Contributors

This release contains contributions from:

jesnie


# Release 2.6.1

This is a bug-fixes release, due to problems with model saving in `2.6.0`.

## Breaking Changes

* Removed `gpflow.utilities.ops.cast`. Use `tf.cast` instead.

## Bug Fixes and Other Changes

* Fixed bug related to `tf.saved_model` and methods wrapped in `@check_shapes`.
* Some documentation formatting fixes.

## Thanks to our Contributors

This release contains contributions from:

jesnie


# Release 2.6.0

The major theme for this release is heteroskedastic likelihoods. Changes have unfortunately caused
some breaking changes, but makes it much easier to use heteroskedastic likelihoods, either by
plugging together built-in GPflow classes, or when writing your own. See our
[updated notebook](https://gpflow.github.io/GPflow/2.6.0/notebooks/advanced/varying_noise.html), for
examples on how to use this.

## Breaking Changes

* All likelihood methods now take an extra `X` argument. If you have written custom likelihoods or
  you have custom code calling likelihoods directly you will need to add this extra argument.
* On the `CGLB` model the `xnew` parameters has changed name to `Xnew`, to be consistent with the
  other models.
* On the `GPLVM` model the variance returned by `predict_f` with `full_cov=True` has changed shape
  from `[batch..., N, N, P]` to `[batch..., P, N, N]` to be consistent with the other models.
* `gpflow.likelihoods.Gaussian.DEFAULT_VARIANCE_LOWER_BOUND` has been replaced with
  `gpflow.likelihoods.scalar_continuous.DEFAULT_LOWER_BOUND`.
* Change to `InducingVariables` API. `InducingVariables` must now have a `shape` property.
* `gpflow.experimental.check_shapes.get_shape.register` has been replaced with
  `gpflow.experimental.check_shapes.register_get_shape`.
* `check_shapes` will no longer automatically wrap shape checking in
  `tf.compat.v1.flags.tf_decorator.make_decorator`. This is likely to affect you if you use
  `check_shapes` with custom Keras models. If you require the decorator you can manually enable it
  with `check_shapes(..., tf_decorator=True)`.

## Known Caveats

* Shape checking is now, by default, disabled within `tf.function`. Use `set_enable_check_shapes` to
  change this behaviour. See the
  [API documentation](https://gpflow.github.io/GPflow/2.6.0/api/gpflow/experimental/check_shapes/index.html#speed-and-interactions-with-tf-function)
  for more details.

## Major Features and Improvements

* Improved handling of variable noise
  - All likelihood methods now take an `X` argument, allowing you to easily implement
    heteroskedastic likelihoods.
  - The `Gaussian` likelihood can now be parametrized by either a `variance` or a `scale`
  - Some existing likelihoods can now take a function (of X) instead of a parameter, allowing them
    to become heteroskedastic. The parameters are:
    - `Gaussian` `variance`
    - `Gaussian` `scale`
    - `StudentT` `scale`
    - `Gamma` `shape`
    - `Beta` `scale`
  - The `GPR` and `SGPR` can now be configured with a custom Gaussian likelihood, allowing you to
    make them heteroskedastic.
  - See the updated
    [notebook](https://gpflow.github.io/GPflow/2.6.0/notebooks/advanced/varying_noise.html).
  - `gpflow.mean_functions` has been renamed `gpflow.functions`, but with an alias, to avoid
    breaking changes.
* `gpflow.experimental.check_shapes`
  - Can now be in three different states - ENABLED, EAGER_MODE_ONLY, and DISABLE.
    The default is EAGER_MODE_ONLY, which only performs shape checks when the code is not compiled.
    Compiling the shape checking code is a major bottleneck and this provides a significant speed-up
    for performance sensitive parts of the code.
  - Now supports multiple variable-rank dimensions at the same time, e.g. `cov: [n..., n...]`.
  - Now supports single broadcast dimensions to have size 0 or 1, instead of only 1.
  - Now supports variable-rank dimensions to be broadcast, even if they're not leading.
  - Now supports `is None` and `is not None` as checks for conditional shapes.
  - Now uses custom function `register_get_shape` instead of `get_shape.register`, for better
    compatibility with TensorFlow.
  - Now supports checking the shapes of `InducingVariable`s.
  - Now adds documentation to function arguments that has declared shapes, but no other
    documentation.
  - All of GPflow is now consistently shape-checked.
* All built-in kernels now consistently support broadcasting.

## Bug Fixes and Other Changes

* Tested with TensorFlow 2.10.
* Add support for Apple Silicon Macs (`arm64`) via the `tensorflow-macos` dependency. (#1850)
* New implementation of GPR and SGPR posterior objects. This primarily improves numerical stability.
  (#1960)
  - For the GPR this is also a speed improvement when using a GPU.
  - For the SGPR this is a mixed bag, performance-wise.
* Improved checking and error reporting for the models than do not support `full_cov` and
  `full_output_cov`.
* Documentation improvements:
  - Improved MCMC notebook.
  - Deleted notebooks that had no contents.
  - Fixed some broken formatting.

## Thanks to our Contributors

This release contains contributions from:

jesnie, corwinpro, st--, vdutor


# Release 2.5.2

This release fixes a performance regression introduced in `2.5.0`.  `2.5.0` used features of Python
that `tensorfow < 2.9.0` do not know how to compile, which negatively impacted performance.

## Bug Fixes and Other Changes

* Fixed some bugs that prevented TensorFlow compilation and had negative performance impact. (#1882)
* Various improvements to documentation. (#1875, #1866, #1877, #1879)

## Thanks to our Contributors

This release contains contributions from:

jesnie


# Release 2.5.1

Fix problem with release process of 2.5.0.

## Bug Fixes and Other Changes

* Fix bug in release process.

## Thanks to our Contributors

This release contains contributions from:

jesnie


# Release 2.5.0

The focus of this release has mostly been bumping the minimally supported versions of Python and
TensorFlow; and development of `gpflow.experimental.check_shapes`.

## Breaking Changes

* Dropped support for Python 3.6. New minimum version is 3.7. (#1803, #1859)
* Dropped support for TensorFlow 2.2 and 2.3. New minimum version is 2.4. (#1803)
* Removed sub-package `gpflow.utilities.utilities`. It was scheduled for deletion in `2.3.0`.
  Use `gpflow.utilities` instead. (#1804)
* Removed method `Likelihood.predict_density`, which has been deprecated since March 24, 2020.
  (#1804)
* Removed property `ScalarLikelihood.num_gauss_hermite_points`, which has been deprecated since
  September 30, 2020. (#1804)

## Known Caveats

* Further improvements to type hints - this may reveal new problems in your code-base if
  you use a type checker, such as `mypy`. (#1795, #1799, #1802, #1812, #1814, #1816)

## Major Features and Improvements

* Significant work on `gpflow.experimental.check_shapes`.

  - Support anonymous dimensions. (#1796)
  - Add a hook to let the user register shapes for custom types. (#1798)
  - Support `Optional` values. (#1797)
  - Make it configurable. (#1810)
  - Add accesors for setting/getting previously applied checks. (#1815)
  - Much improved error messages. (#1822)
  - Add support for user notes on shapes. (#1836)
  - Support checking all elements of collections. (#1840)
  - Enable stand-alone shape checking, without using a decorator. (#1845)
  - Support for broadcasts. (#1849)
  - Add support for checking the shapes of intermediate computations. (#1853)
  - Support conditional shapes. (#1855)

* Significant speed-up of the GPR posterior objects. (#1809, #1811)

* Significant improvements to documentation. Note the new home page:
  https://gpflow.github.io/GPflow/index.html
  (#1828, #1829, #1830, #1831, #1833, #1841, #1842, #1856, #1857)

## Bug Fixes and Other Changes

* Minor improvement to code clarity (variable scoping) in SVGP model. (#1800)
* Improving mathematical formatting in docs (SGPR derivations). (#1806)
* Allow anisotropic kernels to have negative length-scales. (#1843)

## Thanks to our Contributors

This release contains contributions from:

ltiao, uri.granta, frgsimpson, st--, jesnie


# Release 2.4.0

This release mostly focuses on make posterior objects useful for Bayesian Optimisation.
It also adds a new `experimetal` sub-package, with a tool for annotating tensor shapes.


## Breaking Changes

* Slight change to the API of custom posterior objects.
  `gpflow.posteriors.AbstractPosterior._precompute` no longer must return an `alpha` and an
  `Qinv` - instead it returns any arbitrary tuple of `PrecomputedValue`s.
  Correspondingly `gpflow.posteriors.AbstractPosterior._conditional_with_precompute` should no
  longer try to access `self.alpha` and `self.Qinv`, but instead is passed the tuple of tensors
  returned by `_precompute`, as a parameter. (#1763, #1767)

* Slight change to the API of inducing points.
  You should no longer override `gpflow.inducing_variables.InducingVariables.__len__`. Override
  `gpflow.inducing_variables.InducingVariables.num_inducing` instead. `num_inducing` should return a
  `tf.Tensor` which is consistent with previous behaviour, although the type previously was
  annotated as `int`. `__len__` has been deprecated. (#1766, #1792)

## Known Caveats

* Type hints have been added in several places - this may reveal new problems in your code-base if
  you use a type checker, such as `mypy`.
  (#1766, #1769, #1771, #1773, #1775, #1777, #1780, #1783, #1787, #1789)

## Major Features and Improvements

* Add new posterior class to enable faster predictions from the VGP model. (#1761)
* VGP class bug-fixed to work with variable-sized data. Note you can use
  `gpflow.models.vgp.update_vgp_data` to ensure variational parameters are updated sanely. (#1774).
* All posterior classes bug-fixed to work with variable data sizes, for Bayesian Optimisation.
  (#1767)

* Added `experimental` sub-package for features that are still under developmet.
  * Added `gpflow.experimental.check_shapes` for checking tensor shapes.
    (#1760, #1768, #1782, #1785, #1788)

## Bug Fixes and Other Changes

* Make `dataclasses` dependency conditional at install time. (#1759)
* Simplify calculations of some `predict_f`. (#1755)

## Thanks to our Contributors

This release contains contributions from:

jesnie, tmct, joacorapela


# Release 2.3.1

This is a bug-fix release, primarily for the GPR posterior object.

## Bug Fixes and Other Changes

* GPR posterior
  * Fix the calculation in the GPR posterior object (#1734).
  * Fixes leading dimension issues with `GPRPosterior._conditional_with_precompute()` (#1747).

* Make `gpflow.optimizers.Scipy` able to handle unused / unconnected variables. (#1745).

* Build
  * Fixed broken CircleCi build (#1738).
  * Update CircleCi build to use next-gen Docker images (#1740).
  * Fixed broken triggering of docs generation (#1744).
  * Make all slow tests depend on fast tests (#1743).
  * Make `make dev-install` also install the test requirements (#1737).

* Documentation
  * Fixed broken link in `README.md` (#1736).
  * Fix broken build of `cglb.ipynb` (#1742).
  * Add explanation of how to run notebooks locally (#1729).
  * Fix formatting in notebook on Heteroskedastic Likelihood (#1727).
  * Fix broken link in introduction (#1718).

* Test suite
  * Amends `test_gpr_posterior.py` so it will cover leading dimension uses.



## Thanks to our Contributors

This release contains contributions from:

st--, jesnie, johnamcleod, Andrew878


# Release 2.3.0

## Major Features and Improvements

* Refactor posterior base class to support other model types. (#1695)
* Add new posterior class to enable faster predictions from the GPR/SGPR models. (#1696, #1711)
* Construct Parameters from other Parameters and retain properties. (#1699)
* Add CGLB model (#1706)

## Bug Fixes and Other Changes

* Fix unit test failure when using TensorFlow 2.5.0 (#1684)
* Upgrade black formatter to version 20.8b1 (#1694)
* Remove erroneous DeprecationWarnings (#1693)
* Fix SGPR derivation (#1688)
* Fix tests which fail with TensorFlow 2.6.0 (#1714)

## Thanks to our Contributors

This release contains contributions from:

johnamcleod, st--, Andrew878, tadejkrivec, awav, avullo


# Release 2.2.1

Bugfix for creating the new posterior objects with `PrecomputeCacheType.VARIABLE`.


# Release 2.2.0

The main focus of this release is the new "Posterior" object introduced by
PR #1636, which allows for a significant speed-up of post-training predictions
with the `SVGP` model (partially resolving #1599).

* For end-users, by default nothing changes; see Breaking Changes below if you
  have written your own _implementations_ of `gpflow.conditionals.conditional`.
* After training an `SVGP` model, you can call `model.posterior()` to obtain a
  Posterior object that precomputes all quantities not depending on the test
  inputs (e.g. Choleskty of Kuu), and provides a `posterior.predict_f()` method
  that reuses these cached quantities. `model.predict_f()` computes exactly the
  same quantities as before and does **not** give any speed-up.
* `gpflow.conditionals.conditional()` forwards to the same "fused" code-path as
  before.

## Breaking Changes

* `gpflow.conditionals.conditional.register` is deprecated and should not be
  called outside of the GPflow core code.  If you have written your own
  implementations of `gpflow.conditionals.conditional()`, you have two options
  to use your code with GPflow 2.2:
  1. Temporary work-around: Instead of `gpflow.models.SVGP`, use the
     backwards-compatible `gpflow.models.svgp.SVGP_deprecated`.
  2. Convert your conditional() implementation into a subclass of
     `gpflow.posteriors.AbstractPosterior`, and register
     `get_posterior_class()` instead (see the "Variational Fourier Features"
     notebook for an example).

## Known Caveats

* The Posterior object is currently only available for the `SVGP` model. We
  would like to extend this to the other models such as `GPR`, `SGPR`, or `VGP`, but
  this effort is beyond what we can currently provide. If you would be willing
  to contribute to those efforts, please get in touch!
* The Posterior object does not currently provide the `GPModel` convenience
  functions such as `predict_f_samples`, `predict_y`, `predict_log_density`.
  Again, if you're willing to contribute, get in touch!

## Thanks to our Contributors

This release contains contributions from:

stefanosele, johnamcleod, st--


# Release 2.1.5

## Known Caveats

* GPflow requires TensorFlow >= 2.2.

## Deprecations

* The `gpflow.utilities.utilities` submodule has been deprecated and will be removed in GPflow 2.3. User code should access functions directly through `gpflow.utilities` instead (#1650).

## Major Features and Improvements

* Improves compatibility between monitoring API and Scipy optimizer (#1642).
* Adds `_add_noise_cov` method to GPR model class to make it more easily extensible (#1645).

## Bug Fixes

* Fixes a bug in ModelToTensorBoard (#1619) when `max_size=-1` (#1619)
* Fixes a dynamic shape issue in the quadrature code (#1626).
* Fixes #1651, a bug in `fully_correlated_conditional_repeat` (#1652).
* Fixes #1653, a bug in the "fallback" code path for multioutput Kuf (#1654).
* Fixes a bug in the un-whitened code path for the fully correlated conditional function (#1662).
* Fixes a bug in `independent_interdomain_conditional` (#1663).
* Fixes an issue with the gpflow.config API documentation (#1664).

* Test suite
  * Fixes the test suite for TensorFlow 2.4 / TFP 0.12 (#1625).
  * Fixes mypy call (#1637).
  * Fixes a bug in test_method_equivalence.py (#1649).

## Thanks to our Contributors

This release contains contributions from:

johnamcleod, st--, vatsalaggarwal, sam-willis, vdutor
