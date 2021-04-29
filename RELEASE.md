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


# Release 2.2.2 (next upcoming release in progress)

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
