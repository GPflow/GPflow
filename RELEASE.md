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


# Release 2.1.5 (next upcoming release)

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

* Test suite
  * Fixes the test suite for TensorFlow 2.4 / TFP 0.12 (#1625).
  * Fixes mypy call (#1637).
  * Fixes a bug in test_method_equivalence.py (#1649).

## Thanks to our Contributors

This release contains contributions from:

johnamcleod, st--, vatsalaggarwal, sam-willis, vdutor

