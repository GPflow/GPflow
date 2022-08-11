Installation
============

First, a word of warning about TensorFlow versions. GPflow depends on both `TensorFlow
<http://www.tensorflow.org>`_ and `TensorFlow Probability
<https://www.tensorflow.org/probability>`_. These two require very specific versions to be
compatible, and unfortunately this does NOT happen automatically. Even though GPflow will install
these for you, you may not actually get compatible versions, so we recommend you manually and
explicitly install specific versions of these.

+---------------------+---------------------------------+
| TensorFlow version  | TensorFlow Probability version  |
+=====================+=================================+
| 2.4.*               | 0.12.*                          |
+---------------------+---------------------------------+
| 2.5.*               | 0.13.*                          |
+---------------------+---------------------------------+
| 2.6.*               | 0.14.*                          |
+---------------------+---------------------------------+
| 2.7.*               | 0.15.*                          |
+---------------------+---------------------------------+
| 2.8.*               | 0.16.*                          |
+---------------------+---------------------------------+
| 2.9.*               | 0.17.*                          |
+---------------------+---------------------------------+
| 2.10.*              | 0.18.*                          |
+---------------------+---------------------------------+

Second, a word of warning about new Mac computers. On new Mac machines you will need to install
``tensorflow-macos`` instead of the regular ``tensorflow``.

Pre-built installation
----------------------

The recommended way to install GPflow is to get it from `PyPI <https://pypi.org/>`_ using ``pip``.
For example::

  pip install gpflow tensorflow~=2.10.0 tensorflow-probability~=0.18.0

Notice how we explicitly install specific versions of ``tensorflow`` and ``tensorflow-probability``
-- feel free to pick alternative versions if needed, but make sure to pick versions that are
compatible.


Installation from source
------------------------

Alternatively you can install GPflow from source. First check the code out from GitHib::

  git clone git@github.com:GPflow/GPflow.git
  cd GPflow
  pip install . tensorflow~=2.10.0 tensorflow-probability~=0.18.0

Versions
--------

Version history is documented `here <https://github.com/GPflow/GPflow/blob/master/RELEASE.md>`_.
