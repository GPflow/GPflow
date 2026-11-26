GPflow
======

GPflow is a package for building Gaussian Process models in python, using `TensorFlow
<http://www.tensorflow.org>`_. A Gaussian Process is a kind of supervised learning model.
Some advantages of Gaussian Processes are:

* Uncertainty is an inherent part of Gaussian Processes. A Gaussian Process can tell you when
  it does not know the answer.
* Works well with small datasets. If your data is limited Gaussian Procceses can get the most from
  your data.
* Can scale to large datasets. Although Gaussian Processes, admittedly, can be computationally
  intensive there are ways to scale them to large datasets.

GPflow was originally created by `James Hensman <http://www.lancaster.ac.uk/staff/hensmanj/>`_
and `Alexander G. de G. Matthews <http://mlg.eng.cam.ac.uk/?portfolio=alex-matthews>`_.
Today it is primarily maintained by the company `Secondmind <https://www.secondmind.ai/>`_.


Documentation
-------------

If you're new to GPflow we suggest you continue to:

.. toctree::
   :maxdepth: 2

   getting_started

For more in-depth documentation see:

.. toctree::
   :maxdepth: 1

   user_guide
   API reference <api/gpflow/index>
   benchmarks
   bibliography

.. _implemented_models:


What models are implemented?
----------------------------
GPflow has a slew of kernels that can be combined in a straightforward way. As for inference, the options are currently:

Regression
""""""""""
For GP regression with Gaussian noise, it's possible to marginalize the function values exactly: you'll find this in :class:`gpflow.models.GPR`. You can do maximum likelihood or MCMC for the covariance function parameters.

It's also possible to do Sparse GP regression using the :class:`gpflow.models.SGPR` class. This is based on work by Michalis Titsias :cite:p:`titsias2009variational`.

MCMC
""""
For non-Gaussian likelihoods, GPflow has a model that can jointly sample over the function values and the covariance parameters: :class:`gpflow.models.GPMC`. There's also a sparse equivalent in :class:`gpflow.models.SGPMC`, based on :cite:t:`hensman2015mcmc`.

Variational inference
"""""""""""""""""""""
It's often sufficient to approximate the function values as a Gaussian, for which we follow :cite:t:`Opper:2009` in :class:`gpflow.models.VGP`. In addition, there is a sparse version based on :cite:t:`hensman2014scalable` in :class:`gpflow.models.SVGP`. In the Gaussian likelihood case some of the optimization may be done analytically as discussed in :cite:t:`titsias2009variational` and implemented in :class:`gpflow.models.SGPR` . All of the sparse methods in GPflow are solidified in :cite:t:`matthews2016sparse`.

The following table summarizes the model options in GPflow.

+----------------------+----------------------------+----------------------------+------------------------------+
|                      | Gaussian                   | Non-Gaussian (variational) | Non-Gaussian                 |
|                      | Likelihood                 |                            | (MCMC)                       |
+======================+============================+============================+==============================+
| Full-covariance      | :class:`gpflow.models.GPR` | :class:`gpflow.models.VGP` | :class:`gpflow.models.GPMC`  |
+----------------------+----------------------------+----------------------------+------------------------------+
| Sparse approximation | :class:`gpflow.models.SGPR`| :class:`gpflow.models.SVGP`| :class:`gpflow.models.SGPMC` |
+----------------------+----------------------------+----------------------------+------------------------------+

A unified view of many of the relevant references, along with some extensions, and an early discussion of GPflow itself, is given in the PhD thesis of Matthews :cite:p:`matthews2017scalable`.

Interdomain inference and multioutput GPs
"""""""""""""""""""""""""""""""""""""""""
GPflow has an extensive and flexible framework for specifying interdomain inducing variables for variational approximations.
Interdomain variables can greatly improve the effectiveness of a variational approximation, and are used in e.g.
:doc:`notebooks/advanced/convolutional`. In particular, they are crucial for defining sensible sparse
approximations for multioutput GPs (:doc:`notebooks/advanced/multioutput`).

GPflow has a unifying design for using multioutput GPs and specifying interdomain approximations. A review of the
mathematical background and the resulting software design is described in :cite:t:`GPflow2020multioutput`.

GPLVM
"""""
For visualisation, the GPLVM :cite:p:`lawrence2003gaussian` and Bayesian GPLVM :cite:p:`titsias2010bayesian` models are implemented
in GPflow (:doc:`notebooks/advanced/GPLVM`).

Heteroskedastic models
""""""""""""""""""""""
GPflow supports heteroskedastic models by configuring a likelihood object. See examples in :doc:`notebooks/advanced/varying_noise` and :doc:`notebooks/advanced/heteroskedastic`


Contact
-------

* GPflow is an open source project, and you can find this project on `GitHub
  <https://github.com/GPflow/GPflow>`_.
* If you find any bugs, please `file a ticket <https://github.com/GPflow/GPflow/issues/new/choose>`_.
* If you need help, please use `Stack Overflow <https://stackoverflow.com/tags/gpflow>`_.
* If you otherwise need to contact us, the easiest way to get in touch is
  through our `Slack workspace
  <https://join.slack.com/t/gpflow/shared_invite/enQtOTE5MDA0Nzg5NjA2LTYwZWI3MzhjYjNlZWI1MWExYzZjMGNhOWIwZWMzMGY0YjVkYzAyYjQ4NjgzNDUyZTgyNzcwYjAyY2QzMWRmYjE>`_.


If you feel you have some relevant skills and are interested in contributing then please read our
`notes for contributors <https://github.com/GPflow/GPflow/blob/develop/CONTRIBUTING.md>`_ and contact
us. We maintain a `full list of contributors
<https://github.com/GPflow/GPflow/blob/develop/CONTRIBUTORS.md>`_.


Citing GPflow
-------------

To cite GPflow, please reference :cite:t:`GPflow2017`. Sample BibTeX is given below:

.. code-block:: bib

    @ARTICLE{GPflow2017,
        author = {Matthews, Alexander G. de G. and
                  {van der Wilk}, Mark and
                  Nickson, Tom and
                  Fujii, Keisuke. and
                  {Boukouvalas}, Alexis and
                  {Le{\'o}n-Villagr{\'a}}, Pablo and
                  Ghahramani, Zoubin and
                  Hensman, James},
        title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
        journal = {Journal of Machine Learning Research},
        year = {2017},
        month = {apr},
        volume = {18},
        number = {40},
        pages = {1-6},
        url = {http://jmlr.org/papers/v18/16-537.html}
    }

Since the publication of the GPflow paper, the software has been significantly extended
with the framework for interdomain approximations and multioutput priors. We review the
framework and describe the design in :cite:t:`GPflow2020multioutput`, which can be cited by users:

.. code-block:: bib

    @article{GPflow2020multioutput,
      author = {{van der Wilk}, Mark and
                Dutordoir, Vincent and
                John, ST and
                Artemev, Artem and
                Adam, Vincent and
                Hensman, James},
      title = {A Framework for Interdomain and Multioutput {G}aussian Processes},
      year = {2020},
      journal = {arXiv:2003.01115},
      url = {https://arxiv.org/abs/2003.01115}
    }


Acknowledgements
----------------

James Hensman was supported by an MRC fellowship and Alexander G. de G. Matthews was supported by EPSRC grants EP/I036575/1 and EP/N014162/1.
