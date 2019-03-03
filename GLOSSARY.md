## Glossary

GPflow does not always follow standard Python naming conventions,
and instead tries to apply the notation in the relevant GP papers.\
The following is the convention we aim to use in the code.

---

<dl>
  <dt>GPR</dt>
  <dd>Gaussian process regression</dd>

  <dt>SVGP</dt>
  <dd>stochastic variational inference for Gaussian process models</dd>

  <dt>Shape constructions [..., A, B]</dt>
  <dd>the way of describing tensor shapes in docstrings and comments. Example: <i>[..., N, D, D]</i>, this is a tensor with an arbitrary number of leading dimensions indicated using the ellipsis sign, and the last two dimensions are equal</dd>

  <dt>X</dt>
  <dd>(and variations like Xnew) refers to input points; always of rank 2, e.g. shape <i>[N, D]</i>, even when <i>D=1</i></dd>

  <dt>Y</dt>
  <dd>(and variations like Ynew) refers to observed output values, potentially with multiple output dimensions; always of rank 2, e.g. shape <i>[N, P]</i>, even when <i>P=1</i></dd>

  <dt>Z</dt>
  <dd>refers to inducing points</dd>

  <dt>M</dt>
  <dd>stands for the number of inducing features (e.g. length of Z)</dd>

  <dt>N</dt>
  <dd>stands for the number of data or minibatch size in docstrings and shape constructions</dd>

  <dt>P</dt>
  <dd>stands for the number of output dimensions in docstrings and shape constructions</dd>

  <dt>D</dt>
  <dd>stands for the number of input dimensions in docstrings and shape constructions</dd>
</dl>
