<dl>
  <dt>GPR</dt>
  <dd>Gaussian process regression</dd>

  <dt>SVGP</dt>
  <dd>stochastic variational inference for Gaussian process models</dd>

  <dt>[] shape constructions</dt>
  <dd>The way of describing tensor shapes in docstrings and comments. Example: [..., N, D, D], this is a tensor with an arbitrary number of leading dimensions, and the last two dimensions are equal</dd>

  <dt>X</dt>
  <dd>(and variations like Xnew) refers to input points; always of rank 2, e.g. shape [N, D], even when D=1</dd>

  <dt>Y</dt>
  <dd>refers to observed output values, potentially with multiple output dimensions; always of rank 2, e.g. shape [N,
 P], even when P=1</dd>

  <dt>N</dt>
  <dd>stands for data or minibatch size in docstrings and shape constructions</dd>

  <dt>P</dt>
  <dd>stands for the number of output dimensions in docstrings and shape constructions</dd>

  <dt>D</dt>
  <dd>stands for the number of input dimensions in docstrings and shape constructions</dd>

  <dt>Z</dt>
  <dd>inducing points</dd>

  <dt>...</dt>
  <dd>arbitrary leading dimensions</dd>
</dl>
