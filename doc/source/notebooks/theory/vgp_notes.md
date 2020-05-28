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

# Derivation of VGP equations

*James Hensman, 2016*

This notebook contains some implementation notes on the variational Gaussian approximation model in GPflow, `gpflow.models.VGP`. The reference for this work is [Opper and Archambeau 2009, *The variational Gaussian approximation revisited*](http://www.mitpressjournals.org/doi/abs/10.1162/neco.2008.08-07-592); these notes serve to map the conclusions of that paper to their implementation in GPflow. We'll give derivations for the expressions that are implemented in the `VGP` class. 

Two things are not covered by this notebook: prior mean functions, and the extension to multiple independent outputs. Extensions are straightforward in theory but we have taken care in the code to ensure they are handled efficiently. 


## Optimal distribution
The key insight in the work of Opper and Archambeau is that for a Gaussian process with a non-Gaussian likelihood, the optimal Gaussian approximation (in the KL sense) is given by:

\begin{equation}
\hat q(\mathbf f) = \mathcal N\left(\mathbf m, [\mathbf K^{-1} + \textrm{diag}(\boldsymbol \lambda)]^{-1}\right)\,
\end{equation}

We follow their advice in reparameterizing the mean as:

\begin{equation}
\mathbf m = \mathbf K \boldsymbol \alpha
\end{equation}

Additionally, to avoid having to constrain the parameter $\lambda$ to be positive, we take the square. The approximation then becomes:

\begin{equation}
\hat q(\mathbf f) = \mathcal N\left(\mathbf K \boldsymbol \alpha, [\mathbf K^{-1} + \textrm{diag}(\boldsymbol \lambda)^2]^{-1}\right)\,
\end{equation}

The ELBO is:

\begin{equation}
\textrm{ELBO} = \sum_n\mathbb E_{q(f_n)}\left[ \log p(y_n\,|\,f_n)\right] - \textrm{KL}\left[q(\mathbf f)||p(\mathbf f)\right]
\end{equation}

We split the rest of this document into firstly considering the marginals of $q(f)$, and then the KL term. Given these, it is straightforward to compute the ELBO; GPflow uses quadrature to compute one-dimensional expectations where no closed form is available.


## Marginals of $q(\mathbf f)$
Given the above form for $q(\mathbf f)$, what is a quick and stable way to compute the marginals of this Gaussian? The means are trivial, but it would be better if we could obtain the variance without having to perform two matrix inversions. 

Let $\boldsymbol \Lambda = \textrm{diag}(\boldsymbol \lambda)$ and $\boldsymbol \Sigma$ be the covariance in question:  $\boldsymbol \Sigma = [\mathbf K^{-1} + \boldsymbol \Lambda^2]^{-1}$. By the matrix inversion lemma we have:

\begin{align}
\boldsymbol \Sigma &= [\mathbf K^{-1} + \boldsymbol \Lambda^2]^{-1} \\
&= \boldsymbol \Lambda^{-2} - \boldsymbol \Lambda^{-2}[\mathbf K + \boldsymbol \Lambda^{-2}]^{-1}\boldsymbol \Lambda^{-2} \\
&= \boldsymbol \Lambda^{-2} - \boldsymbol \Lambda^{-1}\mathbf A^{-1}\boldsymbol \Lambda^{-1}
\end{align}

where $\mathbf A = \boldsymbol \Lambda\mathbf K \boldsymbol \Lambda + \mathbf I\,.$

Working with this form means that only one matrix decomposition is needed, and taking the Cholesky factor of $\mathbf A$ should be numerically stable because the eigenvalues are bounded by 1.


## KL divergence
The KL divergence term would benefit from a similar reorganisation. The KL is:

\begin{equation}
\textrm{KL} = -0.5 \log |\boldsymbol \Sigma| + 0.5 \log |\mathbf K| +0.5\mathbf m^\top\mathbf K^{-1}\mathbf m + 0.5\textrm{tr}(\mathbf K^{-1} \boldsymbol \Sigma) - 0.5 N
\end{equation}

where $\boldsymbol N$ is the number of data points. Recalling our parameterization $\boldsymbol \alpha$ and combining like terms: 

\begin{equation}
\textrm{KL} = 0.5 (-\log |\mathbf K^{-1}\boldsymbol \Sigma | +\boldsymbol \alpha^\top\mathbf K\boldsymbol \alpha + \textrm{tr}(\mathbf K^{-1} \boldsymbol \Sigma) - N)\,
\end{equation}

with a little manipulation it's possible to show that $\textrm{tr}(\mathbf K^{-1} \boldsymbol \Sigma) = \textrm{tr}(\mathbf A^{-1})$ and $|\mathbf K^{-1} \boldsymbol \Sigma| = |\mathbf A^{-1}|$, giving the final expression:

\begin{equation}
\textrm{KL} = 0.5 (\log |\mathbf A| +\boldsymbol \alpha^\top\mathbf K\boldsymbol \alpha + \textrm{tr}(\mathbf A^{-1}) - N)\,
\end{equation}

This expression is not ideal because we have to compute the diagonal elements of $\mathbf A^{-1}$. We do this with an extra back substitution (into the identity matrix), although it might be possible to do this faster in theory (though not in TensorFlow, to the best of our knowledge).


## Prediction
To make predictions with the Gaussian approximation, we need to integrate:

\begin{equation}
q(f^\star \,|\,\mathbf y) = \int p(f^\star \,|\, \mathbf f)q(\mathbf f)\,\textrm d \mathbf f
\end{equation}

The integral is a Gaussian. We can substitute the equations for these quantities:

\begin{align}
q(f^\star \,|\,\mathbf y) &= \int \mathcal N(f^\star \,|\, \mathbf K_{\star \mathbf f}\mathbf K^{-1}\mathbf f,\, \mathbf K_{\star \star} - \mathbf K_{\star \mathbf f}\mathbf K^{-1}\mathbf K_{\mathbf f \star})\mathcal N (\mathbf f\,|\, \mathbf K \boldsymbol\alpha, \boldsymbol \Sigma)\,\textrm d \mathbf f
q(f^\star \,|\,\mathbf y) \\
&= \mathcal N\left(f^\star \,|\, \mathbf K_{\star \mathbf f}\boldsymbol \alpha,\, \mathbf K_{\star \star} - \mathbf K_{\star \mathbf f}(\mathbf K^{-1} - \mathbf K^{-1}\boldsymbol \Sigma\mathbf K^{-1})\mathbf K_{\mathbf f \star}\right)
\end{align}

where the notation $\mathbf K_{\star \mathbf f}$ means the covariance between the prediction points and the data points, and the matrix $\mathbf K$ is shorthand for $\mathbf K_{\mathbf{ff}}$.

The matrix $\mathbf K^{-1} - \mathbf K^{-1}\boldsymbol \Sigma\mathbf K^{-1}$ can be expanded:

\begin{equation}
\mathbf K^{-1} - \mathbf K^{-1}\boldsymbol \Sigma\mathbf K^{-1} = \mathbf K^{-1} - \mathbf K^{-1}[\mathbf K^{-1} + \boldsymbol\Lambda^2]^{-1}\mathbf K^{-1}\,
\end{equation}

and simplified by recognising the form of the matrix inverse lemma:

\begin{equation}
\mathbf K^{-1} - \mathbf K^{-1}\boldsymbol \Sigma\mathbf K^{-1} = [\mathbf K +  \boldsymbol\Lambda^2]^{-1}\,
\end{equation}

This leads to the final expression for the prediction:

\begin{equation}
q(f^\star \,|\,\mathbf y) = \mathcal N\left(f^\star \,|\, \mathbf K_{\star \mathbf f}\boldsymbol \alpha,\, \mathbf K_{\star \star} - \mathbf K_{\star \mathbf f}[\mathbf K + \boldsymbol \Lambda^2]^{-1}\mathbf K_{\mathbf f \star}\right)
\end{equation}

**NOTE:** The `VGP` class in GPflow has extra functionality to compute the marginal variance of the prediction when the full covariance matrix is not required.
