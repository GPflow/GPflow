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

# Derivation of SGPR equations

*James Hensman, March 2016. Corrections by Alex Matthews, December 2016*

This notebook contains a derivation of the form of the equations for the marginal likelihood bound and predictions for the sparse Gaussian process regression model in GPflow, `gpflow.models.SGPR`.

The primary reference for this work is Titsias 2009 [1], though other works (Hensman et al. 2013 [2], Matthews et al. 2016 [3]) are useful for clarifying the prediction density.

<!-- #region -->
## Marginal likelihood bound
The bound on the marginal likelihood (Titsias 2009) is:

\begin{equation}
\log p(\mathbf y) \geq \log \mathcal N(\mathbf y\,|\,\mathbf 0,\, \mathbf Q_{ff} + \sigma^2 \mathbf I) - \tfrac{1}{2} \sigma^{-2}\textrm{tr}(\mathbf K_{ff} - \mathbf Q_{ff}) \triangleq \mathcal L
\end{equation}
where:
\begin{equation}
\mathbf Q_{ff} = \mathbf K_{fu}\mathbf K_{uu}^{-1}\mathbf K_{uf}
\end{equation}


The kernel matrices $\mathbf K_{ff}$, $\mathbf K_{uu}$, $\mathbf K_{fu}$ represent the kernel evaluated at the data points $\mathbf X$, the inducing input points $\mathbf Z$, and between the data and inducing points respectively. We refer to the value of the GP at the data points $\mathbf X$ as $\mathbf f$, at the inducing points $\mathbf Z$ as $\mathbf u$, and at the remainder of the function as $f^\star$.

To obtain an efficient and stable evaluation on the bound $\mathcal L$, we first apply the Woodbury identity to the effective covariance matrix:

\begin{equation}
[\mathbf Q_{ff} + \sigma^2 \mathbf I ]^{-1} = \sigma^{-2} \mathbf I - \sigma^{-4} \mathbf K_{fu}[\mathbf K_{uu} + \mathbf K_{uf}\mathbf K_{fu}\sigma^{-2}]^{-1}\mathbf K_{uf}
\end{equation}

Now, to obtain a better conditioned matrix for inversion, we rotate by $\mathbf L$, where $\mathbf L\mathbf L^\top = \mathbf K_{uu}$:

\begin{equation}
[\mathbf Q_{ff} + \sigma^2 \mathbf I ]^{-1} = \sigma^{-2} \mathbf I - \sigma^{-4} \mathbf K_{fu}\color{red}{\mathbf L^{-\top} \mathbf L^\top}[\mathbf K_{uu} + \mathbf K_{uf}K_{fu}\sigma^{-2}]^{-1}\color{red}{\mathbf L \mathbf L^{-1}}\mathbf K_{uf}
\end{equation}

This matrix is better conditioned because, for many kernels, it has eigenvalues bounded above and below. For more details, see section 3.4.3 of [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf).

\begin{equation}
\phantom{[\mathbf Q_{ff} + \sigma^2 \mathbf I ]^{-1}} = \sigma^{-2} \mathbf I - \sigma^{-4} \mathbf K_{fu}\color{red}{\mathbf L^{-\top}} [\color{red}{\mathbf L^{-1}}\mathbf (K_{uu} + \mathbf K_{uf}K_{fu}\sigma^{-2})\color{red}{\mathbf L^{-\top}}]^{-1}\color{red}{ \mathbf L^{-1}}\mathbf K_{uf}
\end{equation}

\begin{equation}
\phantom{[\mathbf Q_{ff} + \sigma^2 \mathbf I ]^{-1} }= \sigma^{-2} \mathbf I - \sigma^{-4} \mathbf K_{fu}\color{red}{\mathbf L^{-\top}} [\mathbf I + \color{red}{\mathbf L^{-1}}\mathbf (\mathbf K_{uf}K_{fu})\color{red}{\mathbf L^{-\top}}\sigma^{-2}]^{-1}\color{red}{ \mathbf L^{-1}}\mathbf K_{uf}
\end{equation}

For notational convenience, we'll define $\mathbf L^{-1}\mathbf K_{uf}\sigma^{-1} \triangleq \mathbf A$, and  $[\mathbf I + \mathbf A\mathbf A^\top]\triangleq \mathbf B$:

\begin{equation}
\phantom{[\mathbf Q_{ff} + \sigma^2 \mathbf I ]^{-1} }= \sigma^{-2} \mathbf I - \sigma^{-2} \mathbf A^{\top} [\mathbf I + \mathbf A\mathbf A^\top]^{-1}\mathbf A
\end{equation}

\begin{equation}
\phantom{[\mathbf Q_{ff} + \sigma^2 \mathbf I ]^{-1}}= \sigma^{-2} \mathbf I - \sigma^{-2} \mathbf A^{\top} \mathbf B^{-1}\mathbf A
\end{equation}

We also apply the [matrix determinant lemma](https://en.wikipedia.org/wiki/Matrix_determinant_lemma) to the same:

\begin{equation}
|{\mathbf Q_{ff}} + \sigma^2 {\mathbf I}| = |{\mathbf K_{uu}} + 
 \mathbf K_{uf}\mathbf K_{fu}\sigma^{-2}| \, |\mathbf K_{uu}^{-1}| \, |\sigma^{2}\mathbf I|
\end{equation}

Substituting $\mathbf K_{uu} = {\mathbf {L L}^\top}$:
\begin{equation}
|{\mathbf Q_{ff}} + \sigma^2 {\mathbf I}| = |{\mathbf {L L}^\top} + 
 \mathbf K_{uf}\mathbf K_{fu}\sigma^{-2}| \, |\mathbf L^{-\top}|\,| \mathbf L^{-1}| \, |\sigma^{2}\mathbf I|
\end{equation}

\begin{equation}
|{\mathbf Q_{ff}} + \sigma^2 {\mathbf I}| = |\mathbf I + 
 \mathbf L^{-1}\mathbf K_{uf}\mathbf K_{fu} \mathbf L^{-\top}\sigma^{-2}| \, |\sigma^{2}\mathbf I|
\end{equation}

\begin{equation}
|{\mathbf Q_{ff}} + \sigma^2 {\mathbf I}| = |\mathbf B| \, |\sigma^{2}\mathbf I|
\end{equation}

With these two definitions, we're ready to expand the bound:

\begin{equation}
\mathcal L = \log \mathcal N(\mathbf y\,|\,\mathbf 0,\, \mathbf Q_{ff} + \sigma^2 \mathbf I) - \tfrac{1}{2} \sigma^{-2}\textrm{tr}(\mathbf K_{ff} - \mathbf Q_{ff})
\end{equation}

\begin{equation}
= -\tfrac{N}{2}\log{2\pi} -\tfrac{1}{2}\log|\mathbf Q_{ff}+\sigma^2\mathbf I| - \tfrac{1}{2}\mathbf y^\top [ \mathbf Q_{ff} + \sigma^2 \mathbf I]^{-1}\mathbf y - \tfrac{1}{2} \sigma^{-2}\textrm{tr}(\mathbf K_{ff} - \mathbf Q_{ff})
\end{equation}

\begin{equation}
= -\tfrac{N}{2}\log{2\pi} -\tfrac{1}{2}\log(|\mathbf B||\sigma^{2}\mathbf I|)  - \tfrac{1}{2}\sigma^{-2}\mathbf y^\top (\mathbf I - \sigma^{-2} \mathbf A^{\top} \mathbf B^{-1}\mathbf A)\mathbf y - \tfrac{1}{2} \sigma^{-2}\textrm{tr}(\mathbf K_{ff} - \mathbf Q_{ff})
\end{equation}

\begin{equation}
= -\tfrac{N}{2}\log{2\pi} 
-\tfrac{1}{2}\log|\mathbf B|
-\tfrac{N}{2}\log\sigma^{2}
-\tfrac{1}{2}\sigma^{-2}\mathbf y^\top\mathbf y
+\tfrac{1}{2}\sigma^{-2}\mathbf y^\top\mathbf A^{\top} \mathbf B^{-1}\mathbf A\mathbf y
-\tfrac{1}{2}\sigma^{-2}\textrm{tr}(\mathbf K_{ff})
+ \tfrac{1}{2}\textrm{tr}(\mathbf {AA}^\top)
\end{equation}

where $\sigma^{-2}\textrm{tr}(\mathbf Q) = \textrm{tr}(\mathbf {AA}^\top)$.

Finally, we define $\mathbf c \triangleq \mathbf L_{\mathbf B}^{-1}\mathbf A\mathbf y \sigma^{-1}$, with $\mathbf {L_BL_B}^\top = \mathbf B$, so that:

\begin{equation}
\sigma^{-2}\mathbf y^\top\mathbf A^{\top} \mathbf B^{-1}\mathbf A\mathbf y = 
\mathbf c^\top \mathbf c
\end{equation}

The `SGPR` code implements this equation with small changes for multiple concurrent outputs (columns of the data matrix Y), and also a prior mean function.

<!-- #endregion -->

## Prediction
At prediction time, we need to compute the mean and variance of the variational approximation at some new points $\mathbf X^\star$.

Following Hensman et al. (2013), we know that all the information in the posterior approximation is contained in the Gaussian distribution $q(\mathbf u)$, which represents the distribution of function values at the inducing points $\mathbf Z$. Remember that:

\begin{equation}
q(\mathbf u) = \mathcal N(\mathbf u\,|\,  \mathbf m, \mathbf \Lambda)
\end{equation}

with:

\begin{equation}
\mathbf \Lambda = \mathbf K_{uu}^{-1} + \mathbf K_{uu}^{-1}\mathbf K_{uf}\mathbf K_{fu}\mathbf K_{uu}^{-1} \sigma^{-2}
\end{equation}

\begin{equation}
\mathbf m = \mathbf \Lambda^{-1} \mathbf K_{uu}^{-1}\mathbf K_{uf}\mathbf y\sigma^{-2}
\end{equation}

To make a prediction, we need to integrate:

\begin{equation}
p(\mathbf f^\star) = \int p(\mathbf f^\star \,|\, \mathbf u) q (\mathbf u) \textrm d \mathbf u
\end{equation}

with:

\begin{equation}
p(\mathbf f^\star \,|\, \mathbf u) = \mathcal N(\mathbf f^\star\,|\, \mathbf K_{\star u}\mathbf K_{uu}^{-1}\mathbf u, \,\mathbf K_{\star\star} - \mathbf K_{\star u}\mathbf K_{uu}^{-1}\mathbf K_{u\star})
\end{equation}

The integral results in:

\begin{equation}
p(\mathbf f^\star) = \mathcal N(\mathbf f^\star\,|\, \mathbf K_{\star u}\mathbf K_{uu}^{-1}\mathbf m, \,\mathbf K_{\star\star} - \mathbf K_{\star u}\mathbf K_{uu}^{-1}\mathbf K_{u\star} + \mathbf K_{\star u}\mathbf K_{uu}^{-1}\mathbf \Lambda \mathbf K_{uu}^{-1}\mathbf K_{u\star})
\end{equation}

Note from our above definitions we have:

\begin{equation}
\mathbf K_{uu}^{-1}\mathbf \Lambda \mathbf K_{uu}^{-1} = 
\mathbf L^{-\top}\mathbf B^{-1}\mathbf L^{-1}
\end{equation}

and further:

\begin{equation}
\mathbf K_{uu}^{-1}\mathbf m = \mathbf L^{-\top}\mathbf L_{\mathbf B}^{-\top}\mathbf c
\end{equation}

substituting:

\begin{equation}
p(\mathbf f^\star) = \mathcal N(\mathbf f^\star\,|\, \mathbf K_{\star u}\mathbf L^{-\top}\mathbf L_{\mathbf B}^{-\top}\mathbf c, \,\mathbf K_{\star\star} - \mathbf K_{\star u}\mathbf L^{-1}(\mathbf I - \mathbf B^{-1})\mathbf L^{-1}\mathbf K_{u\star})
\end{equation}

The code in `SGPR` implements this equation, with an additional switch depending on whether the full covariance matrix is required.



## References

[1] Titsias, M: Variational Learning of Inducing Variables in Sparse Gaussian Processes, PMLR 5:567-574, 2009

[2] Hensman et al: Gaussian Processes for Big Data, UAI, 2013

[3] Matthews et al: On Sparse Variational Methods and the Kullback-Leibler Divergence between Stochastic Processes, AISTATS, 2016

