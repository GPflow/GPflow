from __future__ import print_function
import GPflow
import numpy as np
import unittest

class TestEquivalence(unittest.TestCase):
    """
    With a Gaussian likelihood, and inducing points (where appropriate)
    positioned at the data, many of the GPflow methods are equivalent (perhaps
    subject to some optimization). 

    Here, we make 5 models that should be the same, and make sure some similarites hold. The models are:

    1) GP Regression
    2) Variational GP (with the likelihood set to Gaussian)
    3) Sparse variational GP (likelihood is Gaussian, inducing poitns at the data)
    4) Sparse variational GP (as above, but with the whitening rotation of the inducing variables)
    5) Sparse variational GP Regression (as above, but there the inducing variables are 'collapsed' out, as in Titsias 2009)
    """
    def setUp(self):
        rng = np.random.RandomState(0)
        X = rng.rand(20,1)*10
        Y = np.sin(X) + 0.9 * np.cos(X*1.6) + rng.randn(*X.shape)* 0.8
        self.Xtest = rng.rand(10,1)*10

        m1 = GPflow.gpr.GPR(X, Y, kern=GPflow.kernels.RBF(1))
        m2 = GPflow.vgp.VGP(X, Y, GPflow.kernels.RBF(1), likelihood=GPflow.likelihoods.Gaussian())
        m3 = GPflow.svgp.SVGP(X, Y, GPflow.kernels.RBF(1),
                              likelihood=GPflow.likelihoods.Gaussian(),
                              Z=X.copy(), q_diag=False)
        m3.Z.fixed = True
        m4 = GPflow.svgp.SVGP(X, Y, GPflow.kernels.RBF(1),
                              likelihood=GPflow.likelihoods.Gaussian(),
                              Z=X.copy(), q_diag=False, whiten=True)
        m4.Z.fixed=True
        m5 = GPflow.sgpr.SGPR(X, Y, GPflow.kernels.RBF(1),
                              Z=X.copy())
        m5.Z.fixed = True
        m6 = GPflow.sgpr.GPRFITC(X, Y, GPflow.kernels.RBF(1)+GPflow.kernels.White(1), Z=X.copy())
        m6.Z.fixed = True
        self.models = [m1, m2, m3, m4, m5, m6]
        for m in self.models:
            m.optimize(display=False, max_iters=300)
            print('.') # stop travis timing out

    def test_likelihoods(self):
        likelihoods = np.array([-m._objective(m.get_free_state())[0].squeeze() for m in self.models])
        self.failUnless(np.allclose(likelihoods, likelihoods[0], 1e-2))
        
    def test_kernel_params(self):
        variances, lengthscales = [], []
        for m in self.models:
            if hasattr(m.kern, 'rbf'):
                variances.append(m.kern.rbf.variance._array)
                lengthscales.append(m.kern.rbf.lengthscales._array)
            else:
                variances.append(m.kern.variance._array)
                lengthscales.append(m.kern.lengthscales._array)
        variances, lengthscales = np.array(variances), np.array(lengthscales)

        self.failUnless(np.allclose(variances, variances[0], 1e-3))
        self.failUnless(np.allclose(lengthscales, lengthscales[0], 1e-3))


    def test_predictions(self):
        mu0, var0 = self.models[0].predict_y(self.Xtest)
        for m in self.models[1:]:
            mu, var = m.predict_y(self.Xtest)
            self.failUnless(np.allclose(mu, mu0, 1e-2))
            self.failUnless(np.allclose(var, var0, 1e-2))




if __name__=='__main__':
    unittest.main()
