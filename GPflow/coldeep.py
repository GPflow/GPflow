from functools import reduce
import GPflow
import numpy as np
import tensorflow as tf
from GPflow.tf_hacks import eye

# TODO:
# allow non-diagonal q(u)


def cho_solve(L, X):
    return tf.matrix_triangular_solve(tf.transpose(L),
                                      tf.matrix_triangular_solve(L, X), lower=False)


class Layer(GPflow.param.Parameterized):
    def __init__(self, input_dim, output_dim, kern, Z, beta=10.0):
        GPflow.param.Parameterized.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = Z.shape[0]

        assert Z.shape[1] == self.input_dim
        self.kern = kern
        self.Z = GPflow.param.Param(Z)
        self.beta = GPflow.param.Param(beta, GPflow.transforms.positive)

        shape = (self.num_inducing, self.output_dim)
        self.q_mu = GPflow.param.Param(np.zeros(shape))
        q_sqrt = np.array([np.eye(self.num_inducing)
                           for _ in range(self.output_dim)]).swapaxes(0, 2)  # M x M x D
        self.q_sqrt = GPflow.param.Param(q_sqrt)

    def build_kl(self, Kmm):
        return GPflow.kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, Kmm)

    def build_predict(self, Xnew, full_cov=False):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern,
                                               self.q_mu, full_cov=full_cov,
                                               q_sqrt=self.q_sqrt, whiten=False)

    @GPflow.model.AutoFlow((tf.float64, [None, None]))
    def predict_f(self, X):
        return self.build_predict(X)

    @GPflow.model.AutoFlow((tf.float64, [None, None]))
    def predict_f_samples(self, X):
        return self.build_posterior_samples(X)

    def build_posterior_samples(self, Xtest, full_cov=False):
        m, v = self.build_predict(Xtest, full_cov=full_cov)
        if full_cov:
            samples = []
            for i in range(self.output_dim):
                L = tf.cholesky(v[:, :, i])
                samples.append(m[:, i] + tf.matmul(L, tf.random_normal(tf.shape(m)[:1], dtype=tf.float64)))
            return tf.transpose(tf.pack(samples))
        else:
            return m + tf.random_normal(tf.shape(m), dtype=tf.float64)*tf.sqrt(v)


class HiddenLayer(Layer):

    def feed_forward(self, X_in_mean, X_in_var):
        """
        Compute the variational distribution for the outputs of this layer, as
        well as any marginal likelihood terms that occur
        """

        # kernel computations
        psi0 = tf.reduce_sum(self.kern.eKdiag(X_in_mean, X_in_var))
        psi1 = self.kern.eKxz(self.Z, X_in_mean, X_in_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_in_mean, tf.matrix_diag(X_in_var)), 0)

        Kmm = self.kern.K(self.Z) + np.eye(self.num_inducing)*1e-6
        Lmm = tf.cholesky(Kmm)

        # useful computations
        KmmiPsi2 = cho_solve(Lmm, psi2)
        q_chol = tf.matrix_band_part(tf.transpose(self.q_sqrt, (2, 0, 1)), -1, 0)  # force lower triangle
        q_cov = tf.batch_matmul(q_chol, tf.transpose(q_chol, perm=[0, 2, 1]))  # D x M x M
        uuT = tf.matmul(self.q_mu, tf.transpose(self.q_mu)) + tf.reduce_sum(q_cov, 0)

        # trace term, KL
        trace = psi0 - tf.reduce_sum(tf.diag_part(KmmiPsi2))
        self._log_marginal_contribution = -0.5 * self.beta * self.output_dim * trace
        self._log_marginal_contribution -= self.build_kl(Kmm)

        # distribution to feed forward to downstream layers
        psi1Kmmi = tf.transpose(cho_solve(Lmm, tf.transpose(psi1)))
        forward_mean = tf.matmul(psi1Kmmi, self.q_mu)
        tmp = tf.einsum('ij,kjl->ikl', psi1Kmmi, q_chol)
        forward_var = tf.reduce_sum(tf.square(tmp), 2) + 1./self.beta

        # complete the square term
        psi2_centered = psi2 - tf.matmul(tf.transpose(psi1), psi1)
        KmmiuuT = cho_solve(Lmm, uuT)
        KmmiuuTKmmi = cho_solve(Lmm, tf.transpose(KmmiuuT))
        self._log_marginal_contribution += -0.5*self.beta * tf.reduce_sum(psi2_centered * KmmiuuTKmmi)

        return forward_mean, forward_var


class InputLayerFixed(Layer):
    def __init__(self, X, input_dim, output_dim, kern, Z, beta=500.):
        Layer.__init__(self, input_dim=input_dim, output_dim=output_dim, kern=kern, Z=Z, beta=beta)
        self.X = X

    def feed_forward(self):
        # kernel computations
        kdiag = self.kern.Kdiag(self.X)
        Knm = self.kern.K(self.X, self.Z)
        Kmm = self.kern.K(self.Z) + eye(self.num_inducing)*1e-6
        Lmm = tf.cholesky(Kmm)
        A = tf.matrix_triangular_solve(Lmm, tf.transpose(Knm))

        # trace term, KL term
        trace = tf.reduce_sum(kdiag) - tf.reduce_sum(tf.square(A))
        self._log_marginal_contribution = -0.5*self.beta*self.output_dim * trace
        self._log_marginal_contribution -= self.build_kl(Kmm)

        # feed outputs to next layer
        mu, var = self.build_predict(self.X)
        return mu, var + 1./self.beta


class ObservedLayer(Layer):
    def __init__(self, Y, input_dim, output_dim, kern, Z, beta=0.01):
        Layer.__init__(self, input_dim=input_dim, output_dim=output_dim, kern=kern, Z=Z, beta=beta)
        assert Y.shape[1] == output_dim
        self.Y = Y

    def feed_forward(self, X_in_mean, X_in_var):
        # kernel computations
        psi0 = tf.reduce_sum(self.kern.eKdiag(X_in_mean, X_in_var))
        psi1 = self.kern.eKxz(self.Z, X_in_mean, X_in_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_in_mean, tf.matrix_diag(X_in_var)), 0)
        Kmm = self.kern.K(self.Z) + eye(self.num_inducing)*1e-6
        Lmm = tf.cholesky(Kmm)
        q_chol = tf.matrix_band_part(tf.transpose(self.q_sqrt, (2, 0, 1)), -1, 0)  # force lower triangle
        q_cov = tf.batch_matmul(q_chol, tf.transpose(q_chol, perm=[0, 2, 1]))  # D x M x M
        uuT = tf.matmul(self.q_mu, tf.transpose(self.q_mu)) + tf.reduce_sum(q_cov, 0)

        # trace term
        KmmiPsi2 = cho_solve(Lmm, psi2)
        trace = psi0 - tf.reduce_sum(tf.diag_part(KmmiPsi2))
        self._log_marginal_contribution = -0.5 * self.beta * self.output_dim * trace

        # CTS term
        KmmiuuT = cho_solve(Lmm, uuT)
        self._log_marginal_contribution += -0.5 * self.beta * tf.reduce_sum(tf.diag_part(tf.matmul(KmmiPsi2, KmmiuuT)))

        # KL term
        self._log_marginal_contribution -= self.build_kl(Kmm)

        # data fit terms
        A = tf.transpose(cho_solve(Lmm, tf.transpose(psi1)))
        proj_mean = tf.matmul(A, self.q_mu)
        N = tf.cast(tf.shape(X_in_mean)[0], tf.float64)

        self._log_marginal_contribution += -0.5 * N * self.output_dim * tf.log(2 * np.pi / self.beta)
        self._log_marginal_contribution += -0.5 * self.beta * (np.sum(np.square(self.Y)) -
                                                               2.*tf.reduce_sum(self.Y*proj_mean))


class ColDeep(GPflow.model.Model):
    def __init__(self, X, Y, Qs, Ms, ARD_X=False):
        """
        Build a coldeep structure with len(Qs) hidden layers.

        Note that len(Ms) = len(Qs) + 1, since there's always 1 more GP than there
        are hidden layers.
        """

        GPflow.model.Model.__init__(self)
        assert len(Ms) == (1 + len(Qs))

        Nx, D_in = X.shape
        Ny, D_out = Y.shape
        assert Nx == Ny
        self.layers = GPflow.param.ParamList([])
        # input layer
        Z0 = np.linspace(0, 1, Ms[0]).reshape(-1, 1) * (X.max(0)-X.min(0)) + X.min(0)
        self.layers.append(InputLayerFixed(X=X,
                           input_dim=D_in,
                           output_dim=Qs[0],
                           kern=GPflow.ekernels.RBF(D_in, ARD=ARD_X),
                           Z=Z0,
                           beta=100.))
        # hidden layers
        for h in range(len(Qs)-1):
            Z0 = np.tile(np.linspace(-3, 3, Ms[h+1]).reshape(-1, 1), [1, Qs[h]])
            self.layers.append(HiddenLayer(input_dim=Qs[h],
                               output_dim=Qs[h+1],
                               kern=GPflow.ekernels.RBF(Qs[h], ARD=ARD_X),
                               Z=Z0,
                               beta=100.))
        # output layer
        Z0 = np.tile(np.linspace(-3, 3, Ms[-1]).reshape(-1, 1), [1, Qs[-1]])
        self.layers.append(ObservedLayer(Y=Y,
                           input_dim=Qs[-1],
                           output_dim=D_out,
                           kern=GPflow.ekernels.RBF(Qs[-1], ARD=ARD_X),
                           Z=Z0,
                           beta=500.))

    def build_likelihood(self):
        mu, var = self.layers[0].feed_forward()
        for l in self.layers[1:-1]:
            mu, var = l.feed_forward(mu, var)
        self.layers[-1].feed_forward(mu, var)
        return reduce(tf.add, [l._log_marginal_contribution for l in self.layers])

    @GPflow.model.AutoFlow((tf.float64,))
    def predict_sampling(self, Xtest):
        for l in self.layers:
            Xtest = l.build_posterior_samples(Xtest, full_cov=False)
        return Xtest

    @GPflow.model.AutoFlow((tf.float64,))
    def predict_sampling_correlated(self, Xtest):
        for l in self.layers:
            Xtest = l.build_posterior_samples(Xtest, full_cov=True)
        return Xtest

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    X = np.linspace(-3, 3, 100)[:, None]
    Y = np.where(X < 0, -1, 1) + np.random.randn(100, 1) * 0.01
    m = ColDeep(X, Y, (1,), (15, 15))
    for l in m.layers:
        l.Z.fixed = True
        l.kern.fixed = True
        l.kern.lengthscales = 2.5
        l.beta.fixed = True
    m.optimize(maxiter=5000, disp=1)
    plt.figure()
    for i in range(10):
        s = m.predict_sampling(X)
        plt.plot(X, s, 'bo', mew=0, alpha=0.5)
    plt.plot(X, Y, 'kx', mew=1.5, ms=8)

    for l in m.layers:
        Z = l.Z.value
        extra = (Z.max() - Z.min()) * 0.1
        Xtest = np.linspace(Z.min() - extra, Z.max() + extra, 100)[:, None]
        mu, var = l.predict_f(Xtest)
        plt.figure()
        plt.plot(Xtest, mu, 'r', lw=1.5)
        plt.plot(Xtest, mu - 2*np.sqrt(var), 'r--')
        plt.plot(Xtest, mu + 2*np.sqrt(var), 'r--')
        mu, var = l.predict_f(Z)
        plt.errorbar(Z.flatten(), mu.flatten(), yerr=2*np.sqrt(var).flatten(), capsize=0, elinewidth=1.5, ecolor='r', linewidth=0)
