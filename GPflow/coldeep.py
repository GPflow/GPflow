import GPflow
import numpy as np
import tensorflow as tf
from GPflow.tf_hacks import eye

# TODO:
# allow non-diagonal q(u)

def cho_solve(L, X):    
    return tf.matrix_triangular_solve(tf.transpose(L), 
                   tf.matrix_triangular_solve(L, X), lower=False)

class Layer(GPflow.model.Model):
    def __init__(self, input_dim, output_dim, kern, Z, beta=10.0, whiten=True):
        GPflow.param.Parameterized.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = Z.shape[0]
        self.whiten = whiten

        assert Z.shape[1] == self.input_dim
        self.kern = kern
        self.Z = GPflow.param.Param(Z)
        self.beta = GPflow.param.Param(beta, GPflow.transforms.positive)

        self.q_of_U_mean = GPflow.param.Param(np.zeros((self.num_inducing, self.output_dim)))
        self.q_of_U_diags = GPflow.param.Param(np.ones((self.num_inducing, self.output_dim)), GPflow.transforms.positive)


    def build_kl(self, Kmm):
        if self.whiten:
            return GPflow.kullback_leiblers.gauss_kl_white_diag(self.q_of_U_mean, self.q_of_U_diags, self.output_dim)
        else:
            return GPflow.kullback_leiblers.gauss_kl_diag(self.q_of_U_mean, self.q_of_U_diags, Kmm, self.output_dim)
        

    def build_predict(self, Xnew, full_cov=False):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern, self.q_of_U_mean, self.output_dim, full_cov=full_cov, q_sqrt=self.q_of_U_diags, whiten=self.whiten)

    @GPflow.model.AutoFlow(tf.placeholder(tf.float64, [None, None]))
    def predict_f(self, X):
        return self.build_predict(X)

    @GPflow.model.AutoFlow(tf.placeholder(tf.float64, [None, None]))
    def predict_f_samples(self, X):
        return self.build_posterior_samples(X)

    def build_posterior_samples(self, Xtest, full_cov=False):
        m,v = self.build_predict(Xtest, full_cov=full_cov)
        if full_cov:
            samples = []
            for i in range(self.output_dim):
                L = tf.cholesky(v[:,:,i])
                samples.append(m[:,i] + tf.matmul(L, tf.random_normal(tf.shape(m)[:1], dtype=tf.float64)))
            return tf.transpose(tf.pack(samples))    
        else:
            return m + tf.random_normal(tf.shape(m), dtype=tf.float64)*tf.sqrt(v)



class HiddenLayer(Layer):

    def feed_forward(self, X_in_mean, X_in_var):
        """
        Compute the variational distribution for the outputs of this layer, as
        well as any marginal likelihood terms that occur
        """

        #kernel computations
        psi0, psi1, psi2 = GPflow.kernel_expectations.build_psi_stats(self.Z, self.kern, X_in_mean, X_in_var)
        Kmm = self.kern.K(self.Z) + np.eye(self.num_inducing)*1e-6
        L = tf.cholesky(Kmm)
        
        #useful computations
        KmmiPsi2 = cho_solve(L, psi2)
        uuT = tf.matmul(self.q_of_U_mean, tf.transpose(self.q_of_U_mean)) + tf.diag(tf.reduce_sum(self.q_of_U_diags, 1))

        #trace term, KL 
        self._log_marginal_contribution = -0.5*self.beta*self.output_dim*(psi0 - tf.reduce_sum(tf.user_ops.get_diag(KmmiPsi2)))
        self._log_marginal_contribution -= self.build_kl(Kmm)

        #distribution to feed forward to downstream layers
        if self.whiten:
            psi1LiT = tf.transpose(tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True))
            forward_mean = tf.matmul(psi1LiT, self.q_of_U_mean) 
            forward_var = tf.matmul(tf.square(psi1LiT), self.q_of_U_diags) + 1./self.beta

            #complete the square term
            psi2_centered = psi2 - tf.matmul(tf.transpose(psi1), psi1)
            LiTuuT = tf.matrix_triangular_solve(tf.transpose(L), uuT, lower=False)
            LiTuuTLi = tf.matrix_triangular_solve(tf.transpose(L), tf.transpose(LiTuuT), lower=False)
            self._log_marginal_contribution += -0.5*self.beta * tf.reduce_sum(psi2_centered * LiTuuTLi)

        else:
            psi1Kmmi = tf.transpose(cho_solve(L, tf.transpose(psi1)))
            forward_mean = tf.matmul(psi1Kmmi, self.q_of_U_mean) 
            forward_var = tf.matmul(tf.square(psi1Kmmi), self.q_of_U_diags) + 1./self.beta

            #complete the square term
            psi2_centered = psi2 - tf.matmul(tf.transpose(psi1), psi1)
            KmmiuuT = cho_solve(L, uuT)
            KmmiuuTKmmi = cho_solve(L, tf.transpose(KmmiuuT))
            self._log_marginal_contribution += -0.5*self.beta * tf.reduce_sum(psi2_centered * KmmiuuTKmmi)


        return forward_mean, forward_var

class InputLayerFixed(Layer):
    def __init__(self, X, input_dim, output_dim, kern, Z, beta=500.):
        Layer.__init__(self, input_dim=input_dim, output_dim=output_dim, kern=kern, Z=Z, beta=beta)
        self.X = X


    def feed_forward(self):
        #kernel computations
        kdiag = self.kern.Kdiag(self.X)
        Knm = self.kern.K(self.X, self.Z)
        Kmm = self.kern.K(self.Z) + eye(self.num_inducing)*1e-6
        L = tf.cholesky(Kmm)
        A = tf.matrix_triangular_solve(L, tf.transpose(Knm))

        #trace term, KL term
        self._log_marginal_contribution = -0.5*self.beta*self.output_dim*(tf.reduce_sum(kdiag) - tf.reduce_sum(tf.square(A)))
        self._log_marginal_contribution -= self.build_kl(Kmm)
        
        #feed outputs to next layer
        return self.build_predict(self.X)


class ObservedLayer(Layer):
    def __init__(self, Y, input_dim, output_dim, kern, Z, beta=0.01):
        Layer.__init__(self, input_dim=input_dim, output_dim=output_dim, kern=kern, Z=Z, beta=beta)
        assert Y.shape[1] == output_dim
        self.Y = Y

    def feed_forward(self, X_in_mean, X_in_var):
        #kernel computations
        psi0, psi1, psi2 = GPflow.kernel_expectations.build_psi_stats(self.Z, self.kern, X_in_mean, X_in_var)
        Kmm = self.kern.K(self.Z) + eye(self.num_inducing)*1e-6
        uuT = tf.matmul(self.q_of_U_mean, tf.transpose(self.q_of_U_mean)) + tf.diag(tf.reduce_sum(self.q_of_U_diags, 1))
        L = tf.cholesky(Kmm)

        KmmiPsi2 = cho_solve(L, psi2)

        if self.whiten:
            A = tf.transpose(tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True))
            #complete the square term
            LiTuuT = tf.matrix_triangular_solve(tf.transpose(L), uuT, lower=False)
            LiTuuTLi = tf.matrix_triangular_solve(tf.transpose(L), tf.transpose(LiTuuT), lower=False)
            cts_tmp = LiTuuTLi

        else:
            A = tf.transpose(cho_solve(L, tf.transpose(psi1)))
            #complete the square term
            KmmiuuT = cho_solve(L, uuT)
            KmmiuuTKmmi = cho_solve(L, tf.transpose(KmmiuuT))
            cts_tmp = KmmiuuTKmmi


        #trace term
        self._log_marginal_contribution = -0.5*self.beta*self.output_dim*(psi0 - tf.reduce_sum(tf.user_ops.get_diag(KmmiPsi2)))
        #CTS term
        self._log_marginal_contribution += -0.5*self.beta * tf.reduce_sum(psi2 * cts_tmp)
        # KL term
        self._log_marginal_contribution -= self.build_kl(Kmm)
                    
        #data fit terms
        proj_mean = tf.matmul(A, self.q_of_U_mean)
        muY = tf.matmul(self.q_of_U_mean, self.Y.T)
        N = tf.cast(tf.shape(X_in_mean)[0], tf.float64)

        self._log_marginal_contribution += -0.5*N*self.output_dim*np.log(2*np.pi/self.beta)
        self._log_marginal_contribution += -0.5 * self.beta * (np.sum(np.square(self.Y))  - 2.*tf.reduce_sum(self.Y*proj_mean) )


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
        assert Nx==Ny
        self.layers = GPflow.param.ParamList([])
        #input layer
        self.layers.append(InputLayerFixed(X=X,
                          input_dim=D_in,
                          output_dim=Qs[0],
                          kern=GPflow.kernels.RBF(D_in, ARD=ARD_X),
                          Z=np.random.randn(Ms[0], D_in),
                          beta=100.))
        #hidden layers
        for h in range(len(Qs)-1):
            self.layers.append(HiddenLayer(input_dim=Qs[h],
                output_dim=Qs[h+1],
                kern=GPflow.kernels.RBF(Qs[h], ARD=ARD_X),
                Z=np.random.randn(Ms[h+1], Qs[h]),
                beta=100.))
        #output layer
        self.layers.append(ObservedLayer(Y=Y,
            input_dim=Qs[-1],
            output_dim=D_out,
            kern=GPflow.kernels.RBF(Qs[-1], ARD=ARD_X),
            Z=np.random.randn(Ms[-1], Qs[-1]),
            beta=500.))

    def build_likelihood(self):
        mu, var = self.layers[0].feed_forward()
        for l in self.layers[1:-1]:
            mu, var = l.feed_forward(mu, var)
        self.layers[-1].feed_forward(mu, var)
        return reduce( tf.add, [l._log_marginal_contribution for l in self.layers])

    @GPflow.model.AutoFlow(tf.placeholder(tf.float64))
    def predict_sampling(self, Xtest):
        for l in self.layers:
            Xtest = l.build_posterior_samples(Xtest, full_cov=False)
        return Xtest

    @GPflow.model.AutoFlow(tf.placeholder(tf.float64))
    def predict_sampling_correlated(self, Xtest):
        for l in self.layers:
            Xtest = l.build_posterior_samples(Xtest, full_cov=True)
        return Xtest


