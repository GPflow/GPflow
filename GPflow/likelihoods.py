from . import densities
import tensorflow as tf
import numpy as np
from .param import Parameterized, Param
from . import transforms
hermgauss = np.polynomial.hermite.hermgauss



class Likelihood(Parameterized):
    def __init__(self):
        Parameterized.__init__(self)
        self.num_gauss_hermite_points = 20

    def logp(self, F, Y):
        """
        Return the log density of the data given the function values.
        """
        raise NotImplementedError("implement the logp function\
                                  for this likelihood")

    def conditional_mean(self, F):
        """
        Given a value of the latent function, compute the mean of the data

        If this object represents

            p(y|f)

        then this method computes

            \int y p(y|f) dy
        """
        raise NotImplementedError

    def conditional_variance(self, F):
        """
        Given a value of the latent function, compute the variance of the data

        If this object represents

            p(y|f)

        then this method computes

            \int y^2 p(y|f) dy  - [\int y p(y|f) dy] ^ 2

        """
        raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        """
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           \int\int y p(y|f)q(f) df dy

        and the predictive variance

           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_w /= np.sqrt(np.pi)
        gh_w = gh_w.reshape(-1, 1)
        shape = tf.shape(Fmu)
        Fmu, Fvar = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar)]
        X = gh_x[None, :] * tf.sqrt(2.0 * Fvar) + Fmu

        # here's the quadrature for the mean
        E_y = tf.reshape(tf.matmul(self.conditional_mean(X), gh_w), shape)

        # here's the quadrature for the variance
        integrand = self.conditional_variance(X)\
            + tf.square(self.conditional_mean(X))
        V_y = tf.reshape(tf.matmul(integrand, gh_w), shape) - tf.square(E_y)

        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

           \int p(y=Y|f)q(f) df

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)

        gh_w = gh_w.reshape(-1, 1)/np.sqrt(np.pi)
        shape = tf.shape(Fmu)
        Fmu, Fvar, Y = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, Y)]
        X = gh_x[None, :] * tf.sqrt(2.0 * Fvar) + Fmu

        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])  # broadcast Y to match X

        logp = self.logp(X, Y)
        return tf.reshape(tf.log(tf.matmul(tf.exp(logp), gh_w)), shape)

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.


        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """

        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x.reshape(1, -1)
        gh_w = gh_w.reshape(-1, 1)/np.sqrt(np.pi)
        shape = tf.shape(Fmu)
        Fmu, Fvar, Y = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, Y)]
        X = gh_x * tf.sqrt(2.0 * Fvar) + Fmu
        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])  # broadcast Y to match X

        logp = self.logp(X, Y)
        return tf.reshape(tf.matmul(logp, gh_w), shape)


class Gaussian(Likelihood):
    def __init__(self):
        Likelihood.__init__(self)
        self.variance = Param(1.0, transforms.positive)

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return densities.gaussian(Fmu, Y, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5*np.log(2*np.pi) - 0.5*tf.log(self.variance)\
            - 0.5*(tf.square(Y - Fmu) + Fvar)/self.variance


class Poisson(Likelihood):
    def __init__(self, invlink=tf.exp):
        Likelihood.__init__(self)
        self.invlink = invlink

    def logp(self, F, Y):
        return densities.poisson(self.invlink(F), Y)

    def conditional_variance(self, F):
        return self.invlink(F)

    def conditional_mean(self, F):
        return self.invlink(F)

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return Y*Fmu - tf.exp(Fmu + Fvar/2) - tf.lgamma(Y+1)
        else:
            return Likelihood.variational_expectations(self, Fmu, Fvar, Y)


class Exponential(Likelihood):
    def __init__(self, invlink=tf.exp):
        Likelihood.__init__(self)
        self.invlink = invlink

    def logp(self, F, Y):
        return densities.exponential(self.invlink(F), Y)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        return tf.square(self.invlink(F))

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return -tf.exp(-Fmu + Fvar/2) * Y - Fmu
        else:
            return Likelihood.variational_expectations(self, Fmu, Fvar, Y)


class StudentT(Likelihood):
    def __init__(self, deg_free=3.0):
        Likelihood.__init__(self)
        self.deg_free = deg_free
        self.scale = Param(1.0, transforms.positive)

    def logp(self, F, Y):
        return densities.student_t(Y, F, self.scale, self.deg_free)

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return F*0.0 + (self.deg_free / (self.deg_free - 2.0))


def probit(x):
    return 0.5*(1.0+tf.erf(x/np.sqrt(2.0))) * (1-2e-3) + 1e-3


class Bernoulli(Likelihood):
    def __init__(self, invlink=probit):
        Likelihood.__init__(self)
        self.invlink = invlink

    def logp(self, F, Y):
        return densities.bernoulli(self.invlink(F), Y)

    def predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is probit:
            p = probit(Fmu / tf.sqrt(1 + Fvar))
            return p,  p - tf.square(p)
        else:
            # for other invlink, use quadrature
            return Likelihood.predict_mean_and_var(self, Fmu, Fvar)

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return densities.bernoulli(p, Y)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.invlink(F)
        return p - tf.square(p)


class Gamma(Likelihood):
    """
    Use the transformed GP to give the *scale* (inverse rate) of the Gamma
    """
    def __init__(self, invlink=tf.exp):
        Likelihood.__init__(self)
        self.invlink = invlink
        self.shape = Param(1.0, transforms.positive)

    def logp(self, F, Y):
        return densities.gamma(self.shape, self.invlink(F), Y)

    def conditional_mean(self, F):
        return self.shape * self.invlink(F)

    def conditional_variance(self, F):
        scale = self.invlink(F)
        return self.shape * tf.square(scale)

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return -self.shape * Fmu - tf.lgamma(self.shape)\
                + (self.shape - 1.) * tf.log(Y) - Y * tf.exp(-Fmu + Fvar/2.)
        else:
            return Likelihood.variational_expectations(self, Fmu, Fvar, Y)


class Beta(Likelihood):
    """
    This uses a reparameterisation of the Beta density. We have the mean of the
    Beta distribution given by the transformed process:

        m = sigma(f)

    and a scale parameter. The familiar alpha, beta parameters are given by

        m     = alpha / (alpha + beta)
        scale = alpha + beta

    so:
        alpha = scale * m
        beta  = scale * (1-m)
    """
    def __init__(self, invlink=probit, scale=1.0):
        Likelihood.__init__(self)
        self.scale = Param(scale, transforms.positive)
        self.invlink = invlink

    def logp(self, F, Y):
        mean = self.invlink(F)
        alpha = mean * self.scale
        beta = self.scale - alpha
        return densities.beta(alpha, beta, Y)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        mean = self.invlink(F)
        return (mean - tf.square(mean)) / (self.scale + 1.)


class RobustMax(object):
    """
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is

    y = [y_1 ... y_k]

    with

    y_i = (1-eps)  i == argmax(f)
          eps/(k-1)  otherwise.


    """
    def __init__(self, num_classes, epsilon=1e-3):
        self.epsilon = epsilon
        self.num_classes = num_classes
        self._eps_K1 = self.epsilon/(self.num_classes-1)

    def __call__(self, F):
        i = tf.argmax(F, 1)
        return tf.one_hot(i, self.num_classes, 1.-self.epsilon, self._eps_K1)

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), tf.float64)
        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(tf.sqrt(tf.clip_by_value(2. * var_selected, 1e-10, np.inf)), (-1, 1))

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(tf.sqrt(tf.clip_by_value(var, 1e-10, np.inf)), 2)
        cdfs = 0.5 * (1.0 + tf.erf(dist/np.sqrt(2.0)))

        cdfs = cdfs * (1-2e-4) + 1e-4

        # blank out all the distances on the selected latent function
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), tf.float64)
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.matmul(tf.reduce_prod(cdfs, 1), tf.reshape(gh_w/np.sqrt(np.pi), (-1, 1)))


class MultiClass(Likelihood):
    def __init__(self, num_classes, invlink=None):
        """
        A likelihood that can do multi-way classification. 
        Currently the only valid choice
        of inverse-link function (invlink) is an instance of RobustMax.
        """
        Likelihood.__init__(self)
        self.num_classes = num_classes
        if invlink is None:
            invlink = RobustMax(self.num_classes)
            self.invlink = invlink
        elif not isinstance(invlink,RobustMax):
            raise NotImplementedError        

    def logp(self, F, Y):
        if isinstance(self.invlink, RobustMax):
            hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1), Y)
            yes = tf.ones(tf.shape(Y), dtype=tf.float64) - self.invlink.epsilon
            no = tf.zeros(tf.shape(Y), dtype=tf.float64) + self.invlink._eps_K1
            p = tf.select(hits, yes, no)
            return tf.log(p)
        else:
            raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            return p * np.log(1-self.invlink.epsilon) + (1.-p) * np.log(self.invlink._eps_K1)
        else:
            raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        if isinstance(self.invlink, RobustMax):
            # To compute this, we'll compute the density for each possible output
            possible_outputs = [tf.fill(tf.pack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64)) for i in range(self.num_classes)]
            ps = [self.predict_density(Fmu, Fvar, po) for po in possible_outputs]
            ps = tf.transpose(tf.pack([tf.reshape(p, (-1,)) for p in ps]))
            return ps, ps - tf.square(ps)
        else:
            raise NotImplementedError

    def predict_density(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            return p * (1. - self.invlink.epsilon) + (1. - p) * self.invlink._eps_K1
        else:
            raise NotImplementedError

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)
