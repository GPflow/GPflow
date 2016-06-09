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

        then this mehtod computes

            \int y p(y|f) dy
        """
        raise NotImplementedError

    def conditional_variance(self, F):
        """
        Given a value of the latent function, compute the variance of the data

        If this object represents

            p(y|f)

        then this mehtod computes

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
        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])
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
        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])
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
        return tf.ones_like(F) * self.variance

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
    This uses a reparameterization of the Beta density. We have the mean of the
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
