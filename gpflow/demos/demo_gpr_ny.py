

import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

plt.style.use('ggplot')

xmin, xmax = 0, 5
N = 300
X1 = np.linspace(xmin, xmax, N).reshape(-1,1)

# stack
X = np.hstack([X1, X1, X1])

# unstack
X = np.vstack([X1, X1 + xmax, X1+2*xmax])

print(X)

f1 = lambda x: np.cos(x * 4)
f2 = lambda x: np.exp(- (x-1)**2 * 2.)*3

NY = 3
new_years = [1, xmax + 2, 2*xmax + 3, 4*xmax]

Y = np.zeros((N, NY))
for ny in range(NY):
    F = f1(X1) + f2(X1 - new_years[ny] + xmax*ny)
    Y[:, ny] = F.flatten()

# stacked
#Y += np.random.randn(N, NY)*.5

# unstack
Y = Y.T.reshape(-1, 1)


# print(X.shape, Y.shape)
# plt.plot(X, Y, 'kx', mew=2);
# plt.show()

data = (X, Y)

cols = 'rgb'
C = 2



# ## Building the model

from gpflow.kernels import Kernel

class CNY(Kernel):
    def __init__(self, base: Kernel, new_years: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.new_years = new_years  # flat array

    def transform_cny(self, dates):
        # dates: shape [N, 1]
        deltas = dates - self.new_years[None, :]
        closest_cny_idx = tf.argmin(tf.abs(deltas), axis=1)
        closest_cny = tf.gather(self.new_years, closest_cny_idx)
        return dates - tf.reshape(closest_cny, tf.shape(dates))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)

        t = self.transform_cny(X)
        if X2 is not None:
            t2 = self.transform_cny(X2)
        else:
            t2 = None
        return self.base.K(t, t2)

    def K_diag(self, X, presliced=False):
        return self.base.K_diag(X, presliced=presliced)


k1 = gpflow.kernels.Periodic(base=gpflow.kernels.SquaredExponential(), period=xmax-xmin)
k2 = CNY(base=gpflow.kernels.SquaredExponential(), new_years=np.asarray(new_years).astype(float))
k = k1 + k2
print_summary(k)

m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)

print_summary(m)

#m.likelihood.variance.assign(0.01)
#m.kernel.lengthscale.assign(0.3)

opt = gpflow.optimizers.Scipy()

def objective_closure():
    return - m.log_marginal_likelihood()

print(objective_closure())


opt_logs = opt.minimize(objective_closure,
                        m.trainable_variables,
                        options=dict(maxiter=100))
print_summary(m)


## generate test points for prediction
xx = np.linspace(xmin - 0.1, 6*xmax + 0.1, 2000).reshape(-1, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

## generate 10 samples from posterior
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, 'kx', mew=2)
plt.plot(xx, mean, 'C0', lw=2)
plt.fill_between(xx[:,0],
                 mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                 mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                 color='C0', alpha=0.2)

plt.plot(xx, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
#plt.xlim(xmin - 0.1, xmax + 0.1);
plt.show()
