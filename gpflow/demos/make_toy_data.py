import numpy as np
import pandas as pd

np.random.seed(0)
# rho ~ f1(x1) + f2(x2) * f3(x3)
# --------------- model
xrange1 = [0, 2]
f1 = lambda x: np.exp(x / 2) - 1.5
xrange2 = [0, np.pi]
f2 = lambda x: 1 + np.cos(2.*x + np.pi / 3)
xrange3 =[0, 2]
f3 = lambda x: -np.sin(x)
fs = [f1, f2, f3]
xranges = [xrange1, xrange2, xrange3]
C = len(fs)  # number of functions

# --------------- observations
n = 10000  # number of observations
observations = 'poisson'
assert observations in ['poisson', 'binomial', 'gaussian']
# generate input variables (from uniform distribution in defined range)
X = np.zeros((n, C))
for c in range(3):
    X[:, c] = xranges[c][0] + np.random.rand(n,) * np.diff(xranges[c])
# generate predictor
rho = (f1(X[:, 0]) + f2(X[:, 1]) * f3(X[:, 2]))[..., None] # predictor
# store individual contributions
F = np.vstack([fs[c](X[:, c]) for c in range(C)]).T
# generate observations
phi = lambda x:1. / (1 + np.exp(-x))
if observations == 'binomial':
    pval = phi(rho)  # pass through sigmoid
    Y = (pval > np.random.rand(n, 1)).astype(float)  # generate response from Bernoulli distribution
elif observations == 'poisson':
    Y = np.random.poisson(np.exp(rho)).astype(float)
elif observations == 'gaussian':
    Y = np.random.randn(*rho.shape) * .5 + rho
N, C = X.shape
data = (X, Y)

# save data
XFY = np.hstack([X, F, Y])
print(XFY.shape)
columns = ['X1','X2','X3']+['F1','F2','F3']+['Y']
pd.DataFrame(XFY, columns=columns).to_csv("data.csv", index=False)