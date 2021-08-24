import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpflow import inducing_variables
from pathlib import Path
from gpflow.models import CGLB, SGPR, GPR
from gpflow.kernels import SquaredExponential


np.random.seed(333)
tf.random.set_seed(333)

# %%
# %matplotlib inline

# %% [markdown]
# First, we load the `snelson1d` training dataset.

# %%


def load_snelson_data():
    curdir = Path(__file__).parent
    train_inputs = "snelson_train_inputs.dat"
    train_outputs = "snelson_train_outputs.dat"
    datapath = lambda name: str(Path(curdir, "data", name).resolve())
    xfile = datapath(train_inputs)
    yfile = datapath(train_outputs)
    x = np.loadtxt(xfile).reshape(-1, 1)
    y = np.loadtxt(yfile).reshape(-1, 1)
    return (x, y)


data = load_snelson_data()

# %% [markdown]
# The CGLB model introduces less bias in comparison to SGPR model.
# We can show empirically that CGLB has a lower bias by plotting the objective landscape with respect to different values of the lengthscales hyper parameter.

# %%

x = data[0]
n = x.shape[0]
m = 10

iv_indices = np.random.choice(range(n), size=m, replace=False)
iv = x[iv_indices, :]

noise = 0.1
gpr = GPR(data, kernel=SquaredExponential(), noise_variance=noise)
cglb = CGLB(data, kernel=SquaredExponential(), noise_variance=noise, inducing_variable=iv)
sgpr = SGPR(data, kernel=SquaredExponential(), noise_variance=noise, inducing_variable=iv)


def loss_with_changed_parameter(model, parameter, value: float):
    original = parameter.numpy()
    parameter.assign(value)
    loss = model.training_loss()
    parameter.assign(original)
    return loss.numpy()


ls = np.linspace(0.01, 3, 100)
losses_fn = lambda m: [loss_with_changed_parameter(m, m.kernel.lengthscales, l) for l in ls]
gpr_obj = losses_fn(gpr)
sgpr_obj = losses_fn(sgpr)
cglb_obj = losses_fn(cglb)

plt.plot(ls, gpr_obj, label="GPR")
plt.plot(ls, sgpr_obj, label="SGPR")
plt.plot(ls, cglb_obj, label="CGLB")
plt.legend()
plt.show()
