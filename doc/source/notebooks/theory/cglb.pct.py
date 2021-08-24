import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gpflow.models import CGLB, SGPR, GPR
from gpflow.kernels import SquaredExponential
from gpflow.optimizers import Scipy

# %%
# %matplotlib inline

# %% [markdown]
# First, we load the `snelson1d` training dataset.

# %%
def load_snelson_data():
    if 'workbookDir' in globals():
        curdir = Path(".").resolve()
    else:
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
# We can show empirically that CGLB has a lower bias by plotting the objective landscape with respect to different values of the lengthscale hyperparameters.

# %%

x, y = data
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


# %% [markdown]
# The interface of CGLB model is similar to SGPR, however, it has extra options related to the conjugate gradient which is involved in the computation of `yᵀK⁻¹y`.
# * `cg_tolerance`. This is the expected maximum error in the solution produced by the conjugate gradient,
# i.e. the ϵ threshold of the stopping criteria in the CG loop: `rᵀQ⁻¹r < 0.5 ϵ`.
# The default value for `cg_tolerance` is _1.0_. It is a good working value for model training, which has been confirmed in practice. The `predict_f`, `predict_log_density`, and `predict_y` methods have their own `cg_tolerance` options as for predictions you might want to tighten up the CG solution closer to exact.
# * `max_cg_iters`. The maximum number of CG iterations.
# * `restart_cg_step`. The frequency with wich the CG resets the internal state to the initial position using current solution vector `v`.
# * `v_grad_optimization`. CGLB introduces auxilary parameter `v`, and by default optimal `v` is found with the CG. Howover you can include `v` into the list of trainable model parameters.

cglb = CGLB(
    data,
    kernel=SquaredExponential(),
    noise_variance=noise,
    inducing_variable=iv,
    cg_tolerance=1.0,
    max_cg_iters=50,
    restart_cg_iters=50,
)
opt = Scipy()

# %% [markdown]
# We train the model as usual. Variables do not include the `v` auxilary vector.

# %%
variables = cglb.trainable_variables
_ = opt.minimize(cglb.training_loss_closure(), variables, options=dict(maxiter=100))

# %% [markdown]
# Below we compare prediction results for different CG tolerances. The `cg_tolerance=None` means that no CG is run to tune the `v` vector, and `cg_tolerance=0.01` is much lower value than the one used at the model optimization.

# %%
k = 100
xnew = np.linspace(x.min() - 0.5, x.max() + 0.5, k).reshape(-1, 1)

pred_no_tol = cglb.predict_y(xnew, cg_tolerance=None)
pred_tol = cglb.predict_y(xnew, cg_tolerance=0.01)

fig, axes = plt.subplots(2, 1, figsize=(7, 5))

def plot_prediction(ax, x, y, xnew, loc, scale, color, label):
    for k in (1, 2):
        lb = (loc - k * scale).squeeze()
        ub = (loc + k * scale).squeeze()
        alpha = 1 - 0.4 * k
        ax.fill_between(xnew, lb, ub, color=color, alpha=alpha)

    ax.plot(xnew, loc, color=color, label=label)
    ax.scatter(x, y, color="gray", s=8, alpha=0.7)

x, y = x.squeeze(), y.squeeze()
xnew = xnew.squeeze()

mu_no_tol = pred_no_tol[0].numpy().squeeze()
std_no_tol = np.sqrt(pred_no_tol[1].numpy().squeeze())
plot_prediction(axes[0], x, y, xnew, mu_no_tol, std_no_tol, "tab:blue", "CGLB")


mu_tol = pred_tol[0].numpy().squeeze()
std_tol = np.sqrt(pred_tol[1].numpy().squeeze())
plot_prediction(axes[1], x, y, xnew, mu_tol, std_tol, "tab:green", "CGLB-tol=0.01")

plt.legend()
plt.show()