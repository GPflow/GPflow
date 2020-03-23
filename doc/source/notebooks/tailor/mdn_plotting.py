import numpy as np

from scipy.stats import norm


def make_grid(xx, yy):
    """
    Returns two n-by-n matrices. The first one contains all the x values 
    and the second all the y values of a cartesian product between `xx` and `yy`.
    """
    n = len(xx)
    xx, yy = np.meshgrid(xx, yy)
    grid = np.array([xx.ravel(), yy.ravel()]).T
    x = grid[:, 0].reshape(n, n)
    y = grid[:, 1].reshape(n, n)
    return x, y


def plot(model, X, Y, axes, cmap, N_plot=100):
    xx = np.linspace(X.min() - 1, X.max() + 1, N_plot)[:, None]
    yy = np.linspace(Y.min() - 1, Y.max() + 1, N_plot)
    pis, mus, sigmas = [v.numpy() for v in model.eval_network(xx)]

    probs = norm.pdf(yy[:, None, None], loc=mus[None, :, :], scale=sigmas[None, :, :])
    probs = np.sum(probs * pis[None, :, :], axis=-1)
    plot_x, plot_y = make_grid(xx, yy)
    axes[0].set_title("Posterior density and trainings data")
    axes[0].contourf(plot_x, plot_y, np.log(probs), 500, cmap=cmap, vmin=-5, vmax=5)
    axes[0].plot(X, Y, "ro", alpha=0.2, ms=3, label="data")
    axes[0].legend(loc=4)
    axes[1].set_title(r"$\mu_m(x)$ and their relative contribution shown by size")
    axes[1].scatter(
        np.repeat(xx.flatten(), repeats=mus.shape[1]), mus.flatten(), s=pis.flatten() * 20
    )
