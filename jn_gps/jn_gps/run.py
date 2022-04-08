from pathlib import Path
from typing import Any

import jax.numpy as np
import matplotlib.pyplot as plt
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import check_shape as cs
from gpflow.experimental.check_shapes import check_shapes
from jax import grad, jit, random
from matplotlib.axes import Axes

from .clastion.utilities import multi_set, root, to_loss_function
from .covariances import RBF
from .datasets import NoisyFnXYData
from .gpr import GPR
from .means import PolynomialMeanFunction
from .sgpr import SGPR

# TODO:
# * Factory functions?
# * Multiple inputs
# * More kernels
# * VGP
# * SVGP
# * Cool batching of everything
# * Multiple outputs
# * Clastion error messages
# * Clastion inheritance / interfaces
# * Clastion multi_get/set support:
#   - dicts
#   - all elements of collections
# * Allow check_shape in @derived

OUT_DIR = Path(__file__).parent.parent

DTYPE = np.float64


@check_shapes()
def plot_model(model: Any, name: str) -> None:
    if model.x_data.shape[1] != 1:
        return

    n_rows = 3
    n_columns = 1
    plot_width = n_columns * 6.0
    plot_height = n_rows * 4.0
    _fig, (sample_ax, f_ax, y_ax) = plt.subplots(
        nrows=n_rows, ncols=n_columns, figsize=(plot_width, plot_height)
    )

    plot_x = cs(np.linspace(0.0, 10.0, num=100, dtype=DTYPE)[:, None], "[n_plot, 1]")
    model = model(x_predict=plot_x)

    key = random.PRNGKey(20220506)
    n_samples = 5
    key, *keys = random.split(key, num=n_samples + 1)
    for i, k in enumerate(keys):
        plot_y = cs(
            random.multivariate_normal(k, model.f_mean[:, 0], model.f_covariance)[:, None],
            "[n_plot, 1]",
        )
        sample_ax.plot(plot_x, plot_y, label=str(i))
    sample_ax.set_title("Samples")

    @check_shapes(
        "plot_mean: [n_plot, 1]",
        "plot_full_cov: [n_plot, n_plot]",
    )
    def plot_dist(ax: Axes, title: str, plot_mean: AnyNDArray, plot_full_cov: AnyNDArray) -> None:
        plot_cov = cs(np.diag(plot_full_cov), "[n_plot]")
        plot_std = cs(np.sqrt(plot_cov), "[n_plot]")
        plot_lower = cs(plot_mean[:, 0] - plot_std, "[n_plot]")
        plot_upper = cs(plot_mean[:, 0] + plot_std, "[n_plot]")
        (mean_line,) = ax.plot(plot_x, plot_mean)
        color = mean_line.get_color()
        ax.fill_between(plot_x[:, 0], plot_lower, plot_upper, color=color, alpha=0.3)
        ax.scatter(model.x_data, model.y_data, color=color)
        ax.set_title(title)

    plot_dist(f_ax, "f", model.f_mean, model.f_covariance)
    plot_dist(y_ax, "y", model.y_mean, model.y_covariance)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png")
    plt.close()


@check_shapes()
def main() -> None:
    key = random.PRNGKey(20220812)
    dataset_key, x_key = random.split(key)
    datasets = {
        "linear_1d": NoisyFnXYData(
            x_data=random.uniform(x_key, (50, 1), dtype=DTYPE, minval=1.0, maxval=6.0),
            f=lambda x: 0.3 - 0.2 * x,
            noise_scale=0.1,
            key=dataset_key,
        ),
        "quadratic_2d": NoisyFnXYData(
            n_inputs=2,
            f=lambda x: np.sum((x - [0.3, 0.4]) ** 2, axis=-1),
            noise_scale=0.1,
            key=dataset_key,
        ),
    }
    models = [
        (
            GPR(
                mean_func=PolynomialMeanFunction(coeffs=[1.0, 0.0]),
                covariance_func=RBF(variance=1.0),
                noise_var=0.1,
            ),
            (
                root.mean_func.coeffs.u,
                root.noise_var.u,
                root.covariance_func.variance.u,
                root.covariance_func.lengthscale.u,
            ),
        ),
        (
            SGPR(
                mean_func=PolynomialMeanFunction(coeffs=[1.0, 0.0]),
                covariance_func=RBF(variance=1.0),
                noise_var=0.1,
                z=[[1.0], [3.0], [5.0], [7.0], [9.0]],
            ),
            (
                root.mean_func.coeffs.u,
                root.noise_var.u,
                root.covariance_func.variance.u,
                root.covariance_func.lengthscale.u,
                root.z.u,
            ),
        ),
    ]

    for dataset_name, dataset in datasets.items():
        for model, to_train in models:
            key = random.PRNGKey(20220701)
            model_name = model.__class__.__name__

            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(dataset_name, model_name)
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

            prior = model(
                x_data=np.zeros((0, dataset.n_inputs)), y_data=np.zeros((0, dataset.n_inputs))
            )
            plot_model(prior, f"{dataset_name}_{model_name}_prior")

            posterior = prior(x_data=dataset.x_data, y_data=dataset.y_data)
            plot_model(posterior, f"{dataset_name}_{model_name}_posterior")

            loss, params = to_loss_function(
                posterior,
                to_train,
                root.log_likelihood,
            )
            loss_grad = jit(grad(loss))

            for i in range(1_000):
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print(f"Iteration: {i}, loss: {loss(params)}")
                for path, value in params.items():
                    print(f"{path}: {value}")
                param_grads = loss_grad(params)
                params = {k: v + 0.01 * param_grads[k] for k, v in params.items()}

            trained = multi_set(posterior, params)
            plot_model(trained, f"{dataset_name}_{model_name}_trained")


if __name__ == "__main__":
    main()
