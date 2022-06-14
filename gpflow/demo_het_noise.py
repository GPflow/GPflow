from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import gpflow
from gpflow import default_float
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import check_shape as cs
from gpflow.experimental.check_shapes import check_shapes
from gpflow.functions import Linear
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel
from gpflow.models.util import InducingPointsLike

# TODO(jesper):
# * SGPR
# * Demo notebook

Data = Tuple[AnyNDArray, AnyNDArray]

OUT_DIR = Path(__file__).parent


DATA_X_MIN = 0.0
DATA_X_MAX = 1.0
DATA_DIFF = DATA_X_MAX - DATA_X_MIN
PLOT_X_MIN = DATA_X_MIN - DATA_DIFF / 2
PLOT_X_MAX = DATA_X_MAX + DATA_DIFF / 2


check_data_shapes = check_shapes(
    "return[0]: [N, 1]",
    "return[1]: [N, 1]",
)
check_model_shapes = check_shapes(
    "data[0]: [N, 1]",
    "data[1]: [N, 1]",
)


@check_data_shapes
def homo_data() -> Data:
    rng = np.random.default_rng(20220614)
    n = 20
    x = DATA_X_MIN + DATA_DIFF * rng.random((n, 1), dtype=default_float())
    e = 0.3 * rng.standard_normal((n, 1), dtype=default_float())
    y = 0.5 + 0.4 * np.sin(10 * x) + e
    return x, y


@check_data_shapes
def hetero_data() -> Data:
    rng = np.random.default_rng(20220614)
    n = 20
    x = DATA_X_MIN + DATA_DIFF * rng.random((n, 1), dtype=default_float())
    e = (0.2 + 0.5 * x) * rng.standard_normal((n, 1), dtype=default_float())
    y = 0.5 + 0.4 * np.sin(10 * x) + e
    return x, y


@check_data_shapes
def hetero_data2() -> Data:
    rng = np.random.default_rng(20220614)
    n = 20
    x = np.linspace(DATA_X_MIN, DATA_X_MAX, n, dtype=default_float())[:, None]
    e = (0.2 + 0.5 * x) * rng.standard_normal((n, 1), dtype=default_float())
    y = e
    return x, y


@check_data_shapes
def gamma_data() -> Data:
    rng = np.random.default_rng(20220614)
    n = 20
    x = DATA_X_MIN + DATA_DIFF * rng.random((n, 1), dtype=default_float())
    e = (0.2 + 0.5 * x) * rng.standard_normal((n, 1), dtype=default_float())
    y = 1.5 + 0.4 * np.sin(10 * x) + e
    assert (y > 0.0).all(), y
    return x, y


@check_data_shapes
def beta_data() -> Data:
    rng = np.random.default_rng(20220614)
    n = 50
    x = DATA_X_MIN + DATA_DIFF * rng.random((n, 1), dtype=default_float())
    e = (0.6 * x) * rng.standard_normal((n, 1), dtype=default_float())
    y = 0.3 + e
    done = False
    while not done:
        too_small = y < 0
        y[too_small] = -y[too_small]
        too_great = y > 1
        y[too_great] = 2 - y[too_great]
        done = (not too_small.any()) and (not too_great.any())
    assert (y < 1.0).all(), y
    assert (y > 0.0).all(), y
    return x, y


def create_kernel() -> Kernel:
    return gpflow.kernels.RBF(lengthscales=0.2)


def create_inducing() -> InducingPointsLike:
    Z = np.linspace(DATA_X_MIN, DATA_X_MAX, 5)[:, None]
    iv = gpflow.inducing_variables.InducingPoints(Z)
    gpflow.set_trainable(iv.Z, False)
    return iv


def create_constant_noise() -> Gaussian:
    return Gaussian(variance=0.3 ** 2)


def create_linear_1() -> Linear:
    return Linear(A=[[0.5]], b=0.2)
    # return Linear(A=[[0.0]], b=1.0)


def create_linear_noise() -> Gaussian:
    return Gaussian(scale=Linear(A=[[0.5]], b=0.2))


@check_model_shapes
def gpr_default(data: Data) -> GPModel:
    return GPR(
        data,
        kernel=create_kernel(),
    )


@check_model_shapes
def gpr_constant(data: Data) -> GPModel:
    return GPR(
        data,
        kernel=create_kernel(),
        likelihood=create_constant_noise(),
    )


@check_model_shapes
def gpr_linear(data: Data) -> GPModel:
    return GPR(
        data,
        kernel=create_kernel(),
        likelihood=create_linear_noise(),
    )


@check_model_shapes
def vgp_constant(data: Data) -> GPModel:
    return VGP(
        data,
        kernel=create_kernel(),
        likelihood=create_constant_noise(),
    )


@check_model_shapes
def vgp_linear(data: Data) -> GPModel:
    return VGP(
        data,
        kernel=create_kernel(),
        likelihood=create_linear_noise(),
    )


@check_model_shapes
def vgp_student_t(data: Data) -> GPModel:
    return VGP(
        data,
        kernel=create_kernel(),
        likelihood=gpflow.likelihoods.StudentT(),
    )


@check_model_shapes
def vgp_linear_student_t(data: Data) -> GPModel:
    return VGP(
        data,
        kernel=create_kernel(),
        likelihood=gpflow.likelihoods.StudentT(scale=create_linear_1()),
    )


@check_model_shapes
def vgp_gamma(data: Data) -> GPModel:
    return VGP(
        data,
        kernel=create_kernel(),
        likelihood=gpflow.likelihoods.Gamma(),
    )


@check_model_shapes
def vgp_linear_gamma(data: Data) -> GPModel:
    return VGP(
        data,
        kernel=create_kernel(),
        likelihood=gpflow.likelihoods.Gamma(shape=create_linear_1()),
    )


@check_model_shapes
def vgp_beta(data: Data) -> GPModel:
    return VGP(
        data,
        kernel=create_kernel(),
        likelihood=gpflow.likelihoods.Beta(),
    )


@check_model_shapes
def vgp_linear_beta(data: Data) -> GPModel:
    return VGP(
        data,
        kernel=create_kernel(),
        likelihood=gpflow.likelihoods.Beta(scale=create_linear_1()),
    )


@check_model_shapes
def sgpr_default(data: Data) -> GPModel:
    return SGPR(
        data,
        kernel=create_kernel(),
        inducing_variable=create_inducing(),
    )


@check_model_shapes
def sgpr_constant(data: Data) -> GPModel:
    return SGPR(
        data,
        kernel=create_kernel(),
        inducing_variable=create_inducing(),
        likelihood=create_constant_noise(),
    )


@check_model_shapes
def sgpr_linear(data: Data) -> GPModel:
    return SGPR(
        data,
        kernel=create_kernel(),
        inducing_variable=create_inducing(),
        likelihood=create_linear_noise(),
    )


@check_model_shapes
def svgp_constant(data: Data) -> GPModel:
    return SVGP(
        kernel=create_kernel(),
        likelihood=create_constant_noise(),
        inducing_variable=create_inducing(),
    )


@check_model_shapes
def svgp_linear(data: Data) -> GPModel:
    return SVGP(
        kernel=create_kernel(),
        likelihood=create_linear_noise(),
        inducing_variable=create_inducing(),
    )


datas = [
    # homo_data,
    hetero_data,
    hetero_data2,
    # gamma_data,
    # beta_data,
]


models = [
    # gpr_default,
    # gpr_constant,
    # gpr_linear,
    # vgp_constant,
    # vgp_linear,
    # vgp_student_t,
    # vgp_linear_student_t,
    # vgp_gamma,
    # vgp_linear_gamma,
    # vgp_beta,
    # vgp_linear_beta,
    # sgpr_default,
    # sgpr_constant,
    sgpr_linear,
    # svgp_constant,
    # svgp_linear,
]


@check_shapes()
def main() -> None:

    do_compile = True
    do_optimise = True

    for create_data in datas:
        data_name = create_data.__name__
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(data_name)
        data = create_data()
        data_x, data_y = data
        for create_model in models:
            model_name = create_model.__name__
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(data_name, "/", model_name)
            model = create_model(data)
            loss_fn = gpflow.models.training_loss_closure(model, data, compile=do_compile)
            if do_optimise:
                gpflow.optimizers.Scipy().minimize(
                    loss_fn,
                    variables=model.trainable_variables,
                    compile=do_compile,
                )
            gpflow.utilities.print_summary(model)
            print("loss: ", float(loss_fn().numpy()))

            n_rows = 3
            n_columns = 1
            plot_width = n_columns * 6.0
            plot_height = n_rows * 4.0
            _fig, (sample_ax, f_ax, y_ax) = plt.subplots(
                nrows=n_rows, ncols=n_columns, figsize=(plot_width, plot_height)
            )

            plot_x = cs(
                np.linspace(PLOT_X_MIN, PLOT_X_MAX, num=100, dtype=default_float())[:, None],
                "[n_plot, 1]",
            )

            f_samples = model.predict_f_samples(plot_x, 5)
            for i, plot_y in enumerate(f_samples):
                sample_ax.plot(plot_x, plot_y, label=str(i))
            sample_ax.set_title("Samples")
            sample_ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
            sample_ax.set_ylim(-2.0, 2.0)

            @check_shapes(
                "plot_mean: [n_plot, 1]",
                "plot_cov: [n_plot, 1]",
            )
            def plot_dist(
                ax: Axes, title: str, plot_mean: AnyNDArray, plot_cov: AnyNDArray
            ) -> None:
                # pylint: disable=cell-var-from-loop
                plot_mean = cs(plot_mean[:, 0], "[n_plot]")
                plot_cov = cs(plot_cov[:, 0], "[n_plot]")
                plot_std = cs(np.sqrt(plot_cov), "[n_plot]")
                plot_lower = cs(plot_mean - plot_std, "[n_plot]")
                plot_upper = cs(plot_mean + plot_std, "[n_plot]")
                (mean_line,) = ax.plot(plot_x, plot_mean)
                color = mean_line.get_color()
                ax.fill_between(plot_x[:, 0], plot_lower, plot_upper, color=color, alpha=0.3)
                ax.scatter(data_x, data_y, color=color)
                ax.set_title(title)
                ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
                ax.set_ylim(-2.0, 2.0)

            plot_dist(f_ax, "f", *model.predict_f(plot_x, full_cov=False))
            plot_dist(y_ax, "y", *model.predict_y(plot_x, full_cov=False))

            plt.tight_layout()
            plt.savefig(OUT_DIR / f"{data_name}_{model_name}.png")
            plt.close()


if __name__ == "__main__":
    main()
