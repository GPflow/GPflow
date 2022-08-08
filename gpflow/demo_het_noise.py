import datetime
import os
import traceback
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats.qmc import Halton
from tabulate import tabulate
from tensorflow_probability.python.bijectors import Exp
import tensorflow_probability as tfp
# todo use randominit from trieste import GaussianProcessRegression
# todo or train gp first constant noise then retrain
import gpflow
from gpflow import Parameter
from gpflow.datasets import Dataset, get_regression_data, regression_datasets
from gpflow.experimental.check_shapes import check_shapes
from gpflow.functions import Linear, LinearNoise
from gpflow.initialisation import find_best_model_initialization
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel
from gpflow.models.util import InducingPointsLike

TIMESTAMP = datetime.datetime.now().isoformat()
SCRIPT_NAME = Path(__file__).stem
RUN_ID = SCRIPT_NAME + "-" + TIMESTAMP
RESULTS_DIR = Path(os.environ["PROWLER_IO_HOME"]) / "experiment_results"
# RUN_ID = SCRIPT_NAME + str(2) # + "-" + TIMESTAMP
# RESULTS_DIR = Path('C:/src/GPflow') / "experiment_results"
OUTPUT_DIR = RESULTS_DIR / RUN_ID
OUTPUT_DIR.mkdir(parents=True)
LATEST_DIR = RESULTS_DIR / "latest"
if LATEST_DIR.is_symlink():
    LATEST_DIR.unlink()
LATEST_DIR.symlink_to(RUN_ID)
VARIANCE_LOWER_BOUND = 1e-6
N_INITS = 1_000
N_REPS=1
OPT_METHOD =  'l-bfgs-b' # 'bfgs' #
# def homo_data() -> Data:
#     rng = np.random.default_rng(20220614)
#     n = 20
#     x = DATA_X_MIN + DATA_DIFF * rng.random((n, 1), dtype=default_float())
#     e = 0.3 * rng.standard_normal((n, 1), dtype=default_float())
#     y = 0.5 + 0.4 * np.sin(10 * x) + e
#     return x, y
#
#
# def hetero_data() -> Data:
#     rng = np.random.default_rng(20220614)
#     n = 20
#     x = DATA_X_MIN + DATA_DIFF * rng.random((n, 1), dtype=default_float())
#     e = (0.2 + 0.5 * x) * rng.standard_normal((n, 1), dtype=default_float())
#     y = 0.5 + 0.4 * np.sin(10 * x) + e
#     return x, y
#
#
# def hetero_data2() -> Data:
#     rng = np.random.default_rng(20220614)
#     n = 20
#     x: AnyNDArray = np.linspace(DATA_X_MIN, DATA_X_MAX, n, dtype=default_float())[:, None]
#     e = (0.2 + 0.5 * x) * rng.standard_normal((n, 1), dtype=default_float())
#     y = e
#     return x, y
#
#
# def gamma_data() -> Data:
#     rng = np.random.default_rng(20220614)
#     n = 20
#     x = DATA_X_MIN + DATA_DIFF * rng.random((n, 1), dtype=default_float())
#     e = (0.2 + 0.5 * x) * rng.standard_normal((n, 1), dtype=default_float())
#     y = 1.5 + 0.4 * np.sin(10 * x) + e
#     assert (y > 0.0).all(), y
#     return x, y
#
#
# def beta_data() -> Data:
#     rng = np.random.default_rng(20220614)
#     n = 50
#     x = DATA_X_MIN + DATA_DIFF * rng.random((n, 1), dtype=default_float())
#     e = (0.6 * x) * rng.standard_normal((n, 1), dtype=default_float())
#     y = 0.3 + e
#     done = False
#     while not done:
#         too_small = y < 0
#         y[too_small] = -y[too_small]
#         too_great = y > 1
#         y[too_great] = 2 - y[too_great]
#         done = (not too_small.any()) and (not too_great.any())
#     assert (y < 1.0).all(), y
#     assert (y > 0.0).all(), y
#     return x, y


C = TypeVar("C", bound=Callable[..., Any])


def create_model(sparse: bool) -> Callable[[C], C]:
    def wrapper(f: C) -> C:
        f.__sparse__ = sparse  # type: ignore[attr-defined]
        return f

    return wrapper


def create_kernel(data: Dataset, rng: np.random.Generator) -> Kernel:
    D = data.D  # type: ignore[attr-defined]
    kernel = gpflow.kernels.RBF(
        variance=rng.gamma(5.0, 0.2, []),
        lengthscales=rng.gamma(5.0, 0.2, [D]),
    )
    length_prior =  tfp.distributions.LogNormal(loc=np.float64(0.), scale=np.float64(0.5))
    kernel.variance.prior = tfp.distributions.LogNormal(loc=np.float64(-1.), scale=np.float64(1.0))
    kernel.lengthscales = Parameter(kernel.lengthscales.numpy(), transform=Exp(), prior=length_prior)

    return kernel


def create_inducing(data: Dataset, rng: np.random.Generator) -> InducingPointsLike:
    N, D = data.N, data.D  # type: ignore
    n = min(N // 2, 200)
    Z = Halton(D, scramble=False).random(n)
    lower = np.min(data.X_train, axis=0)
    upper = np.max(data.X_train, axis=0)
    Z = Z * (upper - lower) + lower
    return gpflow.inducing_variables.InducingPoints(Z)


def create_constant_noise(data: Dataset, rng: np.random.Generator) -> Gaussian:
    return Gaussian(variance=rng.gamma(5.0, 0.2, []))


def create_linear(data: Dataset, rng: np.random.Generator) -> Linear:
    D = data.D  # type: ignore[attr-defined]

    # First determine the noise amplitude at the origin
    origin_noise = rng.lognormal(-1.0, 1.0, [])

    # Now decide what fraction the noise changes along each axis, maximum permitted value of 1
    fractional_noise_gradient = rng.normal(0.0, 0.01, [D, 1])  # np.zeros((D, 1)) # todo hack

    # Ensure noise_gradient is consistent with the chosen origin noise amplitude
    b_transform = Exp()
    if True: # Use parameterisation where 0th and 1st order terms are not independent
        linear_noise_fn = LinearNoise(
            A=fractional_noise_gradient,
            b=origin_noise,
        )
        b_prior = tfp.distributions.LogNormal(loc=np.float64(-1.), scale=np.float64(1.0))
        linear_noise_fn.b =  Parameter(linear_noise_fn.b.numpy(), transform=b_transform, prior=b_prior)
        # linear_noise_fn.b =  Parameter(1e-2, transform=b_transform)
        linear_noise_fn.A.prior =  tfp.distributions.Cauchy(loc=np.float64(0.), scale=np.float64(0.01)) # 0.05 was old scale # tfp.distributions.Normal(loc=np.float64(0.), scale=np.float64(0.03))  #
    else:
        noise_gradient = fractional_noise_gradient * origin_noise
        linear_noise_fn = Linear(
            A=noise_gradient,
            b=origin_noise,
        )

        linear_noise_fn.b = Parameter(linear_noise_fn.b.numpy(), transform=b_transform)
        linear_noise_fn.b.prior = tfp.distributions.LogNormal(loc=np.float64(-1.), scale=np.float64(1.0))
        linear_noise_fn.A.prior = tfp.distributions.Cauchy(loc=np.float64(0.), scale=np.float64(0.01))  #  tfp.distributions.Normal(loc=np.float64(0.), scale=np.float64(0.1))

    return linear_noise_fn


def create_linear_noise(data: Dataset, rng: np.random.Generator) -> Gaussian:
    return Gaussian(scale=create_linear(data, rng), variance_lower_bound=VARIANCE_LOWER_BOUND)


@create_model(sparse=False)
def gpr_default(data: Dataset, rng: np.random.Generator) -> GPModel:
    return GPR(
        data.train,
        kernel=create_kernel(data, rng),
        noise_variance=rng.gamma(5.0, 0.2, []),
    )


@create_model(sparse=False)
def gpr_constant(data: Dataset, rng: np.random.Generator) -> GPModel:
    return GPR(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=create_constant_noise(data, rng),
    )


@create_model(sparse=False)
def gpr_linear(data: Dataset, rng: np.random.Generator) -> GPModel:
    return GPR(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=create_linear_noise(data, rng),
    )


@create_model(sparse=False)
def vgp_constant(data: Dataset, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=create_constant_noise(data, rng),
    )


@create_model(sparse=False)
def vgp_linear(data: Dataset, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=create_linear_noise(data, rng),
    )


@create_model(sparse=False)
def vgp_student_t(data: Dataset, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=gpflow.likelihoods.StudentT(scale=rng.gamma(5.0, 0.2, [])),
    )


@create_model(sparse=False)
def vgp_linear_student_t(data: Dataset, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=gpflow.likelihoods.StudentT(scale=create_linear(data, rng)),
    )


@create_model(sparse=False)
def vgp_gamma(data: Dataset, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=gpflow.likelihoods.Gamma(shape=rng.gamma(5.0, 0.2, [])),
    )


@create_model(sparse=False)
def vgp_linear_gamma(data: Dataset, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=gpflow.likelihoods.Gamma(shape=create_linear(data, rng)),
    )


@create_model(sparse=False)
def vgp_beta(data: Dataset, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=gpflow.likelihoods.Beta(scale=rng.gamma(5.0, 0.2, [])),
    )


@create_model(sparse=False)
def vgp_linear_beta(data: Dataset, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.train,
        kernel=create_kernel(data, rng),
        likelihood=gpflow.likelihoods.Beta(scale=create_linear(data, rng)),
    )


@create_model(sparse=True)
def sgpr_default(data: Dataset, rng: np.random.Generator) -> GPModel:
    return SGPR(
        data.train,
        kernel=create_kernel(data, rng),
        inducing_variable=create_inducing(data, rng),
        noise_variance=rng.gamma(5.0, 0.2, []),
    )


@create_model(sparse=True)
def sgpr_constant(data: Dataset, rng: np.random.Generator) -> GPModel:
    return SGPR(
        data.train,
        kernel=create_kernel(data, rng),
        inducing_variable=create_inducing(data, rng),
        likelihood=create_constant_noise(data, rng),
    )


@create_model(sparse=True)
def sgpr_linear(data: Dataset, rng: np.random.Generator) -> GPModel:
    return SGPR(
        data.train,
        kernel=create_kernel(data, rng),
        inducing_variable=create_inducing(data, rng),
        likelihood=create_linear_noise(data, rng),
    )


@create_model(sparse=True)
def svgp_constant(data: Dataset, rng: np.random.Generator) -> GPModel:
    return SVGP(
        kernel=create_kernel(data, rng),
        likelihood=create_constant_noise(data, rng),
        inducing_variable=create_inducing(data, rng),
    )


@create_model(sparse=True)
def svgp_linear(data: Dataset, rng: np.random.Generator) -> GPModel:
    return SVGP(
        kernel=create_kernel(data, rng),
        likelihood=create_linear_noise(data, rng),
        inducing_variable=create_inducing(data, rng),
    )


models = [
    gpr_default,
    # gpr_constant,
    gpr_linear,
    # vgp_constant,
    # vgp_linear,
    # vgp_student_t,
    # vgp_linear_student_t,
    # vgp_gamma,
    # vgp_linear_gamma,
    # vgp_beta,
    # vgp_linear_beta,
    sgpr_default,
    # sgpr_constant,
    sgpr_linear,
    # svgp_constant,
    # svgp_linear,
]

def plot_model(data: Dataset, model: GPModel) -> None:
    pass

    # if data.D != 1:
    #     return
    #
    # n_rows = 3
    # n_columns = 1
    # plot_width = n_columns * 6.0
    # plot_height = n_rows * 4.0
    # _fig, (sample_ax, f_ax, y_ax) = plt.subplots(
    #     nrows=n_rows, ncols=n_columns, figsize=(plot_width, plot_height)
    # )
    #
    # plot_x: AnyNDArray = cs(
    #     np.linspace(PLOT_X_MIN, PLOT_X_MAX, num=100, dtype=default_float())[:, None],
    #     "[n_plot, 1]",
    # )
    #
    # f_samples = model.predict_f_samples(plot_x, 5)
    # for i, plot_y in enumerate(f_samples):
    #     sample_ax.plot(plot_x, plot_y, label=str(i))
    # sample_ax.set_title("Samples")
    # sample_ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    # sample_ax.set_ylim(-2.0, 2.0)
    #
    # @check_shapes(
    #     "plot_mean: [n_plot, 1]",
    #     "plot_cov: [n_plot, 1]",
    # )
    # def plot_dist(
    #     ax: Axes, title: str, plot_mean: AnyNDArray, plot_cov: AnyNDArray
    # ) -> None:
    #     # pylint: disable=cell-var-from-loop
    #     plot_mean = cs(plot_mean[:, 0], "[n_plot]")
    #     plot_cov = cs(plot_cov[:, 0], "[n_plot]")
    #     plot_std = cs(np.sqrt(plot_cov), "[n_plot]")
    #     plot_lower = cs(plot_mean - plot_std, "[n_plot]")
    #     plot_upper = cs(plot_mean + plot_std, "[n_plot]")
    #     (mean_line,) = ax.plot(plot_x, plot_mean)
    #     color = mean_line.get_color()
    #     ax.fill_between(plot_x[:, 0], plot_lower, plot_upper, color=color, alpha=0.3)
    #     ax.scatter(data_x, data_y, color=color)
    #     ax.set_title(title)
    #     ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    #     ax.set_ylim(-2.0, 2.0)
    #
    # plot_dist(f_ax, "f", *model.predict_f(plot_x, full_cov=False))
    # plot_dist(y_ax, "y", *model.predict_y(plot_x, full_cov=False))
    #
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / f"{data_name}_{model_name}.png")
    # plt.close()


def run_model(
    data: Dataset, model: GPModel, data_name: str, model_name: str
) -> Mapping[str, float]:
    do_compile = True
    do_optimise = True

    res = {}

    model.predict_y(data.X_test)  # Warm-up TF.

    print("Before:")
    gpflow.utilities.print_summary(model)

    if do_optimise:
        t_before = perf_counter()
        model = find_best_model_initialization(model, N_INITS)

        loss_fn = gpflow.models.training_loss_closure(model, data.train, compile=do_compile)
        opt_log = gpflow.optimizers.Scipy().minimize(
            loss_fn,
            variables=model.trainable_variables,
            compile=do_compile,
            options={"disp": 10, "maxiter": 1_000},
            method=OPT_METHOD,
        )
        t_after = perf_counter()
        n_iter = opt_log.nit # if opt_log.nit > 0 else 1
        t_train = t_after - t_before
        res["opt/n_iter"] = n_iter
        res["time/training/s"] = t_train
        res["time/iter/s"] = t_train / n_iter
        print(f"Training took {t_after - t_before}s / {n_iter} iterations.")
        print("After:")
        gpflow.utilities.print_summary(model)

    t_before = perf_counter()
    m, v = model.predict_y(data.X_test)
    t_after = perf_counter()
    res["time/prediction/s"] = t_after - t_before

    l = norm.logpdf(data.Y_test, loc=m, scale=v ** 0.5)
    res["accuracy/loglik"] = np.average(l)

    lu = norm.logpdf(data.Y_test * data.Y_std, loc=m * data.Y_std, scale=(v ** 0.5) * data.Y_std)
    res["accuracy/loglik/unnormalized"] = np.average(lu)

    d = data.Y_test - m
    du = d * data.Y_std

    res["accuracy/mae"] = np.average(np.abs(d))
    res["accuracy/mae/unnormalized"] = np.average(np.abs(du))

    res["accuracy/rmse"] = np.average(d ** 2) ** 0.5
    res["accuracy/rmse/unnormalized"] = np.average(du ** 2) ** 0.5

    res_values = list(res.values())
    assert np.isfinite(res_values).all(), f"{res_values} not all finite."

    return res


def plot_metrics(metrics_df: pd.DataFrame) -> None:
    for data_name, data_df in metrics_df.groupby("dataset"):
        metric_gr = data_df.groupby(["metric"])
        n_cols = 2
        n_rows = ceil(len(metric_gr) / 2)
        width = 6 * n_cols
        height = 4 * n_rows

        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(width, height), dpi=100)

        for (metric, metric_df), ax in zip(metric_gr, axes.flatten()):
            model_and_values = metric_df.groupby("model").value.apply(list).to_dict()
            model = list(model_and_values.keys())
            values = list(model_and_values.values())
            ax.boxplot(values, labels=model)
            ax.set_title(metric)
            if metric_df.value.min() > 0:
                ax.set_ylim(bottom=0.0)
            if metric_df.value.max() < 0:
                ax.set_ylim(top=0.0)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{data_name}_metrics.png")
        plt.close(fig)


@check_shapes()
def main() -> None:

    res_list = []

    for data_name in regression_datasets:
        # todo try: # prevent crash killing other expts
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(data_name)
        data = get_regression_data(data_name)
        for create_model in models:
            model_name = create_model.__name__
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(f"{data_name}/{model_name} (n={data.N})")
            if data.sparse_only and not create_model.__sparse__:  # type: ignore[attr-defined]
                print("Skipping")
                continue
            rep = 0
            success_reps = 0
            model_plotted = False
            while success_reps < N_REPS:
                print("--------------------------------------------------")
                print(f"{data_name}/{model_name}/{rep} (n={data.N})")
                rng = np.random.default_rng(20220721 + rep)
                model = create_model(data, rng)
                model_ran = False
                res: Mapping[str, float] = {}
                try:
                    res = run_model(data, model, data_name, model_name)
                    model_ran = True
                except Exception:
                    traceback.print_exc()
                    print("Model failed. Trying new random initialisation...")
                if model_ran:
                    success_reps += 1
                    for res_name, res_value in res.items():
                        res_list.append((data_name, model_name, rep, res_name, res_value))
                    if not model_plotted:
                        plot_model(data, model)
                        model_plotted = True
                rep += 1

    res_df = pd.DataFrame(res_list, columns=["dataset", "model", "rep", "metric", "value"])
    print(tabulate(res_df, headers="keys", showindex="never"))
    res_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    plot_metrics(res_df)


if __name__ == "__main__":
    main()
