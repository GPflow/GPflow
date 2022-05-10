from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow as gpf
from gpflow.base import AnyNDArray, RegressionData
from gpflow.ci_utils import ci_niter
from gpflow.experimental.check_shapes import check_shape as cs
from gpflow.experimental.check_shapes import check_shapes
from gpflow.utilities import print_summary

gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")
np.random.seed(0)

OUTPUT_DIR = Path(__file__).parent

MAXITER = ci_niter(2000)

M = 15  # number of inducing points
Zinit = np.linspace(-5, 5, M)[:, None]


@dataclass
class Dataset:
    D: int  # number of input dimensions
    L: int  # number of latent GPs
    P: int  # number of observations = output dimensions
    X: AnyNDArray
    Y: AnyNDArray

    @check_shapes(
        "self.X: [N, D]",
        "self.Y: [N, P]",
    )
    def __post_init__(self) -> None:
        pass

    @property
    def data(self) -> RegressionData:
        return self.X, self.Y


@check_shapes()
def simple_dataset() -> Dataset:
    N = 100  # number of points
    D = 1  # number of input dimensions
    L = 1  # number of latent GPs
    P = 2  # number of observations = output dimensions
    rng = np.random.default_rng(20220510)
    X = cs(rng.random(N)[:, None] * 10 - 5, "[N, D]")
    G: AnyNDArray = cs(np.sin(2 * X), "[N, L]")
    W: AnyNDArray = cs(np.array([[0.5, 0.2]]), "[L, P]")
    F = cs(np.matmul(G, W), "[N, P]")
    Y = cs(F + rng.standard_normal(F.shape) * [0.1, 0.2], "[N, P]")
    return Dataset(
        D=D,
        L=L,
        P=P,
        X=X,
        Y=Y,
    )


def notebook_dataset() -> Dataset:
    N = 100  # number of points
    D = 1  # number of input dimensions
    L = 2  # number of latent GPs
    P = 3  # number of observations = output dimensions
    rng = np.random.default_rng(20220510)
    X = rng.random(N)[:, None] * 10 - 5  # Inputs = N x D
    G: AnyNDArray = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))  # G = N x L
    W: AnyNDArray = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])  # L x P
    F = np.matmul(G, W)  # N x P
    Y = F + rng.standard_normal(F.shape) * [0.2, 0.2, 0.2]
    return Dataset(
        D=D,
        L=L,
        P=P,
        X=X,
        Y=Y,
    )


DATA_GENERATORS = [
    simple_dataset,
    notebook_dataset,
]


def gpr(data: Dataset) -> gpf.models.GPModel:
    kernel = gpf.kernels.SquaredExponential() + gpf.kernels.Linear()
    return gpf.models.gpr.GPR_deprecated(data.data, kernel)


def gpr_shared(data: Dataset) -> gpf.models.GPModel:
    kernel = gpf.kernels.SharedIndependent(
        gpf.kernels.SquaredExponential() + gpf.kernels.Linear(), output_dim=data.P
    )
    return gpf.models.gpr.GPR_deprecated(data.data, kernel)


def gpr_separate(data: Dataset) -> gpf.models.GPModel:
    kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(data.P)]
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    return gpf.models.gpr.GPR_deprecated(data.data, kernel)


def gpr_linear_coreg(data: Dataset) -> gpf.models.GPModel:
    kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(data.L)]
    kernel = gpf.kernels.LinearCoregionalization(kern_list, W=np.random.randn(data.P, data.L))
    return gpf.models.gpr.GPR_deprecated(data.data, kernel)


def svgp_shared_shared(data: Dataset) -> gpf.models.GPModel:
    kernel = gpf.kernels.SharedIndependent(
        gpf.kernels.SquaredExponential() + gpf.kernels.Linear(), output_dim=data.P
    )
    Z = Zinit.copy()
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(Z)
    )
    return gpf.models.SVGP(
        kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=data.P
    )


def svgp_separate_shared(data: Dataset) -> gpf.models.GPModel:
    kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(data.P)]
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    Z = Zinit.copy()
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(Z)
    )
    return gpf.models.SVGP(
        kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=data.P
    )


def svgp_separate_separate(data: Dataset) -> gpf.models.GPModel:
    kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(data.P)]
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    Zs = [Zinit.copy() for _ in range(data.P)]
    iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    return gpf.models.SVGP(
        kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=data.P
    )


def svgp_linear_coreg(data: Dataset) -> gpf.models.GPModel:
    kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(data.L)]
    kernel = gpf.kernels.LinearCoregionalization(kern_list, W=np.random.randn(data.P, data.L))
    Z = Zinit.copy()
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(Z)
    )
    q_mu = np.zeros((M, data.L))
    q_sqrt = np.repeat(np.eye(M)[None, ...], data.L, axis=0) * 1.0
    return gpf.models.SVGP(
        kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, q_mu=q_mu, q_sqrt=q_sqrt
    )


MODEL_GENERATORS = [
    gpr,
    gpr_shared,
    gpr_separate,
    gpr_linear_coreg,
    svgp_shared_shared,
    svgp_separate_shared,
    svgp_separate_separate,
    svgp_linear_coreg,
]


def optimize_model_with_scipy(data: Dataset, model: gpf.models.GPModel) -> None:
    optimizer = gpf.optimizers.Scipy()
    optimizer.minimize(
        gpf.models.training_loss_closure(model, data.data),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": 50, "maxiter": MAXITER},
    )


@check_shapes()
def plot_model(
    data: Dataset, model: gpf.models.GPModel, name: str, lower: float = -20.0, upper: float = 20.0
) -> None:
    fig, ax = plt.subplots(figsize=(20, 16), dpi=100)
    N = 500

    pX = cs(np.linspace(lower, upper, N)[:, None], "[N, 1]")
    pY, pYv = model.predict_y(pX)
    if pY.ndim == 3:
        pY = pY[:, 0, :]

    cs(pY, "[N, P]")
    cs(pYv, "[N, P]")

    ax.plot(data.X, data.Y, "x")
    ax.set_prop_cycle(None)
    ax.plot(pX, pY)
    for i in range(pY.shape[1]):
        top = pY[:, i] + 2.0 * pYv[:, i] ** 0.5
        bot = pY[:, i] - 2.0 * pYv[:, i] ** 0.5
        plt.fill_between(pX[:, 0], top, bot, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("f")
    if hasattr(model, "elbo"):
        ax.set_title(f"ELBO: {model.elbo(data.data):.3}")
    # ax.plot(Z, Z * 0.0, "o")

    pY = cs(model.predict_f_samples(pX, full_cov=True, full_output_cov=True), "[N, P]")
    ax.set_prop_cycle(None)
    ax.plot(pX, pY, ls=":")

    fig.savefig(OUTPUT_DIR / f"{name}.png")
    plt.close(fig)


def main() -> None:
    for create_data in DATA_GENERATORS:
        data = create_data()

        for create_model in MODEL_GENERATORS:
            name = f"{create_data.__name__}__{create_model.__name__}"
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(name)
            model = create_model(data)

            # X = np.random.default_rng(42).random((2, 3, 4, 1))
            # for full_output_cov in [False, True]:
            #     for full_cov in [False, True]:
            #         print(
            #             create_model.__name__,
            #             "full_cov",
            #             full_cov,
            #             "full_output_cov",
            #             full_output_cov,
            #         )
            #         model.predict_f(X, full_cov=full_cov, full_output_cov=full_output_cov)
            #         print("ok")

            optimize_model_with_scipy(data, model)
            print_summary(model)
            plot_model(data, model, name)


if __name__ == "__main__":
    main()
