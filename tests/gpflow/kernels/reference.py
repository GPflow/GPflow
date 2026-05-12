from typing import Callable, Mapping, Sequence

import numpy as np
from check_shapes import check_shapes

from gpflow.base import AnyNDArray

_EmbedFn = Callable[[AnyNDArray, AnyNDArray], AnyNDArray]


@check_shapes(
    "X: [N, D]",
    "X: [N, D_tuple...]",
    "lengthscales: [broadcast D_tuple...]",
    "signal_variance: [broadcast D_tuple...]",
)
def ref_rbf_kernel(
    X: AnyNDArray, lengthscales: AnyNDArray, signal_variance: AnyNDArray
) -> AnyNDArray:
    N, _ = X.shape
    kernel = np.zeros((N, N))
    for row_index in range(N):
        for column_index in range(N):
            vecA = X[row_index, :]
            vecB = X[column_index, :]
            delta = vecA - vecB
            distance_squared = np.dot(delta.T, delta)
            kernel[row_index, column_index] = signal_variance * np.exp(
                -0.5 * distance_squared / lengthscales**2
            )
    return kernel


@check_shapes(
    "X: [N, D]",
    "X: [N, D_tuple...]",
    "weight_variances: [broadcast D_tuple...]",
    "bias_variance: [broadcast D_tuple...]",
    "signal_variance: [broadcast D_tuple...]",
)
def ref_arccosine_kernel(
    X: AnyNDArray,
    order: int,
    weight_variances: AnyNDArray,
    bias_variance: AnyNDArray,
    signal_variance: AnyNDArray,
) -> AnyNDArray:
    num_points = X.shape[0]
    kernel = np.empty((num_points, num_points))
    for row in range(num_points):
        for col in range(num_points):
            x = X[row]
            y = X[col]

            numerator = (weight_variances * x).dot(y) + bias_variance

            x_denominator = np.sqrt((weight_variances * x).dot(x) + bias_variance)
            y_denominator = np.sqrt((weight_variances * y).dot(y) + bias_variance)
            denominator = x_denominator * y_denominator

            theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
            if order == 0:
                J = np.pi - theta
            elif order == 1:
                J = np.sin(theta) + (np.pi - theta) * np.cos(theta)
            elif order == 2:
                J = 3.0 * np.sin(theta) * np.cos(theta)
                J += (np.pi - theta) * (1.0 + 2.0 * np.cos(theta) ** 2)

            kernel[row, col] = (
                signal_variance
                * (1.0 / np.pi)
                * J
                * x_denominator**order
                * y_denominator**order
            )
    return kernel


@check_shapes(
    "X: [N, D]",
    "X: [N, D_tuple...]",
    "lengthscales: [broadcast D_tuple...]",
    "signal_variance: [broadcast D_tuple...]",
    "period: [broadcast D_tuple...]",
)
def ref_periodic_kernel(
    X: AnyNDArray,
    base_name: str,
    lengthscales: AnyNDArray,
    signal_variance: AnyNDArray,
    period: AnyNDArray,
) -> AnyNDArray:
    """
    Calculates K(X) for the periodic kernel based on various base kernels.
    """
    sine_arg = np.pi * (X[:, None, :] - X[None, :, :]) / period
    sine_base = np.sin(sine_arg) / lengthscales
    exp_dist: AnyNDArray
    if base_name in {"RBF", "SquaredExponential"}:
        dist = 0.5 * np.sum(np.square(sine_base), axis=-1)
        exp_dist = np.exp(-dist)
    elif base_name == "Matern12":
        dist = np.sum(np.abs(sine_base), axis=-1)
        exp_dist = np.exp(-dist)
    elif base_name == "Matern32":
        dist = np.sqrt(3) * np.sum(np.abs(sine_base), axis=-1)
        exp_dist = (1 + dist) * np.exp(-dist)
    elif base_name == "Matern52":
        dist = np.sqrt(5) * np.sum(np.abs(sine_base), axis=-1)
        exp_dist = (1 + dist + dist**2 / 3) * np.exp(-dist)
    return signal_variance * exp_dist


def _hierarchical_embedding(
    X: AnyNDArray,
    feature_dims: Sequence[int],
    feature_bounds: AnyNDArray,
    indicator_dims: Sequence[int],
    activity_conditions: Sequence[Mapping[int, int]],
    embed_conditional: _EmbedFn,
) -> AnyNDArray:
    X_feat = X[:, list(feature_dims)]
    lo, hi = feature_bounds[:, 0], feature_bounds[:, 1]
    rng = np.where(np.abs(hi - lo) < 1e-12, 1.0, hi - lo)
    v = (X_feat - lo) / rng

    if indicator_dims:
        ind = np.rint(X[:, list(indicator_dims)]).astype(int)
        mask = np.ones((X.shape[0], len(feature_dims)), dtype=bool)
        for j, req in enumerate(activity_conditions):
            for k, val in req.items():
                mask[:, j] &= ind[:, k] == val
    else:
        mask = np.ones((X.shape[0], len(feature_dims)), dtype=bool)

    cond_idx = [j for j, req in enumerate(activity_conditions) if req]
    uncond_idx = [j for j in range(len(feature_dims)) if j not in set(cond_idx)]

    parts = []
    if uncond_idx:
        parts.append(v[:, uncond_idx])
    if cond_idx:
        v_c = v[:, cond_idx]
        m_c = mask[:, cond_idx].astype(float)
        parts.append(embed_conditional(v_c, m_c))
    return np.concatenate(parts, axis=-1) if parts else np.zeros((X.shape[0], 0))


def ref_arc_hierarchical_kernel(
    X: AnyNDArray,
    feature_dims: Sequence[int],
    feature_bounds: AnyNDArray,
    indicator_dims: Sequence[int],
    activity_conditions: Sequence[Mapping[int, int]],
    angle: AnyNDArray,
    radius: AnyNDArray,
    base_variance: float = 1.0,
) -> AnyNDArray:
    def arc(v_c: AnyNDArray, m_c: AnyNDArray) -> AnyNDArray:
        theta = np.pi * angle * v_c
        return np.concatenate(
            [radius * np.sin(theta) * m_c, radius * np.cos(theta) * m_c], axis=-1
        )

    Z = _hierarchical_embedding(
        X, feature_dims, feature_bounds, indicator_dims, activity_conditions, arc
    )
    diff = Z[:, None, :] - Z[None, :, :]
    r2 = np.sum(diff**2, axis=-1)
    r = np.sqrt(np.maximum(r2, 0.0))
    sqrt5 = np.sqrt(5.0)
    return base_variance * (1.0 + sqrt5 * r + 5.0 / 3.0 * r**2) * np.exp(-sqrt5 * r)


def ref_wedge_hierarchical_kernel(
    X: AnyNDArray,
    feature_dims: Sequence[int],
    feature_bounds: AnyNDArray,
    indicator_dims: Sequence[int],
    activity_conditions: Sequence[Mapping[int, int]],
    theta1: AnyNDArray,
    theta2: AnyNDArray,
    rho: AnyNDArray,
    base_variance: float = 1.0,
) -> AnyNDArray:
    def wedge(v_c: AnyNDArray, m_c: AnyNDArray) -> AnyNDArray:
        comp1 = (theta1 * v_c + theta2 * v_c * np.cos(rho)) * m_c
        comp2 = (theta2 * v_c * np.sin(rho)) * m_c
        return np.concatenate([comp1, comp2], axis=-1)

    Z = _hierarchical_embedding(
        X, feature_dims, feature_bounds, indicator_dims, activity_conditions, wedge
    )
    diff = Z[:, None, :] - Z[None, :, :]
    r2 = np.sum(diff**2, axis=-1)
    r = np.sqrt(np.maximum(r2, 0.0))
    sqrt5 = np.sqrt(5.0)
    return base_variance * (1.0 + sqrt5 * r + 5.0 / 3.0 * r**2) * np.exp(-sqrt5 * r)
