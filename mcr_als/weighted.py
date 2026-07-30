"""Maximum-likelihood PCA translated from the toolbox's ``mlpca.m``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MLPCAResult:
    u: FloatArray
    s: FloatArray
    v: FloatArray
    objective: float
    error_flag: int
    iterations: int

    @property
    def reconstructed(self) -> FloatArray:
        return self.u @ self.s @ self.v.T


def mlpca(
    observations: ArrayLike,
    standard_deviations: ArrayLike,
    rank: int,
    convergence_limit: float = 1.0e-10,
    max_iterations: int = 200_000,
) -> MLPCAResult:
    """Perform alternating maximum-likelihood PCA as in ``mlpca.m``.

    Zero standard deviations retain the source convention and mark missing
    measurements by assigning a very large variance.
    """
    x = np.asarray(observations, dtype=np.float64)
    std = np.asarray(standard_deviations, dtype=np.float64)
    if x.ndim != 2 or std.shape != x.shape:
        raise ValueError("standard_deviations must match the observation matrix")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(std)):
        raise ValueError("MLPCA inputs must be finite")
    if np.any(std < 0.0):
        raise ValueError("standard deviations must be non-negative")
    components = int(rank)
    if components < 1 or components > min(x.shape):
        raise ValueError("rank must be between 1 and the smallest matrix dimension")
    if convergence_limit < 0.0 or max_iterations < 1:
        raise ValueError("invalid MLPCA convergence settings")

    xx = x.copy()
    variances = std * std
    positive = variances[variances > 0.0]
    if positive.size == 0:
        raise ValueError("at least one standard deviation must be positive")
    error_maximum = float(np.max(variances))
    variances[variances == 0.0] = 1.0e10 * error_maximum

    u, singular_values, vh = np.linalg.svd(xx, full_matrices=False)
    u0 = u[:, :components]
    count = 0
    old_objective = 0.0
    error_flag = -1
    objective = float("nan")
    maximum_likelihood = np.zeros_like(xx)

    while error_flag < 0:
        count += 1
        objective = 0.0
        maximum_likelihood = np.zeros_like(xx)
        for column in range(xx.shape[1]):
            weights = 1.0 / variances[:, column]
            weighted_basis = weights[:, None] * u0
            inverse_information = np.linalg.inv(u0.T @ weighted_basis)
            coefficients = inverse_information @ (u0.T @ (weights * xx[:, column]))
            maximum_likelihood[:, column] = u0 @ coefficients
            residual = xx[:, column] - maximum_likelihood[:, column]
            objective += float(residual @ (weights * residual))

        if count % 2 == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                relative_change = float(
                    np.divide(abs(old_objective - objective), objective)
                )
            if relative_change < convergence_limit:
                error_flag = 0
            elif count > max_iterations:
                error_flag = 1

        if error_flag < 0:
            old_objective = objective
            _, _, vh_ml = np.linalg.svd(maximum_likelihood, full_matrices=False)
            v_ml = vh_ml.T
            xx = xx.T
            variances = variances.T
            u0 = v_ml[:, :components]

    u, singular_values, vh = np.linalg.svd(maximum_likelihood, full_matrices=False)
    return MLPCAResult(
        u=np.asarray(u[:, :components], dtype=np.float64),
        s=np.diag(singular_values[:components]),
        v=np.asarray(vh[:components, :].T, dtype=np.float64),
        objective=objective,
        error_flag=error_flag,
        iterations=count,
    )
