"""Kinetic hard-modeling routines translated from the MATLAB toolbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

from ._matlab import mldivide

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class KineticOptimizationResult:
    rate_constants: FloatArray
    concentrations: tuple[FloatArray, ...]
    sum_squared_residuals: float
    jacobian: FloatArray


@dataclass(frozen=True, slots=True)
class KineticFitResult:
    rate_constants: FloatArray
    concentrations: tuple[FloatArray, ...]
    sum_squared_residuals: float
    jacobian: FloatArray
    standard_errors: FloatArray


def kinetic_derivative(
    time: float,
    concentrations: ArrayLike,
    reaction_orders: ArrayLike,
    stoichiometry: ArrayLike,
    rate_constants: ArrayLike,
    choose: ArrayLike | None = None,
) -> FloatArray:
    """Evaluate ``equations.m``; ``choose`` is retained for API parity."""
    del time, choose
    values = np.asarray(concentrations, dtype=np.float64).reshape(-1)
    orders = np.asarray(reaction_orders, dtype=np.float64)
    coefficients = np.asarray(stoichiometry, dtype=np.float64)
    rates = np.asarray(rate_constants, dtype=np.float64).reshape(-1)
    if orders.ndim != 2 or orders.shape[1] != values.size:
        raise ValueError("reaction_orders must be reactions by species")
    if coefficients.shape != (values.size, orders.shape[0]):
        raise ValueError("stoichiometry must be species by reactions")
    if rates.size != orders.shape[0]:
        raise ValueError("one rate constant is required per reaction")
    reaction_rates = rates * np.prod(np.power(values[None, :], orders), axis=1)
    return np.asarray(coefficients @ reaction_rates, dtype=np.float64)


def integrate_kinetics(
    times: ArrayLike,
    initial_concentrations: ArrayLike,
    reaction_orders: ArrayLike,
    stoichiometry: ArrayLike,
    rate_constants: ArrayLike,
    choose: ArrayLike | None = None,
) -> FloatArray:
    """Integrate with the ode45 tolerances used by the MATLAB routines."""
    time_values = np.asarray(times, dtype=np.float64).reshape(-1)
    initial = np.asarray(initial_concentrations, dtype=np.float64).reshape(-1)
    if time_values.size < 1 or not np.all(np.isfinite(time_values)):
        raise ValueError("each kinetic time vector must be finite and non-empty")
    if time_values.size == 1:
        return initial[None, :]
    differences = np.diff(time_values)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError("kinetic times must be strictly monotonic")
    solution = solve_ivp(
        lambda t, c: kinetic_derivative(
            t, c, reaction_orders, stoichiometry, rate_constants, choose
        ),
        (float(time_values[0]), float(time_values[-1])),
        initial,
        method="RK45",
        t_eval=time_values,
        rtol=1.0e-2,
        atol=1.0e-3,
    )
    if not solution.success:
        raise RuntimeError(f"kinetic integration failed: {solution.message}")
    return np.asarray(solution.y.T, dtype=np.float64)


def interesp(colored_concentrations: ArrayLike, colored_mask: ArrayLike) -> FloatArray:
    """Insert zero profiles for non-coloured species (``interesp.m``)."""
    concentrations = np.asarray(colored_concentrations, dtype=np.float64)
    mask = np.asarray(colored_mask, dtype=bool).reshape(-1)
    if concentrations.ndim != 2 or concentrations.shape[1] != np.count_nonzero(mask):
        raise ValueError("colored concentration columns must match colored_mask")
    result = np.zeros((concentrations.shape[0], mask.size), dtype=np.float64)
    result[:, mask] = concentrations
    return result


def rcalcglob(
    rate_constants: ArrayLike,
    initial_concentrations: ArrayLike,
    times: Sequence[ArrayLike],
    experimental_concentrations: Sequence[ArrayLike],
    choose: ArrayLike | None,
    reaction_orders: ArrayLike,
    stoichiometry: ArrayLike,
    colored_mask: ArrayLike,
) -> tuple[FloatArray, tuple[FloatArray, ...]]:
    """Global kinetic residual vector translated from ``rcalcglob.m``."""
    initial = np.asarray(initial_concentrations, dtype=np.float64)
    if initial.ndim == 1:
        initial = initial[None, :]
    if (
        len(times) != initial.shape[0]
        or len(experimental_concentrations) != initial.shape[0]
    ):
        raise ValueError("kinetic experiment counts do not agree")
    mask = np.asarray(colored_mask, dtype=bool).reshape(-1)
    fitted: list[FloatArray] = []
    residual_blocks: list[FloatArray] = []
    for experiment in range(initial.shape[0]):
        calculated = integrate_kinetics(
            times[experiment],
            initial[experiment, :],
            reaction_orders,
            stoichiometry,
            np.abs(np.asarray(rate_constants, dtype=np.float64)),
            choose,
        )
        observed = np.asarray(experimental_concentrations[experiment], dtype=np.float64)
        if observed.shape != calculated.shape or mask.size != calculated.shape[1]:
            raise ValueError("kinetic observed/calculated profile shapes do not agree")
        fitted.append(calculated)
        residual_blocks.append(observed[:, mask] - calculated[:, mask])
    residual_matrix = np.vstack(residual_blocks)
    return residual_matrix.reshape(-1, order="F"), tuple(fitted)


def nglmglob(
    initial_rate_constants: ArrayLike,
    initial_concentrations: ArrayLike,
    times: Sequence[ArrayLike],
    experimental_concentrations: Sequence[ArrayLike],
    choose: ArrayLike | None,
    reaction_orders: ArrayLike,
    stoichiometry: ArrayLike,
    colored_mask: ArrayLike,
) -> KineticOptimizationResult:
    """Nine-step damped Gauss-Newton loop from ``nglmglob.m``."""
    old_ssq = 1.0e50
    marquardt = 10.0
    convergence_limit = 1.0e-4
    differentiation_step = 1.0e-4
    rates = np.asarray(initial_rate_constants, dtype=np.float64).reshape(-1).copy()
    if rates.size == 0 or np.any(rates == 0.0):
        raise ValueError("optimized kinetic rate constants must be non-zero")
    parameter_step = np.zeros_like(rates)
    old_residual: FloatArray | None = None
    jacobian: FloatArray | None = None
    fitted: tuple[FloatArray, ...] = ()
    ssq = float("nan")

    for _ in range(9):
        residual, fitted = rcalcglob(
            rates,
            initial_concentrations,
            times,
            experimental_concentrations,
            choose,
            reaction_orders,
            stoichiometry,
            colored_mask,
        )
        ssq = float(residual @ residual)
        convergence = (old_ssq - ssq) / old_ssq
        if convergence >= 0.0 and abs(convergence) >= convergence_limit:
            marquardt /= 3.0
            old_ssq = ssq
            old_residual = residual
            jacobian = np.empty((residual.size, rates.size), dtype=np.float64)
            for index in range(rates.size):
                rates[index] *= 1.0 + differentiation_step
                shifted, _ = rcalcglob(
                    rates,
                    initial_concentrations,
                    times,
                    experimental_concentrations,
                    choose,
                    reaction_orders,
                    stoichiometry,
                    colored_mask,
                )
                jacobian[:, index] = (shifted - residual) / (
                    differentiation_step * rates[index]
                )
                rates[index] /= 1.0 + differentiation_step
        elif convergence < 0.0:
            marquardt *= 5.0
            rates -= parameter_step
        elif abs(convergence) < convergence_limit:
            if marquardt == 0.0:
                break
            marquardt = 0.0
            old_residual = residual

        if jacobian is None or old_residual is None:
            raise RuntimeError("kinetic optimizer failed to initialize its Jacobian")
        augmented_jacobian = np.vstack([jacobian, marquardt * np.eye(rates.size)])
        augmented_residual = np.concatenate(
            [old_residual, np.zeros(rates.size, dtype=np.float64)]
        )
        parameter_step = -np.asarray(
            mldivide(augmented_jacobian, augmented_residual)
        ).reshape(-1)
        rates += parameter_step

    assert jacobian is not None
    return KineticOptimizationResult(rates, fitted, ssq, jacobian)


def opt_kinglob(
    experimental_concentrations: Sequence[ArrayLike],
    stoichiometry: ArrayLike,
    reaction_orders: ArrayLike,
    initial_concentrations: ArrayLike,
    initial_rate_constants: ArrayLike,
    times: Sequence[ArrayLike],
    choose: ArrayLike | None,
    colored_mask: ArrayLike,
) -> KineticOptimizationResult:
    """Prepare and optimize global kinetic constants (``opt_kinglob.m``)."""
    initial = np.asarray(initial_rate_constants, dtype=np.float64).reshape(-1)
    nonzero = initial[initial != 0.0]
    orders = np.asarray(reaction_orders, dtype=np.float64)
    if nonzero.size != orders.shape[0]:
        raise ValueError(
            "the MATLAB kinetic model requires one non-zero constant per reaction"
        )
    optimized = nglmglob(
        nonzero,
        initial_concentrations,
        times,
        experimental_concentrations,
        choose,
        orders,
        stoichiometry,
        colored_mask,
    )
    return KineticOptimizationResult(
        rate_constants=np.abs(optimized.rate_constants),
        concentrations=optimized.concentrations,
        sum_squared_residuals=optimized.sum_squared_residuals,
        jacobian=optimized.jacobian,
    )
