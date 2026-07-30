"""Correlation/regression constraints translated from ``yregrnew.m``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RegressionStats:
    """Statistics returned for one correlated component."""

    slope: float
    offset: float
    correlation: float
    rmsec: float
    relative_error_percent: float


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    """Calibrated profiles and per-block regression statistics."""

    output: FloatArray
    calibrated: FloatArray
    stats: tuple[tuple[RegressionStats | None, ...], ...]


def yregrnew(
    profiles: ArrayLike,
    reference: ArrayLike,
    component_mask: ArrayLike,
) -> CorrelationResult:
    """Apply the linear correlation constraint from ``yregrnew.m``.

    Finite entries in ``reference`` are calibration points; NaNs mark unknown
    values. Selected calibration points are restored exactly after regression.
    """
    input_profiles = np.asarray(profiles, dtype=np.float64)
    selected_values = np.asarray(reference, dtype=np.float64)
    if input_profiles.ndim != 2 or selected_values.shape != input_profiles.shape:
        raise ValueError("reference and profiles must be matrices of equal shape")
    selected_components = np.asarray(component_mask, dtype=bool).reshape(-1)
    if selected_components.size != input_profiles.shape[1]:
        raise ValueError("component_mask must match the profile column count")

    output = input_profiles.copy()
    calibrated = input_profiles.copy()
    stats: list[RegressionStats | None] = [None] * input_profiles.shape[1]

    for component in range(input_profiles.shape[1]):
        if not selected_components[component]:
            continue
        known = np.flatnonzero(np.isfinite(selected_values[:, component]))
        if known.size < 2:
            continue
        x = selected_values[known, component]
        y = input_profiles[known, component]
        first_fit = np.polyfit(x, y, 1)
        if first_fit[0] == 0.0:
            raise ValueError(f"correlation slope is zero for component {component}")
        calculated = (input_profiles[:, component] - first_fit[1]) / first_fit[0]
        calibrated[:, component] = calculated

        second_fit = np.polyfit(x, calculated[known], 1)
        second_residual = calculated[known] - np.polyval(second_fit, x)
        correlation_matrix = np.corrcoef(x, calculated[known])
        deviation = x - calculated[known]
        denominator = float(x @ x)
        relative_error = (
            float(100.0 * np.sqrt((deviation @ deviation) / denominator))
            if denominator != 0.0
            else float("nan")
        )
        stats[component] = RegressionStats(
            slope=float(second_fit[0]),
            offset=float(second_fit[1]),
            correlation=float(correlation_matrix[0, 1]),
            rmsec=float(np.linalg.norm(second_residual) / np.sqrt(known.size)),
            relative_error_percent=relative_error,
        )

        constrained = calculated.copy()
        constrained[known] = x
        output[:, component] = constrained

    return CorrelationResult(output, calibrated, (tuple(stats),))
