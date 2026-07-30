"""Small compatibility helpers for MATLAB numerical/indexing semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]


def as_float_matrix(value: ArrayLike, name: str) -> FloatArray:
    """Return a finite, two-dimensional, C-contiguous float64 copy."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def mldivide(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """MATLAB ``left \\ right`` for dense float64 matrices.

    MATLAB uses a direct solve for square nonsingular systems and a
    rank-revealing least-squares solve otherwise.  SciPy's ``gelsy`` driver is
    the closest dense QR-with-column-pivoting analogue for the latter.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("left operand must be two-dimensional")
    if b.ndim not in (1, 2):
        raise ValueError("right operand must be one- or two-dimensional")
    if a.shape[0] != b.shape[0]:
        raise ValueError("matrix dimensions do not agree for left division")

    if a.shape[0] == a.shape[1]:
        try:
            return np.asarray(linalg.solve(a, b, assume_a="gen"), dtype=np.float64)
        except linalg.LinAlgError:
            pass
    solution, _, _, _ = linalg.lstsq(a, b, cond=None, lapack_driver="gelsy")
    return np.asarray(solution, dtype=np.float64)


def mrdivide(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """MATLAB ``left / right`` implemented through transposed left division."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matrix right division requires two-dimensional arrays")
    return np.asarray(mldivide(b.T, a.T).T, dtype=np.float64)


def matlab_std(array: ArrayLike, axis: int = 0) -> FloatArray:
    """MATLAB's default sample standard deviation."""
    values = np.asarray(array, dtype=np.float64)
    return np.asarray(np.std(values, axis=axis, ddof=1), dtype=np.float64)


def fortran_take(array: ArrayLike, indices: ArrayLike) -> FloatArray:
    """Take zero-based linear indices in MATLAB/Fortran storage order."""
    values = np.asarray(array)
    idx = np.asarray(indices, dtype=np.intp)
    return np.asarray(values.ravel(order="F")[idx])


def fortran_assign(
    array: ArrayLike, indices: ArrayLike, values: ArrayLike | float
) -> FloatArray:
    """Assign zero-based linear indices in MATLAB/Fortran storage order."""
    original = np.asarray(array, dtype=np.float64)
    flat = np.array(original.ravel(order="F"), dtype=np.float64, copy=True)
    flat[np.asarray(indices, dtype=np.intp)] = values
    return flat.reshape(original.shape, order="F")


def normalize_mask(
    mask: Any,
    shape: tuple[int, ...],
    *,
    default: bool,
    name: str,
) -> NDArray[np.bool_]:
    if mask is None:
        return np.full(shape, default, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape == shape:
        return result.copy()
    if result.ndim == 1 and len(shape) == 2:
        if result.size == shape[1]:
            return np.broadcast_to(result, shape).copy()
        if result.size == shape[0]:
            return np.broadcast_to(result[:, None], shape).copy()
    raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
