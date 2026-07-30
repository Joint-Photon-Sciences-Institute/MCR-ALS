"""Direct translations of the numerical MATLAB helper routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._matlab import as_float_matrix, matlab_std, mldivide

FloatArray = NDArray[np.float64]


def pcarep(
    data: ArrayLike, n_components: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float]:
    """Rank-``n_components`` PCA reproduction from ``pcarep.m``."""
    x = as_float_matrix(data, "data")
    rank = int(n_components)
    if rank < 1 or rank > min(x.shape):
        raise ValueError(
            f"n_components must be between 1 and {min(x.shape)}, got {rank}"
        )
    u, singular_values, vh = np.linalg.svd(x, full_matrices=False)
    u = u[:, :rank]
    s = np.diag(singular_values[:rank])
    v = vh[:rank, :].T
    reproduced = (u * singular_values[:rank]) @ vh[:rank, :]
    residual = x - reproduced
    total = float(np.sum(x * x))
    sigma = float(np.sqrt(np.sum(residual * residual) / total) * 100.0)
    return u, s, v, reproduced, sigma


def fnnls(
    xtx: ArrayLike, xty: ArrayLike, tolerance: float | None = None
) -> tuple[FloatArray, FloatArray]:
    """Fast non-negative least squares translated from ``fnnls.m``."""
    gram = as_float_matrix(xtx, "xtx")
    rhs = np.asarray(xty, dtype=np.float64).reshape(-1)
    n = gram.shape[1]
    if gram.shape != (n, n) or rhs.size != n:
        raise ValueError("xtx must be square and xty must have matching length")
    tol: float = (
        float(
            10.0 * np.finfo(np.float64).eps * np.linalg.norm(gram, 1) * max(gram.shape)
        )
        if tolerance is None
        else float(tolerance)
    )

    passive = np.zeros(n, dtype=bool)
    x = np.zeros(n, dtype=np.float64)
    w = rhs - gram @ x
    iterations = 0
    max_iterations = 30 * n

    while np.any(~passive) and np.any(w[~passive] > tol):
        candidates = np.flatnonzero(~passive)
        selected = candidates[int(np.argmax(w[candidates]))]
        passive[selected] = True

        z = np.zeros(n, dtype=np.float64)
        pp = np.flatnonzero(passive)
        z[pp] = np.asarray(mldivide(gram[np.ix_(pp, pp)], rhs[pp])).reshape(-1)

        while np.any(z[pp] <= tol) and iterations < max_iterations:
            iterations += 1
            qq = np.flatnonzero((z <= tol) & passive)
            alpha = np.min(x[qq] / (x[qq] - z[qq]))
            x = x + alpha * (z - x)
            passive[(np.abs(x) < tol) & passive] = False
            pp = np.flatnonzero(passive)
            z.fill(0.0)
            if pp.size:
                z[pp] = np.asarray(mldivide(gram[np.ix_(pp, pp)], rhs[pp])).reshape(-1)

        x = z.copy()
        w = rhs - gram @ x

    return x, w


def nnls(
    matrix: ArrayLike, rhs: ArrayLike, tolerance: float | None = None
) -> tuple[FloatArray, FloatArray]:
    """Lawson-Hanson NNLS translated from the toolbox's ``nnls.m``."""
    a = as_float_matrix(matrix, "matrix")
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    m, n = a.shape
    if b.size != m:
        raise ValueError("rhs length must equal the number of matrix rows")
    tol: float = (
        float(10.0 * np.finfo(np.float64).eps * np.linalg.norm(a, 1) * max(a.shape))
        if tolerance is None
        else float(tolerance)
    )

    passive = np.zeros(n, dtype=bool)
    x = np.zeros(n, dtype=np.float64)
    w = a.T @ (b - a @ x)
    iterations = 0
    max_iterations = 3 * n

    while np.any(~passive) and np.any(w[~passive] > tol):
        candidates = np.flatnonzero(~passive)
        selected = candidates[int(np.argmax(w[candidates]))]
        passive[selected] = True

        pp = np.flatnonzero(passive)
        ep = np.zeros_like(a)
        ep[:, pp] = a[:, pp]
        z = np.linalg.pinv(ep) @ b
        z[~passive] = 0.0

        while np.any(z[pp] <= tol):
            iterations += 1
            if iterations > max_iterations:
                raise RuntimeError(
                    "NNLS iteration count exceeded; try raising the tolerance"
                )
            qq = np.flatnonzero((z <= tol) & passive)
            alpha = np.min(x[qq] / (x[qq] - z[qq]))
            x = x + alpha * (z - x)
            passive[(np.abs(x) < tol) & passive] = False
            pp = np.flatnonzero(passive)
            ep.fill(0.0)
            ep[:, pp] = a[:, pp]
            z = np.linalg.pinv(ep) @ b
            z[~passive] = 0.0

        x = z.copy()
        w = a.T @ (b - a @ x)

    return x, w


def unimod(concentrations: ArrayLike, tolerance_factor: float, mode: int) -> FloatArray:
    """Force profiles to the unimodal shape used by ``unimod.m``."""
    original = np.asarray(concentrations, dtype=np.float64)
    was_vector = original.ndim == 1
    if original.ndim == 1:
        conc = original[:, None].copy()
    elif original.ndim == 2:
        conc = original.copy()
    else:
        raise ValueError("concentrations must be one- or two-dimensional")
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0, 1, or 2")

    n_rows, n_components = conc.shape
    maxima = np.argmax(conc, axis=0)
    for component in range(n_components):
        peak = int(maxima[component])
        running = conc[peak, component]
        k = peak
        while k > 0:
            k -= 1
            if conc[k, component] <= running:
                running = conc[k, component]
                continue
            if conc[k, component] > running * tolerance_factor:
                if mode == 0:
                    conc[k, component] = 1.0e-30
                elif mode == 1:
                    conc[k, component] = conc[k + 1, component]
                elif running > 0:
                    averaged = (conc[k, component] + conc[k + 1, component]) / 2.0
                    conc[k : k + 2, component] = averaged
                    # Literal MATLAB code jumps two positions to recheck the
                    # newly averaged section.  Clamp the rare boundary case.
                    k = min(k + 2, n_rows - 1)
                else:
                    conc[k, component] = 0.0
                running = conc[k, component]

        running = conc[peak, component]
        k = peak
        while k < n_rows - 1:
            k += 1
            if conc[k, component] <= running:
                running = conc[k, component]
                continue
            if conc[k, component] > running * tolerance_factor:
                if mode == 0:
                    conc[k, component] = 1.0e-30
                elif mode == 1:
                    conc[k, component] = conc[k - 1, component]
                elif running > 0:
                    averaged = (conc[k, component] + conc[k - 1, component]) / 2.0
                    conc[k - 1 : k + 1, component] = averaged
                    k = max(k - 2, 0)
                else:
                    conc[k, component] = 0.0
                running = conc[k, component]

    return conc[:, 0] if was_vector else conc


def _target_is_scalar_zero(target: ArrayLike | float | None) -> bool:
    if target is None:
        return True
    values = np.asarray(target)
    return values.size == 0 or (values.size == 1 and float(values.flat[0]) == 0.0)


def _apply_closure_group(
    conc: FloatArray,
    selection: NDArray[np.bool_],
    kind: int,
    scalar_target: float,
    vector_target: ArrayLike | float | None,
) -> None:
    if not np.any(selection):
        return
    group = conc[:, selection]
    total = np.sum(group, axis=1)
    total[total == 0.0] = 1.0
    vector_mode = not _target_is_scalar_zero(vector_target)
    target = (
        np.asarray(vector_target, dtype=np.float64).reshape(-1)
        if vector_mode
        else np.full(conc.shape[0], float(scalar_target), dtype=np.float64)
    )
    if target.size != conc.shape[0]:
        raise ValueError("vector closure target must match the profile length")

    if kind == 1:
        conc[:, selection] = group * (target / total)[:, None]
    elif kind == 2:
        scale = np.asarray(mldivide(group, target)).reshape(-1)
        conc[:, selection] = group * scale[None, :]
    elif kind == 3:
        conc[:, selection] = group * (target / np.max(total))[:, None]
    else:
        raise ValueError("closure kind must be 1, 2, or 3")


def closure(
    concentrations: ArrayLike,
    n_closures: int,
    selection1: ArrayLike,
    kind1: int,
    target1: float,
    target2: float = 0.0,
    selection2: ArrayLike | None = None,
    kind2: int = 1,
    vector1: ArrayLike | float | None = None,
    vector2: ArrayLike | float | None = None,
) -> FloatArray:
    """Apply one or two closure conditions.

    This follows ``closure.m`` while repairing its undefined ``y`` and
    accidental ``cclos`` references in vector/two-group least-squares paths.
    """
    conc = as_float_matrix(concentrations, "concentrations")
    count = int(n_closures)
    if count not in (1, 2):
        raise ValueError("n_closures must be 1 or 2")
    first = np.asarray(selection1, dtype=bool).reshape(-1)
    second = (
        np.zeros(conc.shape[1], dtype=bool)
        if selection2 is None
        else np.asarray(selection2, dtype=bool).reshape(-1)
    )
    if first.size != conc.shape[1] or second.size != conc.shape[1]:
        raise ValueError("closure selections must match the component count")
    if count == 2 and np.any(first & second):
        raise ValueError("one species is included in both closures")

    _apply_closure_group(conc, first, int(kind1), float(target1), vector1)
    if count == 2:
        _apply_closure_group(conc, second, int(kind2), float(target2), vector2)
    return conc


def closurels(
    concentrations: ArrayLike,
    n_closures: int,
    selection1: ArrayLike,
    kind1: int,
    target1: float,
    target2: float = 0.0,
    selection2: ArrayLike | None = None,
    kind2: int = 1,
    vector1: ArrayLike | float | None = None,
    vector2: ArrayLike | float | None = None,
) -> FloatArray:
    """Compatibility entry point for the toolbox's ``closurels.m``.

    That file contains the equality and least-squares subset of ``closure.m``
    (and declares the function under the name ``closure``). The Python name is
    disambiguated while retaining its call signature.
    """
    if int(kind1) not in (1, 2) or (int(n_closures) == 2 and int(kind2) not in (1, 2)):
        raise ValueError("closurels supports equality and least-squares kinds")
    return closure(
        concentrations,
        n_closures,
        selection1,
        kind1,
        target1,
        target2,
        selection2,
        kind2,
        vector1,
        vector2,
    )


def normv2(spectra: ArrayLike) -> FloatArray:
    """Normalize every row by its Euclidean norm (``normv2.m``)."""
    values = as_float_matrix(spectra, "spectra")
    with np.errstate(divide="ignore", invalid="ignore"):
        return values / np.sqrt(np.sum(values * values, axis=1))[:, None]


def normv3(spectra: ArrayLike) -> FloatArray:
    """Normalize absolute row values by their absolute sum (``normv3.m``)."""
    values = as_float_matrix(spectra, "spectra")
    absolute = np.abs(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return absolute / np.sum(absolute, axis=1)[:, None]


def _orient_leading_singular_vectors(
    u: FloatArray, vh: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Use the sign convention assumed by the legacy ``trilin.m`` code."""
    u = u.copy()
    vh = vh.copy()
    if np.sum(u[:, 0]) > 0.0:
        u[:, 0] *= -1.0
        vh[0, :] *= -1.0
    return u, vh


def trilin(
    profile: ArrayLike, n_experiments: int, shape: int = 1
) -> tuple[FloatArray, FloatArray]:
    """Force equal profiles across experiments (``trilin.m``)."""
    raw = np.asarray(profile, dtype=np.float64)
    was_vector = raw.ndim == 1
    if raw.ndim == 1:
        c = raw[:, None].copy()
    elif raw.ndim == 2 and raw.shape[1] >= 1:
        c = raw.copy()
    else:
        raise ValueError("profile must be a vector or two-dimensional matrix")
    ne = int(n_experiments)
    if ne < 1 or c.shape[0] % ne:
        raise ValueError("profile length must be divisible by n_experiments")
    if shape not in (1, 2):
        raise ValueError("shape must be 1 (synchronized) or 2 (peak shifted)")

    n_rows = c.shape[0] // ne
    component = c[:, -1]
    folded = component.reshape((n_rows, ne), order="F")
    aligned = folded.copy()
    shifts = np.zeros(ne, dtype=np.intp)
    reference = 0

    if shape == 2:
        maxima = np.max(folded, axis=0)
        positions = np.argmax(folded, axis=0)
        positive = np.flatnonzero(maxima > 0.0)
        if positive.size:
            reference = int(positive[np.argmin(positions[positive])])
        reference_position = int(positions[reference])
        for experiment in range(ne):
            if experiment != reference and maxima[experiment] > 0.0:
                shift = int(positions[experiment] - reference_position)
                shifts[experiment] = shift
                if shift:
                    aligned[: n_rows - shift, experiment] = folded[shift:, experiment]
                    aligned[n_rows - shift :, experiment] = 0.0

    u, singular_values, vh = np.linalg.svd(aligned, full_matrices=False)
    u, vh = _orient_leading_singular_vectors(u, vh)
    totals = -singular_values[0] * vh[0, :]
    reconstructed = -u[:, [0]] * totals[None, :]

    output_folded = reconstructed.copy()
    if shape == 2:
        for experiment, shift in enumerate(shifts):
            if experiment != reference and shift:
                output_folded[:shift, experiment] = 0.0
                output_folded[shift:, experiment] = reconstructed[
                    : n_rows - shift, experiment
                ]

    c[:, -1] = output_folded.reshape(-1, order="F")
    return (c[:, 0] if was_vector else c), totals


def quadril(
    profile: ArrayLike,
    ne1: int,
    ne2: int,
    ne3: int,
    shape: int = 1,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Force the quadrilinear profile model from ``quadril.m``."""
    raw = np.asarray(profile, dtype=np.float64)
    was_vector = raw.ndim == 1
    c = raw.reshape(-1).copy()
    d1, d2, d3 = int(ne1), int(ne2), int(ne3)
    # Literal dimension swap at the beginning of quadril.m.
    d1, d3 = d3, d1
    if d1 * d2 * d3 != c.size:
        raise ValueError("mode dimensions do not fit the profile length")
    if shape not in (1, 2):
        raise ValueError("shape must be 1 or 2")
    if shape == 2:
        raise NotImplementedError(
            "quadril.m calls an unavailable peakshift helper for shape=2"
        )

    ns1 = c.size // d1
    folded1 = c.reshape((ns1, d1), order="F")
    u, singular_values, vh = np.linalg.svd(folded1, full_matrices=False)
    mode1 = singular_values[0] * vh[0, :]

    ns2 = ns1 // d2
    folded2 = u[:, 0].reshape((ns2, d2), order="F")
    u_second, singular_values_second, vh_second = np.linalg.svd(
        folded2, full_matrices=False
    )
    mode2 = singular_values_second[0] * vh_second[0, :]
    mode3 = u_second[:, 0]
    folded2_reconstructed = np.outer(mode3, mode2)
    long_profile = folded2_reconstructed.reshape(-1, order="F")
    reconstructed = np.outer(long_profile, mode1).reshape(-1, order="F")
    if was_vector:
        result = reconstructed
    else:
        result = reconstructed.reshape(raw.shape)
    return result, mode1, mode2, mode3


def tuck2(
    concentrations: ArrayLike,
    n_matrices: int,
    shape: int,
    mode: int,
) -> tuple[FloatArray, FloatArray]:
    """Tucker interaction helper translated from ``tuck2.m``."""
    c = as_float_matrix(concentrations, "concentrations")
    if shape != 4:
        raise ValueError("tuck2 is defined only for shape=4")
    if mode not in (1, 3):
        raise ValueError("mode must be 1 or 3")
    nmat = int(n_matrices)
    n_total_rows, n_components = c.shape
    if nmat < 1 or n_total_rows % nmat:
        raise ValueError("row count must be divisible by n_matrices")
    n_rows = n_total_rows // nmat

    if mode == 1:
        blocks = [
            c[:, component].reshape((n_rows, nmat), order="F")
            for component in range(n_components)
        ]
        folded = np.hstack(blocks)
    else:
        blocks = [
            c[:, component].reshape((n_rows, nmat), order="F")
            for component in range(n_components)
        ]
        folded = np.vstack(blocks)

    u, singular_values, vh = np.linalg.svd(folded, full_matrices=False)
    totals = singular_values[0] * vh[0, :]
    if mode == 1:
        tnew = totals.reshape((nmat, n_components), order="F")
        cnew = np.kron(tnew, u[:, [0]])
    else:
        tnew = totals
        unew = np.column_stack(
            [u[i * n_rows : (i + 1) * n_rows, 0] for i in range(n_components)]
        )
        cnew = np.kron(tnew[:, None], unew)
    return np.asarray(cnew, dtype=np.float64), np.asarray(tnew, dtype=np.float64)


def wmat(
    correlation: ArrayLike,
    pure_indices: ArrayLike,
    rank: int,
    variable: int,
) -> FloatArray:
    """Build the determinant weight matrix used by ``pure.m``.

    Indices are zero-based in the Python API.
    """
    c = as_float_matrix(correlation, "correlation")
    imp = np.asarray(pure_indices, dtype=np.intp).reshape(-1)
    irank = int(rank)
    jvar = int(variable)
    if irank < 1 or imp.size < irank - 1:
        raise ValueError("pure_indices must contain rank-1 entries")
    dm = np.empty((irank, irank), dtype=np.float64)
    dm[0, 0] = c[jvar, jvar]
    for k in range(1, irank):
        kvar = int(imp[k - 1])
        dm[0, k] = c[jvar, kvar]
        dm[k, 0] = c[kvar, jvar]
        for kk in range(1, irank):
            kkvar = int(imp[kk - 1])
            dm[k, kk] = c[kvar, kkvar]
    return dm


def pure(
    data: ArrayLike, n_components: int, noise_percent: float
) -> tuple[FloatArray, NDArray[np.intp]]:
    """SIMPLISMA pure-variable estimates from ``pure.m`` without GUI pauses."""
    d = as_float_matrix(data, "data")
    n_rows, n_columns = d.shape
    rank = int(n_components)
    if rank < 1 or rank > n_columns:
        raise ValueError("n_components must be between 1 and the column count")

    standard_deviation = matlab_std(d, axis=0)
    mean = np.mean(d, axis=0)
    ll = standard_deviation**2 + mean**2
    noise = float(np.max(mean) * (noise_percent / 100.0))
    purity0 = standard_deviation / (mean + noise)
    length = np.sqrt(standard_deviation**2 + (mean + noise) ** 2)
    scaled = d / length[None, :]
    correlation = (scaled.T @ scaled) / n_rows

    weight0 = ll / (length**2)
    purity0 = weight0 * purity0
    indices = np.empty(rank, dtype=np.intp)
    indices[0] = int(np.argmax(purity0))
    for i in range(1, rank):
        weighted_purity = np.empty(n_columns, dtype=np.float64)
        for variable in range(n_columns):
            determinant = np.linalg.det(wmat(correlation, indices[:i], i + 1, variable))
            weighted_purity[variable] = purity0[variable] * determinant
        indices[i] = int(np.argmax(weighted_purity))

    selected = d[:, indices].T
    return normv2(selected), indices


@dataclass(frozen=True, slots=True)
class EFAResult:
    profiles: FloatArray | None
    forward: FloatArray
    backward: FloatArray


def efa(
    data: ArrayLike,
    n_rows: int | None = None,
    n_factors: int | None = None,
) -> EFAResult:
    """Headless evolving factor analysis translated from ``efa.m``."""
    d = as_float_matrix(data, "data")
    count = d.shape[0] if n_rows is None else int(n_rows)
    if count < 2 or count > d.shape[0]:
        raise ValueError("n_rows must be between 2 and the data row count")
    x = d[:count, :]
    width = min(x.shape)
    forward = np.zeros((count - 1, width), dtype=np.float64)
    backward = np.zeros_like(forward)
    reversed_x = x[::-1, :]
    for n in range(2, count + 1):
        values = np.linalg.svd(x[:n, :], compute_uv=False) ** 2
        forward[n - 2, : values.size] = values
        values = np.linalg.svd(reversed_x[:n, :], compute_uv=False) ** 2
        backward[n - 2, : values.size] = values

    profiles: FloatArray | None = None
    if n_factors is not None:
        factors = int(n_factors)
        if factors < 1 or factors > width:
            raise ValueError("n_factors must be between 1 and the EFA width")
        arranged = np.empty((count - 1, factors), dtype=np.float64)
        for factor in range(factors):
            backward_factor = factors - 1 - factor
            for row in range(count - 1):
                value = min(
                    forward[row, factor],
                    backward[count - 2 - row, backward_factor],
                )
                arranged[row, factor] = 1.0e-30 if value == 0.0 else value
        profiles = np.empty((count, factors), dtype=np.float64)
        profiles[0, :] = arranged[0, :]
        profiles[1:, :] = arranged

    return EFAResult(profiles=profiles, forward=forward, backward=backward)


def lof_trilinear(
    data: ArrayLike, mode1: ArrayLike, mode2: ArrayLike, mode3: ArrayLike
) -> tuple[float, float]:
    """R² and lack-of-fit percentages from ``loftril.m``."""
    cube = np.asarray(data, dtype=np.float64)
    u = np.asarray(mode1, dtype=np.float64)
    v = np.asarray(mode2, dtype=np.float64)
    t = np.asarray(mode3, dtype=np.float64)
    if cube.ndim == 2:
        calculated = u @ v.T
    elif cube.ndim == 3:
        calculated = np.einsum("ir,jr,kr->ijk", u, v, t)
    else:
        raise ValueError("data must be a matrix or three-way cube")
    residual = cube - calculated
    sum_data = float(np.sum(cube * cube))
    sum_residual = float(np.sum(residual * residual))
    lof = float(np.sqrt(sum_residual / sum_data) * 100.0)
    r_squared = float((sum_data - sum_residual) / sum_data * 100.0)
    return r_squared, lof


def lof_quadrilinear(
    data: ArrayLike,
    mode1: ArrayLike,
    mode2: ArrayLike,
    mode3: ArrayLike,
    mode4: ArrayLike,
) -> tuple[float, float]:
    """R² and lack-of-fit percentages from ``lofquadril.m``."""
    hypercube = np.asarray(data, dtype=np.float64)
    if hypercube.ndim != 4:
        raise ValueError("data must be a four-way hypercube")
    calculated = np.einsum(
        "ir,jr,kr,lr->ijkl",
        np.asarray(mode1, dtype=np.float64),
        np.asarray(mode2, dtype=np.float64),
        np.asarray(mode3, dtype=np.float64),
        np.asarray(mode4, dtype=np.float64),
    )
    finite = np.isfinite(hypercube)
    residual = hypercube[finite] - calculated[finite]
    sum_data = float(np.sum(hypercube[finite] ** 2))
    sum_residual = float(np.sum(residual**2))
    lof = float(np.sqrt(sum_residual / sum_data) * 100.0)
    r_squared = float((sum_data - sum_residual) / sum_data * 100.0)
    return r_squared, lof
