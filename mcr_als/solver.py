"""Alternating least-squares engine ported from ``alsOptimization.m``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize  # type: ignore[import-untyped]

from .correlation import CorrelationResult, RegressionStats, yregrnew
from .kinetics import KineticFitResult, opt_kinglob
from ._matlab import (
    as_float_matrix,
    fortran_assign,
    fortran_take,
    mldivide,
    mrdivide,
    normalize_mask,
)
from .helpers import (
    closure,
    fnnls,
    normv2,
    normv3,
    pcarep,
    quadril,
    trilin,
    tuck2,
    unimod,
)
from .options import (
    ClosureBlock,
    ClosureCondition,
    CorrelationOptions,
    KineticModelOptions,
    KineticOptions,
    MCRALSOptions,
    MultiExperimentOptions,
    NonnegativityOptions,
    TuckerOptions,
    UnimodalityOptions,
    ValueConstraint,
)
from .weighted import MLPCAResult, mlpca

FloatArray = NDArray[np.float64]


class UnsupportedFeatureError(NotImplementedError):
    """Backward-compatible exception name retained for earlier API clients."""


@dataclass(frozen=True, slots=True)
class IterationHistory:
    """Per-iteration values, including non-optimal/diverging iterations."""

    lack_of_fit_experimental: FloatArray
    r_squared_percent: FloatArray
    sigma_change_percent: FloatArray
    sigma_experimental: FloatArray
    sigma_pca: FloatArray
    spectra: FloatArray
    concentrations: FloatArray


@dataclass(frozen=True, slots=True)
class MCRALSResult:
    """Best-so-far result produced by :func:`mcr_als`."""

    concentrations: FloatArray
    spectra: FloatArray
    residual_pca: FloatArray
    residual_experimental: FloatArray
    lack_of_fit: FloatArray
    r_squared: float
    component_areas: FloatArray
    relative_areas: FloatArray
    iterations: int
    best_iteration: int
    status: Literal["converged", "diverged", "max_iterations"]
    initial_estimate_mode: Literal["concentrations", "spectra"]
    pca_reproduced_data: FloatArray
    pca_lack_of_fit: float
    pca_u: FloatArray
    pca_s: FloatArray
    pca_v: FloatArray
    kinetic_history: tuple[tuple[KineticFitResult, ...], ...]
    correlation_history: tuple[CorrelationResult, ...]
    weighted_preprocessing: MLPCAResult | None
    history: IterationHistory

    @property
    def reconstructed_data(self) -> FloatArray:
        return self.concentrations @ self.spectra


def _validate_options(options: MCRALSOptions) -> None:
    if options.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if options.tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if options.divergence_limit < 0:
        raise ValueError("divergence_limit must be non-negative")
    if options.normalization not in ("none", "maximum", "euclidean", "sum"):
        raise ValueError(f"unknown normalization: {options.normalization!r}")
    if options.kinetic is not None and not isinstance(options.kinetic, KineticOptions):
        raise TypeError("kinetic must be a KineticOptions instance or None")


def _validate_blocks(
    raw_blocks: Sequence[tuple[int, int]] | None,
    length: int,
    name: str,
) -> tuple[tuple[int, int], ...]:
    if raw_blocks is None:
        return ((0, length),)
    blocks = tuple((int(start), int(stop)) for start, stop in raw_blocks)
    if not blocks:
        raise ValueError(f"{name} must contain at least one block")
    expected = 0
    for start, stop in blocks:
        if start != expected or stop <= start or stop > length:
            raise ValueError(
                f"{name} must be contiguous half-open intervals covering 0:{length}"
            )
        expected = stop
    if expected != length:
        raise ValueError(f"{name} must cover all {length} entries")
    return blocks


def _partitions(
    options: MultiExperimentOptions,
    data_shape: tuple[int, int],
    n_components: int,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    NDArray[np.bool_],
]:
    rows = _validate_blocks(options.row_blocks, data_shape[0], "row_blocks")
    columns = _validate_blocks(options.column_blocks, data_shape[1], "column_blocks")
    presence = normalize_mask(
        options.presence,
        (len(rows), n_components),
        default=True,
        name="presence",
    )
    return rows, columns, presence


def _infer_initial_estimate(
    data: FloatArray, estimate: ArrayLike
) -> tuple[FloatArray, FloatArray, Literal["concentrations", "spectra"]]:
    initial = as_float_matrix(estimate, "initial_estimate")
    n_rows, n_columns = data.shape
    initial_rows, initial_columns = initial.shape
    concentrations: FloatArray | None = None
    spectra: FloatArray | None = None
    mode: Literal["concentrations", "spectra"] | None = None

    # These are deliberately independent and ordered like lines 49-53 of
    # alsOptimization.m.  Consequently a square ambiguous estimate follows
    # MATLAB and the last matching spectral orientation wins.
    if initial_rows == n_rows:
        concentrations = initial.copy()
        mode = "concentrations"
    if initial_columns == n_rows:
        concentrations = initial.T.copy()
        mode = "concentrations"
    if initial_columns == n_columns:
        spectra = initial.copy()
        mode = "spectra"
    if initial_rows == n_columns:
        spectra = initial.T.copy()
        mode = "spectra"

    if mode is None:
        raise ValueError(
            "initial_estimate does not match either data dimension; expected "
            "(rows, components), (components, rows), (components, columns), "
            "or (columns, components)"
        )
    if mode == "concentrations":
        assert concentrations is not None
        spectra = np.asarray(mldivide(concentrations, data), dtype=np.float64)
    else:
        assert spectra is not None
        concentrations = np.asarray(mrdivide(data, spectra), dtype=np.float64)
    return concentrations, spectra, mode


def _component_block_mask(
    mask: ArrayLike | None,
    n_blocks: int,
    n_components: int,
    *,
    spectra: bool,
    name: str,
) -> NDArray[np.bool_]:
    block_component = normalize_mask(
        mask,
        (n_blocks, n_components),
        default=True,
        name=name,
    )
    return block_component.T if spectra else block_component


def _apply_nonnegative_concentrations(
    concentrations: FloatArray,
    spectra: FloatArray,
    reproduced_data: FloatArray,
    blocks: tuple[tuple[int, int], ...],
    config: NonnegativityOptions,
) -> FloatArray:
    if not config.enabled:
        return concentrations
    if config.algorithm not in ("truncate", "nnls", "fnnls"):
        raise ValueError(f"unknown concentration NNLS algorithm: {config.algorithm!r}")
    result = concentrations.copy()
    n_components = result.shape[1]
    mask = _component_block_mask(
        config.mask,
        len(blocks),
        n_components,
        spectra=False,
        name="nonnegativity_c.mask",
    )
    for block_index, (start, stop) in enumerate(blocks):
        selected = mask[block_index, :]
        if config.algorithm == "truncate":
            block = result[start:stop, :]
            constrained = block[:, selected]
            constrained[constrained < 0.0] = 0.0
            block[:, selected] = constrained
            result[start:stop, :] = block
        elif np.all(selected):
            if config.algorithm == "nnls":
                for row in range(start, stop):
                    solution, _ = optimize.nnls(spectra.T, reproduced_data[row, :])
                    result[row, :] = solution
            else:
                gram = spectra @ spectra.T
                for row in range(start, stop):
                    solution, _ = fnnls(gram, spectra @ reproduced_data[row, :].T)
                    result[row, :] = solution
        # MATLAB applies the NNLS/FNNLS algorithms only when every component in
        # the block is selected.  Partial masks are intentionally left alone.
    return result


def _apply_nonnegative_spectra(
    spectra: FloatArray,
    concentrations: FloatArray,
    reproduced_data: FloatArray,
    blocks: tuple[tuple[int, int], ...],
    config: NonnegativityOptions,
) -> FloatArray:
    if not config.enabled:
        return spectra
    if config.algorithm not in ("truncate", "nnls", "fnnls"):
        raise ValueError(f"unknown spectral NNLS algorithm: {config.algorithm!r}")
    result = spectra.copy()
    n_components = result.shape[0]
    mask = _component_block_mask(
        config.mask,
        len(blocks),
        n_components,
        spectra=True,
        name="nonnegativity_s.mask",
    )
    for block_index, (start, stop) in enumerate(blocks):
        selected = mask[:, block_index]
        if config.algorithm == "truncate":
            block = result[:, start:stop]
            constrained = block[selected, :]
            constrained[constrained < 0.0] = 0.0
            block[selected, :] = constrained
            result[:, start:stop] = block
        elif np.all(selected):
            if config.algorithm == "nnls":
                for column in range(start, stop):
                    solution, _ = optimize.nnls(
                        concentrations, reproduced_data[:, column]
                    )
                    result[:, column] = solution
            else:
                gram = concentrations.T @ concentrations
                for column in range(start, stop):
                    solution, _ = fnnls(
                        gram, concentrations.T @ reproduced_data[:, column]
                    )
                    result[:, column] = solution
    return result


def _apply_unimodality_concentrations(
    concentrations: FloatArray,
    blocks: tuple[tuple[int, int], ...],
    config: UnimodalityOptions,
) -> FloatArray:
    if not config.enabled:
        return concentrations
    result = concentrations.copy()
    mask = _component_block_mask(
        config.mask,
        len(blocks),
        result.shape[1],
        spectra=False,
        name="unimodality_c.mask",
    )
    for block_index, (start, stop) in enumerate(blocks):
        for component in range(result.shape[1]):
            if mask[block_index, component]:
                result[start:stop, component] = unimod(
                    result[start:stop, component],
                    config.tolerance,
                    config.mode,
                )
    return result


def _apply_unimodality_spectra(
    spectra: FloatArray,
    blocks: tuple[tuple[int, int], ...],
    config: UnimodalityOptions,
) -> FloatArray:
    if not config.enabled:
        return spectra
    result = spectra.copy()
    mask = _component_block_mask(
        config.mask,
        len(blocks),
        result.shape[0],
        spectra=True,
        name="unimodality_s.mask",
    )
    for block_index, (start, stop) in enumerate(blocks):
        for component in range(result.shape[0]):
            if mask[component, block_index]:
                result[component, start:stop] = unimod(
                    result[component, start:stop],
                    config.tolerance,
                    config.mode,
                )
    return result


_CLOSURE_KIND = {"equality": 1, "least_squares": 2, "lower_equal": 3}


def _closure_target(
    condition: ClosureCondition,
    start: int,
    stop: int,
    total_length: int,
) -> tuple[float, FloatArray | None]:
    raw = np.asarray(condition.target, dtype=np.float64)
    if raw.size == 1:
        return float(raw.reshape(-1)[0]), None
    vector = raw.reshape(-1)
    if vector.size == total_length:
        vector = vector[start:stop]
    elif vector.size != stop - start:
        raise ValueError(
            "closure vector target must match its block or the complete mode"
        )
    return 0.0, vector


def _closure_blocks(
    configured: Sequence[ClosureBlock],
    count: int,
    n_components: int,
) -> tuple[ClosureBlock, ...]:
    if not configured:
        default = ClosureBlock(
            ClosureCondition(
                components=np.ones(n_components, dtype=bool),
                kind="equality",
                target=1.0,
            )
        )
        return tuple(default for _ in range(count))
    blocks = tuple(configured)
    if len(blocks) == 1 and count > 1:
        return tuple(blocks[0] for _ in range(count))
    if len(blocks) != count:
        raise ValueError(f"closure.blocks must contain 1 or {count} configurations")
    return blocks


def _apply_closure(
    values: FloatArray,
    *,
    concentration_mode: bool,
    partitions: tuple[tuple[int, int], ...],
    configured: Sequence[ClosureBlock],
) -> FloatArray:
    n_components = values.shape[1] if concentration_mode else values.shape[0]
    total_length = values.shape[0] if concentration_mode else values.shape[1]
    blocks = _closure_blocks(configured, len(partitions), n_components)
    result = values.copy()
    for (start, stop), block in zip(partitions, blocks):
        local = result[start:stop, :] if concentration_mode else result[:, start:stop].T
        first_selection = np.asarray(block.first.components, dtype=bool).reshape(-1)
        if first_selection.size != n_components:
            raise ValueError("closure component selection has the wrong length")
        if block.first.kind not in _CLOSURE_KIND:
            raise ValueError(f"unknown closure kind: {block.first.kind!r}")
        target1, vector1 = _closure_target(block.first, start, stop, total_length)
        if block.second is None:
            local = closure(
                local,
                1,
                first_selection,
                _CLOSURE_KIND[block.first.kind],
                target1,
                vector1=vector1,
            )
        else:
            second_selection = np.asarray(block.second.components, dtype=bool).reshape(
                -1
            )
            if second_selection.size != n_components:
                raise ValueError("second closure selection has the wrong length")
            if block.second.kind not in _CLOSURE_KIND:
                raise ValueError(f"unknown closure kind: {block.second.kind!r}")
            target2, vector2 = _closure_target(block.second, start, stop, total_length)
            local = closure(
                local,
                2,
                first_selection,
                _CLOSURE_KIND[block.first.kind],
                target1,
                target2,
                second_selection,
                _CLOSURE_KIND[block.second.kind],
                vector1,
                vector2,
            )
        if concentration_mode:
            result[start:stop, :] = local
        else:
            result[:, start:stop] = local.T
    return result


def _constraint_indices(
    constraint: ValueConstraint, shape: tuple[int, int]
) -> NDArray[np.intp]:
    values = np.asarray(constraint.values)
    if constraint.indices is None:
        if values.size == 1:
            return np.arange(np.prod(shape), dtype=np.intp)
        try:
            broadcast = np.broadcast_to(values, shape)
        except ValueError as exc:
            raise ValueError("constraint values cannot be broadcast to target") from exc
        if np.issubdtype(broadcast.dtype, np.floating) and np.any(
            ~np.isfinite(broadcast)
        ):
            return np.flatnonzero(np.isfinite(broadcast).ravel(order="F"))
        return np.arange(np.prod(shape), dtype=np.intp)

    raw = np.asarray(constraint.indices)
    if raw.dtype == np.bool_:
        if raw.shape != shape:
            raise ValueError("boolean constraint indices must match target shape")
        return np.flatnonzero(raw.ravel(order="F"))
    indices = np.asarray(raw, dtype=np.intp).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= np.prod(shape)):
        raise IndexError("constraint linear index is out of range")
    return indices


def _apply_value_constraint(
    target: FloatArray, constraint: ValueConstraint | None
) -> FloatArray:
    if constraint is None:
        return target
    if constraint.kind not in ("equal", "upper", "lower"):
        raise ValueError(f"unknown value constraint kind: {constraint.kind!r}")
    indices = _constraint_indices(constraint, (target.shape[0], target.shape[1]))
    raw_values = np.asarray(constraint.values, dtype=np.float64)
    if raw_values.size == 1:
        selected_values: ArrayLike = float(raw_values.reshape(-1)[0])
    else:
        try:
            reference = np.broadcast_to(raw_values, target.shape)
        except ValueError as exc:
            raise ValueError("constraint values cannot be broadcast to target") from exc
        selected_values = fortran_take(reference, indices)

    current = fortran_take(target, indices)
    values_array = np.broadcast_to(selected_values, current.shape)
    if constraint.kind == "equal":
        replacement = values_array
    elif constraint.kind == "upper":
        replacement = np.minimum(current, values_array)
    else:
        replacement = np.maximum(current, values_array)
    return fortran_assign(target, indices, replacement)


def _apply_correlation(
    concentrations: FloatArray,
    blocks: tuple[tuple[int, int], ...],
    config: CorrelationOptions | None,
) -> tuple[FloatArray, CorrelationResult | None]:
    if config is None or not config.enabled:
        return concentrations, None
    if config.reference is None:
        raise ValueError("correlation.reference is required")
    if config.model not in ("global", "local"):
        raise ValueError("correlation.model must be 'global' or 'local'")

    reference = np.asarray(config.reference, dtype=np.float64)
    if reference.shape != concentrations.shape:
        raise ValueError("correlation.reference must match concentrations")
    component_mask = normalize_mask(
        config.component_mask,
        (concentrations.shape[1],),
        default=True,
        name="correlation.component_mask",
    )
    if config.model == "global" or len(blocks) == 1:
        result = yregrnew(concentrations, reference, component_mask)
        return result.output, result

    output = concentrations.copy()
    calibrated = concentrations.copy()
    stats_rows: list[tuple[RegressionStats | None, ...]] = []
    block_results: list[CorrelationResult] = []
    for start, stop in blocks:
        result = yregrnew(
            concentrations[start:stop, :],
            reference[start:stop, :],
            component_mask,
        )
        block_results.append(result)
        output[start:stop, :] = result.output
        calibrated[start:stop, :] = result.calibrated
        stats_rows.append(result.stats[0])

    if config.matrix_effect:
        reference_stats = block_results[0].stats[0]
        for block_index, (start, stop) in enumerate(blocks[1:], start=1):
            current_stats = block_results[block_index].stats[0]
            for component in range(concentrations.shape[1]):
                base = reference_stats[component]
                current = current_stats[component]
                if base is None or current is None or base.slope == 0.0:
                    continue
                output[start:stop, component] = (
                    output[start:stop, component] * current.slope
                    + current.offset
                    - base.offset
                ) / base.slope
        output[reference == 0.0] = 0.0

    aggregate = CorrelationResult(
        output=output.copy(),
        calibrated=calibrated,
        stats=tuple(stats_rows),
    )
    return output, aggregate


def _normalize_correlation_spectra(
    spectra: FloatArray, component_mask: ArrayLike | None
) -> FloatArray:
    mask = normalize_mask(
        component_mask,
        (spectra.shape[0],),
        default=True,
        name="correlation.component_mask",
    )
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return spectra
    reference_maximum = np.max(spectra[selected[-1], :])
    result = spectra.copy()
    for component in np.flatnonzero(~mask):
        maximum = np.max(result[component, :])
        if maximum != 0.0:
            result[component, :] *= reference_maximum / maximum
    return result


def _apply_tucker(
    concentrations: FloatArray,
    row_blocks: tuple[tuple[int, int], ...],
    config: TuckerOptions | None,
) -> FloatArray:
    if config is None or not config.enabled:
        return concentrations
    n_matrices = (
        len(row_blocks) if config.n_matrices is None else int(config.n_matrices)
    )
    if n_matrices < 1 or concentrations.shape[0] % n_matrices:
        raise ValueError("Tucker n_matrices must divide the concentration rows")
    result = concentrations.copy()
    for mode_config in config.modes:
        if mode_config.mode not in (1, 3):
            raise ValueError("Tucker mode must be 1 or 3")
        groups = np.asarray(mode_config.groups, dtype=np.intp).reshape(-1)
        if groups.size != concentrations.shape[1]:
            raise ValueError("Tucker groups must match the component count")
        for group in np.unique(groups[groups >= 0]):
            components = np.flatnonzero(groups == group)
            constrained, _ = tuck2(
                result[:, components],
                n_matrices,
                4,
                mode_config.mode,
            )
            result[:, components] = constrained
    return result


def _kinetic_experiment_indices(
    model: KineticModelOptions,
    row_blocks: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    mask = normalize_mask(
        model.experiment_mask,
        (len(row_blocks),),
        default=True,
        name="kinetic.experiment_mask",
    )
    selected = tuple(int(index) for index in np.flatnonzero(mask))
    if not selected:
        raise ValueError("a kinetic model must select at least one experiment")
    return selected


def _validate_kinetic_times(
    vectors: Sequence[ArrayLike],
    selected: tuple[int, ...],
    row_blocks: tuple[tuple[int, int], ...],
) -> tuple[FloatArray, ...]:
    if len(vectors) != len(selected):
        raise ValueError("kinetic time vectors do not match selected experiments")
    output: list[FloatArray] = []
    for raw, block_index in zip(vectors, selected):
        vector = np.asarray(raw, dtype=np.float64).reshape(-1)
        start, stop = row_blocks[block_index]
        if vector.size != stop - start:
            raise ValueError(
                "each kinetic time vector must match its experiment row block"
            )
        output.append(vector)
    return tuple(output)


def _kinetic_time_vectors(
    raw_time: ArrayLike | Sequence[ArrayLike],
    selected: tuple[int, ...],
    row_blocks: tuple[tuple[int, int], ...],
    total_rows: int,
) -> tuple[FloatArray, ...]:
    numeric: FloatArray | None
    try:
        numeric = np.asarray(raw_time, dtype=np.float64)
    except (TypeError, ValueError):
        numeric = None

    if numeric is not None and (numeric.ndim == 1 or 1 in numeric.shape):
        vector = numeric.reshape(-1)
        if vector.size == total_rows:
            return tuple(
                vector[row_blocks[index][0] : row_blocks[index][1]].copy()
                for index in selected
            )
        lengths = tuple(
            row_blocks[index][1] - row_blocks[index][0] for index in selected
        )
        if lengths and all(length == vector.size for length in lengths):
            return tuple(vector.copy() for _ in selected)
        raise ValueError(
            "kinetic time must cover all rows or match every selected row block"
        )

    if numeric is not None and numeric.ndim == 2:
        if numeric.shape[0] == len(row_blocks):
            return _validate_kinetic_times(
                tuple(numeric[index, :] for index in selected), selected, row_blocks
            )
        if numeric.shape[0] == len(selected):
            return _validate_kinetic_times(tuple(numeric), selected, row_blocks)
        if numeric.shape[1] == len(row_blocks):
            return _validate_kinetic_times(
                tuple(numeric[:, index] for index in selected), selected, row_blocks
            )
        if numeric.shape[1] == len(selected):
            return _validate_kinetic_times(
                tuple(numeric[:, index] for index in range(len(selected))),
                selected,
                row_blocks,
            )

    if not isinstance(raw_time, Sequence):
        raise ValueError("kinetic time must be a vector or sequence of vectors")
    vectors = tuple(raw_time)
    if len(vectors) == len(row_blocks):
        vectors = tuple(vectors[index] for index in selected)
    return _validate_kinetic_times(vectors, selected, row_blocks)


def _kinetic_initial_concentrations(
    raw_initial: ArrayLike,
    selected: tuple[int, ...],
    n_experiments: int,
    n_species: int,
) -> FloatArray:
    initial = np.asarray(raw_initial, dtype=np.float64)
    if initial.ndim == 1:
        if initial.size != n_species:
            raise ValueError("kinetic initial concentrations must match species")
        return np.tile(initial, (len(selected), 1))
    if initial.ndim != 2 or initial.shape[1] != n_species:
        raise ValueError(
            "kinetic initial concentrations must be experiments by species"
        )
    if initial.shape[0] == n_experiments:
        return initial[np.asarray(selected, dtype=np.intp), :].copy()
    if initial.shape[0] == len(selected):
        return initial.copy()
    raise ValueError("kinetic initial concentrations do not match experiments")


def _apply_kinetic(
    concentrations: FloatArray,
    row_blocks: tuple[tuple[int, int], ...],
    config: KineticOptions | None,
    current_rates: Sequence[FloatArray],
) -> tuple[FloatArray, tuple[KineticFitResult, ...], tuple[FloatArray, ...]]:
    if config is None or not config.enabled:
        return concentrations, (), tuple(current_rates)
    if len(current_rates) != len(config.models):
        raise ValueError("kinetic rate-state count does not match model count")

    result = concentrations.copy()
    fits: list[KineticFitResult] = []
    updated_rates: list[FloatArray] = []
    for model_index, model in enumerate(config.models):
        if not isinstance(model, KineticModelOptions):
            raise TypeError("kinetic models must be KineticModelOptions instances")
        orders = np.asarray(model.reaction_orders, dtype=np.float64)
        stoichiometry = np.asarray(model.stoichiometry, dtype=np.float64)
        if orders.ndim != 2:
            raise ValueError("kinetic reaction_orders must be two-dimensional")
        n_reactions, n_species = orders.shape
        if stoichiometry.shape != (n_species, n_reactions):
            raise ValueError("kinetic stoichiometry must be species by reactions")

        raw_mapping = np.asarray(model.component_mapping, dtype=np.float64).reshape(-1)
        if raw_mapping.size != result.shape[1] or not np.all(np.isfinite(raw_mapping)):
            raise ValueError("kinetic component_mapping must match ALS components")
        mapping = np.rint(raw_mapping).astype(np.intp)
        if not np.array_equal(raw_mapping, mapping.astype(np.float64)):
            raise ValueError("kinetic component_mapping entries must be integers")
        if np.any(mapping >= n_species):
            raise ValueError("kinetic component_mapping contains an unknown species")
        mapped_components = np.flatnonzero(mapping >= 0)
        mapped_species = mapping[mapped_components]
        if np.unique(mapped_species).size != mapped_species.size:
            raise ValueError("each kinetic species can map to only one ALS component")
        colored = normalize_mask(
            model.colored_mask,
            (n_species,),
            default=True,
            name="kinetic.colored_mask",
        )
        if not np.array_equal(np.sort(mapped_species), np.flatnonzero(colored)):
            raise ValueError(
                "kinetic component_mapping must map every colored species exactly once"
            )

        selected = _kinetic_experiment_indices(model, row_blocks)
        times = _kinetic_time_vectors(
            model.time, selected, row_blocks, concentrations.shape[0]
        )
        initial = _kinetic_initial_concentrations(
            model.initial_concentrations,
            selected,
            len(row_blocks),
            n_species,
        )
        observed: list[FloatArray] = []
        for block_index in selected:
            start, stop = row_blocks[block_index]
            profile = np.zeros((stop - start, n_species), dtype=np.float64)
            for component, species in zip(mapped_components, mapped_species):
                profile[:, species] = result[start:stop, component]
            observed.append(profile)

        optimized = opt_kinglob(
            observed,
            stoichiometry,
            orders,
            initial,
            current_rates[model_index],
            times,
            model.choose,
            colored,
        )
        updated_rates.append(optimized.rate_constants.copy())
        for fitted, block_index in zip(optimized.concentrations, selected):
            start, stop = row_blocks[block_index]
            for component, species in zip(mapped_components, mapped_species):
                result[start:stop, component] = fitted[:, species]

        degrees_of_freedom = (
            sum(profile.size for profile in optimized.concentrations)
            - optimized.rate_constants.size
        )
        if degrees_of_freedom > 0:
            residual_sigma = np.sqrt(
                optimized.sum_squared_residuals / degrees_of_freedom
            )
            information = optimized.jacobian.T @ optimized.jacobian
            try:
                covariance = np.linalg.inv(information)
            except np.linalg.LinAlgError:
                covariance = np.linalg.pinv(information)
            standard_errors = residual_sigma * np.sqrt(
                np.maximum(np.diag(covariance), 0.0)
            )
        else:
            standard_errors = np.full(
                optimized.rate_constants.shape, np.nan, dtype=np.float64
            )
        fits.append(
            KineticFitResult(
                rate_constants=optimized.rate_constants.copy(),
                concentrations=tuple(
                    profile.copy() for profile in optimized.concentrations
                ),
                sum_squared_residuals=optimized.sum_squared_residuals,
                jacobian=optimized.jacobian.copy(),
                standard_errors=np.asarray(standard_errors, dtype=np.float64),
            )
        )
    return result, tuple(fits), tuple(updated_rates)


def _apply_trilinearity_concentrations(
    concentrations: FloatArray,
    n_row_blocks: int,
    options: MCRALSOptions,
    total_concentrations: FloatArray,
) -> FloatArray:
    config = options.trilinearity
    if not config.enabled or config.direction not in ("concentration", "both"):
        return concentrations
    if concentrations.shape[0] % n_row_blocks:
        raise ValueError(
            "trilinearity requires equal-length concentration experiment blocks"
        )
    selected = normalize_mask(
        config.component_mask,
        (concentrations.shape[1],),
        default=True,
        name="trilinearity.component_mask",
    )
    result = concentrations.copy()
    for component in range(result.shape[1]):
        if not selected[component]:
            continue
        if config.quadrilinear_dimensions is None:
            result[:, component], totals = trilin(
                result[:, component], n_row_blocks, config.shape
            )
            total_concentrations[component, :] = totals
        else:
            dimensions = config.quadrilinear_dimensions
            result[:, component], _, _, _ = quadril(
                result[:, component], *dimensions, config.shape
            )
    return result


def _apply_trilinearity_spectra(
    spectra: FloatArray,
    n_column_blocks: int,
    options: MCRALSOptions,
) -> FloatArray:
    config = options.trilinearity
    if not config.enabled or config.direction not in ("spectra", "both"):
        return spectra
    if spectra.shape[1] % n_column_blocks:
        raise ValueError(
            "trilinearity requires equal-length spectral experiment blocks"
        )
    selected = normalize_mask(
        config.component_mask,
        (spectra.shape[0],),
        default=True,
        name="trilinearity.component_mask",
    )
    result = spectra.copy()
    for component in range(result.shape[0]):
        if selected[component]:
            profile, _ = trilin(result[component, :], n_column_blocks, config.shape)
            result[component, :] = profile
    return result


def _normalize_spectra(spectra: FloatArray, mode: str) -> FloatArray:
    if mode == "none":
        return spectra
    if mode == "maximum":
        with np.errstate(divide="ignore", invalid="ignore"):
            return spectra / np.max(spectra, axis=1)[:, None]
    if mode == "euclidean":
        return normv2(spectra)
    if mode == "sum":
        return normv3(spectra)
    raise ValueError(f"unknown normalization: {mode!r}")


def mcr_als(
    data: ArrayLike,
    initial_estimate: ArrayLike,
    options: MCRALSOptions | None = None,
) -> MCRALSResult:
    """Run MCR-ALS with MATLAB-compatible update and constraint ordering.

    Calculations are float64 and the returned solution is the best improving
    iteration, as in ``alsOptimization.m``. The desktop GUI calls this same
    public engine.
    """
    config = MCRALSOptions() if options is None else options
    if not isinstance(config, MCRALSOptions):
        raise TypeError("options must be an MCRALSOptions instance or None")
    _validate_options(config)
    original_data = as_float_matrix(data, "data")
    weighted_preprocessing: MLPCAResult | None = None
    weighted = config.weighted
    if weighted is not None and weighted.enabled:
        if weighted.standard_deviations is None:
            raise ValueError("weighted.standard_deviations is required")
        initial_shape = np.asarray(initial_estimate).shape
        if len(initial_shape) != 2:
            raise ValueError("initial_estimate must be two-dimensional")
        weighted_preprocessing = mlpca(
            original_data,
            weighted.standard_deviations,
            min(initial_shape),
            weighted.convergence_limit,
            weighted.max_iterations,
        )
        experimental = weighted_preprocessing.reconstructed
    else:
        experimental = original_data
    concentrations, spectra, initial_mode = _infer_initial_estimate(
        experimental, initial_estimate
    )
    n_components = spectra.shape[0]
    if concentrations.shape != (experimental.shape[0], n_components):
        raise ValueError("initial estimate produced inconsistent component dimensions")
    if n_components < 1 or n_components > min(experimental.shape):
        raise ValueError(
            "the inferred component count must not exceed the data matrix rank"
        )

    row_blocks, column_blocks, presence = _partitions(
        config.multi,
        (experimental.shape[0], experimental.shape[1]),
        n_components,
    )
    pca_u, pca_s, pca_v, reproduced, pca_lof = pcarep(experimental, n_components)
    sum_squares_experimental = float(np.sum(experimental * experimental))
    sum_squares_pca = float(np.sum(reproduced * reproduced))
    best_sigma = float(np.sqrt(sum_squares_experimental))
    total_concentrations = np.ones((n_components, len(row_blocks)), dtype=np.float64)
    relative = np.ones_like(total_concentrations)

    history_lof: list[float] = []
    history_r2: list[float] = []
    history_change: list[float] = []
    history_sigma_experimental: list[float] = []
    history_sigma_pca: list[float] = []
    history_spectra: list[FloatArray] = []
    history_concentrations: list[FloatArray] = []
    correlation_history: list[CorrelationResult] = []
    kinetic_history: list[tuple[KineticFitResult, ...]] = []
    kinetic_config = config.kinetic
    if kinetic_config is not None and kinetic_config.enabled:
        kinetic_rates = tuple(
            np.asarray(model.initial_rate_constants, dtype=np.float64).reshape(-1)
            for model in kinetic_config.models
        )
    else:
        kinetic_rates = ()

    best_concentrations: FloatArray | None = None
    best_spectra: FloatArray | None = None
    best_residual_pca: FloatArray | None = None
    best_residual_experimental: FloatArray | None = None
    best_lof: FloatArray | None = None
    best_areas: FloatArray | None = None
    best_relative: FloatArray | None = None
    best_r_squared = float("nan")
    best_iteration = 0
    divergence_count = 0
    status: Literal["converged", "diverged", "max_iterations"] = "max_iterations"
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        iterations = iteration

        # Concentration least-squares estimate and constraints.
        concentrations = np.asarray(mrdivide(reproduced, spectra), dtype=np.float64)
        concentrations = _apply_nonnegative_concentrations(
            concentrations,
            spectra,
            reproduced,
            row_blocks,
            config.nonnegativity_c,
        )
        concentrations = _apply_trilinearity_concentrations(
            concentrations,
            len(row_blocks),
            config,
            total_concentrations,
        )
        concentrations = _apply_tucker(concentrations, row_blocks, config.tucker)
        for block_index, (start, stop) in enumerate(row_blocks):
            concentrations[start:stop, ~presence[block_index, :]] = 0.0
        concentrations = _apply_unimodality_concentrations(
            concentrations, row_blocks, config.unimodality_c
        )
        if config.closure.enabled and config.closure.mode == "concentration":
            concentrations = _apply_closure(
                concentrations,
                concentration_mode=True,
                partitions=row_blocks,
                configured=config.closure.blocks,
            )
        concentrations = _apply_value_constraint(
            concentrations, config.concentration_values
        )
        concentrations, correlation_result = _apply_correlation(
            concentrations, row_blocks, config.correlation
        )
        if correlation_result is not None:
            correlation_history.append(correlation_result)
        concentrations, kinetic_fits, kinetic_rates = _apply_kinetic(
            concentrations, row_blocks, kinetic_config, kinetic_rates
        )
        if kinetic_fits:
            kinetic_history.append(kinetic_fits)

        # MATLAB recomputes all areas on iteration one.  With a trilinear model,
        # subsequent constrained components retain the totals returned by
        # trilin.m while unconstrained values keep their previous areas.
        if not config.trilinearity.enabled or iteration == 1:
            for component in range(n_components):
                for block_index, (start, stop) in enumerate(row_blocks):
                    total_concentrations[component, block_index] = np.sum(
                        concentrations[start:stop, component]
                    )
        for component in range(n_components):
            if total_concentrations[component, 0] > 0.0:
                relative[component, :] = (
                    total_concentrations[component, :]
                    / total_concentrations[component, 0]
                )
            else:
                relative[component, :] = total_concentrations[component, :]

        # Spectral least-squares estimate and constraints.
        spectra = np.asarray(mldivide(concentrations, reproduced), dtype=np.float64)
        spectra = _apply_nonnegative_spectra(
            spectra,
            concentrations,
            reproduced,
            column_blocks,
            config.nonnegativity_s,
        )
        spectra = _apply_trilinearity_spectra(spectra, len(column_blocks), config)
        spectra = _apply_unimodality_spectra(
            spectra, column_blocks, config.unimodality_s
        )
        spectra = _apply_value_constraint(spectra, config.spectral_values)
        if config.closure.enabled and config.closure.mode == "spectra":
            spectra = _apply_closure(
                spectra,
                concentration_mode=False,
                partitions=column_blocks,
                configured=config.closure.blocks,
            )
        if not config.closure.enabled:
            correlation = config.correlation
            if (
                correlation is not None
                and correlation.enabled
                and correlation.normalize_spectra
            ):
                spectra = _normalize_correlation_spectra(
                    spectra, correlation.component_mask
                )
            else:
                spectra = _normalize_spectra(spectra, config.normalization)

        calculated = concentrations @ spectra
        residual_pca = reproduced - calculated
        residual_experimental = experimental - calculated
        ss_residual_pca = float(np.sum(residual_pca * residual_pca))
        ss_residual_experimental = float(
            np.sum(residual_experimental * residual_experimental)
        )
        sigma_pca = float(
            np.sqrt(ss_residual_pca / (experimental.shape[0] * experimental.shape[1]))
        )
        sigma_experimental = float(
            np.sqrt(
                ss_residual_experimental
                / (experimental.shape[0] * experimental.shape[1])
            )
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            change = float(np.divide(best_sigma - sigma_pca, sigma_pca) * 100.0)
        lack_of_fit = np.array(
            [
                np.sqrt(ss_residual_pca / sum_squares_pca) * 100.0,
                np.sqrt(ss_residual_experimental / sum_squares_experimental) * 100.0,
            ],
            dtype=np.float64,
        )
        r_squared = float(
            (sum_squares_experimental - ss_residual_experimental)
            / sum_squares_experimental
        )

        history_lof.append(float(lack_of_fit[1]))
        history_r2.append(100.0 * r_squared)
        history_change.append(change)
        history_sigma_experimental.append(sigma_experimental)
        history_sigma_pca.append(sigma_pca)
        history_spectra.append(spectra.copy())
        history_concentrations.append(concentrations.copy())

        if change < 0.0:
            divergence_count += 1
        else:
            divergence_count = 0

        if change > 0.0 or iteration == 1:
            best_sigma = sigma_pca
            best_concentrations = concentrations.copy()
            best_spectra = spectra.copy()
            best_residual_pca = residual_pca.copy()
            best_residual_experimental = residual_experimental.copy()
            best_lof = lack_of_fit.copy()
            best_areas = total_concentrations.copy()
            best_relative = relative.T.copy()
            best_r_squared = r_squared
            best_iteration = iteration

        if abs(change) < config.tolerance:
            status = "converged"
            break
        if divergence_count > config.divergence_limit:
            status = "diverged"
            break

    assert best_concentrations is not None
    assert best_spectra is not None
    assert best_residual_pca is not None
    assert best_residual_experimental is not None
    assert best_lof is not None
    assert best_areas is not None
    assert best_relative is not None

    history = IterationHistory(
        lack_of_fit_experimental=np.asarray(history_lof, dtype=np.float64),
        r_squared_percent=np.asarray(history_r2, dtype=np.float64),
        sigma_change_percent=np.asarray(history_change, dtype=np.float64),
        sigma_experimental=np.asarray(history_sigma_experimental, dtype=np.float64),
        sigma_pca=np.asarray(history_sigma_pca, dtype=np.float64),
        spectra=np.vstack(history_spectra),
        concentrations=np.hstack(history_concentrations),
    )
    return MCRALSResult(
        concentrations=best_concentrations,
        spectra=best_spectra,
        residual_pca=best_residual_pca,
        residual_experimental=best_residual_experimental,
        lack_of_fit=best_lof,
        r_squared=best_r_squared,
        component_areas=best_areas,
        relative_areas=best_relative,
        iterations=iterations,
        best_iteration=best_iteration,
        status=status,
        initial_estimate_mode=initial_mode,
        pca_reproduced_data=reproduced,
        pca_lack_of_fit=pca_lof,
        pca_u=pca_u,
        pca_s=pca_s,
        pca_v=pca_v,
        weighted_preprocessing=weighted_preprocessing,
        correlation_history=tuple(correlation_history),
        history=history,
        kinetic_history=tuple(kinetic_history),
    )
