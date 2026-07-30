"""Configuration dataclasses for :func:`mcr_als.mcr_als`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike

NonnegativeAlgorithm = Literal["truncate", "nnls", "fnnls"]
UnimodalityMode = Literal[0, 1, 2]
ClosureKind = Literal["equality", "least_squares", "lower_equal"]
ClosureMode = Literal["concentration", "spectra"]
Normalization = Literal["none", "maximum", "euclidean", "sum"]
ConstraintKind = Literal["equal", "upper", "lower"]
TrilinearityDirection = Literal["concentration", "spectra", "both"]
CorrelationModel = Literal["global", "local"]


@dataclass(slots=True)
class NonnegativityOptions:
    """Non-negativity constraint for one ALS mode.

    ``mask`` is component-by-block for spectra and block-by-component for
    concentrations.  A one-dimensional component mask is broadcast.
    """

    enabled: bool = False
    algorithm: NonnegativeAlgorithm = "truncate"
    mask: ArrayLike | None = None


@dataclass(slots=True)
class UnimodalityOptions:
    """Unimodality constraint matching ``unimod.m``.

    Modes are 0 (set violations to ``1e-30``), 1 (copy the adjacent value),
    and 2 (average adjacent values).
    """

    enabled: bool = False
    tolerance: float = 1.1
    mode: UnimodalityMode = 1
    mask: ArrayLike | None = None


@dataclass(slots=True)
class ClosureCondition:
    """One closure group within an experiment block."""

    components: Sequence[bool] | ArrayLike
    kind: ClosureKind = "equality"
    target: float | ArrayLike = 1.0


@dataclass(slots=True)
class ClosureBlock:
    """One or two disjoint closure groups for a row/column block."""

    first: ClosureCondition
    second: ClosureCondition | None = None


@dataclass(slots=True)
class ClosureOptions:
    enabled: bool = False
    mode: ClosureMode = "concentration"
    blocks: Sequence[ClosureBlock] = field(default_factory=tuple)


@dataclass(slots=True)
class ValueConstraint:
    """Equality, upper-bound, or lower-bound values in one ALS mode.

    ``indices`` are zero-based linear indices interpreted in MATLAB
    column-major order.  A boolean array with the same shape as ``values`` may
    be used instead.
    """

    values: ArrayLike
    indices: ArrayLike | None = None
    kind: ConstraintKind = "equal"


@dataclass(slots=True)
class MultiExperimentOptions:
    """Partitions for row- and column-augmented experiments.

    Bounds are half-open Python intervals ``(start, stop)``.  ``presence`` has
    shape ``(number_of_row_blocks, number_of_components)`` and zeros force a
    component to zero in the corresponding concentration block.
    """

    row_blocks: Sequence[tuple[int, int]] | None = None
    column_blocks: Sequence[tuple[int, int]] | None = None
    presence: ArrayLike | None = None

    @classmethod
    def from_lengths(
        cls,
        *,
        row_lengths: Sequence[int] | None = None,
        column_lengths: Sequence[int] | None = None,
        presence: ArrayLike | None = None,
    ) -> "MultiExperimentOptions":
        def bounds(lengths: Sequence[int] | None) -> tuple[tuple[int, int], ...] | None:
            if lengths is None:
                return None
            result: list[tuple[int, int]] = []
            start = 0
            for raw_length in lengths:
                length = int(raw_length)
                if length <= 0:
                    raise ValueError("experiment block lengths must be positive")
                result.append((start, start + length))
                start += length
            return tuple(result)

        return cls(
            row_blocks=bounds(row_lengths),
            column_blocks=bounds(column_lengths),
            presence=presence,
        )


@dataclass(slots=True)
class TrilinearityOptions:
    """Three-/four-way constraints translated from ``trilin.m``/``quadril.m``."""

    enabled: bool = False
    direction: TrilinearityDirection = "concentration"
    shape: Literal[1, 2] = 1
    component_mask: ArrayLike | None = None
    quadrilinear_dimensions: tuple[int, int, int] | None = None


@dataclass(slots=True)
class CorrelationOptions:
    """Linear calibration constraint translated from ``correlCons``.

    Finite entries in ``reference`` are known concentrations and NaNs are
    unknown. ``component_mask`` selects the regressed components.
    """

    enabled: bool = True
    reference: ArrayLike | None = None
    component_mask: ArrayLike | None = None
    model: CorrelationModel = "global"
    matrix_effect: bool = False
    normalize_spectra: bool = False


@dataclass(slots=True)
class WeightedOptions:
    """MLPCA preprocessing options translated from ``weighted.m``."""

    enabled: bool = True
    standard_deviations: ArrayLike | None = None
    convergence_limit: float = 1.0e-10
    max_iterations: int = 200_000


@dataclass(slots=True)
class KineticModelOptions:
    """One globally fitted kinetic hard model.

    ``reaction_orders`` is reactions-by-species and ``stoichiometry`` is
    species-by-reactions. ``component_mapping`` assigns a zero-based model
    species to each ALS component; negative entries leave a component
    unconstrained. ``colored_mask`` selects the observed model species.

    ``time`` may be one vector covering every row, one vector reused for all
    selected experiments, or a sequence containing one vector per experiment.
    Initial concentrations may likewise be one vector or one row per selected
    experiment. ``experiment_mask`` selects row blocks for a multi-experiment
    fit and defaults to all blocks.
    """

    reaction_orders: ArrayLike
    stoichiometry: ArrayLike
    initial_rate_constants: ArrayLike
    initial_concentrations: ArrayLike
    time: ArrayLike | Sequence[ArrayLike]
    component_mapping: ArrayLike
    colored_mask: ArrayLike
    experiment_mask: ArrayLike | None = None
    choose: ArrayLike | None = None
    name: str = ""


@dataclass(slots=True)
class KineticOptions:
    """Kinetic hard-model constraints translated from the MATLAB GUI branch."""

    enabled: bool = True
    models: Sequence[KineticModelOptions] = field(default_factory=tuple)


@dataclass(slots=True)
class TuckerModeOptions:
    """One Tucker interaction mode.

    ``groups`` assigns a zero-based interaction group to each component;
    negative values omit a component from this mode.
    """

    mode: Literal[1, 3]
    groups: ArrayLike


@dataclass(slots=True)
class TuckerOptions:
    """Tucker interaction constraints translated from ``tuck2.m``."""

    enabled: bool = True
    n_matrices: int | None = None
    modes: Sequence[TuckerModeOptions] = field(default_factory=tuple)


@dataclass(slots=True)
class MCRALSOptions:
    """Options for the numerical MCR-ALS engine."""

    max_iterations: int = 50
    tolerance: float = 0.1
    divergence_limit: int = 20
    normalization: Normalization = "none"
    nonnegativity_c: NonnegativityOptions = field(default_factory=NonnegativityOptions)
    nonnegativity_s: NonnegativityOptions = field(default_factory=NonnegativityOptions)
    unimodality_c: UnimodalityOptions = field(default_factory=UnimodalityOptions)
    unimodality_s: UnimodalityOptions = field(default_factory=UnimodalityOptions)
    closure: ClosureOptions = field(default_factory=ClosureOptions)
    concentration_values: ValueConstraint | None = None
    spectral_values: ValueConstraint | None = None
    multi: MultiExperimentOptions = field(default_factory=MultiExperimentOptions)
    trilinearity: TrilinearityOptions = field(default_factory=TrilinearityOptions)

    correlation: CorrelationOptions | None = None
    kinetic: KineticOptions | None = None
    weighted: WeightedOptions | None = None
    tucker: TuckerOptions | None = None
