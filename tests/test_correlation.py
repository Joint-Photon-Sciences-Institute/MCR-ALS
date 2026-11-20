from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from mcr_als import (
    CorrelationOptions,
    MCRALSOptions,
    MultiExperimentOptions,
    mcr_als,
    yregrnew,
)


def test_yregrnew_matches_linear_calibration_and_restores_known_values() -> None:
    profiles = np.array(
        [
            [1.0, 9.0],
            [2.0, 8.0],
            [3.0, 7.0],
            [4.0, 6.0],
            [5.0, 5.0],
        ]
    )
    reference = np.array(
        [
            [0.0, np.nan],
            [np.nan, np.nan],
            [1.0, np.nan],
            [np.nan, np.nan],
            [2.0, np.nan],
        ]
    )
    result = yregrnew(profiles, reference, [True, False])
    assert_allclose(result.calibrated[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0], atol=2e-15)
    assert_allclose(result.output[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0])
    assert_allclose(result.output[:, 1], profiles[:, 1])
    stats = result.stats[0][0]
    assert stats is not None
    assert_allclose(
        [stats.slope, stats.offset, stats.correlation, stats.rmsec],
        [1.0, 0.0, 1.0, 0.0],
        atol=1e-14,
    )


def test_global_correlation_is_applied_in_als_constraint_order() -> None:
    concentrations = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.0, 1.0]]
    )
    spectra = np.array([[1.0, 0.8, 0.2, 0.1], [0.0, 0.2, 0.7, 1.0]])
    data = concentrations @ spectra
    reference = np.full_like(concentrations, np.nan)
    reference[[0, 2, 4], 0] = concentrations[[0, 2, 4], 0]
    result = mcr_als(
        data,
        spectra + 0.02,
        MCRALSOptions(
            max_iterations=2,
            correlation=CorrelationOptions(
                reference=reference,
                component_mask=[True, False],
            ),
        ),
    )
    assert_allclose(
        result.concentrations[[0, 2, 4], 0],
        reference[[0, 2, 4], 0],
        atol=0.0,
    )
    assert len(result.correlation_history) == result.iterations


def test_local_correlation_keeps_separate_block_statistics() -> None:
    profiles = np.array([[1.0], [2.0], [3.0], [2.0], [4.0], [6.0]])
    spectra = np.array([[1.0, 0.5, 0.2]])
    data = profiles @ spectra
    reference = np.array([[0.0], [np.nan], [1.0], [0.0], [np.nan], [1.0]])
    result = mcr_als(
        data,
        spectra,
        MCRALSOptions(
            max_iterations=1,
            multi=MultiExperimentOptions.from_lengths(row_lengths=[3, 3]),
            correlation=CorrelationOptions(
                reference=reference,
                component_mask=[True],
                model="local",
            ),
        ),
    )
    assert len(result.correlation_history[0].stats) == 2
    assert_allclose(result.concentrations[[0, 2, 3, 5], 0], [0.0, 1.0, 0.0, 1.0])


def test_known_reference_values_are_imposed_on_every_component() -> None:
    # alsOptimization.m ends the correlation step with conc(iisel)=csel(iisel)
    # for every finite reference entry: components left out of the regression
    # and components with a single calibration point are constrained as well.
    rng = np.random.default_rng(0)
    concentrations = rng.uniform(0.2, 2.0, (8, 3))
    spectra = np.array(
        [
            [1.0, 0.8, 0.2, 0.1, 0.0],
            [0.0, 0.2, 0.7, 1.0, 0.3],
            [0.3, 0.0, 0.1, 0.4, 1.0],
        ]
    )
    data = (concentrations * np.array([3.0, 5.0, 2.0])) @ spectra
    reference = np.full_like(concentrations, np.nan)
    reference[:5, 0] = concentrations[:5, 0]
    reference[[1, 6], 1] = [0.25, 0.75]
    reference[3, 2] = 0.5
    result = mcr_als(
        data,
        spectra + 0.01,
        MCRALSOptions(
            max_iterations=3,
            correlation=CorrelationOptions(
                reference=reference,
                component_mask=[True, False, True],
            ),
        ),
    )
    known = np.isfinite(reference)
    assert_allclose(result.concentrations[known], reference[known], atol=0.0)
    last = result.correlation_history[-1]
    assert last.stats[0][0] is not None
    assert last.stats[0][1] is None
    assert last.stats[0][2] is None
    assert_allclose(last.output[known], reference[known], atol=0.0)


def test_local_correlation_imposes_known_values_in_every_block() -> None:
    rng = np.random.default_rng(1)
    concentrations = rng.uniform(0.2, 2.0, (8, 2))
    spectra = np.array([[1.0, 0.8, 0.2, 0.1], [0.0, 0.2, 0.7, 1.0]])
    data = (concentrations * np.array([3.0, 5.0])) @ spectra
    reference = np.full_like(concentrations, np.nan)
    reference[[0, 1, 2, 4, 5, 6], 0] = concentrations[[0, 1, 2, 4, 5, 6], 0]
    reference[[3, 7], 1] = [0.4, 0.6]
    result = mcr_als(
        data,
        spectra + 0.01,
        MCRALSOptions(
            max_iterations=2,
            multi=MultiExperimentOptions.from_lengths(row_lengths=[4, 4]),
            correlation=CorrelationOptions(
                reference=reference,
                component_mask=[True, False],
                model="local",
            ),
        ),
    )
    known = np.isfinite(reference)
    assert_allclose(result.concentrations[known], reference[known], atol=0.0)
