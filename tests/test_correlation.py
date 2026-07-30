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
    assert_allclose(result.calibrated[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0], atol=3e-16)
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
