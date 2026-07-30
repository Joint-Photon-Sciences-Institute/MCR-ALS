from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from mcr_als import MCRALSOptions, WeightedOptions, mcr_als, mlpca


def weighted_example() -> tuple[np.ndarray, np.ndarray]:
    observations = np.array(
        [
            [1.0, 2.0, 0.5, 1.5],
            [2.0, 3.9, 1.1, 2.8],
            [3.0, 6.1, 1.4, 4.6],
            [4.0, 7.8, 2.1, 5.9],
            [5.0, 10.2, 2.4, 7.6],
        ]
    )
    standard_deviations = np.array(
        [
            [0.10, 0.20, 0.15, 0.10],
            [0.12, 0.25, 0.10, 0.14],
            [0.15, 0.20, 0.12, 0.18],
            [0.10, 0.30, 0.14, 0.12],
            [0.18, 0.22, 0.11, 0.16],
        ]
    )
    return observations, standard_deviations


def test_mlpca_returns_rank_limited_weighted_reconstruction() -> None:
    observations, standard_deviations = weighted_example()
    result = mlpca(
        observations,
        standard_deviations,
        2,
        convergence_limit=1e-12,
        max_iterations=1000,
    )
    assert result.error_flag == 0
    assert result.reconstructed.shape == observations.shape
    assert np.linalg.matrix_rank(result.reconstructed, tol=1e-10) == 2
    assert result.objective >= 0.0


def test_weighted_als_uses_mlpca_reconstruction_as_analysis_data() -> None:
    observations, standard_deviations = weighted_example()
    initial_spectra = np.array([[1.0, 2.0, 0.5, 1.5], [0.2, 0.5, 1.0, 0.7]])
    direct = mlpca(observations, standard_deviations, 2, 1e-12, 1000)
    result = mcr_als(
        observations,
        initial_spectra,
        MCRALSOptions(
            max_iterations=1,
            weighted=WeightedOptions(
                standard_deviations=standard_deviations,
                convergence_limit=1e-12,
                max_iterations=1000,
            ),
        ),
    )
    assert result.weighted_preprocessing is not None
    assert_allclose(
        result.weighted_preprocessing.reconstructed,
        direct.reconstructed,
        rtol=1e-12,
        atol=1e-12,
    )
