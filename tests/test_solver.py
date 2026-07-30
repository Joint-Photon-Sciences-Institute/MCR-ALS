from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mcr_als import (
    ClosureBlock,
    ClosureCondition,
    ClosureOptions,
    MCRALSOptions,
    MultiExperimentOptions,
    NonnegativityOptions,
    TuckerModeOptions,
    TuckerOptions,
    ValueConstraint,
    mcr_als,
    pcarep,
    tuck2,
)


def example_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    concentrations = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.7, 0.3],
            [1.0, 0.0],
        ]
    )
    spectra = np.array([[1.0, 0.8, 0.2, 0.1, 0.0], [0.0, 0.2, 0.5, 0.9, 1.0]])
    noise = np.array(
        [
            [0.002, -0.001, 0.0, 0.001, -0.002],
            [0.0, 0.001, -0.001, 0.002, 0.0],
            [-0.001, 0.0, 0.002, -0.001, 0.001],
            [0.001, -0.002, 0.001, 0.0, -0.001],
            [0.0, 0.001, -0.002, 0.001, 0.0],
            [-0.002, 0.0, 0.001, -0.001, 0.002],
            [0.001, 0.002, -0.001, 0.0, -0.002],
            [-0.001, 0.0, 0.002, -0.002, 0.001],
        ]
    )
    return concentrations @ spectra + noise, concentrations, spectra


def test_one_iteration_matches_matlab_update_formulas() -> None:
    data, _, spectra = example_data()
    initial_spectra = spectra + np.array(
        [[0.02, -0.01, 0.01, 0.0, 0.01], [0.01, 0.0, -0.01, 0.01, -0.02]]
    )
    options = MCRALSOptions(max_iterations=1)
    result = mcr_als(data, initial_spectra, options)

    _, _, _, reproduced, _ = pcarep(data, 2)
    expected_c = np.linalg.lstsq(initial_spectra.T, reproduced.T, rcond=None)[0].T
    expected_s = np.linalg.lstsq(expected_c, reproduced, rcond=None)[0]
    assert_allclose(result.concentrations, expected_c, rtol=2e-12, atol=2e-13)
    assert_allclose(result.spectra, expected_s, rtol=2e-12, atol=2e-13)
    assert_allclose(
        result.residual_pca,
        reproduced - expected_c @ expected_s,
        rtol=2e-12,
        atol=5e-16,
    )
    assert result.status == "max_iterations"
    assert result.best_iteration == 1


def test_initial_orientation_is_inferred_for_both_modes() -> None:
    data, concentrations, spectra = example_data()
    concentration_result = mcr_als(
        data,
        concentrations,
        MCRALSOptions(max_iterations=1),
    )
    spectral_result = mcr_als(data, spectra, MCRALSOptions(max_iterations=1))
    transposed_result = mcr_als(data, spectra.T, MCRALSOptions(max_iterations=1))
    assert concentration_result.initial_estimate_mode == "concentrations"
    assert spectral_result.initial_estimate_mode == "spectra"
    assert transposed_result.initial_estimate_mode == "spectra"


def test_result_metrics_and_history_shapes_follow_matlab_formulas() -> None:
    data, _, spectra = example_data()
    result = mcr_als(
        data,
        spectra + 0.01,
        MCRALSOptions(max_iterations=6, tolerance=1e-12),
    )
    calculated = result.concentrations @ result.spectra
    residual = data - calculated
    ss_data = np.sum(data**2)
    ss_residual = np.sum(residual**2)
    assert_allclose(result.r_squared, (ss_data - ss_residual) / ss_data)
    assert_allclose(result.lack_of_fit[1], np.sqrt(ss_residual / ss_data) * 100.0)
    assert result.history.spectra.shape == (
        result.iterations * 2,
        data.shape[1],
    )
    assert result.history.concentrations.shape == (
        data.shape[0],
        result.iterations * 2,
    )


def test_multi_experiment_presence_closure_and_nonnegativity() -> None:
    _, concentrations, spectra = example_data()
    concentrations = concentrations.copy()
    concentrations[4:, 1] = 0.0
    concentrations[4:, 0] = [0.2, 0.4, 0.7, 1.0]
    data = concentrations @ spectra
    options = MCRALSOptions(
        max_iterations=5,
        nonnegativity_c=NonnegativityOptions(enabled=True),
        nonnegativity_s=NonnegativityOptions(enabled=True),
        closure=ClosureOptions(
            enabled=True,
            mode="concentration",
            blocks=(ClosureBlock(ClosureCondition([True, True], "equality", 1.0)),),
        ),
        multi=MultiExperimentOptions.from_lengths(
            row_lengths=[4, 4],
            presence=[[1, 1], [1, 0]],
        ),
    )
    result = mcr_als(data, spectra + 0.02, options)
    assert np.min(result.concentrations) >= -1e-14
    assert np.min(result.spectra) >= -1e-14
    assert_allclose(np.sum(result.concentrations[:4], axis=1), 1.0, atol=1e-13)
    assert_allclose(np.sum(result.concentrations[4:], axis=1), 1.0, atol=1e-13)
    assert_allclose(result.concentrations[4:, 1], 0.0, atol=0.0)
    assert result.component_areas.shape == (2, 2)
    assert result.relative_areas.shape == (2, 2)


def test_spectral_normalizations() -> None:
    data, _, spectra = example_data()
    maximum = mcr_als(
        data,
        spectra + 0.03,
        MCRALSOptions(max_iterations=1, normalization="maximum"),
    )
    euclidean = mcr_als(
        data,
        spectra + 0.03,
        MCRALSOptions(max_iterations=1, normalization="euclidean"),
    )
    total_sum = mcr_als(
        data,
        spectra + 0.03,
        MCRALSOptions(max_iterations=1, normalization="sum"),
    )
    assert_allclose(np.max(maximum.spectra, axis=1), 1.0)
    assert_allclose(np.linalg.norm(euclidean.spectra, axis=1), 1.0)
    assert_allclose(np.sum(np.abs(total_sum.spectra), axis=1), 1.0)
    assert np.min(total_sum.spectra) >= 0.0


def test_value_constraints_use_fortran_linear_indices() -> None:
    data, _, spectra = example_data()
    values = np.zeros((8, 2))
    # MATLAB linear index 2 (Python index 1) is row 2, component 1.
    values[1, 0] = 0.25
    result = mcr_als(
        data,
        spectra,
        MCRALSOptions(
            max_iterations=1,
            concentration_values=ValueConstraint(values, indices=[1], kind="equal"),
        ),
    )
    assert result.concentrations[1, 0] == 0.25


def test_nonnegative_fnnls_algorithms_run_end_to_end() -> None:
    data, _, spectra = example_data()
    result = mcr_als(
        data,
        spectra - 0.1,
        MCRALSOptions(
            max_iterations=3,
            nonnegativity_c=NonnegativityOptions(enabled=True, algorithm="fnnls"),
            nonnegativity_s=NonnegativityOptions(enabled=True, algorithm="fnnls"),
        ),
    )
    assert np.min(result.concentrations) >= -1e-12
    assert np.min(result.spectra) >= -1e-12


def test_trilinear_constraint_uses_equal_row_blocks() -> None:
    base_c = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 0.0], [1.0, 0.5]])
    concentrations = np.vstack([base_c, 2.0 * base_c])
    spectra = np.array([[1.0, 0.5, 0.1], [0.1, 0.5, 1.0]])
    data = concentrations @ spectra
    from mcr_als import TrilinearityOptions

    result = mcr_als(
        data,
        spectra + 0.01,
        MCRALSOptions(
            max_iterations=2,
            multi=MultiExperimentOptions.from_lengths(row_lengths=[4, 4]),
            trilinearity=TrilinearityOptions(
                enabled=True, direction="concentration", shape=1
            ),
        ),
    )
    first = result.concentrations[:4]
    second = result.concentrations[4:]
    for component in range(2):
        ratio = np.dot(first[:, component], second[:, component]) / np.dot(
            first[:, component], first[:, component]
        )
        assert_allclose(second[:, component], ratio * first[:, component], atol=1e-11)


def test_tucker_constraint_matches_tuck2_at_its_matlab_order_position() -> None:
    concentrations = np.array(
        [
            [1.0, 0.2],
            [0.7, 0.5],
            [0.2, 1.0],
            [0.5, 0.9],
            [1.1, 0.3],
            [0.4, 0.8],
        ]
    )
    spectra = np.array([[1.0, 0.7, 0.2, 0.1], [0.1, 0.3, 0.8, 1.0]])
    data = concentrations @ spectra
    initial = spectra + np.array([[0.02, 0.0, -0.01, 0.01], [0.0, 0.01, 0.0, -0.02]])
    _, _, _, reproduced, _ = pcarep(data, 2)
    unconstrained = np.linalg.lstsq(initial.T, reproduced.T, rcond=None)[0].T
    expected, _ = tuck2(unconstrained, 2, 4, 1)
    result = mcr_als(
        data,
        initial,
        MCRALSOptions(
            max_iterations=1,
            multi=MultiExperimentOptions.from_lengths(row_lengths=[3, 3]),
            tucker=TuckerOptions(modes=(TuckerModeOptions(mode=1, groups=[0, 0]),)),
        ),
    )
    assert_allclose(result.concentrations, expected, rtol=2e-12, atol=2e-13)


def test_invalid_kinetic_configuration_fails_explicitly() -> None:
    data, _, spectra = example_data()
    options = MCRALSOptions()
    options.kinetic = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="KineticOptions"):
        mcr_als(data, spectra, options)


def test_invalid_partition_fails_before_iteration() -> None:
    data, _, spectra = example_data()
    options = MCRALSOptions(multi=MultiExperimentOptions(row_blocks=[(0, 3), (4, 8)]))
    with pytest.raises(ValueError, match="contiguous"):
        mcr_als(data, spectra, options)
