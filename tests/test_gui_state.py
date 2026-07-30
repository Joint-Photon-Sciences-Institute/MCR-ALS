from __future__ import annotations

import importlib

import numpy as np
from numpy.testing import assert_allclose
from scipy.io import savemat

from mcr_als.gui_state import (
    GUIState,
    KineticModelInput,
    build_options,
    create_initial_estimate,
    kinetic_model_from_input,
    load_matrix,
    load_numeric_array,
    parse_int_vector,
    parse_matrix_text,
    state_from_dict,
    state_to_dict,
)


def test_gui_module_import_does_not_create_a_window() -> None:
    module = importlib.import_module("mcr_als.gui")
    assert callable(module.main)


def test_matrix_and_integer_parsers_accept_matlab_style_text() -> None:
    matrix = parse_matrix_text("[1, 2, 3; 4 5 6]", name="example")
    assert_allclose(matrix, [[1, 2, 3], [4, 5, 6]])
    assert_allclose(parse_int_vector("1 0 -1"), [1, 0, -1])


def test_matrix_loader_supports_numpy_matlab_and_csv(tmp_path) -> None:
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    npy_path = tmp_path / "matrix.npy"
    npz_path = tmp_path / "matrix.npz"
    mat_path = tmp_path / "matrix.mat"
    csv_path = tmp_path / "matrix.csv"
    np.save(npy_path, expected)
    np.savez(npz_path, small=np.ones((1, 1)), selected=expected)
    savemat(mat_path, {"small": np.ones((1, 1)), "selected": expected})
    np.savetxt(csv_path, expected, delimiter=",")
    assert_allclose(load_matrix(npy_path), expected)
    assert_allclose(load_numeric_array(f"{npz_path}::selected"), expected)
    assert_allclose(load_numeric_array(f"{mat_path}::selected"), expected)
    assert_allclose(load_matrix(csv_path), expected)


def test_kinetic_editor_uses_legacy_one_based_mapping() -> None:
    model = kinetic_model_from_input(
        KineticModelInput(
            reaction_orders="1 0",
            stoichiometry="-1; 1",
            initial_rate_constants="0.2",
            initial_concentrations="1 0",
            time="0 1 2",
            component_mapping="2 0 1",
            colored_mask="1 1",
            experiment_mask="1 0",
        )
    )
    assert_allclose(model.component_mapping, [1, -1, 0])
    assert_allclose(model.experiment_mask, [1, 0])


def test_build_options_covers_gui_advanced_models(tmp_path) -> None:
    correlation_reference = np.full((6, 2), np.nan)
    correlation_reference[[0, 2, 3, 5], :] = [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.8],
        [0.8, 0.0],
    ]
    correlation_path = tmp_path / "correlation.npy"
    deviations_path = tmp_path / "deviations.npy"
    values_path = tmp_path / "values.npy"
    np.save(correlation_path, correlation_reference)
    np.save(deviations_path, np.ones((6, 4)) * 0.1)
    values = np.full((6, 2), np.nan)
    values[0, 0] = 1.0
    np.save(values_path, values)
    state = GUIState(
        row_lengths="3 3",
        presence="1 1; 1 1",
        closure_enabled=True,
        closure_components="1 1",
        closure_target="1",
        concentration_values_path=str(values_path),
        correlation_enabled=True,
        correlation_reference_path=str(correlation_path),
        correlation_component_mask="1 1",
        correlation_model="local",
        trilinearity_enabled=True,
        trilinearity_component_mask="1 0",
        tucker_enabled=True,
        tucker_n_matrices=2,
        tucker_mode1_groups="1 1",
        weighted_enabled=True,
        standard_deviations_path=str(deviations_path),
        kinetic_enabled=True,
        kinetic_models=[
            KineticModelInput(
                reaction_orders="1 0",
                stoichiometry="-1; 1",
                initial_rate_constants="0.2",
                initial_concentrations="1 0; 0.8 0",
                time="0 1 2",
                component_mapping="1 2",
                colored_mask="1 1",
                experiment_mask="1 1",
            )
        ],
    )
    options = build_options(state, (6, 4), 2)
    assert options.multi.row_blocks == ((0, 3), (3, 6))
    assert options.closure.enabled
    assert options.correlation is not None
    assert options.correlation.model == "local"
    assert options.weighted is not None
    assert options.tucker is not None
    assert_allclose(options.tucker.modes[0].groups, [0, 0])
    assert options.kinetic is not None
    assert_allclose(options.kinetic.models[0].component_mapping, [0, 1])


def test_initial_generators_and_configuration_round_trip() -> None:
    concentrations = np.array([[1.0, 0.0], [0.7, 0.3], [0.3, 0.7], [0.0, 1.0]])
    spectra = np.array([[1.0, 0.5, 0.1], [0.1, 0.5, 1.0]])
    data = concentrations @ spectra
    simplisma = create_initial_estimate(
        GUIState(initial_method="simplisma", components=2), data
    )
    efa = create_initial_estimate(GUIState(initial_method="efa", components=2), data)
    assert simplisma.shape == (2, data.shape[0])
    assert efa.shape == (data.shape[0], 2)
    state = GUIState(
        data_path="data.csv",
        max_iterations=17,
        kinetic_models=[KineticModelInput(name="A to B")],
    )
    restored = state_from_dict(state_to_dict(state))
    assert restored.data_path == state.data_path
    assert restored.max_iterations == 17
    assert restored.kinetic_models[0].name == "A to B"
