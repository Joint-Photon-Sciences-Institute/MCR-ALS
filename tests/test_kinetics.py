from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mcr_als import (
    KineticModelOptions,
    KineticOptions,
    MCRALSOptions,
    MultiExperimentOptions,
    integrate_kinetics,
    interesp,
    kinetic_derivative,
    mcr_als,
    nglmglob,
    opt_kinglob,
    rcalcglob,
)


def first_order_model() -> tuple[np.ndarray, np.ndarray]:
    orders = np.array([[1.0, 0.0]])
    stoichiometry = np.array([[-1.0], [1.0]])
    return orders, stoichiometry


def test_kinetic_derivative_matches_first_order_mass_balance() -> None:
    orders, stoichiometry = first_order_model()
    derivative = kinetic_derivative(0.0, [0.8, 0.2], orders, stoichiometry, [0.5])
    assert_allclose(derivative, [-0.4, 0.4], rtol=0.0, atol=1e-15)


def test_integrate_kinetics_matches_first_order_solution() -> None:
    orders, stoichiometry = first_order_model()
    times = np.linspace(0.0, 5.0, 21)
    calculated = integrate_kinetics(times, [1.0, 0.0], orders, stoichiometry, [0.4])
    expected_a = np.exp(-0.4 * times)
    assert_allclose(calculated[:, 0], expected_a, rtol=1.2e-2, atol=5e-4)
    assert_allclose(calculated[:, 1], 1.0 - expected_a, rtol=1.2e-2, atol=5e-4)


def test_interesp_inserts_noncolored_species() -> None:
    colored = np.array([[1.0, 3.0], [2.0, 4.0]])
    expected = np.array([[1.0, 0.0, 3.0], [2.0, 0.0, 4.0]])
    assert_allclose(interesp(colored, [1, 0, 1]), expected)


def test_rcalcglob_returns_matlab_column_major_residuals() -> None:
    orders, stoichiometry = first_order_model()
    times = np.linspace(0.0, 3.0, 7)
    fitted = integrate_kinetics(times, [1.0, 0.0], orders, stoichiometry, [0.4])
    observed = fitted.copy()
    observed[:, 0] += np.arange(times.size) * 0.01
    residual, profiles = rcalcglob(
        [0.4],
        [[1.0, 0.0]],
        [times],
        [observed],
        [1, 2],
        orders,
        stoichiometry,
        [1, 0],
    )
    assert_allclose(residual, np.arange(times.size) * 0.01, atol=2e-16)
    assert_allclose(profiles[0], fitted)


def test_nglmglob_and_opt_kinglob_recover_global_rate_constant() -> None:
    orders, stoichiometry = first_order_model()
    times = np.linspace(0.0, 5.0, 11)
    observed = integrate_kinetics(times, [1.0, 0.0], orders, stoichiometry, [0.4])
    direct = nglmglob(
        [0.2],
        [[1.0, 0.0]],
        [times],
        [observed],
        [1, 2],
        orders,
        stoichiometry,
        [1, 1],
    )
    wrapped = opt_kinglob(
        [observed],
        stoichiometry,
        orders,
        [[1.0, 0.0]],
        [0.2],
        [times],
        [1, 2],
        [1, 1],
    )
    assert_allclose(direct.rate_constants, [0.4], rtol=2e-8, atol=2e-10)
    assert_allclose(wrapped.rate_constants, direct.rate_constants)
    assert wrapped.sum_squared_residuals < 1e-20
    assert wrapped.jacobian.shape == (observed.size, 1)


def test_solver_applies_kinetics_to_multiple_experiments() -> None:
    orders, stoichiometry = first_order_model()
    times = np.linspace(0.0, 4.0, 9)
    initial_concentrations = np.array([[1.0, 0.0], [0.6, 0.0]])
    blocks = tuple(
        integrate_kinetics(times, initial, orders, stoichiometry, [0.35])
        for initial in initial_concentrations
    )
    concentrations = np.vstack(blocks)
    spectra = np.array([[1.0, 0.5, 0.1], [0.1, 0.5, 1.0]])
    data = concentrations @ spectra
    model = KineticModelOptions(
        reaction_orders=orders,
        stoichiometry=stoichiometry,
        initial_rate_constants=[0.15],
        initial_concentrations=initial_concentrations,
        time=(times, times),
        component_mapping=[0, 1],
        colored_mask=[1, 1],
        experiment_mask=[1, 1],
        name="A to B",
    )
    options = MCRALSOptions(
        max_iterations=1,
        multi=MultiExperimentOptions.from_lengths(row_lengths=[times.size, times.size]),
        kinetic=KineticOptions(models=(model,)),
    )
    result = mcr_als(data, spectra, options)
    fit = result.kinetic_history[0][0]
    assert_allclose(fit.rate_constants, [0.35], rtol=2e-8, atol=2e-10)
    assert_allclose(result.concentrations, concentrations, rtol=3e-12, atol=3e-13)
    assert fit.standard_errors.shape == (1,)


def test_solver_rejects_incomplete_kinetic_component_mapping() -> None:
    orders, stoichiometry = first_order_model()
    times = np.linspace(0.0, 2.0, 5)
    concentrations = integrate_kinetics(times, [1.0, 0.0], orders, stoichiometry, [0.4])
    spectra = np.array([[1.0, 0.2], [0.2, 1.0]])
    model = KineticModelOptions(
        orders,
        stoichiometry,
        [0.2],
        [1.0, 0.0],
        times,
        [0, -1],
        [1, 1],
    )
    with pytest.raises(ValueError, match="every colored species"):
        mcr_als(
            concentrations @ spectra,
            spectra,
            MCRALSOptions(max_iterations=1, kinetic=KineticOptions(models=(model,))),
        )
