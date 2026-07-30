from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat

from mcr_als import (
    ClosureBlock,
    ClosureCondition,
    ClosureOptions,
    MCRALSOptions,
    MultiExperimentOptions,
    NonnegativityOptions,
    UnimodalityOptions,
    closure,
    fnnls,
    integrate_kinetics,
    interesp,
    kinetic_derivative,
    mlpca,
    nnls,
    normv2,
    normv3,
    quadril,
    opt_kinglob,
    trilin,
    tuck2,
    unimod,
    wmat,
    mcr_als,
    yregrnew,
)

FIXTURE = Path(__file__).parent / "fixtures" / "matlab_reference.mat"


@pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="run matlab/generate_reference_fixtures.m on a MATLAB machine",
)
def test_against_matlab_reference_fixture() -> None:
    reference = loadmat(FIXTURE, squeeze_me=True)
    options = MCRALSOptions(
        max_iterations=int(reference["max_iterations"]),
        tolerance=float(reference["tolerance"]),
        nonnegativity_c=NonnegativityOptions(enabled=True, algorithm="truncate"),
        nonnegativity_s=NonnegativityOptions(enabled=True, algorithm="truncate"),
        unimodality_c=UnimodalityOptions(enabled=True, tolerance=1.1, mode=1),
        closure=ClosureOptions(
            enabled=True,
            mode="concentration",
            blocks=(ClosureBlock(ClosureCondition([True, True], "equality", 1.0)),),
        ),
        multi=MultiExperimentOptions.from_lengths(row_lengths=[4, 4]),
    )
    result = mcr_als(reference["data"], reference["initial_spectra"], options)

    # MATLAB and SciPy can select slightly different QR/SVD code paths.  These
    # tolerances are tight enough to detect algorithm/order changes while
    # allowing normal BLAS/LAPACK last-bit differences.
    assert_allclose(
        result.concentrations,
        reference["concentrations"],
        rtol=2e-10,
        atol=2e-11,
    )
    assert_allclose(
        result.spectra,
        reference["spectra"],
        rtol=2e-10,
        atol=2e-11,
    )
    assert_allclose(
        result.lack_of_fit,
        np.asarray(reference["lack_of_fit"]).reshape(-1),
        rtol=2e-10,
        atol=2e-11,
    )
    assert_allclose(result.r_squared, reference["r_squared"], rtol=2e-10)

    matrix = np.array(
        [[1.0, 0.2, 0.0], [0.0, 1.0, 0.4], [1.0, 0.0, 1.0], [0.2, 0.1, 0.8]]
    )
    rhs = np.array([0.1, 1.0, 0.4, 0.2])
    fnnls_x, fnnls_w = fnnls(matrix.T @ matrix, matrix.T @ rhs)
    nnls_x, nnls_w = nnls(matrix, rhs)
    assert_allclose(fnnls_x, reference["helper_fnnls_x"], rtol=2e-10, atol=2e-11)
    assert_allclose(fnnls_w, reference["helper_fnnls_w"], rtol=2e-10, atol=2e-11)
    assert_allclose(nnls_x, reference["helper_nnls_x"], rtol=2e-10, atol=2e-11)
    assert_allclose(nnls_w, reference["helper_nnls_w"], rtol=2e-10, atol=2e-11)

    profile = np.array([0.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.0])
    assert_allclose(unimod(profile, 1.0, 1), reference["helper_unimod"])
    closure_input = np.array([[1.0, 2.0, 9.0], [2.0, 2.0, 8.0], [3.0, 1.0, 7.0]])
    assert_allclose(
        closure(closure_input, 1, [1, 1, 0], 1, 10.0),
        reference["helper_closure"],
    )

    norm_input = np.array([[3.0, 4.0], [-1.0, 1.0]])
    assert_allclose(normv2(norm_input), reference["helper_normv2"])
    assert_allclose(normv3(norm_input), reference["helper_normv3"])

    base = np.array([0.0, 1.0, 3.0, 1.0])
    trilinear, _ = trilin(np.concatenate([base, 2.0 * base, 0.5 * base]), 3, 1)
    assert_allclose(trilinear, reference["helper_trilin"], rtol=2e-10, atol=2e-11)

    mode1 = np.array([1.0, 2.0])
    mode2 = np.array([0.5, 1.5, 2.5])
    mode3 = np.array([1.0, 3.0])
    long_profile = np.outer(mode3, mode2).reshape(-1, order="F")
    quadrilinear_input = np.outer(long_profile, mode1).reshape(-1, order="F")
    quadrilinear, _, _, _ = quadril(quadrilinear_input, 2, 3, 2, 1)
    assert_allclose(
        quadrilinear,
        reference["helper_quadril"],
        rtol=2e-10,
        atol=2e-11,
    )

    correlation = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]])
    assert_allclose(wmat(correlation, [1], 2, 2), reference["helper_wmat"])

    correlation_profiles = np.array(
        [[1.0, 9.0], [2.0, 8.0], [3.0, 7.0], [4.0, 6.0], [5.0, 5.0]]
    )
    correlation_reference = np.array(
        [
            [0.0, np.nan],
            [np.nan, np.nan],
            [1.0, np.nan],
            [np.nan, np.nan],
            [2.0, np.nan],
        ]
    )
    regression = yregrnew(correlation_profiles, correlation_reference, [1, 0])
    assert_allclose(regression.output, reference["helper_corr_output"], atol=2e-14)
    assert_allclose(
        regression.calibrated, reference["helper_corr_calibrated"], atol=2e-14
    )
    stats = regression.stats[0][0]
    assert stats is not None
    assert_allclose(
        [
            stats.slope,
            stats.offset,
            stats.correlation,
            stats.rmsec,
            stats.relative_error_percent,
        ],
        reference["helper_corr_stats"],
        rtol=2e-10,
        atol=2e-11,
    )

    weighted_data = np.array(
        [
            [1.0, 2.0, 0.5, 1.5],
            [2.0, 3.9, 1.1, 2.8],
            [3.0, 6.1, 1.4, 4.6],
            [4.0, 7.8, 2.1, 5.9],
            [5.0, 10.2, 2.4, 7.6],
        ]
    )
    weighted_std = np.array(
        [
            [0.10, 0.20, 0.15, 0.10],
            [0.12, 0.25, 0.10, 0.14],
            [0.15, 0.20, 0.12, 0.18],
            [0.10, 0.30, 0.14, 0.12],
            [0.18, 0.22, 0.11, 0.16],
        ]
    )
    weighted_result = mlpca(weighted_data, weighted_std, 2, 1e-12, 1000)
    assert_allclose(
        weighted_result.reconstructed,
        reference["helper_weighted_reconstruction"],
        rtol=2e-10,
        atol=2e-11,
    )
    assert_allclose(
        weighted_result.objective, reference["helper_weighted_objective"], rtol=2e-9
    )
    assert weighted_result.error_flag == int(reference["helper_weighted_error"])

    tucker_input = np.array(
        [
            [1.0, 0.2],
            [0.7, 0.5],
            [0.2, 1.0],
            [0.5, 0.9],
            [1.1, 0.3],
            [0.4, 0.8],
        ]
    )
    tucker_output, _ = tuck2(tucker_input, 2, 4, 1)
    assert_allclose(
        tucker_output, reference["helper_tucker_output"], rtol=2e-10, atol=2e-11
    )

    kinetic_orders = np.array([[1.0, 0.0]])
    kinetic_stoichiometry = np.array([[-1.0], [1.0]])
    kinetic_times = np.linspace(0.0, 5.0, 11)
    derivative = kinetic_derivative(
        0.0,
        [0.8, 0.2],
        kinetic_orders,
        kinetic_stoichiometry,
        [0.5],
    )
    assert_allclose(derivative, reference["helper_kin_derivative"], atol=2e-15)
    assert_allclose(
        interesp([[1.0, 3.0], [2.0, 4.0]], [1, 0, 1]),
        reference["helper_kin_interesp"],
    )
    integrated = integrate_kinetics(
        kinetic_times,
        [1.0, 0.0],
        kinetic_orders,
        kinetic_stoichiometry,
        [0.4],
    )
    assert_allclose(integrated, reference["helper_kin_profile"], rtol=1.5e-2, atol=5e-4)
    kinetic_fit = opt_kinglob(
        [reference["helper_kin_profile"]],
        kinetic_stoichiometry,
        kinetic_orders,
        [[1.0, 0.0]],
        [0.2],
        [kinetic_times],
        [1, 2],
        [1, 1],
    )
    assert_allclose(
        kinetic_fit.rate_constants,
        reference["helper_kin_rates"],
        rtol=2e-3,
    )
    assert_allclose(
        kinetic_fit.concentrations[0],
        reference["helper_kin_fit"],
        rtol=1.5e-2,
        atol=5e-4,
    )
