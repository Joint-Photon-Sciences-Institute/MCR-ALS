from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import nnls as scipy_nnls

from mcr_als import (
    closure,
    closurels,
    efa,
    fnnls,
    lof_quadrilinear,
    lof_trilinear,
    nnls,
    normv2,
    normv3,
    pcarep,
    pure,
    quadril,
    trilin,
    tuck2,
    unimod,
    wmat,
)


def test_pcarep_matches_truncated_svd_and_lof_formula() -> None:
    data = np.array(
        [[1.0, 2.0, 0.0], [2.0, 1.0, 1.0], [3.0, 0.0, 2.0], [4.0, 1.0, 3.0]]
    )
    u, s, v, reproduced, sigma = pcarep(data, 2)
    assert u.shape == (4, 2)
    assert s.shape == (2, 2)
    assert v.shape == (3, 2)
    assert_allclose(reproduced, u @ s @ v.T, rtol=1e-14, atol=1e-14)
    residual = data - reproduced
    assert_allclose(
        sigma,
        np.sqrt(np.sum(residual**2) / np.sum(data**2)) * 100.0,
        rtol=1e-14,
    )


def test_fnnls_matches_scipy_nnls_solution_and_dual() -> None:
    matrix = np.array(
        [[1.0, 0.2, 0.0], [0.0, 1.0, 0.4], [1.0, 0.0, 1.0], [0.2, 0.1, 0.8]]
    )
    rhs = np.array([0.1, 1.0, 0.4, 0.2])
    expected, _ = scipy_nnls(matrix, rhs)
    result, dual = fnnls(matrix.T @ matrix, matrix.T @ rhs)
    assert_allclose(result, expected, rtol=1e-11, atol=1e-12)
    assert_allclose(dual, matrix.T @ rhs - matrix.T @ matrix @ result, atol=1e-13)
    assert np.all(result >= -1e-14)


def test_legacy_nnls_matches_scipy_nnls_solution_and_dual() -> None:
    matrix = np.array(
        [[1.0, 0.2, 0.0], [0.0, 1.0, 0.4], [1.0, 0.0, 1.0], [0.2, 0.1, 0.8]]
    )
    rhs = np.array([0.1, 1.0, 0.4, 0.2])
    expected, _ = scipy_nnls(matrix, rhs)
    result, dual = nnls(matrix, rhs)
    assert_allclose(result, expected, rtol=1e-11, atol=1e-12)
    assert_allclose(dual, matrix.T @ (rhs - matrix @ result), atol=1e-13)


def test_unimod_copy_mode_matches_matlab_sweep_order() -> None:
    profile = np.array([0.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.0])
    expected = np.array([0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 2.0])
    assert_allclose(unimod(profile, 1.0, 1), expected)


def test_unimod_operates_column_by_column() -> None:
    profiles = np.column_stack(
        [
            [0.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.0],
            [1.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
        ]
    )
    result = unimod(profiles, 1.0, 1)
    assert_allclose(result[:, 0], [0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 2.0])
    assert_allclose(result[:, 1], profiles[:, 1])


def test_closure_equality_scalar_and_vector_targets() -> None:
    concentrations = np.array([[1.0, 2.0, 9.0], [2.0, 2.0, 8.0], [3.0, 1.0, 7.0]])
    scalar = closure(concentrations, 1, [1, 1, 0], 1, 10.0)
    assert_allclose(np.sum(scalar[:, :2], axis=1), 10.0)
    assert_allclose(scalar[:, 2], concentrations[:, 2])

    targets = np.array([1.0, 2.0, 3.0])
    vector = closure(
        concentrations,
        1,
        [1, 1, 0],
        1,
        0.0,
        vector1=targets,
    )
    assert_allclose(np.sum(vector[:, :2], axis=1), targets)


def test_closure_least_squares_and_two_disjoint_groups() -> None:
    concentrations = np.array(
        [[1.0, 2.0, 2.0, 1.0], [2.0, 1.0, 1.0, 2.0], [1.5, 1.5, 2.0, 2.0]]
    )
    result = closure(
        concentrations,
        2,
        [1, 1, 0, 0],
        2,
        3.0,
        4.0,
        [0, 0, 1, 1],
        2,
    )
    first_scale = np.linalg.lstsq(concentrations[:, :2], np.full(3, 3.0), rcond=None)[0]
    second_scale = np.linalg.lstsq(concentrations[:, 2:], np.full(3, 4.0), rcond=None)[
        0
    ]
    assert_allclose(result[:, :2], concentrations[:, :2] * first_scale)
    assert_allclose(result[:, 2:], concentrations[:, 2:] * second_scale)


def test_closure_lower_equal_uses_maximum_initial_sum() -> None:
    concentrations = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]])
    result = closure(concentrations, 1, [1, 1], 3, 8.0)
    assert_allclose(np.sum(result, axis=1), [4.0, 8.0, 8.0])


def test_closurels_matches_supported_closure_subset() -> None:
    concentrations = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 2.0]])
    expected = closure(concentrations, 1, [1, 1], 2, 4.0)
    assert_allclose(
        closurels(concentrations, 1, [1, 1], 2, 4.0),
        expected,
    )


def test_normv2_and_normv3_match_row_formulas() -> None:
    values = np.array([[3.0, 4.0], [-1.0, 1.0]])
    assert_allclose(normv2(values), [[0.6, 0.8], [-(2**-0.5), 2**-0.5]])
    assert_allclose(normv3(values), [[3 / 7, 4 / 7], [0.5, 0.5]])


def test_trilin_exact_rank_one_profiles_and_total_ratios() -> None:
    base = np.array([0.0, 1.0, 3.0, 1.0])
    profile = np.concatenate([base, 2.0 * base, 0.5 * base])
    result, totals = trilin(profile, 3, 1)
    assert_allclose(result, profile, rtol=1e-13, atol=1e-13)
    assert_allclose(totals / totals[0], [1.0, 2.0, 0.5], rtol=1e-13)


def test_trilin_peak_shift_preserves_peak_positions() -> None:
    base = np.array([0.0, 1.0, 3.0, 1.0, 0.0])
    shifted = np.array([0.0, 0.0, 1.0, 3.0, 1.0])
    result, _ = trilin(np.concatenate([base, shifted]), 2, 2)
    folded = result.reshape((5, 2), order="F")
    assert_array_equal(np.argmax(folded, axis=0), [2, 3])


def test_quadril_reproduces_exact_separable_profile() -> None:
    # quadril.m swaps ne1 and ne3 internally.
    mode1 = np.array([1.0, 2.0])  # original mode 3
    mode2 = np.array([0.5, 1.5, 2.5])
    mode3 = np.array([1.0, 3.0])  # original mode 1
    long_profile = np.outer(mode3, mode2).reshape(-1, order="F")
    profile = np.outer(long_profile, mode1).reshape(-1, order="F")
    result, _, _, _ = quadril(profile, 2, 3, 2, 1)
    assert_allclose(result, profile, rtol=1e-13, atol=1e-13)


def test_tuck2_mode1_reproduces_rank_one_interaction() -> None:
    row_profile = np.array([[1.0], [2.0], [1.0]])
    totals = np.array([[1.0, 2.0], [0.5, 1.5]])
    concentrations = np.kron(totals, row_profile)
    result, _ = tuck2(concentrations, 2, 4, 1)
    assert_allclose(result, concentrations, rtol=1e-13, atol=1e-13)


def test_tuck2_mode3_reproduces_rank_one_interaction() -> None:
    row_profiles = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 0.5]])
    totals = np.array([1.0, 0.5])
    concentrations = np.kron(totals[:, None], row_profiles)
    result, _ = tuck2(concentrations, 2, 4, 3)
    assert_allclose(result, concentrations, rtol=1e-13, atol=1e-13)


def test_wmat_uses_zero_based_python_indices() -> None:
    correlation = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]])
    result = wmat(correlation, [1], 2, 2)
    assert_allclose(result, [[1.0, 0.4], [0.4, 1.0]])


def test_pure_returns_selected_normalized_rows_and_indices() -> None:
    data = np.array(
        [
            [1.0, 0.1, 0.9, 0.2],
            [2.0, 0.2, 0.5, 0.3],
            [3.0, 0.9, 0.2, 0.4],
            [4.0, 1.5, 0.1, 0.5],
        ]
    )
    profiles, indices = pure(data, 2, 1.0)
    assert profiles.shape == (2, 4)
    assert indices.shape == (2,)
    assert_allclose(profiles, normv2(data[:, indices].T))


def test_efa_forward_backward_and_arranged_profiles() -> None:
    data = np.array(
        [
            [1.0, 0.0, 0.1],
            [2.0, 0.2, 0.0],
            [3.0, 1.0, 0.2],
            [2.0, 2.0, 0.5],
            [1.0, 3.0, 1.0],
        ]
    )
    result = efa(data, n_rows=5, n_factors=2)
    expected_first = np.linalg.svd(data[:2], compute_uv=False) ** 2
    assert_allclose(result.forward[0, :2], expected_first)
    assert result.forward.shape == (4, 3)
    assert result.backward.shape == (4, 3)
    assert result.profiles is not None
    assert result.profiles.shape == (5, 2)
    assert_allclose(result.profiles[0], result.profiles[1])


def test_lof_trilinear_is_exact_for_cp_model() -> None:
    u = np.array([[1.0, 0.0], [0.5, 1.0]])
    v = np.array([[1.0, 0.2], [2.0, 0.1], [0.0, 1.0]])
    t = np.array([[1.0, 0.5], [0.7, 2.0]])
    data = np.einsum("ir,jr,kr->ijk", u, v, t)
    r_squared, lof = lof_trilinear(data, u, v, t)
    assert_allclose([r_squared, lof], [100.0, 0.0], atol=1e-13)


def test_lof_quadrilinear_ignores_nonfinite_observations() -> None:
    u1 = np.array([[1.0], [2.0]])
    u2 = np.array([[1.0], [0.5]])
    u3 = np.array([[1.0], [3.0]])
    u4 = np.array([[2.0], [1.0]])
    data = np.einsum("ir,jr,kr,lr->ijkl", u1, u2, u3, u4)
    data[0, 0, 0, 0] = np.nan
    r_squared, lof = lof_quadrilinear(data, u1, u2, u3, u4)
    assert_allclose([r_squared, lof], [100.0, 0.0], atol=1e-13)
