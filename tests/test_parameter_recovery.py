"""
tests/test_parameter_recovery.py
=================================
Parameter-recovery tests for AutoTraj.

Each test simulates data from a known model, runs AutoTraj, and checks that
the recovered parameters are within generous tolerances of the ground truth.
The goal is to catch catastrophic math errors and estimation failures —
not to demand machine precision.

Tolerances used throughout
--------------------------
- Intercept / beta recovery:  abs_tol = 0.3 (1-group) or 0.5 (multi-group)
- Group proportions:          within 10 pp of truth
- Assignment accuracy:        > 85 % (2-group) or > 80 % (3-group)
- Sigma (CNORM):              within 20 % of truth
- BIC model-selection:        exact group count must match

All tests run with n_starts=3 to balance speed vs. local-optima risk.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Make project root importable when pytest is run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_single_model, run_autotraj, get_subject_assignments
from tests.simulate import (
    simulate_logit_trajectories,
    simulate_cnorm_trajectories,
    simulate_poisson_trajectories,
    simulate_logit_with_mixing_covariates,
    simulate_logit_with_tvc,
    simulate_logit_with_covariates_and_tvc,
    simulate_logit_with_biased_sampling_weights,
    make_two_group_logit,
    make_two_group_poisson,
    make_two_group_cnorm,
)

# ---------------------------------------------------------------------------
# Shared time grids
# ---------------------------------------------------------------------------
T10  = np.linspace(-1.0, 1.0, 10)
T15  = np.linspace(-1.0, 1.0, 15)
T20  = np.linspace(-1.0, 1.0, 20)

N_STARTS = 3   # multi-start restarts for all tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assignment_accuracy(assignments_df: pd.DataFrame,
                          truth: dict) -> float:
    """Fraction of subjects assigned to the correct group.

    Groups may be flipped (label-switching), so we also try the permuted
    assignment and return the higher accuracy.
    """
    true_map = truth['assignments']
    df = assignments_df[assignments_df['ID'].isin(true_map)].copy()
    if len(df) == 0:
        return 0.0

    pred = df.set_index('ID')['Assigned_Group']
    true_series = pd.Series(true_map)
    # Align
    common = pred.index.intersection(true_series.index)
    pred   = pred.loc[common]
    true_v = true_series.loc[common]

    n_groups = pred.max()

    # Try all label permutations (works well for 2-3 groups)
    from itertools import permutations
    best_acc = 0.0
    for perm in permutations(range(1, n_groups + 1)):
        mapping = {orig: new for orig, new in enumerate(perm, 1)}
        remapped = pred.map(mapping)
        acc = (remapped == true_v).mean()
        best_acc = max(best_acc, acc)
    return float(best_acc)


def _best_model(models):
    """Return the model dict with the highest BIC (Nagin convention)."""
    return max(models, key=lambda m: m['bic_nagin'])


def _beta_at(model_dict, group_idx: int, coef_idx: int) -> float:
    """Extract a single beta coefficient from a converged model dict."""
    orders = model_dict['orders']
    k      = len(orders)
    params = model_dict['result'].x
    # beta block starts at index k-1, then accumulate group sizes
    idx = k - 1
    for g in range(group_idx):
        idx += orders[g] + 1
    return float(params[idx + coef_idx])


# ---------------------------------------------------------------------------
# TEST 1: LOGIT 1-group, intercept only
# ---------------------------------------------------------------------------
@pytest.mark.recovery
def test_logit_1group_intercept_only():
    """Recovered intercept must be within 0.3 of true beta0=-2.0."""
    true_beta0 = -2.0
    df, truth = simulate_logit_trajectories(
        n_subjects=200,
        time_points=T10,
        group_params=[{'betas': [true_beta0]}],
        group_proportions=[1.0],
        seed=1,
    )

    m = run_single_model(df, orders_list=[0], dist='LOGIT', n_starts=N_STARTS)

    assert m['result'].success or m['result'].status == 2, \
        "Optimizer did not converge"
    assert np.isfinite(m['ll']), f"LL is not finite: {m['ll']}"
    assert m['ll'] < 0, f"LL should be negative for binary data; got {m['ll']}"

    recovered = _beta_at(m, 0, 0)
    assert abs(recovered - true_beta0) < 0.3, (
        f"Intercept recovery failed: true={true_beta0}, got={recovered:.4f}"
    )


# ---------------------------------------------------------------------------
# TEST 2: LOGIT 1-group, linear
# ---------------------------------------------------------------------------
@pytest.mark.recovery
def test_logit_1group_linear():
    """Both betas of a linear 1-group LOGIT recovered within 0.5."""
    true_betas = [-1.0, 1.5]
    df, truth = simulate_logit_trajectories(
        n_subjects=200,
        time_points=T10,
        group_params=[{'betas': true_betas}],
        group_proportions=[1.0],
        seed=2,
    )

    m = run_single_model(df, orders_list=[1], dist='LOGIT', n_starts=N_STARTS)

    assert m['result'].success or m['result'].status == 2
    assert np.isfinite(m['ll']) and m['ll'] < 0

    for coef_idx, true_val in enumerate(true_betas):
        got = _beta_at(m, 0, coef_idx)
        assert abs(got - true_val) < 0.5, (
            f"beta[{coef_idx}] recovery failed: true={true_val}, got={got:.4f}"
        )


# ---------------------------------------------------------------------------
# TEST 3: LOGIT 2-group recovery
# ---------------------------------------------------------------------------
@pytest.mark.recovery
@pytest.mark.slow
def test_logit_2group_recovery():
    """AutoSearch selects 2-group model and achieves >=85% assignment accuracy."""
    true_params = [
        {'betas': [-2.0,  0.5]},   # Group 1: low-risk, slowly rising
        {'betas': [ 0.5, -0.3]},   # Group 2: high-risk, slowly falling
    ]
    df, truth = simulate_logit_trajectories(
        n_subjects=500,
        time_points=T15,
        group_params=true_params,
        group_proportions=[0.60, 0.40],
        seed=3,
    )

    top_models, _ = run_autotraj(
        df, min_groups=1, max_groups=3,
        min_order=0, max_order=2,
        min_group_pct=5.0, p_val_thresh=0.10,
        dist='LOGIT', n_starts=N_STARTS,
    )

    assert len(top_models) > 0, "AutoSearch returned no valid models"

    best = _best_model(top_models)
    k = len(best['orders'])
    assert k == 2, f"Expected 2-group model; BIC selected k={k}"

    # Group proportion recovery (within 10 pp)
    pis = best['pis']
    assert abs(pis[0] - 0.60) < 0.10 or abs(pis[1] - 0.60) < 0.10, (
        f"Group proportions off: {[f'{p:.1%}' for p in pis]}, true=[60%, 40%]"
    )

    # Assignment accuracy
    assignments = get_subject_assignments(best, df)
    acc = _assignment_accuracy(assignments, truth)
    assert acc >= 0.85, f"Assignment accuracy too low: {acc:.1%} (need >=85%)"


# ---------------------------------------------------------------------------
# TEST 4: LOGIT 3-group recovery
# ---------------------------------------------------------------------------
@pytest.mark.recovery
@pytest.mark.slow
def test_logit_3group_recovery():
    """AutoSearch selects 3-group model from well-separated data, >80% accuracy."""
    # All three groups have both intercept and slope clearly non-zero so that
    # the p-value filter (which checks the highest-order term) does not
    # erroneously reject the 3-group models.
    # Trajectories on [-1, 1]:
    #   Group 1: logit(-4→-2) ≈  2–12 %   (low-risk, rising)
    #   Group 2: logit( 0→1 ) ≈ 50–73 %   (moderate, rising)
    #   Group 3: logit( 3.5→1.5) ≈ 97–82% (high-risk, falling)
    # No pair of trajectories crosses → clear separation throughout.
    true_params = [
        {'betas': [-3.0,  1.0]},   # Group 1: low risk, rising
        {'betas': [ 0.5,  0.5]},   # Group 2: moderate risk, gently rising
        {'betas': [ 2.5, -1.0]},   # Group 3: high risk, falling
    ]
    df, truth = simulate_logit_trajectories(
        n_subjects=800,
        time_points=T20,
        group_params=true_params,
        group_proportions=[0.40, 0.35, 0.25],
        seed=4,
    )

    top_models, _ = run_autotraj(
        df, min_groups=1, max_groups=4,
        min_order=0, max_order=1,
        min_group_pct=5.0, p_val_thresh=0.10,
        dist='LOGIT', n_starts=N_STARTS,
    )

    assert len(top_models) > 0, "AutoSearch returned no valid models"

    best = _best_model(top_models)
    k = len(best['orders'])
    assert k == 3, f"Expected 3-group model; BIC selected k={k}"

    assignments = get_subject_assignments(best, df)
    acc = _assignment_accuracy(assignments, truth)
    assert acc >= 0.80, f"Assignment accuracy too low: {acc:.1%} (need >=80%)"


# ---------------------------------------------------------------------------
# TEST 5: CNORM 1-group, sigma recovery
# ---------------------------------------------------------------------------
@pytest.mark.recovery
def test_cnorm_1group_recovery():
    """Sigma recovered within 20% of true value (sigma=2.0)."""
    true_sigma  = 2.0
    true_beta0  = 1.5
    df, truth = simulate_cnorm_trajectories(
        n_subjects=200,
        time_points=T10,
        group_params=[{'betas': [true_beta0]}],
        group_proportions=[1.0],
        sigma=true_sigma,
        cnorm_min=-5.0,
        cnorm_max=8.0,
        seed=5,
    )

    m = run_single_model(
        df, orders_list=[0], dist='CNORM',
        cnorm_min=-5.0, cnorm_max=8.0,
        n_starts=N_STARTS,
    )

    assert m['result'].success or m['result'].status == 2
    assert np.isfinite(m['ll'])

    # Sigma is stored as log-sigma in last param; exp to get sigma
    raw_sigma = m['result'].x[-1]
    recovered_sigma = float(np.exp(raw_sigma))
    rel_err = abs(recovered_sigma - true_sigma) / true_sigma
    assert rel_err < 0.20, (
        f"Sigma recovery failed: true={true_sigma}, got={recovered_sigma:.4f} "
        f"({rel_err:.1%} error)"
    )

    # Intercept recovery
    recovered_beta0 = _beta_at(m, 0, 0)
    assert abs(recovered_beta0 - true_beta0) < 0.5, (
        f"Intercept recovery failed: true={true_beta0}, got={recovered_beta0:.4f}"
    )


# ---------------------------------------------------------------------------
# TEST 6: CNORM 2-group recovery
# ---------------------------------------------------------------------------
@pytest.mark.recovery
@pytest.mark.slow
def test_cnorm_2group_recovery():
    """CNORM 2-group: sigma recovered, groups assigned with >=80% accuracy."""
    true_sigma = 0.8
    df, truth = simulate_cnorm_trajectories(
        n_subjects=500,
        time_points=T15,
        group_params=[
            {'betas': [1.0, -2.5]},   # Group 1: starts high (~3.5 at t=-1), falls sharply
            {'betas': [4.5,  0.0]},   # Group 2: flat at 4.5 — well-separated
        ],
        group_proportions=[0.55, 0.45],
        sigma=true_sigma,
        cnorm_min=0.0,
        cnorm_max=6.0,
        seed=6,
    )

    top_models, _ = run_autotraj(
        df, min_groups=1, max_groups=3,
        min_order=0, max_order=2,
        min_group_pct=5.0, p_val_thresh=0.10,
        dist='CNORM', cnorm_min=0.0, cnorm_max=6.0,
        n_starts=N_STARTS,
    )

    assert len(top_models) > 0, "AutoSearch returned no valid CNORM models"

    best = _best_model(top_models)
    k = len(best['orders'])
    assert k == 2, f"Expected 2-group model; BIC selected k={k}"

    # Sigma check (within 30% — autotraj may fit different order poly)
    raw_sigma = best['result'].x[-1]
    recovered_sigma = float(np.exp(raw_sigma))
    rel_err = abs(recovered_sigma - true_sigma) / true_sigma
    assert rel_err < 0.30, (
        f"Sigma recovery failed: true={true_sigma}, got={recovered_sigma:.4f} "
        f"({rel_err:.1%} error)"
    )

    # Assignment accuracy
    assignments = get_subject_assignments(best, df)
    acc = _assignment_accuracy(assignments, truth)
    assert acc >= 0.80, f"Assignment accuracy too low: {acc:.1%} (need >=80%)"


# ---------------------------------------------------------------------------
# TEST 7: Poisson 2-group recovery
# ---------------------------------------------------------------------------
@pytest.mark.recovery
@pytest.mark.slow
def test_poisson_2group_recovery():
    """Poisson 2-group: correct k selected, intercepts recover count levels."""
    # Group 1: low counts exp(0.5) ≈ 1.6;  Group 2: high counts exp(2.0) ≈ 7.4
    true_params = [
        {'betas': [0.5,  0.3]},
        {'betas': [2.0, -0.2]},
    ]
    df, truth = simulate_poisson_trajectories(
        n_subjects=500,
        time_points=T10,
        group_params=true_params,
        group_proportions=[0.60, 0.40],
        seed=7,
    )

    top_models, _ = run_autotraj(
        df, min_groups=1, max_groups=3,
        min_order=0, max_order=2,
        min_group_pct=5.0, p_val_thresh=0.10,
        dist='POISSON', n_starts=N_STARTS,
    )

    assert len(top_models) > 0, "AutoSearch returned no valid Poisson models"

    best = _best_model(top_models)
    k = len(best['orders'])
    assert k == 2, f"Expected 2-group model; BIC selected k={k}"

    # Intercepts should clearly separate the two count levels
    b0_g0 = _beta_at(best, 0, 0)
    b0_g1 = _beta_at(best, 1, 0)
    # One intercept should be near 0.5 and the other near 2.0
    recovered_b0s = sorted([b0_g0, b0_g1])
    true_b0s_sorted = sorted([true_params[0]['betas'][0], true_params[1]['betas'][0]])
    for rec, true_val in zip(recovered_b0s, true_b0s_sorted):
        assert abs(rec - true_val) < 0.5, (
            f"Poisson intercept recovery failed: true={true_val}, got={rec:.4f}"
        )


# ---------------------------------------------------------------------------
# TEST 8: BIC selects correct K on 2-group data
# ---------------------------------------------------------------------------
@pytest.mark.recovery
@pytest.mark.slow
def test_bic_selects_correct_k():
    """BIC should prefer k=2 when data is generated from exactly 2 groups."""
    df, truth = simulate_logit_trajectories(
        n_subjects=600,
        time_points=T15,
        group_params=[
            {'betas': [-2.5,  0.5]},
            {'betas': [ 1.0, -0.4]},
        ],
        group_proportions=[0.55, 0.45],
        seed=8,
    )

    top_models, all_eval = run_autotraj(
        df, min_groups=1, max_groups=4,
        min_order=0, max_order=2,
        min_group_pct=5.0, p_val_thresh=0.10,
        dist='LOGIT', n_starts=N_STARTS,
    )

    assert len(top_models) > 0, "AutoSearch returned no valid models"

    best = _best_model(top_models)
    k = len(best['orders'])
    model_summary = [(len(m['orders']), round(m['bic_nagin'], 1)) for m in top_models]
    assert k == 2, (
        f"BIC selected k={k} but data was generated from k=2. "
        f"All valid models (k, BIC): {model_summary}"
    )


# ---------------------------------------------------------------------------
# TEST 9: Homogeneous data selects k=1 (anti-Skardhamar test)
# ---------------------------------------------------------------------------
@pytest.mark.recovery
@pytest.mark.slow
def test_single_group_data_selects_1():
    """BIC should NOT hallucinate groups in homogeneous 1-group data.

    This guards against the Skardhamar critique: GBTM artificially creates
    groups even when the population is homogeneous.  When data truly comes
    from a single trajectory, BIC should select k=1.
    """
    df, truth = simulate_logit_trajectories(
        n_subjects=500,
        time_points=T15,
        group_params=[{'betas': [-1.0, 0.5]}],   # one group, linear trend
        group_proportions=[1.0],
        seed=9,
    )

    top_models, _ = run_autotraj(
        df, min_groups=1, max_groups=3,
        min_order=0, max_order=2,
        min_group_pct=5.0, p_val_thresh=0.05,
        dist='LOGIT', n_starts=N_STARTS,
    )

    assert len(top_models) > 0, (
        "AutoSearch returned no valid models — 1-group homogeneous data should "
        "always produce at least one valid model"
    )

    best = _best_model(top_models)
    k = len(best['orders'])
    assert k == 1, (
        f"GBTM hallucinated {k} groups on homogeneous 1-group data. "
        f"This is the Skardhamar over-grouping problem. "
        f"BIC (Nagin): {best['bic_nagin']:.2f}"
    )


# ---------------------------------------------------------------------------
# V3.0: mixing covariates (Gamma) and time-varying covariates (delta)
# ---------------------------------------------------------------------------

def test_logit_2group_mixing_covariate_recovery():
    """Recovered Gamma (mixing-covariate coefficients) matches ground truth,
    and pi_g(x) evaluated at representative covariate values matches the
    simulated covariate-dependent group probabilities (not just point-estimate
    closeness on Gamma itself, since pi is the user-facing quantity)."""
    true_params = [
        {'betas': [-1.5]},          # Group 1 (lower intercept -> sorted first)
        {'betas': [1.0, -0.3]},     # Group 2
    ]
    gamma_matrix = [[0.0, 0.0], [0.2, 1.2]]  # theta_1(x) = 0.2 + 1.2*x
    df, truth = simulate_logit_with_mixing_covariates(
        n_subjects=800, time_points=T15, group_params=true_params,
        gamma_matrix=gamma_matrix, cov_mean=0.0, cov_sd=1.0, seed=11,
    )

    m = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS,
                          baseline_cov_cols=['X1'])

    assert m['result'].success or m['result'].status == 2
    assert m['n_mix'] == 2

    gamma_est = m['result'].x[0:2]  # (k-1)*n_mix = 1*2 = 2 params: [intercept, slope]
    assert abs(gamma_est[0] - 0.2) < 0.5, f"Gamma intercept off: {gamma_est[0]:.3f} (true 0.2)"
    assert abs(gamma_est[1] - 1.2) < 0.5, f"Gamma slope off: {gamma_est[1]:.3f} (true 1.2)"

    # pi_1(x) at representative x values (k=2 -> softmax reduces to logistic(theta_1))
    for x_val in (-1.0, 0.0, 1.0):
        true_theta1 = 0.2 + 1.2 * x_val
        true_pi1 = 1.0 / (1.0 + np.exp(-true_theta1))
        est_theta1 = gamma_est[0] + gamma_est[1] * x_val
        est_pi1 = 1.0 / (1.0 + np.exp(-est_theta1))
        assert abs(est_pi1 - true_pi1) < 0.15, (
            f"pi_1(x={x_val}) off: est={est_pi1:.3f}, true={true_pi1:.3f}"
        )


def test_logit_2group_tvc_recovery():
    """Recovered delta (TVC deflection) matches ground truth per group, and
    beta/assignment recovery is not degraded by the added TVC term (guards
    against gradient-slot mix-ups between the beta and delta blocks)."""
    true_params = [
        {'betas': [-1.5]},
        {'betas': [1.0, -0.3]},
    ]
    delta_true = [0.8, -1.0]
    df, truth = simulate_logit_with_tvc(
        n_subjects=800, time_points=T15, group_params=true_params,
        group_proportions=[0.6, 0.4], delta_per_group=delta_true, tvc_sd=1.0, seed=13,
    )

    m = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS, tvc_cols=['Z1'])

    assert m['result'].success or m['result'].status == 2
    assert m['n_tvc'] == 1

    k, n_mix = 2, 1
    num_betas = 1 + 2  # orders [0, 1] -> group0 has 1 coef, group1 has 2 coefs
    delta_start = (k - 1) * n_mix + num_betas
    delta_est = m['result'].x[delta_start: delta_start + k]

    assert abs(delta_est[0] - delta_true[0]) < 0.5, f"delta_1 off: {delta_est[0]:.3f} (true {delta_true[0]})"
    assert abs(delta_est[1] - delta_true[1]) < 0.5, f"delta_2 off: {delta_est[1]:.3f} (true {delta_true[1]})"

    # Beta/assignment recovery not degraded by the added TVC term
    assignments = get_subject_assignments(m, df)
    acc = _assignment_accuracy(assignments, truth)
    assert acc >= 0.80, f"Assignment accuracy degraded by TVC term: {acc:.1%}"


def test_logit_2group_covariate_and_tvc_joint_recovery():
    """Both new V3.0 blocks (Gamma and delta) recovered simultaneously — a
    direct regression test for index-arithmetic bugs from inserting two new
    blocks into the flat parameter vector at once."""
    true_params = [
        {'betas': [-1.5]},
        {'betas': [1.0, -0.3]},
    ]
    gamma_matrix = [[0.0, 0.0], [0.2, 1.2]]
    delta_true = [0.8, -1.0]
    df, truth = simulate_logit_with_covariates_and_tvc(
        n_subjects=1000, time_points=T15, group_params=true_params,
        gamma_matrix=gamma_matrix, delta_per_group=delta_true,
        cov_mean=0.0, cov_sd=1.0, tvc_sd=1.0, seed=17,
    )

    m = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS,
                          baseline_cov_cols=['X1'], tvc_cols=['Z1'])

    assert m['result'].success or m['result'].status == 2
    assert m['n_mix'] == 2 and m['n_tvc'] == 1

    gamma_est = m['result'].x[0:2]
    assert abs(gamma_est[0] - 0.2) < 0.6
    assert abs(gamma_est[1] - 1.2) < 0.6

    k, n_mix = 2, 2
    num_betas = 1 + 2
    delta_start = (k - 1) * n_mix + num_betas
    delta_est = m['result'].x[delta_start: delta_start + k]
    assert abs(delta_est[0] - delta_true[0]) < 0.6
    assert abs(delta_est[1] - delta_true[1]) < 0.6


def test_backward_compat_zero_covariates_matches_v15():
    """Fitting with no covariates (default) vs. explicit empty covariate lists
    must produce bit-identical results — the concrete regression guard for
    V3.0's 'P=0, Q=0 reduces to V1.5.0 exactly' design invariant."""
    df, _ = simulate_logit_trajectories(
        n_subjects=300, time_points=T15,
        group_params=[{'betas': [-1.5]}, {'betas': [1.0, -0.3]}],
        group_proportions=[0.6, 0.4], seed=21,
    )

    m_old = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS)
    m_new = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS,
                              baseline_cov_cols=[], tvc_cols=[])

    assert m_old['ll'] == pytest.approx(m_new['ll'], abs=1e-8)
    assert np.allclose(m_old['result'].x, m_new['result'].x, atol=1e-8), (
        "Parameter vectors differ between implicit and explicit no-covariate calls"
    )
    assert np.allclose(m_old['se_model'], m_new['se_model'], atol=1e-6)


def test_null_mixing_covariate_does_not_destabilize():
    """An irrelevant baseline covariate (no true effect on group membership)
    should recover a near-zero Gamma slope and must not destabilize
    convergence or assignment accuracy."""
    df, truth = simulate_logit_trajectories(
        n_subjects=500, time_points=T15,
        group_params=[{'betas': [-1.5]}, {'betas': [1.0, -0.3]}],
        group_proportions=[0.6, 0.4], seed=31,
    )
    rng = np.random.default_rng(999)
    noise_by_id = {sid: rng.normal() for sid in df['ID'].unique()}
    df = df.copy()
    df['Noise'] = df['ID'].map(noise_by_id)

    m = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS,
                          baseline_cov_cols=['Noise'])

    assert m['result'].success or m['result'].status == 2
    gamma_slope = m['result'].x[1]  # (k-1)*n_mix=2 params: [intercept, slope]
    assert abs(gamma_slope) < 0.6, f"Null covariate slope spuriously large: {gamma_slope:.3f}"

    assignments = get_subject_assignments(m, df)
    acc = _assignment_accuracy(assignments, truth)
    assert acc >= 0.80, f"Null covariate destabilized assignment accuracy: {acc:.1%}"


def test_baseline_covariate_constancy_guard():
    """A covariate that genuinely varies within subject must be rejected with
    a clear error when supplied as a baseline covariate — it should be
    supplied as a TVC instead. Regression test for the identifiability
    guard in build_baseline_covariate_matrix."""
    df, _ = simulate_logit_with_tvc(
        n_subjects=50, time_points=T15,
        group_params=[{'betas': [-1.5]}, {'betas': [1.0, -0.3]}],
        group_proportions=[0.6, 0.4], delta_per_group=[0.5, -0.5], seed=41,
    )
    with pytest.raises(ValueError, match="varies within subject"):
        run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=1, baseline_cov_cols=['Z1'])


# ---------------------------------------------------------------------------
# V4.0: survey (sampling) weights
# ---------------------------------------------------------------------------

def test_weighted_recovery_corrects_sampling_bias():
    """Weighted fit on a biased (informatively undersampled) sample must recover
    the TRUE population group proportions more closely than an unweighted fit
    on the same biased sample — the test that demonstrates weights are
    actually doing something, not just plumbing."""
    true_params = [
        {'betas': [-1.5]},
        {'betas': [1.0, -0.3]},
    ]
    df, truth = simulate_logit_with_biased_sampling_weights(
        n_population=3000, time_points=T15, group_params=true_params,
        group_proportions=[0.5, 0.5], keep_probs=[1.0, 0.3], seed=51,
    )

    m_unweighted = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS)
    m_weighted = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS,
                                   weight_col='Weight')

    true_prop = 0.5
    # Label switching may put either group first; compare against the closer match.
    unweighted_err = min(abs(p - true_prop) for p in m_unweighted['pis'])
    weighted_err = min(abs(p - true_prop) for p in m_weighted['pis'])

    assert weighted_err < unweighted_err, (
        f"Weighted fit should recover the true 50/50 population split better than "
        f"unweighted: unweighted pis={m_unweighted['pis']}, weighted pis={m_weighted['pis']}"
    )
    assert weighted_err < 0.15, f"Weighted proportion recovery too far off: {m_weighted['pis']}"


def test_backward_compat_uniform_weights_matches_unweighted():
    """Fitting with weight_col=None vs. an all-1.0 weight column must produce
    bit-identical results — the concrete regression guard for V4.0's
    'w_i=1 reduces to V3.0 exactly' design invariant."""
    df, _ = simulate_logit_trajectories(
        n_subjects=300, time_points=T15,
        group_params=[{'betas': [-1.5]}, {'betas': [1.0, -0.3]}],
        group_proportions=[0.6, 0.4], seed=61,
    )
    df = df.copy()
    df['W'] = 1.0

    m_unweighted = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS)
    m_weighted = run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=N_STARTS,
                                   weight_col='W')

    assert m_unweighted['ll'] == pytest.approx(m_weighted['ll'], abs=1e-8)
    assert np.allclose(m_unweighted['result'].x, m_weighted['result'].x, atol=1e-8), (
        "Parameter vectors differ between unweighted and all-ones-weighted calls"
    )
    assert np.allclose(m_unweighted['se_model'], m_weighted['se_model'], atol=1e-6)
    assert np.allclose(m_unweighted['se_robust'], m_weighted['se_robust'], atol=1e-6)


def test_weight_column_constancy_guard():
    """A weight column that varies within subject must be rejected with a clear
    error — survey weights are inherently one-per-subject."""
    df, _ = simulate_logit_trajectories(
        n_subjects=50, time_points=T15,
        group_params=[{'betas': [-1.5]}, {'betas': [1.0, -0.3]}],
        group_proportions=[0.6, 0.4], seed=71,
    )
    rng = np.random.default_rng(0)
    df = df.copy()
    df['W'] = rng.uniform(0.5, 2.0, size=len(df))  # varies row-by-row, not per subject
    with pytest.raises(ValueError, match="varies within subject"):
        run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=1, weight_col='W')


def test_non_positive_weight_rejected():
    """A weight column containing zero, negative, or missing values must be
    rejected with a clear error."""
    df, _ = simulate_logit_trajectories(
        n_subjects=50, time_points=T15,
        group_params=[{'betas': [-1.5]}, {'betas': [1.0, -0.3]}],
        group_proportions=[0.6, 0.4], seed=72,
    )
    df = df.copy()
    first_id = df['ID'].unique()[0]
    weight_by_id = {sid: (0.0 if sid == first_id else 1.0) for sid in df['ID'].unique()}
    df['W'] = df['ID'].map(weight_by_id)
    with pytest.raises(ValueError, match="strictly positive"):
        run_single_model(df, orders_list=[0, 1], dist='LOGIT', n_starts=1, weight_col='W')
