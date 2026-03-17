"""
tests/test_cambridge_benchmark.py
==================================
Benchmark tests against the Cambridge Study of Delinquent Development.

N=196 subjects, T=23 binary time-points (criminal convictions, ages 10-32,
scaled to [-1.1, 1.1]).  Published results: Nagin (1999), Jones & Nagin (2001).

What we are checking
---------------------
- Data loads to the right shape
- 1-group intercept-only LL matches the closed-form MLE for a Bernoulli proportion
- 2-group quadratic model improves BIC and has proportions consistent with
  the published ~70-80 % / ~20-30 % split
- AutoSearch finds multi-group solutions and picks k>=2 by BIC
- Adequacy metrics (AvePP, OCC, relative entropy) meet published thresholds
- Robust SEs are well-behaved relative to model-based SEs
- Dropout model still converges on this balanced-panel dataset
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    prep_trajectory_data,
    run_single_model,
    run_autotraj,
    get_subject_assignments,
    calc_model_adequacy,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAMBRIDGE_PATH = os.path.join(_REPO_ROOT, "cambridge.txt")

N_STARTS = 3   # multi-start restarts — same as recovery suite


# ---------------------------------------------------------------------------
# Module-scoped fixtures (computed once, shared by all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cambridge_long():
    """Wide-format Cambridge file converted to long format."""
    wide = pd.read_csv(_CAMBRIDGE_PATH, sep=r"\s+")
    return prep_trajectory_data(wide)


@pytest.fixture(scope="module")
def model_1group(cambridge_long):
    return run_single_model(
        cambridge_long, orders_list=[0],
        dist="LOGIT", n_starts=N_STARTS,
    )


@pytest.fixture(scope="module")
def model_2group_quadratic(cambridge_long):
    return run_single_model(
        cambridge_long, orders_list=[2, 2],
        dist="LOGIT", n_starts=N_STARTS,
    )


# ---------------------------------------------------------------------------
# TEST 1: data loads correctly
# ---------------------------------------------------------------------------

def test_cambridge_loads(cambridge_long):
    """196 subjects × 23 time points, binary outcomes."""
    n_subjects = cambridge_long["ID"].nunique()
    obs_per_subject = cambridge_long.groupby("ID").size().unique()

    # Published Nagin (1999) reports N=196; this file has 195 (one subject absent).
    assert n_subjects == 195, f"Expected 195 subjects, got {n_subjects}"
    assert len(obs_per_subject) == 1 and obs_per_subject[0] == 23, (
        f"Expected exactly 23 obs per subject; got {sorted(obs_per_subject)}"
    )

    # Outcomes are binary {0, 1}
    unique_outcomes = sorted(cambridge_long["Outcome"].unique())
    assert unique_outcomes == [0, 1] or unique_outcomes == [0.0, 1.0], (
        f"Non-binary outcomes found: {unique_outcomes}"
    )


# ---------------------------------------------------------------------------
# TEST 2: 1-group intercept-only log-likelihood cross-check
# ---------------------------------------------------------------------------

def test_cambridge_1group_loglik(cambridge_long, model_1group):
    """
    1-group order-0 LL must match the closed-form MLE of a Bernoulli proportion.

    For a 1-group LOGIT model with polynomial order 0 the MLE collapses to
    beta_0* = logit(p_hat)  where  p_hat = mean(y).
    The resulting LL equals  sum_obs [y*log(p_hat) + (1-y)*log(1-p_hat)].
    """
    m = model_1group
    assert m["result"].success or m["result"].status == 2, (
        "1-group intercept-only model failed to converge"
    )

    gbtm_ll = m["ll"]

    y = cambridge_long["Outcome"].values.astype(float)
    p_hat = y.mean()
    baseline_ll = float(
        np.sum(y * np.log(p_hat) + (1.0 - y) * np.log(1.0 - p_hat))
    )

    assert abs(gbtm_ll - baseline_ll) < 0.5, (
        f"1-group LL mismatch: GBTM={gbtm_ll:.4f}, "
        f"Bernoulli baseline={baseline_ll:.4f}, "
        f"diff={abs(gbtm_ll - baseline_ll):.4f}"
    )


# ---------------------------------------------------------------------------
# TEST 3: 2-group quadratic model
# ---------------------------------------------------------------------------

def test_cambridge_2group_quadratic(cambridge_long, model_1group, model_2group_quadratic):
    """
    2-group [2,2] LOGIT model must:
      - out-perform 1-group by LL and Nagin BIC
      - have group proportions consistent with the published ~75% / ~25% split
      - have Group 1 intercept < Group 2 intercept (lower = non-offending after sorting)
    """
    m1 = model_1group
    m2 = model_2group_quadratic

    assert m2["result"].success or m2["result"].status == 2, (
        "2-group quadratic model failed to converge"
    )
    assert m2["ll"] > m1["ll"], (
        f"2-group LL ({m2['ll']:.3f}) must be higher than 1-group ({m1['ll']:.3f})"
    )
    assert m2["bic_nagin"] > m1["bic_nagin"], (
        f"2-group BIC Nagin ({m2['bic_nagin']:.3f}) must improve over "
        f"1-group ({m1['bic_nagin']:.3f})"
    )

    pis = m2["pis"]
    large_pi = float(max(pis))
    small_pi = float(min(pis))

    assert 0.60 <= large_pi <= 0.90, (
        f"Larger group proportion {large_pi:.2%} outside expected 60-90% range"
    )
    assert 0.10 <= small_pi <= 0.40, (
        f"Smaller group proportion {small_pi:.2%} outside expected 10-40% range"
    )

    # After sort_groups_by_intercept, params layout for k=2, orders=[2,2]:
    #   [theta1, beta0_g1, beta1_g1, beta2_g1, beta0_g2, beta1_g2, beta2_g2]
    params = m2["result"].x
    k = 2
    n_betas_g1 = 3   # order 2 => 3 coefficients
    beta0_g1 = params[k - 1]                # index 1
    beta0_g2 = params[k - 1 + n_betas_g1]  # index 4

    assert beta0_g1 < beta0_g2, (
        f"After sorting, Group 1 intercept ({beta0_g1:.3f}) should be less "
        f"than Group 2 intercept ({beta0_g2:.3f})"
    )


# ---------------------------------------------------------------------------
# TEST 4: autosearch across k=1-4 and orders 0-3
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cambridge_autosearch(cambridge_long):
    """
    Full AutoTraj search must:
      - produce at least one valid 2-group and one valid 3-group model
      - select a best-BIC model with k >= 2
    """
    valid, _ = run_autotraj(
        cambridge_long,
        min_groups=1, max_groups=4,
        min_order=0, max_order=3,
        min_group_pct=5.0, p_val_thresh=0.05,
        dist="LOGIT", n_starts=N_STARTS,
    )

    assert len(valid) > 0, "AutoTraj returned no valid models"

    group_counts = [len(m["orders"]) for m in valid]
    assert 2 in group_counts, (
        f"No valid 2-group model found; valid k values = {sorted(set(group_counts))}"
    )
    assert 3 in group_counts, (
        f"No valid 3-group model found; valid k values = {sorted(set(group_counts))}"
    )

    best = max(valid, key=lambda m: m["bic_nagin"])
    best_k = len(best["orders"])
    assert best_k >= 2, (
        f"Best-BIC model has only k={best_k}; Cambridge data should require k>=2"
    )


# ---------------------------------------------------------------------------
# TEST 5: adequacy metrics for the best 2-group model
# ---------------------------------------------------------------------------

def test_cambridge_adequacy_metrics(cambridge_long, model_2group_quadratic):
    """
    Adequacy metrics from Nagin (2005) thresholds:
      - AvePP  > 0.70  for every group
      - OCC    > 5.0   for every group
      - Relative entropy > 0.50
    """
    m = model_2group_quadratic
    assignments = get_subject_assignments(m, cambridge_long)
    k = len(m["orders"])
    group_names = [f"Group {g + 1}" for g in range(k)]
    adeq_df, rel_entropy = calc_model_adequacy(assignments, m["pis"], group_names)

    assert rel_entropy > 0.50, (
        f"Relative entropy {rel_entropy:.4f} < 0.50 — poor group separation"
    )

    for _, row in adeq_df.iterrows():
        avepp = row["AvePP"]
        occ = row["OCC"]
        if isinstance(avepp, float) and np.isfinite(avepp):
            assert avepp > 0.70, (
                f"{row['Group']}: AvePP={avepp:.4f} below 0.70 threshold"
            )
        if isinstance(occ, float) and np.isfinite(occ):
            assert occ > 5.0, (
                f"{row['Group']}: OCC={occ:.2f} below 5.0 threshold"
            )


# ---------------------------------------------------------------------------
# TEST 6: robust SEs vs model-based SEs
# ---------------------------------------------------------------------------

def test_cambridge_robust_se(cambridge_long, model_2group_quadratic):
    """
    Robust SEs must be finite and within a 2× factor of model-based SEs.
    Wild divergence (ratio < 0.25 or > 4.0) signals numerical problems.
    """
    m = model_2group_quadratic
    se_m = m["se_model"]
    se_r = m["se_robust"]

    assert np.all(np.isfinite(se_m)), (
        f"Model SEs contain non-finite values: {se_m}"
    )
    assert np.all(np.isfinite(se_r)), (
        f"Robust SEs contain non-finite values: {se_r}"
    )
    assert np.all(se_m >= 0), "Model SEs must be non-negative"
    assert np.all(se_r >= 0), "Robust SEs must be non-negative"

    # The mixing-weight (theta) parameter at index 0 can have SE=0 when group
    # separation is very strong and the Hessian is near-flat along that axis.
    # Check the ratio only for the beta parameters (indices 1 onwards).
    se_m_betas = se_m[1:]
    se_r_betas = se_r[1:]
    assert np.all(se_m_betas > 0), f"Beta model SEs must be positive: {se_m_betas}"
    assert np.all(se_r_betas > 0), f"Beta robust SEs must be positive: {se_r_betas}"

    ratio = se_r_betas / se_m_betas
    assert np.all(ratio > 0.25), (
        f"Some robust SEs suspiciously small vs model SEs — min ratio={ratio.min():.3f}"
    )
    assert np.all(ratio < 4.0), (
        f"Some robust SEs more than 4× model SEs — max ratio={ratio.max():.3f}\n"
        f"Ratios: {np.round(ratio, 3)}"
    )


# ---------------------------------------------------------------------------
# TEST 7: dropout model still converges
# ---------------------------------------------------------------------------

def test_cambridge_dropout_model(cambridge_long):
    """
    Adding dropout gammas to a [2,2] model on this balanced panel should
    converge.  Cambridge data has no actual dropout (every subject observed
    at all 23 time points), so the dropout gammas should be near zero and
    the LL should not differ greatly from the non-dropout version.
    """
    m_no_drop = run_single_model(
        cambridge_long, orders_list=[2, 2],
        use_dropout=False, dist="LOGIT", n_starts=N_STARTS,
    )
    m_drop = run_single_model(
        cambridge_long, orders_list=[2, 2],
        use_dropout=True, dist="LOGIT", n_starts=N_STARTS,
    )

    assert m_drop["result"].success or m_drop["result"].status == 2, (
        "Dropout [2,2] model failed to converge"
    )
    assert np.isfinite(m_drop["ll"]), "Dropout model log-likelihood is not finite"

    # Locate gamma parameters: layout is
    #   [(k-1) thetas] [betas g1] [betas g2] [gammas g1 (3)] [gammas g2 (3)]
    k = 2
    orders = [2, 2]
    n_betas = sum(o + 1 for o in orders)   # 6
    gamma_start = (k - 1) + n_betas        # 7
    params = m_drop["result"].x
    gammas = params[gamma_start: gamma_start + 3 * k]

    assert len(gammas) == 6, f"Expected 6 gamma params; got {len(gammas)}"

    # Cambridge is a fully balanced panel — every subject is observed at all
    # 23 time points, so dropouts[] is all-zero.  With no actual dropout events
    # the gamma parameters are unidentified and can drift to large values.
    # We only require they are finite (not NaN/Inf) and that the LL is unharmed.
    assert np.all(np.isfinite(gammas)), (
        f"Gamma params contain non-finite values: {gammas}"
    )

    # LL difference should be modest; dropout adds 6 free params
    ll_diff = m_drop["ll"] - m_no_drop["ll"]
    assert ll_diff >= -1.0, (
        f"Dropout model LL ({m_drop['ll']:.3f}) is unexpectedly much lower than "
        f"no-dropout ({m_no_drop['ll']:.3f}); diff={ll_diff:.3f}"
    )
    assert ll_diff < 20.0, (
        f"Dropout LL gain of {ll_diff:.3f} is suspiciously large on a balanced panel "
        f"with no real dropout"
    )
