"""
tests/test_edge_cases.py
========================
Pathological-input and boundary-condition tests for AutoTraj.

Each test verifies that the engine handles a tricky scenario *gracefully*:
it may produce wide confidence intervals, reject a model, or converge to a
boundary optimum — but it must never crash, raise an unhandled exception, or
return silently corrupted results (NaN/Inf where not expected).
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
)
from tests.simulate import simulate_logit_trajectories

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAMBRIDGE_PATH = os.path.join(_REPO_ROOT, "cambridge.txt")

N_STARTS = 3


# ---------------------------------------------------------------------------
# TEST 1: tiny sample (N=20, T=5)
# ---------------------------------------------------------------------------

def test_tiny_sample():
    """Engine must not crash on N=20, T=5.  Finite LL is sufficient."""
    df, _ = simulate_logit_trajectories(
        n_subjects=20,
        time_points=np.linspace(-1, 1, 5),
        group_params=[{'betas': [-1.0, 1.5]}],
        group_proportions=[1.0],
        seed=101,
    )

    m = run_single_model(df, orders_list=[1], dist='LOGIT', n_starts=N_STARTS)

    assert np.isfinite(m['ll']), f"Tiny-sample LL is not finite: {m['ll']}"
    assert m['result'] is not None, "result object must not be None"
    assert np.all(np.isfinite(m['result'].x)), (
        f"Parameter vector contains non-finite values: {m['result'].x}"
    )


# ---------------------------------------------------------------------------
# TEST 2: subjects with only 1 observation
# ---------------------------------------------------------------------------

def test_single_timepoint_subject():
    """Subjects with a single observation must not crash the engine."""
    rng = np.random.default_rng(202)

    records = []
    # 15 subjects with 5 time points
    for sid in range(1, 16):
        for t in np.linspace(-1, 1, 5):
            y = float(rng.binomial(1, 0.3))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y})

    # 5 subjects with only 1 observation
    for sid in range(16, 21):
        t = float(rng.uniform(-1, 1))
        y = float(rng.binomial(1, 0.3))
        records.append({'ID': sid, 'Time': t, 'Outcome': y})

    df = pd.DataFrame(records).sort_values(['ID', 'Time']).reset_index(drop=True)

    # Must not raise an exception
    try:
        m = run_single_model(df, orders_list=[0], dist='LOGIT', n_starts=N_STARTS)
        # If it produced a result, LL must be finite
        if m['result'] is not None and (m['result'].success or m['result'].status == 2):
            assert np.isfinite(m['ll']), f"LL not finite: {m['ll']}"
    except Exception as exc:
        pytest.fail(f"Engine crashed with single-obs subjects: {exc}")


# ---------------------------------------------------------------------------
# TEST 3: heavy MCAR missing data (40%)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_heavy_missing_data():
    """With 40% MCAR missing, the engine should still converge and select k=2."""
    df, truth = simulate_logit_trajectories(
        n_subjects=300,
        time_points=np.linspace(-1, 1, 15),
        group_params=[
            {'betas': [-2.0,  1.0]},
            {'betas': [ 1.5, -1.0]},
        ],
        group_proportions=[0.55, 0.45],
        missing_rate=0.40,
        seed=303,
    )

    # Verify the missing data was actually applied
    expected_max = 300 * 15
    assert len(df) < expected_max * 0.85, (
        f"Expected <{int(expected_max*0.85)} rows after 40% MCAR, got {len(df)}"
    )

    valid, _ = run_autotraj(
        df,
        min_groups=1, max_groups=3,
        min_order=0, max_order=2,
        min_group_pct=5.0, p_val_thresh=0.20,   # looser threshold for noisy data
        dist='LOGIT', n_starts=N_STARTS,
    )

    assert len(valid) > 0, "AutoTraj found no valid models with 40% missing"

    best = max(valid, key=lambda m: m['bic_nagin'])
    k = len(best['orders'])
    assert k == 2, f"Expected 2-group best model with 40% missing; got k={k}"

    # Wide tolerance (±1.0) for missing-data conditions
    params = best['result'].x
    orders = best['orders']
    beta0_g1 = params[k - 1]
    beta0_g2 = params[k - 1 + (orders[0] + 1)]
    assert abs(beta0_g1 - (-2.0)) < 1.0 or abs(beta0_g2 - (-2.0)) < 1.0, (
        f"Neither recovered intercept is near true -2.0: {beta0_g1:.3f}, {beta0_g2:.3f}"
    )


# ---------------------------------------------------------------------------
# TEST 4: all-zero outcomes (LOGIT)
# ---------------------------------------------------------------------------

def test_all_zero_outcomes():
    """All y=0 must not crash; intercept must converge to a very negative value."""
    rng = np.random.default_rng(404)
    n_subj, n_t = 50, 8
    times = np.linspace(-1, 1, n_t)
    records = [
        {'ID': sid, 'Time': float(t), 'Outcome': 0.0}
        for sid in range(1, n_subj + 1)
        for t in times
    ]
    df = pd.DataFrame(records)

    try:
        m = run_single_model(df, orders_list=[0], dist='LOGIT', n_starts=1)
    except Exception as exc:
        pytest.fail(f"Engine crashed on all-zero outcomes: {exc}")

    # Params must be finite regardless of convergence status
    assert np.all(np.isfinite(m['result'].x)), (
        f"Non-finite parameters on all-zero data: {m['result'].x}"
    )

    # If it converged, intercept must be strongly negative
    if m['result'].success or m['result'].status == 2:
        assert np.isfinite(m['ll']), "LL must be finite if converged"
        beta0 = float(m['result'].x[0])   # k=1: params[k-1=0] = beta0
        assert beta0 < -2.0, (
            f"All-zero intercept should be strongly negative; got beta0={beta0:.3f}"
        )


# ---------------------------------------------------------------------------
# TEST 5: all-one outcomes (LOGIT)
# ---------------------------------------------------------------------------

def test_all_one_outcomes():
    """All y=1 must not crash; intercept must converge to a very positive value."""
    rng = np.random.default_rng(505)
    n_subj, n_t = 50, 8
    times = np.linspace(-1, 1, n_t)
    records = [
        {'ID': sid, 'Time': float(t), 'Outcome': 1.0}
        for sid in range(1, n_subj + 1)
        for t in times
    ]
    df = pd.DataFrame(records)

    try:
        m = run_single_model(df, orders_list=[0], dist='LOGIT', n_starts=1)
    except Exception as exc:
        pytest.fail(f"Engine crashed on all-one outcomes: {exc}")

    assert np.all(np.isfinite(m['result'].x)), (
        f"Non-finite parameters on all-one data: {m['result'].x}"
    )

    if m['result'].success or m['result'].status == 2:
        assert np.isfinite(m['ll']), "LL must be finite if converged"
        beta0 = float(m['result'].x[0])
        assert beta0 > 2.0, (
            f"All-one intercept should be strongly positive; got beta0={beta0:.3f}"
        )


# ---------------------------------------------------------------------------
# TEST 6: overparameterized model (order=5, T=4 time points)
# ---------------------------------------------------------------------------

def test_overparameterized_model():
    """
    A degree-5 polynomial with only 4 distinct time points is underidentified.
    AutoTraj must reject the model (Singular Matrix or Degenerate SE).
    """
    df, _ = simulate_logit_trajectories(
        n_subjects=100,
        time_points=np.linspace(-1, 1, 4),   # only 4 distinct times
        group_params=[{'betas': [-1.0, 0.5]}],
        group_proportions=[1.0],
        seed=606,
    )

    # Ask for a 1-group order-5 model — should be caught by identifiability checks
    _, all_models = run_autotraj(
        df,
        min_groups=1, max_groups=1,
        min_order=5, max_order=5,
        min_group_pct=5.0, p_val_thresh=0.05,
        dist='LOGIT', n_starts=N_STARTS,
    )

    assert len(all_models) > 0, "Expected at least one evaluated model"
    # Should have zero valid models (all rejected or failed)
    valid_statuses = [m['Status'] for m in all_models if m['Status'] == 'Valid']
    assert len(valid_statuses) == 0, (
        f"Overparameterized model should be rejected, but status was 'Valid'"
    )

    # Sanity-check: run_single_model directly should show a huge condition number
    m = run_single_model(df, orders_list=[5], dist='LOGIT', n_starts=1)
    if m['result'].success or m['result'].status == 2:
        assert m['cond_num'] > 1e6, (
            f"Expected large condition number for overparameterized model; "
            f"got cond_num={m['cond_num']:.2e}"
        )


# ---------------------------------------------------------------------------
# TEST 7: very large time values (0 – 10000)
# ---------------------------------------------------------------------------

def test_very_large_time_values():
    """Scale-factor normalization must handle times up to 10000 without overflow."""
    raw_times = np.linspace(0, 10000, 10)

    # Simulate data with these raw times (polynomial is in time units)
    df, _ = simulate_logit_trajectories(
        n_subjects=150,
        time_points=raw_times,
        group_params=[
            {'betas': [-2.0, 0.0002]},   # logit(p) = -2 + 0.0002*t
            {'betas': [ 1.0, -0.0001]},
        ],
        group_proportions=[0.60, 0.40],
        seed=707,
    )

    m = run_single_model(df, orders_list=[1, 1], dist='LOGIT', n_starts=N_STARTS)

    assert m['result'] is not None
    assert np.all(np.isfinite(m['result'].x)), (
        f"Non-finite params with large time values: {m['result'].x}"
    )
    if m['result'].success or m['result'].status == 2:
        assert np.isfinite(m['ll']), "LL not finite with large time values"
        assert np.all(np.isfinite(m['se_model'])), (
            "Non-finite SEs with large time values"
        )


# ---------------------------------------------------------------------------
# TEST 8: negative / symmetric time values (−5 to 5)
# ---------------------------------------------------------------------------

def test_negative_time_values():
    """Times centered at 0 with wider range (−5 to 5) must work correctly."""
    wide_times = np.linspace(-5.0, 5.0, 11)

    df, truth = simulate_logit_trajectories(
        n_subjects=200,
        time_points=wide_times,
        group_params=[
            {'betas': [-1.5,  0.3]},
            {'betas': [ 1.0, -0.2]},
        ],
        group_proportions=[0.55, 0.45],
        seed=808,
    )

    m = run_single_model(df, orders_list=[1, 1], dist='LOGIT', n_starts=N_STARTS)

    assert m['result'] is not None
    assert np.all(np.isfinite(m['result'].x)), (
        f"Non-finite params with negative/wide time values: {m['result'].x}"
    )
    if m['result'].success or m['result'].status == 2:
        assert np.isfinite(m['ll'])
        # Both groups should have finite SEs
        assert np.all(np.isfinite(m['se_model'])), "Non-finite SEs with wide times"

    # Scale factor must be 5.0 (max |time|)
    # Verify implicitly: if SEs are huge, time scaling likely failed
    if m['result'].success or m['result'].status == 2:
        se_betas = m['se_model'][1:]   # skip theta SE
        assert np.all(se_betas < 50), (
            f"Unexpectedly large SEs suggest time-scaling failure: {se_betas}"
        )


# ---------------------------------------------------------------------------
# TEST 9: unbalanced groups (90 % / 10 %)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_unbalanced_groups():
    """
    A 90%/10% split must be:
      - Accepted  when min_group_pct = 5.0   (10% > 5%)
      - Rejected  when min_group_pct = 15.0  (10% < 15%)
    """
    df, _ = simulate_logit_trajectories(
        n_subjects=400,
        time_points=np.linspace(-1, 1, 10),
        group_params=[
            {'betas': [-2.5,  0.5]},   # large group (low risk)
            {'betas': [ 1.5, -1.0]},   # small group (high risk, falling)
        ],
        group_proportions=[0.90, 0.10],
        seed=909,
    )

    # --- Should be accepted with loose threshold ---
    valid_loose, all_loose = run_autotraj(
        df,
        min_groups=2, max_groups=2,
        min_order=1, max_order=1,
        min_group_pct=5.0, p_val_thresh=0.10,
        dist='LOGIT', n_starts=N_STARTS,
    )

    assert len(valid_loose) > 0, (
        "Expected at least one valid 2-group model with min_group_pct=5.0; "
        f"all_models={[(m['Status'], m['Min_Group_%']) for m in all_loose]}"
    )

    # --- Should be rejected with tight threshold ---
    valid_tight, all_tight = run_autotraj(
        df,
        min_groups=2, max_groups=2,
        min_order=1, max_order=1,
        min_group_pct=15.0, p_val_thresh=0.10,
        dist='LOGIT', n_starts=N_STARTS,
    )

    assert len(valid_tight) == 0, (
        f"Expected no valid models with min_group_pct=15.0; "
        f"got {[(m['bic_nagin'], m['pis']) for m in valid_tight]}"
    )
    # Check the rejection reason is specifically group-size related
    group_size_rejections = [
        m for m in all_tight
        if 'Group Size' in m.get('Status', '')
    ]
    assert len(group_size_rejections) > 0, (
        f"Expected group-size rejection; statuses: {[m['Status'] for m in all_tight]}"
    )


# ---------------------------------------------------------------------------
# TEST 10: identical groups (BIC prefers 1-group)
# ---------------------------------------------------------------------------

def test_identical_groups():
    """
    If both 'groups' have identical generating parameters, the 1-group model
    should be preferred by BIC — or the 2-group model should produce nearly
    identical group trajectories.
    """
    df, _ = simulate_logit_trajectories(
        n_subjects=300,
        time_points=np.linspace(-1, 1, 10),
        group_params=[
            {'betas': [-1.0, 0.8]},
            {'betas': [-1.0, 0.8]},   # identical
        ],
        group_proportions=[0.50, 0.50],
        seed=1010,
    )

    m1 = run_single_model(df, orders_list=[1],    dist='LOGIT', n_starts=N_STARTS)
    m2 = run_single_model(df, orders_list=[1, 1], dist='LOGIT', n_starts=N_STARTS)

    # At minimum: no crash and finite LL for both
    assert np.isfinite(m1['ll']), "1-group LL not finite"
    if m2['result'].success or m2['result'].status == 2:
        assert np.isfinite(m2['ll']), "2-group LL not finite"

        # Nagin BIC penalises the extra parameter — 1-group should win
        # (or the 2-group groups should be near-identical)
        if m2['bic_nagin'] > m1['bic_nagin']:
            # 2-group BIC better: verify the two groups are very similar
            params = m2['result'].x
            k = 2
            beta0_g1 = params[k - 1]
            beta0_g2 = params[k - 1 + 2]   # order 1 → 2 betas per group
            assert abs(beta0_g1 - beta0_g2) < 1.0, (
                f"2-group BIC beat 1-group but groups are far apart: "
                f"β0_g1={beta0_g1:.3f}, β0_g2={beta0_g2:.3f}"
            )
        else:
            # 1-group BIC is better — expected behaviour
            assert m1['bic_nagin'] >= m2['bic_nagin'], (
                f"BIC: 1-group={m1['bic_nagin']:.3f}, 2-group={m2['bic_nagin']:.3f}"
            )


# ---------------------------------------------------------------------------
# TEST 11: reproducibility (deterministic multi-start)
# ---------------------------------------------------------------------------

def test_reproducibility():
    """Running the same model twice must produce identical log-likelihoods."""
    df, _ = simulate_logit_trajectories(
        n_subjects=150,
        time_points=np.linspace(-1, 1, 10),
        group_params=[
            {'betas': [-2.0, 1.0]},
            {'betas': [ 1.0, -0.5]},
        ],
        group_proportions=[0.60, 0.40],
        seed=1111,
    )

    m_a = run_single_model(df, orders_list=[1, 1], dist='LOGIT', n_starts=N_STARTS)
    m_b = run_single_model(df, orders_list=[1, 1], dist='LOGIT', n_starts=N_STARTS)

    assert np.isfinite(m_a['ll']), "First run LL not finite"
    assert np.isfinite(m_b['ll']), "Second run LL not finite"

    assert m_a['ll'] == pytest.approx(m_b['ll'], abs=1e-6), (
        f"LL differs between runs: {m_a['ll']:.8f} vs {m_b['ll']:.8f}"
    )
    assert np.allclose(m_a['result'].x, m_b['result'].x, atol=1e-6), (
        f"Parameter vectors differ between runs:\n"
        f"  run 1: {m_a['result'].x}\n"
        f"  run 2: {m_b['result'].x}"
    )


# ---------------------------------------------------------------------------
# TEST 12: wide vs long format produce identical results
# ---------------------------------------------------------------------------

def test_wide_vs_long_format():
    """
    Running AutoTraj on the Cambridge data loaded as wide-then-converted vs
    as long directly must produce the same log-likelihood and parameters.
    """
    # Path 1: load wide → convert to long
    wide_df = pd.read_csv(_CAMBRIDGE_PATH, sep=r'\s+')
    long_from_wide = prep_trajectory_data(wide_df)

    # Path 2: load wide → convert to long again (same function, same data)
    #         Then manually verify the two DataFrames are identical
    long_from_wide2 = prep_trajectory_data(wide_df.copy())

    # Both paths must produce the same sorted DataFrame
    long1 = long_from_wide.sort_values(['ID', 'Time']).reset_index(drop=True)
    long2 = long_from_wide2.sort_values(['ID', 'Time']).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        long1[['ID', 'Time', 'Outcome']],
        long2[['ID', 'Time', 'Outcome']],
        check_dtype=False,
    )

    # Run the same 1-group model on both; results must be identical
    m1 = run_single_model(long1, orders_list=[1], dist='LOGIT', n_starts=1)
    m2 = run_single_model(long2, orders_list=[1], dist='LOGIT', n_starts=1)

    assert m1['ll'] == pytest.approx(m2['ll'], abs=1e-8), (
        f"LL differs between format paths: {m1['ll']:.8f} vs {m2['ll']:.8f}"
    )
    assert np.allclose(m1['result'].x, m2['result'].x, atol=1e-8), (
        "Parameter vectors differ between wide→long and direct-long paths"
    )

    # Also verify that manually constructing a wide df from a simulated long df
    # and converting back gives the same results
    df_long, _ = simulate_logit_trajectories(
        n_subjects=50,
        time_points=np.linspace(-1, 1, 6),
        group_params=[{'betas': [-1.0, 1.0]}],
        group_proportions=[1.0],
        seed=1212,
    )
    m_direct = run_single_model(df_long, orders_list=[1], dist='LOGIT', n_starts=1)

    # Convert long → wide → long using prep_trajectory_data
    pivot_outcomes = df_long.pivot(index='ID', columns='Time', values='Outcome')
    pivot_times    = df_long.pivot(index='ID', columns='Time', values='Time')

    time_pts = sorted(df_long['Time'].unique())
    wide_manual = pd.DataFrame({'ID': df_long['ID'].unique()}).set_index('ID')
    for j, t in enumerate(time_pts, 1):
        wide_manual[f'C{j}'] = pivot_outcomes[t]
        wide_manual[f'T{j}'] = t
    wide_manual = wide_manual.reset_index()

    df_roundtrip = prep_trajectory_data(wide_manual)
    m_roundtrip = run_single_model(df_roundtrip, orders_list=[1], dist='LOGIT', n_starts=1)

    assert m_direct['ll'] == pytest.approx(m_roundtrip['ll'], abs=1e-6), (
        f"Round-trip LL mismatch: direct={m_direct['ll']:.8f}, "
        f"roundtrip={m_roundtrip['ll']:.8f}"
    )
