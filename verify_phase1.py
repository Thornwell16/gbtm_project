#!/usr/bin/env python3
"""
Phase 1 Verification Script for AutoTraj
=========================================
Run this BEFORE moving to Phase 2 (Prompts 5-8).
This confirms the math fixes from Prompts 1-4 are working correctly.

Usage:
    python verify_phase1.py

If all checks pass, you're clear to proceed to Phase 2.
If any check FAILS, fix it before moving on — Phase 2 builds on Phase 1.
"""

import sys
import os
import traceback
import numpy as np
import pandas as pd

# Add the repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    prep_trajectory_data,
    run_autotraj,
    run_single_model,
    get_subject_assignments,
    calc_model_adequacy,
    extract_flat_arrays,
    create_design_matrix_jit,
    calc_logit_prob_jit,
)

PASS = 0
FAIL = 0
WARN = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ PASS: {name}")
        PASS += 1
    else:
        print(f"  ❌ FAIL: {name}")
        if detail:
            print(f"         {detail}")
        FAIL += 1

def warn(name, detail=""):
    global WARN
    print(f"  ⚠️  WARN: {name}")
    if detail:
        print(f"         {detail}")
    WARN += 1


def load_cambridge():
    """Load Cambridge data — the canonical GBTM benchmark."""
    try:
        df = pd.read_csv("cambridge.txt", sep=r'\s+', encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except FileNotFoundError:
        print("ERROR: cambridge.txt not found. Run this from the repo root directory.")
        sys.exit(1)


# =========================================================================
# TEST 1: Cambridge data loads and preps correctly
# =========================================================================
print("\n" + "="*70)
print("TEST 1: Data Loading & Preparation")
print("="*70)

raw_df = load_cambridge()
long_df = prep_trajectory_data(raw_df, 'ID', 'C', 'T').dropna(subset=['Time', 'Outcome'])

n_subjects = long_df['ID'].nunique()
n_timepoints = long_df['Time'].nunique()

check("Cambridge loads", raw_df is not None)
check(f"Subject count = 196 (got {n_subjects})", abs(n_subjects - 196) <= 5,
      "Some subjects may be missing but should be close to 196")
check(f"Time points = 23 (got {n_timepoints})", n_timepoints == 23)
check("Outcomes are binary (0/1)", set(long_df['Outcome'].dropna().unique()).issubset({0, 1, 0.0, 1.0}),
      f"Unique values: {sorted(long_df['Outcome'].dropna().unique())}")


# =========================================================================
# TEST 2: Single model runs and produces valid output
# =========================================================================
print("\n" + "="*70)
print("TEST 2: Single Model Execution (1-group intercept-only)")
print("="*70)

try:
    m1 = run_single_model(long_df, [0], use_dropout=False, dist='LOGIT')
    check("1-group model converges", m1['result'].success or m1['result'].status == 2)
    check("Log-likelihood is finite", np.isfinite(m1['ll']), f"LL = {m1['ll']}")
    check("Log-likelihood is negative", m1['ll'] < 0, f"LL = {m1['ll']}")
    check("SE array exists and is finite", m1['se_model'] is not None and np.all(np.isfinite(m1['se_model'])))

    # The overall rate of offending in Cambridge is roughly 10-15%
    # So the intercept should be negative (log-odds < 0 means p < 0.5)
    intercept = m1['result'].x[-1] if len(m1['result'].x) > 0 else None
    if intercept is not None:
        check("Intercept is negative (low base rate)", intercept < 0,
              f"Intercept = {intercept:.4f}")
except Exception as e:
    print(f"  ❌ FAIL: 1-group model crashed: {e}")
    traceback.print_exc()
    FAIL += 1


# =========================================================================
# TEST 3: 2-group model on Cambridge data
# =========================================================================
print("\n" + "="*70)
print("TEST 3: 2-Group Quadratic Model on Cambridge Data")
print("="*70)

try:
    m2 = run_single_model(long_df, [2, 2], use_dropout=False, dist='LOGIT')
    check("2-group model converges", m2['result'].success or m2['result'].status == 2)
    check("Log-likelihood is finite", np.isfinite(m2['ll']), f"LL = {m2['ll']}")
    check("BIC improves over 1-group", m2['bic'] > m1['bic'],
          f"2-group BIC={m2['bic']:.2f} vs 1-group BIC={m1['bic']:.2f}")

    # Check group proportions
    pis = m2['pis']
    check("Group proportions sum to ~1.0", abs(sum(pis) - 1.0) < 0.01,
          f"Sum = {sum(pis):.4f}")
    check("Larger group is 60-90%", max(pis) > 0.6 and max(pis) < 0.95,
          f"Group proportions: {[f'{p:.1%}' for p in pis]}")
    check("Smaller group is 5-40%", min(pis) > 0.05 and min(pis) < 0.40,
          f"Group proportions: {[f'{p:.1%}' for p in pis]}")

    # Check adequacy
    assign_df = get_subject_assignments(m2, long_df)
    adq_df, rel_ent = calc_model_adequacy(assign_df, pis, ["Group 1", "Group 2"])
    check("Relative entropy > 0.5", rel_ent > 0.5, f"Entropy = {rel_ent:.3f}")

    # Check AvePP
    for _, row in adq_df.iterrows():
        avepp = row['AvePP']
        if isinstance(avepp, (int, float)) and not np.isnan(avepp):
            check(f"AvePP for {row['Group']} > 0.7", avepp > 0.7,
                  f"AvePP = {avepp:.3f}")

except Exception as e:
    print(f"  ❌ FAIL: 2-group model crashed: {e}")
    traceback.print_exc()
    FAIL += 1


# =========================================================================
# TEST 4: BIC conventions (Prompt 1 fix)
# =========================================================================
print("\n" + "="*70)
print("TEST 4: BIC/AIC Convention Fix (Prompt 1)")
print("="*70)

try:
    # Check that both BIC conventions exist in the model dict
    has_nagin_bic = 'bic' in m2 or 'bic_nagin' in m2
    check("Model dict has BIC (Nagin convention)", has_nagin_bic,
          f"Keys: {[k for k in m2.keys() if 'bic' in k.lower() or 'aic' in k.lower()]}")

    # Check for standard convention — look for any key containing 'standard'
    bic_keys = [k for k in m2.keys() if 'bic' in k.lower()]
    aic_keys = [k for k in m2.keys() if 'aic' in k.lower()]
    print(f"  ℹ️  BIC-related keys in model dict: {bic_keys}")
    print(f"  ℹ️  AIC-related keys in model dict: {aic_keys}")

    # Verify BIC values are reasonable (not NaN, not positive for Nagin convention)
    bic_val = m2.get('bic') or m2.get('bic_nagin')
    if bic_val is not None:
        check("BIC (Nagin) is finite", np.isfinite(bic_val), f"BIC = {bic_val:.2f}")
        check("BIC (Nagin) is negative", bic_val < 0,
              f"BIC = {bic_val:.2f} — Nagin BIC should be negative (LL - penalty)")

    # If standard BIC exists, verify it's positive and consistent
    bic_std = m2.get('bic_standard')
    if bic_std is not None:
        check("BIC (Standard) is finite", np.isfinite(bic_std), f"BIC_std = {bic_std:.2f}")
        check("BIC (Standard) is positive", bic_std > 0,
              f"BIC_std = {bic_std:.2f} — Standard BIC should be positive (-2LL + p*ln(N))")
    else:
        warn("BIC (Standard) key not found",
             "Prompt 1 should have added bic_standard. Check if Claude Code used a different key name.")

except Exception as e:
    print(f"  ❌ FAIL: BIC convention test crashed: {e}")
    traceback.print_exc()
    FAIL += 1


# =========================================================================
# TEST 5: Multiple starting values (Prompt 2 fix)
# =========================================================================
print("\n" + "="*70)
print("TEST 5: Multiple Random Starts (Prompt 2)")
print("="*70)

try:
    # Check if n_starts parameter exists in the function signature
    import inspect
    sig_single = inspect.signature(run_single_model)
    sig_auto = inspect.signature(run_autotraj)

    has_nstarts_single = 'n_starts' in sig_single.parameters
    has_nstarts_auto = 'n_starts' in sig_auto.parameters

    check("run_single_model has n_starts parameter", has_nstarts_single,
          f"Parameters: {list(sig_single.parameters.keys())}")
    check("run_autotraj has n_starts parameter", has_nstarts_auto,
          f"Parameters: {list(sig_auto.parameters.keys())}")

    if not has_nstarts_single and not has_nstarts_auto:
        warn("Multi-start may have been implemented differently",
             "Check if Claude Code used a different approach (e.g., internal loop without exposing parameter)")

    # Run a 2-group model with explicit n_starts if available
    if has_nstarts_single:
        m2_multi = run_single_model(long_df, [1, 1], use_dropout=False, dist='LOGIT', n_starts=5)
        check("Multi-start model converges", m2_multi['result'].success or m2_multi['result'].status == 2)
        check("Multi-start LL is finite", np.isfinite(m2_multi['ll']))

except Exception as e:
    print(f"  ❌ FAIL: Multi-start test crashed: {e}")
    traceback.print_exc()
    FAIL += 1


# =========================================================================
# TEST 6: Label sorting (Prompt 4 fix)
# =========================================================================
print("\n" + "="*70)
print("TEST 6: Label Sorting — Groups Ordered by Intercept (Prompt 4)")
print("="*70)

try:
    # Run 2-group model and check that Group 1 has the lower intercept
    m2_sort = run_single_model(long_df, [1, 1], use_dropout=False, dist='LOGIT')

    if m2_sort['result'].success or m2_sort['result'].status == 2:
        orders = m2_sort['orders']
        params = m2_sort['result'].x
        k = len(orders)

        # Extract intercepts
        current_idx = k - 1
        intercepts = []
        for g in range(k):
            intercepts.append(params[current_idx])
            current_idx += orders[g] + 1

        check("Group 1 intercept <= Group 2 intercept",
              intercepts[0] <= intercepts[1] + 0.01,  # small tolerance
              f"Intercepts: {[f'{b:.3f}' for b in intercepts]}")

        if intercepts[0] > intercepts[1] + 0.01:
            warn("Labels may not be sorted",
                 "Prompt 4 should have added sort_groups_by_intercept(). "
                 "Check if the function exists in main.py.")
    else:
        warn("2-group model didn't converge for label sort test")

except Exception as e:
    print(f"  ❌ FAIL: Label sorting test crashed: {e}")
    traceback.print_exc()
    FAIL += 1


# =========================================================================
# TEST 7: Input validation (Prompt 4 fix)
# =========================================================================
print("\n" + "="*70)
print("TEST 7: Input Validation (Prompt 4)")
print("="*70)

# We can't test Streamlit UI validation from a script, but we can check
# that the functions don't crash on bad input

try:
    # Test: non-binary data with LOGIT
    bad_df = long_df.copy()
    bad_df.loc[bad_df.index[0], 'Outcome'] = 0.5  # Not binary

    # This should either raise an error, return a failed model, or handle it
    # The important thing is it doesn't crash with a cryptic error
    try:
        m_bad = run_single_model(bad_df, [0], use_dropout=False, dist='LOGIT')
        warn("Non-binary LOGIT data did not raise error",
             "run_single_model accepted non-binary data. Validation may only be in app.py (OK).")
    except (ValueError, Exception) as e:
        check("Non-binary LOGIT data caught by engine", True, f"Error: {e}")

except Exception as e:
    warn(f"Input validation test issue: {e}")


# =========================================================================
# TEST 8: CNORM basic check (Prompt 3 fix)
# =========================================================================
print("\n" + "="*70)
print("TEST 8: CNORM Distribution Basic Check (Prompt 3)")
print("="*70)

try:
    # Create a simple continuous dataset for CNORM testing
    np.random.seed(42)
    n = 200
    ids = np.repeat(np.arange(1, n+1), 10)
    times = np.tile(np.linspace(-1, 1, 10), n)
    true_mu = -0.5 + 1.0 * times  # linear trajectory
    true_sigma = 0.5
    outcomes = np.random.normal(true_mu, true_sigma)

    cnorm_df = pd.DataFrame({'ID': ids, 'Time': times, 'Outcome': outcomes})
    cnorm_df = cnorm_df.sort_values(['ID', 'Time'])

    m_cnorm = run_single_model(cnorm_df, [1], use_dropout=False, dist='CNORM',
                                cnorm_min=outcomes.min(), cnorm_max=outcomes.max())

    check("CNORM model converges", m_cnorm['result'].success or m_cnorm['result'].status == 2)
    check("CNORM log-likelihood is finite", np.isfinite(m_cnorm['ll']),
          f"LL = {m_cnorm['ll']}")

    # Check sigma recovery (this is the key Prompt 3 test)
    params = m_cnorm['result'].x
    recovered_sigma = np.exp(params[-1])  # last param is log(sigma)
    sigma_error = abs(recovered_sigma - true_sigma) / true_sigma

    check(f"CNORM sigma recovered within 30% (true={true_sigma}, got={recovered_sigma:.3f})",
          sigma_error < 0.30,
          f"Error = {sigma_error:.1%}")

    if sigma_error > 0.30:
        warn("Sigma recovery is poor — Prompt 3 gradient fix may not have been applied correctly",
             "Check that the CNORM gradient for log(sigma) includes proper chain rule factors")

    # Check intercept and slope recovery
    k = 1
    beta_start = k - 1  # = 0 for 1-group
    recovered_intercept = params[beta_start]
    recovered_slope = params[beta_start + 1]

    check(f"CNORM intercept ~-0.5 (got {recovered_intercept:.3f})",
          abs(recovered_intercept - (-0.5)) < 0.3)
    check(f"CNORM slope ~1.0 (got {recovered_slope:.3f})",
          abs(recovered_slope - 1.0) < 0.3)

except Exception as e:
    print(f"  ❌ FAIL: CNORM test crashed: {e}")
    traceback.print_exc()
    FAIL += 1


# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "="*70)
print("PHASE 1 VERIFICATION SUMMARY")
print("="*70)
print(f"  ✅ Passed: {PASS}")
print(f"  ❌ Failed: {FAIL}")
print(f"  ⚠️  Warnings: {WARN}")
print()

if FAIL == 0:
    print("🎉 ALL CHECKS PASSED — You are clear to proceed to Phase 2.")
    print()
    print("Next steps:")
    print("  1. git add -A && git commit -m 'Phase 1 complete: math fixes verified'")
    print("  2. git push origin main")
    print("  3. Start Prompt 5 (Poisson distribution)")
elif FAIL <= 2:
    print("⚠️  MOSTLY PASSING — Review the failures above.")
    print("If failures are minor (e.g., key naming differences), you can probably")
    print("proceed to Phase 2 but fix the issues when convenient.")
    print()
    print("If failures are in BIC values, sigma recovery, or convergence,")
    print("DO NOT proceed — those are math errors that compound in later phases.")
else:
    print("🛑 MULTIPLE FAILURES — Fix these before proceeding to Phase 2.")
    print("Phase 2 adds new distributions that depend on the math being correct.")
    print("Re-run the relevant prompts or debug manually.")

sys.exit(0 if FAIL == 0 else 1)