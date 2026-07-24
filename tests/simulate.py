"""
tests/simulate.py — Ground-truth simulation framework for AutoTraj
===================================================================
Generates longitudinal datasets with *known* parameters so that
recovery tests can compare AutoTraj's estimates against ground truth.

Each public function follows the same contract:

    long_df, truth = simulate_*(...)

    long_df   : pd.DataFrame with columns [ID, Time, Outcome], sorted
                by (ID, Time).  Missing observations are dropped rows
                (not NaN), so FIML will see the correct structure.

    truth     : dict with at least:
                  'assignments'  : {subject_id -> 1-based group number}
                  'group_params' : the input group_params list
                  'proportions'  : the input proportions list
                  plus distribution-specific extras (sigma, omega, ...)

Design notes
------------
- Random-number generation is fully seeded and self-contained so that
  identical calls always produce identical datasets.
- Time points are **not** scaled internally; callers control the scale.
- Group numbering is 1-based to match AutoTraj output conventions.
- The 'betas' list in each group_params dict is ordered [b0, b1, b2, ...]
  matching the polynomial design matrix [1, t, t^2, ...].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple

# Type aliases
LongDF     = pd.DataFrame
TruthDict  = Dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _design_row(t: float, order: int) -> np.ndarray:
    """Return [1, t, t^2, ..., t^order]."""
    return np.array([t ** p for p in range(order + 1)])


def _logistic(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable logistic function."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _poly_eval(betas: Sequence[float], t: float) -> float:
    """Evaluate polynomial sum_p betas[p] * t^p."""
    return sum(b * (t ** p) for p, b in enumerate(betas))


def _assign_groups(n_subjects: int,
                   proportions: Sequence[float],
                   rng: np.random.Generator) -> np.ndarray:
    """Return 0-based group indices for n_subjects drawn from proportions."""
    props = np.asarray(proportions, dtype=float)
    props /= props.sum()          # normalise in case of floating-point drift
    return rng.choice(len(props), size=n_subjects, p=props)


def _apply_mcar(records: List[dict],
                missing_rate: float,
                rng: np.random.Generator) -> List[dict]:
    """Drop each record independently with probability missing_rate (MCAR)."""
    if missing_rate <= 0.0:
        return records
    keep = rng.random(len(records)) >= missing_rate
    return [r for r, k in zip(records, keep) if k]


def _build_df(records: List[dict]) -> LongDF:
    """Convert list of {ID, Time, Outcome} dicts to a sorted DataFrame."""
    df = pd.DataFrame(records, columns=['ID', 'Time', 'Outcome'])
    df = df.sort_values(['ID', 'Time']).reset_index(drop=True)
    return df


def _build_df_with_extra(records: List[dict], extra_cols: Sequence[str]) -> LongDF:
    """Like _build_df but preserves additional columns (V3.0 covariate/TVC simulators)."""
    df = pd.DataFrame(records, columns=['ID', 'Time', 'Outcome'] + list(extra_cols))
    df = df.sort_values(['ID', 'Time']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 1. LOGIT (binary)
# ---------------------------------------------------------------------------

def simulate_logit_trajectories(
    n_subjects: int,
    time_points: Sequence[float],
    group_params: List[Dict],
    group_proportions: Sequence[float],
    missing_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[LongDF, TruthDict]:
    """Simulate binary outcomes from a LOGIT GBTM.

    Parameters
    ----------
    n_subjects        : number of subjects
    time_points       : 1-D sequence of time values shared by all subjects
    group_params      : list of dicts, each containing 'betas': [b0, b1, ...]
    group_proportions : mixture weights (must sum to 1; will be normalised)
    missing_rate      : MCAR probability that any single observation is dropped
    seed              : random seed for full reproducibility

    Returns
    -------
    long_df : DataFrame [ID, Time, Outcome]  (Outcome in {0, 1})
    truth   : {
                'assignments'  : {id -> 1-based group},
                'group_params' : group_params,
                'proportions'  : normalised proportions,
              }
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(time_points, dtype=float)
    group_idx = _assign_groups(n_subjects, group_proportions, rng)

    props_norm = np.asarray(group_proportions, dtype=float)
    props_norm = props_norm / props_norm.sum()

    records: List[dict] = []
    assignments: Dict[int, int] = {}

    for i in range(n_subjects):
        sid = i + 1
        g   = int(group_idx[i])
        betas = group_params[g]['betas']
        assignments[sid] = g + 1   # 1-based

        for t in times:
            eta = _poly_eval(betas, t)
            p   = float(_logistic(eta))
            y   = float(rng.binomial(1, p))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y})

    records = _apply_mcar(records, missing_rate, rng)
    long_df = _build_df(records)

    truth: TruthDict = {
        'assignments':  assignments,
        'group_params': group_params,
        'proportions':  props_norm.tolist(),
    }
    return long_df, truth


# ---------------------------------------------------------------------------
# 2. CNORM (censored normal / tobit)
# ---------------------------------------------------------------------------

def simulate_cnorm_trajectories(
    n_subjects: int,
    time_points: Sequence[float],
    group_params: List[Dict],
    group_proportions: Sequence[float],
    sigma: float,
    cnorm_min: float,
    cnorm_max: float,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[LongDF, TruthDict]:
    """Simulate censored-normal outcomes from a CNORM GBTM.

    Outcomes are drawn from N(mu, sigma) and then clamped to [cnorm_min,
    cnorm_max].  Values at the boundary represent censored observations.

    Parameters
    ----------
    sigma       : shared residual standard deviation (positive)
    cnorm_min   : lower censoring bound (values <= cnorm_min → cnorm_min)
    cnorm_max   : upper censoring bound (values >= cnorm_max → cnorm_max)

    Returns
    -------
    long_df : DataFrame [ID, Time, Outcome]
    truth   : {
                'assignments', 'group_params', 'proportions',
                'sigma', 'cnorm_min', 'cnorm_max',
              }
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")
    if cnorm_min >= cnorm_max:
        raise ValueError(f"cnorm_min ({cnorm_min}) must be < cnorm_max ({cnorm_max})")

    rng = np.random.default_rng(seed)
    times = np.asarray(time_points, dtype=float)
    group_idx = _assign_groups(n_subjects, group_proportions, rng)

    props_norm = np.asarray(group_proportions, dtype=float)
    props_norm = props_norm / props_norm.sum()

    records: List[dict] = []
    assignments: Dict[int, int] = {}

    for i in range(n_subjects):
        sid   = i + 1
        g     = int(group_idx[i])
        betas = group_params[g]['betas']
        assignments[sid] = g + 1

        for t in times:
            mu = _poly_eval(betas, t)
            y  = float(rng.normal(mu, sigma))
            # Censor: clamp to [cnorm_min, cnorm_max]
            y  = float(np.clip(y, cnorm_min, cnorm_max))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y})

    records = _apply_mcar(records, missing_rate, rng)
    long_df = _build_df(records)

    truth: TruthDict = {
        'assignments':  assignments,
        'group_params': group_params,
        'proportions':  props_norm.tolist(),
        'sigma':        sigma,
        'cnorm_min':    cnorm_min,
        'cnorm_max':    cnorm_max,
    }
    return long_df, truth


# ---------------------------------------------------------------------------
# 3. POISSON (count data, log link)
# ---------------------------------------------------------------------------

def simulate_poisson_trajectories(
    n_subjects: int,
    time_points: Sequence[float],
    group_params: List[Dict],
    group_proportions: Sequence[float],
    missing_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[LongDF, TruthDict]:
    """Simulate count outcomes from a Poisson GBTM (log link).

    mu = exp(beta @ X(t));  y ~ Poisson(mu)

    Returns
    -------
    long_df : DataFrame [ID, Time, Outcome]  (Outcome is non-negative integer)
    truth   : {'assignments', 'group_params', 'proportions'}
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(time_points, dtype=float)
    group_idx = _assign_groups(n_subjects, group_proportions, rng)

    props_norm = np.asarray(group_proportions, dtype=float)
    props_norm = props_norm / props_norm.sum()

    records: List[dict] = []
    assignments: Dict[int, int] = {}

    for i in range(n_subjects):
        sid   = i + 1
        g     = int(group_idx[i])
        betas = group_params[g]['betas']
        assignments[sid] = g + 1

        for t in times:
            eta = _poly_eval(betas, t)
            mu  = np.exp(np.clip(eta, -20.0, 20.0))   # guard against overflow
            y   = float(rng.poisson(mu))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y})

    records = _apply_mcar(records, missing_rate, rng)
    long_df = _build_df(records)

    truth: TruthDict = {
        'assignments':  assignments,
        'group_params': group_params,
        'proportions':  props_norm.tolist(),
    }
    return long_df, truth


# ---------------------------------------------------------------------------
# 4. ZIP (zero-inflated Poisson)
# ---------------------------------------------------------------------------

def simulate_zip_trajectories(
    n_subjects: int,
    time_points: Sequence[float],
    group_params: List[Dict],
    group_proportions: Sequence[float],
    zero_inflation_rates: Sequence[float],
    missing_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[LongDF, TruthDict]:
    """Simulate zero-inflated Poisson outcomes from a ZIP GBTM.

    For each observation:
      - with probability omega_g : y = 0  (structural zero)
      - with probability 1-omega_g: y ~ Poisson(exp(beta @ X(t)))

    Parameters
    ----------
    zero_inflation_rates : per-group structural zero-inflation probability
                           omega_g in (0, 1).  Length must equal number of
                           groups.

    Returns
    -------
    long_df : DataFrame [ID, Time, Outcome]
    truth   : {
                'assignments', 'group_params', 'proportions',
                'zero_inflation_rates',  # omega per group
                'zetas',                 # logit(omega) per group
              }
    """
    k = len(group_params)
    if len(zero_inflation_rates) != k:
        raise ValueError(
            f"zero_inflation_rates has length {len(zero_inflation_rates)} "
            f"but group_params has {k} groups."
        )
    omegas = np.asarray(zero_inflation_rates, dtype=float)
    if np.any(omegas < 0) or np.any(omegas >= 1):
        raise ValueError("zero_inflation_rates must be in [0, 1).")

    rng = np.random.default_rng(seed)
    times = np.asarray(time_points, dtype=float)
    group_idx = _assign_groups(n_subjects, group_proportions, rng)

    props_norm = np.asarray(group_proportions, dtype=float)
    props_norm = props_norm / props_norm.sum()

    # Store true zeta = logit(omega) for comparison with recovered params
    # Guard against omega=0 (log(0) undefined)
    safe_omegas = np.clip(omegas, 1e-9, 1.0 - 1e-9)
    true_zetas = np.log(safe_omegas / (1.0 - safe_omegas))

    records: List[dict] = []
    assignments: Dict[int, int] = {}

    for i in range(n_subjects):
        sid   = i + 1
        g     = int(group_idx[i])
        betas = group_params[g]['betas']
        omega = float(omegas[g])
        assignments[sid] = g + 1

        for t in times:
            if rng.random() < omega:
                y = 0.0    # structural zero
            else:
                eta = _poly_eval(betas, t)
                mu  = np.exp(np.clip(eta, -20.0, 20.0))
                y   = float(rng.poisson(mu))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y})

    records = _apply_mcar(records, missing_rate, rng)
    long_df = _build_df(records)

    truth: TruthDict = {
        'assignments':         assignments,
        'group_params':        group_params,
        'proportions':         props_norm.tolist(),
        'zero_inflation_rates': omegas.tolist(),
        'zetas':               true_zetas.tolist(),   # logit scale (model param scale)
    }
    return long_df, truth


# ---------------------------------------------------------------------------
# 5. Informative dropout (MNAR)
# ---------------------------------------------------------------------------

def simulate_dropout_data(
    n_subjects: int,
    time_points: Sequence[float],
    group_params: List[Dict],
    group_proportions: Sequence[float],
    dropout_gammas: Sequence[float],
    seed: int = 42,
) -> Tuple[LongDF, TruthDict]:
    """Simulate binary trajectories with MNAR (informative) dropout.

    Dropout at time t > first time point is governed by a logistic model:

        P(dropout_it = 1) = logistic(gamma0 + gamma1*t + gamma2*y_{i,t-1})

    Once a subject drops out, all subsequent observations are omitted.
    The first time point is always observed (no early dropout).

    Parameters
    ----------
    dropout_gammas : [gamma0, gamma1, gamma2] shared across all groups.
                     gamma0 < 0 keeps the baseline dropout probability low.
                     gamma1 controls time-varying dropout risk.
                     gamma2 controls outcome-dependent dropout.

    Returns
    -------
    long_df : DataFrame [ID, Time, Outcome]  (only non-missing rows)
    truth   : {
                'assignments', 'group_params', 'proportions',
                'dropout_gammas',
                'dropout_rates',  # empirical fraction of subjects who dropped
              }
    """
    if len(dropout_gammas) != 3:
        raise ValueError("dropout_gammas must have exactly 3 elements: [gamma0, gamma1, gamma2].")

    rng = np.random.default_rng(seed)
    times = np.asarray(time_points, dtype=float)
    T = len(times)
    group_idx = _assign_groups(n_subjects, group_proportions, rng)

    props_norm = np.asarray(group_proportions, dtype=float)
    props_norm = props_norm / props_norm.sum()

    gamma0, gamma1, gamma2 = float(dropout_gammas[0]), float(dropout_gammas[1]), float(dropout_gammas[2])

    records: List[dict] = []
    assignments: Dict[int, int] = {}
    n_dropouts = 0

    for i in range(n_subjects):
        sid   = i + 1
        g     = int(group_idx[i])
        betas = group_params[g]['betas']
        assignments[sid] = g + 1

        y_prev   = None
        dropped  = False

        for obs_idx, t in enumerate(times):
            if dropped:
                break   # all subsequent observations missing

            # Compute outcome
            eta = _poly_eval(betas, t)
            p   = float(_logistic(eta))
            y   = float(rng.binomial(1, p))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y})

            # Evaluate dropout probability for the *next* time point
            # (no dropout decision at the first observation)
            if obs_idx > 0 and y_prev is not None:
                z_drop = gamma0 + gamma1 * float(t) + gamma2 * y_prev
                p_drop = float(_logistic(z_drop))
                if rng.random() < p_drop:
                    dropped = True
                    n_dropouts += 1

            y_prev = y

    long_df = _build_df(records)

    truth: TruthDict = {
        'assignments':    assignments,
        'group_params':   group_params,
        'proportions':    props_norm.tolist(),
        'dropout_gammas': list(dropout_gammas),
        'dropout_rates':  n_dropouts / n_subjects,
    }
    return long_df, truth


# ---------------------------------------------------------------------------
# V3.0: mixing covariates (Gamma) and time-varying covariates (delta)
# ---------------------------------------------------------------------------

def simulate_logit_with_mixing_covariates(
    n_subjects: int,
    time_points: Sequence[float],
    group_params: List[Dict],
    gamma_matrix: Sequence[Sequence[float]],
    cov_mean: float = 0.0,
    cov_sd: float = 1.0,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[LongDF, TruthDict]:
    """Simulate LOGIT outcomes where group membership depends on a baseline covariate X1.

    theta_g(x_i) = Gamma_{g,0} + Gamma_{g,1}*x_i for g>0; theta_0(x_i) = 0.
    pi_g(x_i) = softmax(theta(x_i))_g. Group assignment is drawn per-subject from
    this covariate-dependent distribution (not from a fixed proportions vector).

    Parameters
    ----------
    gamma_matrix : (k, 2) array-like; row g = [Gamma_g0, Gamma_g1] for g=1..k-1.
                   Row 0 (reference group) is ignored — treated as zeros.

    Returns
    -------
    long_df : DataFrame [ID, Time, Outcome, X1]  (X1 is the time-invariant covariate)
    truth   : {'assignments', 'group_params', 'gamma_matrix', 'baseline_cov': {id: x}}
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(time_points, dtype=float)
    k = len(group_params)
    gamma = np.asarray(gamma_matrix, dtype=float)

    x = rng.normal(cov_mean, cov_sd, size=n_subjects)

    thetas = np.zeros((n_subjects, k))
    for g in range(1, k):
        thetas[:, g] = gamma[g, 0] + gamma[g, 1] * x
    max_t = thetas.max(axis=1, keepdims=True)
    exp_t = np.exp(thetas - max_t)
    pis = exp_t / exp_t.sum(axis=1, keepdims=True)

    records: List[dict] = []
    assignments: Dict[int, int] = {}
    baseline_cov: Dict[int, float] = {}

    for i in range(n_subjects):
        sid = i + 1
        g = int(rng.choice(k, p=pis[i]))
        betas = group_params[g]['betas']
        assignments[sid] = g + 1
        baseline_cov[sid] = float(x[i])

        for t in times:
            eta = _poly_eval(betas, t)
            p = float(_logistic(eta))
            y = float(rng.binomial(1, p))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y, 'X1': float(x[i])})

    records = _apply_mcar(records, missing_rate, rng)
    long_df = _build_df_with_extra(records, ['X1'])

    truth: TruthDict = {
        'assignments':  assignments,
        'group_params': group_params,
        'gamma_matrix': gamma.tolist(),
        'baseline_cov': baseline_cov,
    }
    return long_df, truth


def simulate_logit_with_tvc(
    n_subjects: int,
    time_points: Sequence[float],
    group_params: List[Dict],
    group_proportions: Sequence[float],
    delta_per_group: Sequence[float],
    tvc_sd: float = 1.0,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[LongDF, TruthDict]:
    """Simulate LOGIT outcomes with one time-varying covariate Z1 deflecting eta.

    eta_igt = poly_eval(betas_g, t) + delta_g * z_it, with z_it ~ N(0, tvc_sd)
    drawn independently per (subject, time) — genuinely time-varying by
    construction (non-trivial within-subject variance).

    Returns
    -------
    long_df : DataFrame [ID, Time, Outcome, Z1]
    truth   : {'assignments', 'group_params', 'proportions', 'delta_per_group'}
    """
    if len(delta_per_group) != len(group_params):
        raise ValueError("delta_per_group must have one entry per group.")

    rng = np.random.default_rng(seed)
    times = np.asarray(time_points, dtype=float)
    group_idx = _assign_groups(n_subjects, group_proportions, rng)
    props_norm = np.asarray(group_proportions, dtype=float)
    props_norm = props_norm / props_norm.sum()

    records: List[dict] = []
    assignments: Dict[int, int] = {}

    for i in range(n_subjects):
        sid = i + 1
        g = int(group_idx[i])
        betas = group_params[g]['betas']
        delta_g = float(delta_per_group[g])
        assignments[sid] = g + 1

        for t in times:
            z = float(rng.normal(0.0, tvc_sd))
            eta = _poly_eval(betas, t) + delta_g * z
            p = float(_logistic(eta))
            y = float(rng.binomial(1, p))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y, 'Z1': z})

    records = _apply_mcar(records, missing_rate, rng)
    long_df = _build_df_with_extra(records, ['Z1'])

    truth: TruthDict = {
        'assignments':     assignments,
        'group_params':    group_params,
        'proportions':     props_norm.tolist(),
        'delta_per_group': [float(d) for d in delta_per_group],
    }
    return long_df, truth


def simulate_logit_with_covariates_and_tvc(
    n_subjects: int,
    time_points: Sequence[float],
    group_params: List[Dict],
    gamma_matrix: Sequence[Sequence[float]],
    delta_per_group: Sequence[float],
    cov_mean: float = 0.0,
    cov_sd: float = 1.0,
    tvc_sd: float = 1.0,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[LongDF, TruthDict]:
    """Combined simulator: mixing covariate X1 (group membership) + TVC Z1
    (trajectory deflection) together, for a joint-recovery regression test that
    catches parameter-vector index cross-talk bugs between the two new blocks.

    Returns
    -------
    long_df : DataFrame [ID, Time, Outcome, X1, Z1]
    truth   : {'assignments', 'group_params', 'gamma_matrix', 'delta_per_group',
               'baseline_cov'}
    """
    if len(delta_per_group) != len(group_params):
        raise ValueError("delta_per_group must have one entry per group.")

    rng = np.random.default_rng(seed)
    times = np.asarray(time_points, dtype=float)
    k = len(group_params)
    gamma = np.asarray(gamma_matrix, dtype=float)

    x = rng.normal(cov_mean, cov_sd, size=n_subjects)
    thetas = np.zeros((n_subjects, k))
    for g in range(1, k):
        thetas[:, g] = gamma[g, 0] + gamma[g, 1] * x
    max_t = thetas.max(axis=1, keepdims=True)
    exp_t = np.exp(thetas - max_t)
    pis = exp_t / exp_t.sum(axis=1, keepdims=True)

    records: List[dict] = []
    assignments: Dict[int, int] = {}
    baseline_cov: Dict[int, float] = {}

    for i in range(n_subjects):
        sid = i + 1
        g = int(rng.choice(k, p=pis[i]))
        betas = group_params[g]['betas']
        delta_g = float(delta_per_group[g])
        assignments[sid] = g + 1
        baseline_cov[sid] = float(x[i])

        for t in times:
            z = float(rng.normal(0.0, tvc_sd))
            eta = _poly_eval(betas, t) + delta_g * z
            p = float(_logistic(eta))
            y = float(rng.binomial(1, p))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y, 'X1': float(x[i]), 'Z1': z})

    records = _apply_mcar(records, missing_rate, rng)
    long_df = _build_df_with_extra(records, ['X1', 'Z1'])

    truth: TruthDict = {
        'assignments':     assignments,
        'group_params':    group_params,
        'gamma_matrix':    gamma.tolist(),
        'delta_per_group': [float(d) for d in delta_per_group],
        'baseline_cov':    baseline_cov,
    }
    return long_df, truth


# ---------------------------------------------------------------------------
# V4.0: survey/sampling weights
# ---------------------------------------------------------------------------

def simulate_logit_with_biased_sampling_weights(
    n_population: int,
    time_points: Sequence[float],
    group_params: List[Dict],
    group_proportions: Sequence[float],
    keep_probs: Sequence[float],
    seed: int = 42,
) -> Tuple[LongDF, TruthDict]:
    """Simulate a full population, then draw a biased sample where each group is
    retained with a different (known) probability, attaching the correct
    inverse-probability weight per sampled subject.

    This models informative sampling: e.g. keep_probs=[1.0, 0.3] always keeps
    group-1 population members but discards 70% of group-2 members, so an
    unweighted fit on the resulting sample sees a group split badly biased
    away from the true population proportions. The correct survey weight for
    a sampled subject in group g is w = 1 / keep_probs[g] (Horvitz-Thompson);
    a correctly weighted fit should recover the true population proportions
    despite the biased sample.

    Parameters
    ----------
    keep_probs : per-group probability that a population subject of that group
                 is retained in the sample.

    Returns
    -------
    long_df : DataFrame [ID, Time, Outcome, Weight] for SAMPLED subjects only
              (IDs are renumbered 1..n_sampled in sampling order).
    truth   : {'assignments', 'group_params', 'proportions' (true population
              proportions, NOT the biased sample's), 'keep_probs'}
    """
    if len(keep_probs) != len(group_params):
        raise ValueError("keep_probs must have one entry per group.")

    rng = np.random.default_rng(seed)
    times = np.asarray(time_points, dtype=float)
    group_idx = _assign_groups(n_population, group_proportions, rng)
    props_norm = np.asarray(group_proportions, dtype=float)
    props_norm = props_norm / props_norm.sum()

    records: List[dict] = []
    assignments: Dict[int, int] = {}
    next_sid = 1

    for i in range(n_population):
        g = int(group_idx[i])
        keep_prob = float(keep_probs[g])
        if rng.random() >= keep_prob:
            continue  # not sampled

        sid = next_sid
        next_sid += 1
        betas = group_params[g]['betas']
        assignments[sid] = g + 1
        weight = 1.0 / keep_prob

        for t in times:
            eta = _poly_eval(betas, t)
            p = float(_logistic(eta))
            y = float(rng.binomial(1, p))
            records.append({'ID': sid, 'Time': float(t), 'Outcome': y, 'Weight': weight})

    long_df = _build_df_with_extra(records, ['Weight'])

    truth: TruthDict = {
        'assignments':  assignments,
        'group_params': group_params,
        'proportions':  props_norm.tolist(),
        'keep_probs':   [float(p) for p in keep_probs],
    }
    return long_df, truth


# ---------------------------------------------------------------------------
# V5.0: joint dual-trajectory (two outcomes linked by a joint pi_gh)
# ---------------------------------------------------------------------------

def simulate_joint_two_outcome_trajectories(
    n_subjects: int,
    time_points_y: Sequence[float],
    time_points_z: Sequence[float],
    group_params_y: List[Dict],
    group_params_z: List[Dict],
    pi_gh,
    dist_y: str = 'LOGIT',
    dist_z: str = 'LOGIT',
    seed: int = 42,
):
    """Simulate a Nagin-style joint dual-trajectory dataset: two outcomes Y
    and Z whose latent group memberships are drawn JOINTLY from a
    (K_Y, K_Z) probability matrix pi_gh, rather than independently.

    Supports LOGIT and CNORM for dist_y/dist_z (sufficient for the V5.0 test
    suite). group_params_y[g]/group_params_z[h] each need 'betas'; CNORM
    additionally needs 'sigma', 'cnorm_min', 'cnorm_max'.

    Parameters
    ----------
    pi_gh : (K_Y, K_Z) array-like joint class probabilities (renormalised to
            sum to 1). Should be explicitly NON-independent (not close to an
            outer product of its own marginals) — a joint-recovery test using
            an independent pi_gh cannot distinguish "the joint model works"
            from "two separate single-outcome models happen to work".

    Returns
    -------
    df_y, df_z : DataFrame [ID, Time, Outcome] for outcomes Y and Z, same ID
                 set, aligned by construction.
    truth : {'assignments_y', 'assignments_z' (1-based dicts), 'pi_gh'
             (renormalised), 'group_params_y', 'group_params_z'}
    """
    rng = np.random.default_rng(seed)
    pi_gh_arr = np.asarray(pi_gh, dtype=float)
    pi_gh_arr = pi_gh_arr / pi_gh_arr.sum()
    k_y, k_z = pi_gh_arr.shape

    flat_probs = pi_gh_arr.flatten()
    joint_idx = rng.choice(k_y * k_z, size=n_subjects, p=flat_probs)
    g_idx = joint_idx // k_z
    h_idx = joint_idx % k_z

    times_y = np.asarray(time_points_y, dtype=float)
    times_z = np.asarray(time_points_z, dtype=float)

    def _simulate_one(betas, t, dist, group_spec):
        eta = _poly_eval(betas, t)
        if dist == 'LOGIT':
            p = float(_logistic(eta))
            return float(rng.binomial(1, p))
        elif dist == 'CNORM':
            sigma = group_spec.get('sigma', 1.0)
            cmin = group_spec.get('cnorm_min', -1e9)
            cmax = group_spec.get('cnorm_max', 1e9)
            y = float(rng.normal(eta, sigma))
            return float(np.clip(y, cmin, cmax))
        else:
            raise ValueError(f"simulate_joint_two_outcome_trajectories does not support dist={dist!r}")

    records_y: List[dict] = []
    records_z: List[dict] = []
    assignments_y: Dict[int, int] = {}
    assignments_z: Dict[int, int] = {}

    for i in range(n_subjects):
        sid = i + 1
        g = int(g_idx[i])
        h = int(h_idx[i])
        assignments_y[sid] = g + 1
        assignments_z[sid] = h + 1

        betas_y = group_params_y[g]['betas']
        for t in times_y:
            y_val = _simulate_one(betas_y, t, dist_y, group_params_y[g])
            records_y.append({'ID': sid, 'Time': float(t), 'Outcome': y_val})

        betas_z = group_params_z[h]['betas']
        for t in times_z:
            z_val = _simulate_one(betas_z, t, dist_z, group_params_z[h])
            records_z.append({'ID': sid, 'Time': float(t), 'Outcome': z_val})

    df_y = _build_df(records_y)
    df_z = _build_df(records_z)

    truth = {
        'assignments_y': assignments_y,
        'assignments_z': assignments_z,
        'pi_gh': pi_gh_arr,
        'group_params_y': group_params_y,
        'group_params_z': group_params_z,
    }
    return df_y, df_z, truth


# ---------------------------------------------------------------------------
# Convenience: canonical test-case presets
# ---------------------------------------------------------------------------

def make_two_group_logit(n_subjects: int = 500, seed: int = 42) -> Tuple[LongDF, TruthDict]:
    """Ready-made 2-group LOGIT dataset for quick regression tests.

    Group 1 (60%): flat low-risk trajectory  logit(p) ≈ -1.5
    Group 2 (40%): rising high-risk trajectory logit(p) = -2.0 + 3.5*t
    Time points: 10 evenly spaced values in [-1, 1].
    """
    return simulate_logit_trajectories(
        n_subjects=n_subjects,
        time_points=np.linspace(-1, 1, 10),
        group_params=[
            {'betas': [-1.5]},
            {'betas': [-2.0, 3.5]},
        ],
        group_proportions=[0.60, 0.40],
        seed=seed,
    )


def make_two_group_poisson(n_subjects: int = 400, seed: int = 42) -> Tuple[LongDF, TruthDict]:
    """Ready-made 2-group Poisson dataset.

    Group 1 (60%): low-count, rising  log(mu) = 0.5 + 0.3*t
    Group 2 (40%): high-count, falling log(mu) = 2.0 - 0.2*t
    """
    return simulate_poisson_trajectories(
        n_subjects=n_subjects,
        time_points=np.linspace(-1, 1, 10),
        group_params=[
            {'betas': [0.5, 0.3]},
            {'betas': [2.0, -0.2]},
        ],
        group_proportions=[0.60, 0.40],
        seed=seed,
    )


def make_two_group_zip(n_subjects: int = 400, seed: int = 42) -> Tuple[LongDF, TruthDict]:
    """Ready-made 2-group ZIP dataset.

    Group 1 (60%): 30% structural zeros, moderate counts
    Group 2 (40%): 10% structural zeros, high counts
    """
    return simulate_zip_trajectories(
        n_subjects=n_subjects,
        time_points=np.linspace(-1, 1, 10),
        group_params=[
            {'betas': [1.0, 0.5]},
            {'betas': [2.5, -0.3]},
        ],
        group_proportions=[0.60, 0.40],
        zero_inflation_rates=[0.30, 0.10],
        seed=seed,
    )


def make_two_group_cnorm(n_subjects: int = 400, seed: int = 42) -> Tuple[LongDF, TruthDict]:
    """Ready-made 2-group CNORM dataset.

    Group 1 (60%): declining   mu = 2.0 - 1.5*t
    Group 2 (40%): flat-high   mu = 4.0
    Censored to [0, 5], sigma = 0.8
    """
    return simulate_cnorm_trajectories(
        n_subjects=n_subjects,
        time_points=np.linspace(-1, 1, 10),
        group_params=[
            {'betas': [2.0, -1.5]},
            {'betas': [4.0]},
        ],
        group_proportions=[0.60, 0.40],
        sigma=0.8,
        cnorm_min=0.0,
        cnorm_max=5.0,
        seed=seed,
    )
