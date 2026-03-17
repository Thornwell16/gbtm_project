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
