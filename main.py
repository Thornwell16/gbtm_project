"""
main.py — AutoTraj Core Engine
================================
Group-Based Trajectory Modeling (GBTM) engine using finite mixture models
with JIT-compiled log-likelihood and analytical Jacobian.

Mathematical Model
------------------
GBTM models a longitudinal outcome y_it for subject i at time t as arising
from one of K latent groups.  Each group g has a polynomial trajectory:

    η_{igt} = β_{g0} + β_{g1}·t + β_{g2}·t² + … + β_{g,p_g}·t^{p_g}

The group membership is unknown; the model estimates the probability π_g
that a randomly chosen subject belongs to group g.  The marginal likelihood
for subject i is:

    L_i = Σ_{g=1}^{K} π_g · Π_t P(y_{it} | g, t)

where P(y | g, t) depends on the chosen distribution family (see below).
The total log-likelihood is ℓ = Σ_i log L_i.

Supported Distributions
------------------------
LOGIT    : Binary outcomes.  P(y=1|η) = σ(η) = 1/(1+e^{-η}).
CNORM    : Censored normal (Tobit).  y ~ N(μ, σ²) clipped to [y_min, y_max].
           σ is estimated as exp(raw_σ) to enforce positivity.
POISSON  : Count outcomes.  y ~ Poisson(exp(η)), log link.
ZIP      : Zero-inflated Poisson.  P(y|g,t) = ω_g·[y=0] + (1-ω_g)·Poisson(y; exp(η)).
           ω_g = σ(ζ_g) is a per-group, time-constant zero-inflation probability.

Full derivations for all four distributions, including gradient proofs and
the complete parameter vector layout, are in MATH.md.

Parameterization
----------------
The optimizer works with an unconstrained parameter vector θ:

  θ[0 .. k-2]              : log-ratio mixing weights (θ_g); π_g = softmax(θ)_g.
                              θ_0 ≡ 0 (implicit reference group).
  θ[k-1 .. k-1+Σ(p_g+1)-1] : trajectory betas, group-major order.
  θ[gamma_start ..]         : dropout gammas [γ₀, γ₁, γ₂] × k (if use_dropout=True).
  θ[-1]                     : log(σ) for CNORM; not present for other distributions.
  θ[-k ..]                  : per-group zeta logits ζ_g for ZIP.

Time Scaling
------------
Before optimization, all time values are divided by scale_factor = max|t|.
This keeps polynomial terms in the range [−1, 1], avoiding ill-conditioned
Hessians.  After optimization, betas are unscaled by the diagonal matrix D
(D_{p,p} = scale_factor^{-p}) before reporting.

Optimization
------------
SciPy BFGS with the analytical Jacobian (full gradient computed in a single
Numba JIT pass — no finite-difference gradient).  Multi-start: n_starts
random perturbations of a deterministic base point; the run with the lowest
NLL is kept.

Standard Errors
---------------
Model-based SEs: diagonal of  V_model = D · H⁻¹ · D  where H is the
  numerical Hessian (central finite-differences, adaptive step size).
Robust SEs (Huber-White sandwich): diagonal of
  V_robust = D · H⁻¹ · G · H⁻¹ · D  where G = Σ_i g_i gᵢᵀ (per-subject
  outer products of the analytical gradient).

BIC / AIC Conventions
---------------------
Two conventions run in parallel throughout the code:

  Nagin (higher is better):   BIC_N = ℓ - ½·p·log(N),   AIC_N = ℓ - p
  Standard (lower is better): BIC_S = -2ℓ + p·log(N),   AIC_S = -2ℓ + 2p

where N is the number of subjects and p is the number of free parameters.
Model selection uses BIC_N by default.

References
----------
Nagin, D.S. (1999). Analyzing developmental trajectories: A semiparametric,
  group-based approach. Psychological Methods, 4(2), 139–157.
Jones, B.L., & Nagin, D.S. (2001). A SAS procedure for group-based trajectory
  modeling. Sociological Methods & Research, 29(3), 374–393.
Nagin, D.S. (2005). Group-Based Modeling of Development. Harvard University Press.
White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator.
  Econometrica, 48(4), 817–838.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t as t_dist
import itertools
from numba import njit
import math
import os
from concurrent.futures import ThreadPoolExecutor

# --- C-LEVEL MATH HELPERS FOR CNORM ---

@njit(cache=True)
def fast_norm_logpdf(x, mu, sigma):
    """Log of the standard normal PDF evaluated at the standardised residual.

    Computes log φ((x - mu)/sigma) - log(sigma) = log N(x; mu, sigma²).

    Args:
        x:     Observed value.
        mu:    Mean of the normal distribution.
        sigma: Standard deviation (must be > 0).

    Returns:
        float: log P(X = x) under N(mu, sigma²).
    """
    variance = sigma ** 2
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - ((x - mu) ** 2) / (2 * variance)

@njit(cache=True)
def fast_norm_pdf(x):
    """Standard normal PDF: φ(x) = (1/√(2π)) · exp(-x²/2).

    Args:
        x: Standardised value z = (y - μ)/σ.

    Returns:
        float: φ(x), the standard normal density at x.
    """
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x ** 2))

@njit(cache=True)
def fast_norm_logcdf(x):
    """Log of the standard normal CDF: log Φ(x).

    Uses erfc for numerical stability; clamps at 1e-15 to avoid log(0).

    Args:
        x: Standardised value z = (y - μ)/σ.

    Returns:
        float: log Φ(x), used in left-censored CNORM log-likelihood.
    """
    cdf_val = 0.5 * math.erfc(-x / math.sqrt(2.0))
    if cdf_val < 1e-15: cdf_val = 1e-15
    return np.log(cdf_val)

@njit(cache=True)
def fast_norm_logsf(x):
    """Log of the standard normal survival function: log(1 - Φ(x)).

    Uses erfc for numerical stability; clamps at 1e-15 to avoid log(0).

    Args:
        x: Standardised value z = (y - μ)/σ.

    Returns:
        float: log(1 - Φ(x)), used in right-censored CNORM log-likelihood.
    """
    sf_val = 0.5 * math.erfc(x / math.sqrt(2.0))
    if sf_val < 1e-15: sf_val = 1e-15
    return np.log(sf_val)

# --- C-LEVEL MATH HELPERS FOR ZIP ---

@njit(cache=True)
def fast_zip_logpmf_grad(y, z, tau):
    """Log PMF and gradients for a single ZIP observation.

    Computes log P(y | λ, ω) for the zero-inflated Poisson mixture and the
    partial derivatives needed by the outer gradient accumulation loop.

    ZIP mixture PMF:
        P(y=0) = ω + (1-ω)·e^{-λ}
        P(y>0) = (1-ω)·Poisson(y; λ)

    where λ = exp(z) is the Poisson rate and ω = σ(tau) is the structural
    zero probability (sigma = logistic function).

    Args:
        y:   Observed count (float; 0.0 or positive integer cast to float).
        z:   Log-rate linear predictor η = X·β (clamped to ±50 for overflow).
        tau: Per-group zero-inflation logit ζ_g (clamped to ±25).

    Returns:
        Tuple[float, float, float]:
            ll      : log P(y | λ, ω) — the log-likelihood contribution.
            err_mu  : ∂ll/∂z  — gradient with respect to the log-rate predictor.
            err_tau : ∂ll/∂tau — gradient with respect to the zeta logit.
    """
    # Clamp log-rate predictor to avoid overflow in exp(z)
    if z > 50.0: z = 50.0
    if z < -50.0: z = -50.0
    lam = np.exp(z)   # Poisson rate λ = exp(η)

    # Clamp zeta and compute structural zero probability ω = σ(tau)
    if tau > 25.0: tau = 25.0
    if tau < -25.0: tau = -25.0
    rho = 1.0 / (1.0 + np.exp(-tau))   # ω = logistic(ζ)
    
    if y == 0.0:
        # P(y=0) = ω + (1-ω)·e^{-λ}  — combined structural and count zeros
        exp_neg_lam = np.exp(-lam)
        p0 = rho + (1.0 - rho) * exp_neg_lam
        p0 = max(1e-15, p0)   # numerical floor to prevent log(0)
        ll = np.log(p0)

        # ∂log(p0)/∂λ = -(1-ω)·e^{-λ}/p0;  chain rule: ∂λ/∂z = λ  →  err_mu = ∂ll/∂z
        dLL_dlam = -(1.0 - rho) * exp_neg_lam / p0
        err_mu = dLL_dlam * lam

        # ∂log(p0)/∂ω = (1 - e^{-λ})/p0;  chain rule: ∂ω/∂ζ = ω(1-ω)  →  err_tau = ∂ll/∂ζ
        dLL_drho = (1.0 - exp_neg_lam) / p0
        err_tau = dLL_drho * rho * (1.0 - rho)
    else:
        # P(y>0) = (1-ω)·Poisson(y; λ) → log P = log(1-ω) + y·z - λ - log(y!)
        one_minus_rho = max(1e-15, 1.0 - rho)
        ll = np.log(one_minus_rho) + y * z - lam - math.lgamma(y + 1.0)

        # ∂ll/∂z = y - λ  (canonical Poisson gradient w.r.t. log-rate predictor)
        err_mu = y - lam
        # ∂ll/∂ζ = ∂log(1-ω)/∂ζ = -ω  (since ∂(1-ω)/∂ζ = -ω(1-ω) and 1/(1-ω) × -(1-ω)ω = -ω)
        err_tau = -rho

    return ll, err_mu, err_tau

# --- DATA PREP ---

def load_cambridge_data():
    """Load the Cambridge Study of Delinquent Development dataset.

    Reads cambridge.txt from the current working directory.  The file is in
    wide format with columns ID, C1–C23 (binary conviction outcomes),
    T1–T23 (pre-scaled time values), DARING, and REARING.

    Returns:
        pd.DataFrame: Wide-format DataFrame (N=195 rows × 49 columns).
    """
    df = pd.read_csv("cambridge.txt", sep=r'\s+')
    return df


def prep_trajectory_data(df, id_col='ID', outcome_prefix='C', time_prefix='T'):
    """Convert a wide-format longitudinal DataFrame to long format.

    Expects columns named <outcome_prefix><j> and <time_prefix><j> for each
    measurement period j (e.g. C1, C2, …, T1, T2, …).  Extra covariates
    such as DARING or REARING are kept as-is in the output.

    Args:
        df:             Wide-format DataFrame.
        id_col:         Name of the subject-ID column (default 'ID').
        outcome_prefix: Stub prefix for outcome columns (default 'C').
        time_prefix:    Stub prefix for time columns (default 'T').

    Returns:
        pd.DataFrame: Long-format DataFrame with columns ID, Time, Outcome,
            Measurement_Period, plus any extra covariates.  Sorted by
            (ID, Measurement_Period).
    """
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    id_col = id_col.strip()
    outcome_prefix = outcome_prefix.strip()
    time_prefix = time_prefix.strip()
    long_df = pd.wide_to_long(df, stubnames=[outcome_prefix, time_prefix], i=id_col, j='Measurement_Period', suffix=r'\d+').reset_index()
    long_df = long_df.rename(columns={outcome_prefix: 'Outcome', time_prefix: 'Time', id_col: 'ID'})
    long_df = long_df.sort_values(by=['ID', 'Measurement_Period'])
    return long_df


def extract_flat_arrays(df):
    """Flatten a long-format DataFrame into contiguous NumPy arrays for the JIT engine.

    Subjects must be sorted by ID so that all rows for one subject are
    contiguous.  The function detects unequal observation counts
    automatically and encodes subject boundaries in subj_breaks.

    Dropout detection: a subject is flagged as a dropout if their last
    observed time is strictly less than the maximum study time.  The dropout
    flag (1.0) is placed at the last-observation index for that subject.

    Args:
        df: Long-format DataFrame with columns ID, Time, Outcome (and
            optionally others which are ignored).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            times:       (N_obs,) float64 — observation times.
            outcomes:    (N_obs,) float64 — outcome values.
            dropouts:    (N_obs,) float64 — 1.0 at the last row of a subject
                         who drops out before max_study_time; 0.0 elsewhere.
            subj_breaks: (N_subjects+1,) int64 — subject boundary indices such
                         that subject i occupies rows [subj_breaks[i], subj_breaks[i+1]).
    """
    ids = df['ID'].values
    times = df['Time'].values.astype(np.float64)
    outcomes = df['Outcome'].values.astype(np.float64)
    max_study_time = np.max(times)
    # Find row indices where the subject ID changes
    changes = np.where(ids[:-1] != ids[1:])[0] + 1
    subj_breaks = np.concatenate(([0], changes, [len(df)])).astype(np.int64)
    dropouts = np.zeros(len(df), dtype=np.float64)
    n_subjects = len(subj_breaks) - 1
    for i in range(n_subjects):
        end_idx = subj_breaks[i+1] - 1
        if times[end_idx] < max_study_time:
            dropouts[end_idx] = 1.0   # mark this subject as a dropout
    return times, outcomes, dropouts, subj_breaks


def extract_joint_flat_arrays(df_y, df_z, id_col='ID'):
    """Flatten TWO long-format DataFrames into aligned flat-array pairs for the
    V5.0 joint dual-trajectory kernel (calc_joint_dual_outcome_gradients_jit).

    Requires the identical subject-ID set in both df_y and df_z (a subject
    present in one outcome's frame but entirely absent from the other's is
    not supported — raises a clear error naming the offending IDs; this is an
    explicit scope boundary, not an oversight, per MATH.md §9). Independent
    per-subject time grids/observation counts ARE supported: df_y and df_z
    need not share the same Time column values or even the same number of
    rows per subject, covering both the common case (both outcomes measured
    at the same annual wave) and genuinely different measurement schedules
    per outcome.

    Args:
        df_y: Long-format DataFrame (ID, Time, Outcome) for outcome Y.
        df_z: Long-format DataFrame (ID, Time, Outcome) for outcome Z.
        id_col: Name of the subject-ID column in both frames (default 'ID').

    Returns:
        Tuple of two 4-tuples, each exactly extract_flat_arrays's return shape,
        plus the canonical subject-ID order both share:
            (times_y, outcomes_y, dropouts_y, subj_breaks_y),
            (times_z, outcomes_z, dropouts_z, subj_breaks_z),
            canonical_ids
        subj_breaks_y and subj_breaks_z describe the SAME N subjects in the
        SAME order (position i is the same subject in both), so the joint
        kernel can index them in lockstep even though each may have a
        different number of observations per subject.
    """
    ids_y = set(df_y[id_col].unique())
    ids_z = set(df_z[id_col].unique())
    if ids_y != ids_z:
        only_y = sorted(ids_y - ids_z)[:5]
        only_z = sorted(ids_z - ids_y)[:5]
        raise ValueError(
            f"Outcome Y and outcome Z must have the identical subject-ID set for the "
            f"V5.0 joint dual-trajectory model. Subjects only in Y: {only_y}"
            f"{'...' if len(ids_y - ids_z) > 5 else ''}; only in Z: {only_z}"
            f"{'...' if len(ids_z - ids_y) > 5 else ''}. Partial outcome-missingness "
            f"across subjects is not supported."
        )

    canonical_ids = pd.unique(df_y[id_col].values)  # order of first appearance in df_y
    rank_map = {v: i for i, v in enumerate(canonical_ids)}

    df_y_sorted = (
        df_y.assign(_rank=df_y[id_col].map(rank_map))
        .sort_values(['_rank', 'Time'], kind='stable')
        .drop(columns='_rank')
    )
    df_z_sorted = (
        df_z.assign(_rank=df_z[id_col].map(rank_map))
        .sort_values(['_rank', 'Time'], kind='stable')
        .drop(columns='_rank')
    )

    times_y, outcomes_y, dropouts_y, subj_breaks_y = extract_flat_arrays(df_y_sorted)
    times_z, outcomes_z, dropouts_z, subj_breaks_z = extract_flat_arrays(df_z_sorted)

    return (
        (times_y, outcomes_y, dropouts_y, subj_breaks_y),
        (times_z, outcomes_z, dropouts_z, subj_breaks_z),
        canonical_ids,
    )


def extract_tvc_array(df, tvc_cols):
    """Extract the (N_obs, n_tvc) time-varying-covariate matrix from a long-format df.

    Row alignment with times/outcomes/dropouts/subj_breaks (from extract_flat_arrays)
    is guaranteed by construction: this must be called on the *same* long-format
    df, in the same row order, as extract_flat_arrays (V3.0; see MATH.md §2c).

    Args:
        df:       Long-format DataFrame (same one passed to extract_flat_arrays).
        tvc_cols: List of column names to use as TVCs, or None/[] for no TVCs.

    Returns:
        np.ndarray: (len(df), len(tvc_cols)) C-contiguous float64 array. When
            tvc_cols is empty, returns a (len(df), 0) array — the V1.5.0-equivalent
            no-TVC default consumed by calc_universal_subject_gradients_jit.
    """
    if not tvc_cols:
        return np.zeros((len(df), 0), dtype=np.float64)
    return np.ascontiguousarray(df[list(tvc_cols)].values, dtype=np.float64)


def build_baseline_covariate_matrix(df, baseline_cov_cols, id_col='ID'):
    """Build the (N_subjects, P+1) mixing-covariate design matrix for group membership.

    Row order matches subj_breaks from extract_flat_arrays: subjects in order of
    first appearance in the (already ID-sorted, contiguous-per-subject) df. Column
    0 is always the intercept (V3.0; see MATH.md §2a).

    Baseline covariates must be time-invariant (constant within each subject) —
    this is validated explicitly and raises a clear error otherwise, since a
    covariate that actually varies within subject should be supplied as a
    time-varying covariate (TVC, see extract_tvc_array) instead.

    Args:
        df:                Long-format DataFrame (same one passed to extract_flat_arrays).
        baseline_cov_cols: List of column names to use as baseline covariates, or
                           None/[] for the no-covariate (intercept-only) case.
        id_col:            Name of the subject-ID column (default 'ID').

    Returns:
        np.ndarray: (N_subjects, P+1) C-contiguous float64 design matrix, intercept
            first. With baseline_cov_cols empty, returns ones((N_subjects, 1)) —
            the V1.5.0-equivalent default consumed by
            calc_universal_subject_gradients_jit.
    """
    ids = df[id_col].values
    subject_order = pd.unique(ids)  # order of first appearance (matches subj_breaks)
    n_subjects = len(subject_order)

    if not baseline_cov_cols:
        return np.ones((n_subjects, 1), dtype=np.float64)

    grouped = df.groupby(id_col, sort=False)
    n_unique = grouped[list(baseline_cov_cols)].nunique(dropna=True)
    for col in baseline_cov_cols:
        offending = n_unique.index[n_unique[col] > 1].tolist()
        if offending:
            raise ValueError(
                f"Baseline covariate '{col}' varies within subject for "
                f"{len(offending)} subject(s) (e.g. ID={offending[0]!r}) — baseline "
                f"covariates must be time-invariant. Supply this as a time-varying "
                f"covariate (TVC) instead if it genuinely changes over time."
            )

    first_vals = grouped[list(baseline_cov_cols)].first().reindex(subject_order)
    X = np.column_stack([
        np.ones(n_subjects),
        first_vals[list(baseline_cov_cols)].values.astype(np.float64),
    ])
    return np.ascontiguousarray(X, dtype=np.float64)


def extract_weights_array(df, weight_col, id_col='ID'):
    """Build the (N_subjects,) per-subject survey/sampling weight array (V4.0).

    Row order matches subj_breaks from extract_flat_arrays: subjects in order of
    first appearance in the (already ID-sorted, contiguous-per-subject) df.

    Survey weights are inherently per-subject (each sampled unit gets one weight),
    so — like baseline covariates — a weight column is validated to be
    time-invariant (constant within each subject), and strictly positive (an
    inverse-probability weight of zero or less is not meaningful; users who want
    to exclude a subject should filter that row out instead).

    Args:
        df:         Long-format DataFrame (same one passed to extract_flat_arrays).
        weight_col: Column name to use as the survey weight, or None for the
                    unweighted (all-ones) case.
        id_col:     Name of the subject-ID column (default 'ID').

    Returns:
        np.ndarray: (N_subjects,) C-contiguous float64 weight array. With
            weight_col=None, returns ones(N_subjects) — the V3.0-equivalent
            default consumed by calc_universal_subject_gradients_jit.
    """
    ids = df[id_col].values
    subject_order = pd.unique(ids)  # order of first appearance (matches subj_breaks)
    n_subjects = len(subject_order)

    if weight_col is None:
        return np.ones(n_subjects, dtype=np.float64)

    grouped = df.groupby(id_col, sort=False)
    n_unique = grouped[weight_col].nunique(dropna=True)
    offending = n_unique.index[n_unique > 1].tolist()
    if offending:
        raise ValueError(
            f"Weight column '{weight_col}' varies within subject for "
            f"{len(offending)} subject(s) (e.g. ID={offending[0]!r}) — survey "
            f"weights must be time-invariant (one weight per sampled subject)."
        )

    weights = grouped[weight_col].first().reindex(subject_order).values.astype(np.float64)
    if np.any(weights <= 0.0) or np.any(np.isnan(weights)):
        bad_ids = subject_order[np.where((weights <= 0.0) | np.isnan(weights))[0]][:5].tolist()
        raise ValueError(
            f"Weight column '{weight_col}' must be strictly positive for every "
            f"subject; found non-positive or missing values (e.g. ID(s)={bad_ids})."
        )

    return np.ascontiguousarray(weights, dtype=np.float64)


@njit(cache=True)
def create_design_matrix_jit(times, order):
    """Build a polynomial design matrix X of shape (n, order+1).

    Row i of X is [1, t_i, t_i², …, t_i^order].

    Args:
        times: (n,) array of time values (typically pre-scaled to [-1, 1]).
        order: Polynomial order (0 = intercept only, 1 = linear, …).

    Returns:
        np.ndarray: (n, order+1) design matrix.
    """
    n = len(times)
    X = np.empty((n, order + 1))
    for i in range(n):
        for p in range(order + 1): X[i, p] = times[i] ** p
    return X


@njit(cache=True)
def calc_logit_prob_jit(betas, X):
    """Compute logistic probabilities P(y=1) = σ(X·β) with numerical clamping.

    Uses two numerically stable branches to avoid overflow:
        z ≥ 0:  σ(z) = 1 / (1 + e^{-z})
        z < 0:  σ(z) = e^z / (1 + e^z)

    Args:
        betas: (p+1,) coefficient vector for one group.
        X:     (n, p+1) design matrix from create_design_matrix_jit.

    Returns:
        np.ndarray: (n,) predicted probabilities in (0, 1).
    """
    z = X @ betas
    probs = np.empty_like(z)
    for i in range(len(z)):
        if z[i] > 25.0: z[i] = 25.0       # clamp to prevent overflow in exp(-z)
        if z[i] < -25.0: z[i] = -25.0
        if z[i] >= 0: probs[i] = 1.0 / (1.0 + np.exp(-z[i]))
        else:
            exp_z = np.exp(z[i])
            probs[i] = exp_z / (1.0 + exp_z)
    return probs


@njit(cache=True)
def logsumexp_jit(a):
    """Numerically stable log-sum-exp: log Σ_i exp(a_i).

    Subtracts max(a) before exponentiating to prevent overflow.

    Args:
        a: 1-D array of log-probabilities or log-weights.

    Returns:
        float: log Σ exp(a_i).
    """
    max_val = np.max(a)
    sum_exp = 0.0
    for i in range(len(a)): sum_exp += np.exp(a[i] - max_val)
    return max_val + np.log(sum_exp)

# --- SINGLE-OUTCOME PER-SUBJECT SUBROUTINES (V5.0 refactor) ---
#
# These two functions are the single-outcome per-group likelihood and gradient
# computation, extracted out of calc_universal_subject_gradients_jit so it can
# be reused unchanged for BOTH outcomes of the V5.0 joint dual-trajectory model
# (calc_joint_dual_outcome_gradients_jit calls each once, for Y and for Z).
# calc_universal_subject_gradients_jit itself now calls these too (for its one
# outcome), rather than keeping a second, independently-maintained copy of the
# 4-distribution x dropout x CNORM/ZIP-tail math — see MATH.md §9.
#
# params_outcome layout (a contiguous slice/view, NOT including any mixing-
# covariate Gamma block): [beta (group-major) | delta (TVC, k*n_tvc, if
# n_tvc>0) | gamma_drop (3k, if use_dropout) | tail (CNORM raw_sigma or ZIP
# zeta, indexed from the END of params_outcome)]. This is exactly the shape
# of (a) the single-outcome kernel's own params[(k-1)*n_mix:] suffix, and (b)
# each outcome's own contiguous block in the V5.0 joint parameter vector
# (MATH.md §9's Y-BLOCK/Z-BLOCK) — by construction of both layouts, so no
# adjustment is needed when calling these functions from either kernel.

@njit(cache=True)
def calc_single_outcome_group_ll_jit(params_outcome, times, outcomes, dropouts, start, end,
                                      orders, tvc_Z, n_tvc, use_dropout, dist_code,
                                      cnorm_min, cnorm_max):
    """Per-group conditional log-likelihood and score arrays for ONE outcome,
    for a single subject's observation window [start, end).

    Args:
        params_outcome: (p_outcome,) contiguous slice — see module-level layout note.
        times, outcomes, dropouts: full flat arrays for this outcome (NOT
            subject-sliced — start/end index into them, same convention as
            calc_universal_subject_gradients_jit).
        start, end:  this subject's row range within the flat arrays.
        orders:      (K,) int32 per-group polynomial orders for this outcome.
        tvc_Z:       (N_obs, n_tvc) float64 TVC matrix (zeros((N_obs,0)) if none).
        n_tvc:       number of TVCs for this outcome (0 for the V5.0 joint model,
                     which does not compose with TVCs — see MATH.md §9 scope note).
        use_dropout: bool — this outcome's own dropout toggle.
        dist_code:   this outcome's own distribution selector (0-3).
        cnorm_min, cnorm_max: this outcome's own CNORM bounds (0.0 if not CNORM).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            L_g:        (K,) log P(outcome_i | group=g), dropout terms included.
            err_mu_ig:  (K, n_obs) score residual w.r.t. eta (beta/delta gradient input).
            err_aux_ig: (K, n_obs) score residual w.r.t. raw_sigma or zeta_g.
    """
    k = len(orders)
    n_obs = end - start

    num_betas = 0
    for g in range(k): num_betas += orders[g] + 1
    delta_start_idx = num_betas
    gamma_start_idx = delta_start_idx + k * n_tvc

    sigma = 1.0
    var = 1.0
    sigma_idx = -1
    zeta_start_idx = -1

    if dist_code == 1:
        sigma_idx = len(params_outcome) - 1
        raw_sigma = params_outcome[sigma_idx]
        sigma = np.exp(raw_sigma) if raw_sigma < 20 else np.exp(20)
        var = sigma ** 2
    elif dist_code == 3:
        zeta_start_idx = len(params_outcome) - k

    zeta_g = 0.0

    L_g = np.zeros(k)
    err_mu_ig = np.zeros((k, n_obs))
    err_aux_ig = np.zeros((k, n_obs))

    current_beta_idx = 0
    current_gamma_idx = gamma_start_idx

    for g in range(k):
        order = orders[g]
        n_betas = order + 1
        group_betas = params_outcome[current_beta_idx : current_beta_idx + n_betas]
        current_beta_idx += n_betas
        group_delta = params_outcome[delta_start_idx + g * n_tvc : delta_start_idx + (g + 1) * n_tvc]

        if use_dropout:
            gamma_0 = params_outcome[current_gamma_idx]
            gamma_1 = params_outcome[current_gamma_idx + 1]
            gamma_2 = params_outcome[current_gamma_idx + 2]
            current_gamma_idx += 3

        if dist_code == 3:
            zeta_g = params_outcome[zeta_start_idx + g]

        ll_g = 0.0

        for obs in range(n_obs):
            idx = start + obs
            t_val = times[idx]
            y_val = outcomes[idx]

            mu = 0.0
            for p in range(order + 1): mu += group_betas[p] * (t_val ** p)
            for q in range(n_tvc): mu += group_delta[q] * tvc_Z[idx, q]

            if dist_code == 0:  # LOGIT
                if mu > 25.0: mu = 25.0
                if mu < -25.0: mu = -25.0
                prob = 1.0 / (1.0 + np.exp(-mu)) if mu >= 0 else np.exp(mu) / (1.0 + np.exp(mu))
                prob = max(1e-12, min(1.0 - 1e-12, prob))
                if mu >= 0: ll_g += y_val * mu - (mu + np.log(1.0 + np.exp(-mu)))
                else: ll_g += y_val * mu - np.log(1.0 + np.exp(mu))
                err_mu_ig[g, obs] = y_val - prob

            elif dist_code == 2:  # POISSON
                if mu > 20.0: mu = 20.0
                if mu < -20.0: mu = -20.0
                exp_eta = np.exp(mu)
                ll_g += y_val * mu - exp_eta - math.lgamma(y_val + 1.0)
                err_mu_ig[g, obs] = y_val - exp_eta

            elif dist_code == 1:  # CNORM
                if y_val <= cnorm_min:
                    z = (cnorm_min - mu) / sigma
                    cdf_val = max(1e-15, 0.5 * math.erfc(-z / math.sqrt(2.0)))
                    imr = fast_norm_pdf(z) / cdf_val
                    ll_g += np.log(cdf_val)
                    err_mu_ig[g, obs] = -(1.0 / sigma) * imr
                    err_aux_ig[g, obs] = -z * imr
                elif y_val >= cnorm_max:
                    z = (cnorm_max - mu) / sigma
                    sf_val = max(1e-15, 0.5 * math.erfc(z / math.sqrt(2.0)))
                    imr = fast_norm_pdf(z) / sf_val
                    ll_g += np.log(sf_val)
                    err_mu_ig[g, obs] = (1.0 / sigma) * imr
                    err_aux_ig[g, obs] = z * imr
                else:
                    z = (y_val - mu) / sigma
                    ll_g += fast_norm_logpdf(y_val, mu, sigma)
                    err_mu_ig[g, obs] = (y_val - mu) / var
                    err_aux_ig[g, obs] = -1.0 + (z ** 2)

            elif dist_code == 3:  # ZIP
                if mu > 20.0: mu = 20.0
                if mu < -20.0: mu = -20.0
                ll_val, err_m, err_t = fast_zip_logpmf_grad(y_val, mu, zeta_g)
                ll_g += ll_val
                err_mu_ig[g, obs] = err_m
                err_aux_ig[g, obs] = err_t

            if use_dropout and obs > 0:
                y_prev = outcomes[idx - 1]
                z_drop = gamma_0 + (gamma_1 * t_val) + (gamma_2 * y_prev)
                if z_drop > 25.0: z_drop = 25.0
                if z_drop < -25.0: z_drop = -25.0
                if z_drop >= 0: ll_g += -z_drop - np.log(1.0 + np.exp(-z_drop))
                else: ll_g += -np.log(1.0 + np.exp(z_drop))

        if use_dropout:
            last_idx = end - 1
            if dropouts[last_idx] == 1.0:
                t_last = times[last_idx]
                y_last = outcomes[last_idx]
                z_drop = gamma_0 + (gamma_1 * t_last) + (gamma_2 * y_last)
                if z_drop > 25.0: z_drop = 25.0
                if z_drop < -25.0: z_drop = -25.0
                if z_drop >= 0: ll_g += -np.log(1.0 + np.exp(-z_drop))
                else: ll_g += z_drop - np.log(1.0 + np.exp(z_drop))

        L_g[g] = ll_g

    return L_g, err_mu_ig, err_aux_ig


@njit(cache=True)
def accumulate_single_outcome_gradient_jit(grad_outcome_row, params_outcome, times, outcomes,
                                            dropouts, start, end, orders, tvc_Z, n_tvc,
                                            use_dropout, dist_code, err_mu_ig, err_aux_ig,
                                            posterior_weight):
    """Accumulate ONE outcome's gradient contribution (beta, delta, dropout
    gamma, CNORM sigma or ZIP zeta) into grad_outcome_row, weighted by
    posterior_weight[g]. Mutates grad_outcome_row in place (a view sharing
    memory with the caller's full gradient row).

    posterior_weight[g] is P(g|i) in the single-outcome model, or the
    MARGINAL posterior (Sum_h P(g,h|i) for outcome Y, Sum_g P(g,h|i) for
    outcome Z) in the V5.0 joint model — MATH.md §9 shows the joint
    per-outcome gradient reduces to exactly this single-outcome formula with
    the marginal substituted for the posterior, which is what justifies
    reusing this same function unchanged for both outcomes.

    grad_outcome_row/params_outcome use the same params_outcome-relative
    offsets as calc_single_outcome_group_ll_jit (beta starts at 0, tail at
    the end) — grad_outcome_row must have the same length as params_outcome.
    """
    k = len(orders)
    n_obs = end - start

    num_betas = 0
    for g in range(k): num_betas += orders[g] + 1
    delta_start_idx = num_betas
    gamma_start_idx = delta_start_idx + k * n_tvc

    sigma_idx = -1
    zeta_start_idx = -1
    if dist_code == 1:
        sigma_idx = len(params_outcome) - 1
    elif dist_code == 3:
        zeta_start_idx = len(params_outcome) - k

    current_beta_idx = 0
    current_gamma_idx = gamma_start_idx

    for g in range(k):
        order = orders[g]
        n_betas = order + 1
        delta_base = delta_start_idx + g * n_tvc
        if use_dropout:
            gamma_0 = params_outcome[current_gamma_idx]
            gamma_1 = params_outcome[current_gamma_idx + 1]
            gamma_2 = params_outcome[current_gamma_idx + 2]

        for obs in range(n_obs):
            idx = start + obs
            t_val = times[idx]

            weighted_err_mu = err_mu_ig[g, obs] * posterior_weight[g]
            for p in range(order + 1):
                grad_outcome_row[current_beta_idx + p] += -1.0 * weighted_err_mu * (t_val ** p)
            for q in range(n_tvc):
                grad_outcome_row[delta_base + q] += -1.0 * weighted_err_mu * tvc_Z[idx, q]

            if dist_code == 1:
                grad_outcome_row[sigma_idx] += -1.0 * err_aux_ig[g, obs] * posterior_weight[g]
            elif dist_code == 3:
                grad_outcome_row[zeta_start_idx + g] += -1.0 * err_aux_ig[g, obs] * posterior_weight[g]

            if use_dropout and obs > 0:
                y_prev = outcomes[idx - 1]
                z_drop = gamma_0 + (gamma_1 * t_val) + (gamma_2 * y_prev)
                p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                err_drop = (0.0 - p_drop) * posterior_weight[g]
                grad_outcome_row[current_gamma_idx] += -1.0 * err_drop * 1.0
                grad_outcome_row[current_gamma_idx + 1] += -1.0 * err_drop * t_val
                grad_outcome_row[current_gamma_idx + 2] += -1.0 * err_drop * y_prev

        if use_dropout:
            last_idx = end - 1
            if dropouts[last_idx] == 1.0:
                t_last = times[last_idx]
                y_last = outcomes[last_idx]
                z_drop = gamma_0 + (gamma_1 * t_last) + (gamma_2 * y_last)
                p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                err_drop = (1.0 - p_drop) * posterior_weight[g]
                grad_outcome_row[current_gamma_idx] += -1.0 * err_drop * 1.0
                grad_outcome_row[current_gamma_idx + 1] += -1.0 * err_drop * t_last
                grad_outcome_row[current_gamma_idx + 2] += -1.0 * err_drop * y_last
            current_gamma_idx += 3
        current_beta_idx += n_betas


# --- CORE LIKELIHOOD/GRADIENT ENGINE (UNIVERSAL) ---

@njit(cache=True, nogil=True)
def calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max, baseline_X, tvc_Z, n_mix, n_tvc, weights):
    """Compute total NLL, flat gradient, and per-subject gradient matrix in one pass.

    This is the single performance-critical kernel that drives every model fit.
    It supports all four distributions (LOGIT / CNORM / Poisson / ZIP) and the
    optional informative-dropout augmentation through a dist_code dispatch.

    Algorithm overview
    ------------------
    For each subject i:
      1. Compute subject i's mixing proportions π_g(x_i) from baseline_X[i, :]
         (V3.0: per-subject mixing covariates; reduces to a fixed π_g when
         n_mix == 1, i.e. baseline_X is intercept-only).
      2. For each group g, compute the conditional log-likelihood L_{ig} =
         Σ_t log P(y_{it} | g, t) + (optional dropout terms), where the
         linear predictor η includes an optional TVC deflection term
         Σ_q δ_{g,q}·z_{i,q,t} (V3.0; vanishes when n_tvc == 0).
      3. Compute the posterior probability P(g | i) ∝ π_g(x_i) · exp(L_{ig}).
      4. Accumulate the total log-likelihood: ℓ += log Σ_g π_g(x_i) · exp(L_{ig}).
      5. Compute the per-subject gradient contributions for all parameter blocks.

    The Jacobian (gradient of the NLL) is returned as both a flat vector
    (used directly by SciPy BFGS) and a per-subject matrix (used to build the
    outer-product G matrix for the sandwich estimator).

    dist_code values
    ----------------
    0 : LOGIT    — binary outcomes, logit link.
    1 : CNORM    — censored normal (Tobit), σ parameterised as exp(raw_σ).
    2 : POISSON  — count outcomes, log link.
    3 : ZIP      — zero-inflated Poisson; one ζ_g scalar per group.

    Args:
        params:      (p,) unconstrained parameter vector in scaled-time units.
                     Layout (MATH.md §2): [Γ][β][δ][γ_drop][raw_σ or ζ].
        times:       (N_obs,) pre-scaled observation times (divided by scale_factor).
        outcomes:    (N_obs,) outcome values.
        dropouts:    (N_obs,) 1.0 at last obs of a dropout subject, else 0.0.
        subj_breaks: (N_subjects+1,) boundary indices from extract_flat_arrays.
        orders:      (K,) int32 array — polynomial order for each group.
        zip_iorder:  Legacy parameter, unused (ZIP now uses per-group zeta).
        use_dropout: bool — whether to include the informative-dropout likelihood.
        dist_code:   int  — distribution selector (0–3, see above).
        cnorm_min:   float — lower censoring bound (CNORM only; 0.0 otherwise).
        cnorm_max:   float — upper censoring bound (CNORM only; 0.0 otherwise).
        baseline_X:  (N_subjects, n_mix) float64 — per-subject mixing-covariate
                     design matrix, intercept-first. Pass ones((N,1)) for the
                     no-covariate (V1.5.0-equivalent) case.
        tvc_Z:       (N_obs, n_tvc) float64 — per-observation time-varying
                     covariate matrix, same row order/alignment as times/outcomes.
                     Pass zeros((N_obs, 0)) for the no-TVC case.
        n_mix:       int — number of mixing-covariate columns (P+1, incl. intercept).
        n_tvc:       int — number of time-varying covariates (Q).
        weights:     (N_subjects,) float64 — per-subject survey/sampling weight (V4.0).
                     Pass ones(N_subjects) for the unweighted (V3.0-equivalent) case.
                     Scales subject i's entire NLL and gradient-row contribution;
                     adds no new parameters to theta (MATH.md §1 weighted-likelihood note).

    Returns:
        Tuple[float, np.ndarray, np.ndarray]:
            nll:        Negative total log-likelihood (scalar minimised by BFGS).
            grad_flat:  (p,) gradient of nll w.r.t. params (analytical Jacobian).
            grad_subj:  (N_subjects, p) per-subject gradient matrix for sandwich SE.

    Notes:
        - All exp() calls on the linear predictor are clamped before
          exponentiation to prevent IEEE 754 overflow.
        - Posterior probabilities are computed in log-space for numerical
          stability (log-sum-exp trick).
        - The returned grad_flat is the gradient of the NLL (positive for
          descent); BFGS minimises NLL so this is the correct sign.
    """
    # dist_code: 0=LOGIT, 1=CNORM, 2=POISSON, 3=ZIP
    #
    # CNORM SIGMA PARAMETERIZATION & CHAIN RULE
    # -----------------------------------------
    # sigma is constrained positive via the log transform:
    #   raw_sigma = log(sigma)  →  sigma = exp(raw_sigma)
    # The optimizer works in raw_sigma space. err_aux_ig[g, obs] accumulates
    # d(LL_g)/d(raw_sigma) — already the gradient w.r.t. the unconstrained
    # parameter — NOT d(LL_g)/d(sigma). The chain rule factor (sigma) is
    # already absorbed into each expression:
    #
    #   Case 1 — Uncensored (cnorm_min < y < cnorm_max), z = (y-mu)/sigma:
    #     log_pdf = -log(sigma) - 0.5*log(2π) - z²/2
    #     d(log_pdf)/d(sigma)     = (1/sigma)(-1 + z²)
    #     d(log_pdf)/d(raw_sigma) = (-1 + z²)              ← stored as err_aux_ig ✓
    #
    #   Case 2 — Left-censored (y <= cnorm_min), z = (cnorm_min-mu)/sigma,
    #             IMR = φ(z)/Φ(z):
    #     LL = log Φ(z)
    #     d(LL)/d(sigma)     = -z·IMR / sigma
    #     d(LL)/d(raw_sigma) = -z·IMR                      ← stored as err_aux_ig ✓
    #
    #   Case 3 — Right-censored (y >= cnorm_max), z = (cnorm_max-mu)/sigma,
    #             IMR = φ(z)/(1-Φ(z)):
    #     LL = log(1 - Φ(z))
    #     d(LL)/d(sigma)     = z·IMR / sigma
    #     d(LL)/d(raw_sigma) = z·IMR                       ← stored as err_aux_ig ✓
    #
    # Therefore the accumulation line:
    #   grad_subj[i, sigma_idx] += -1.0 * err_aux_ig[g, obs] * posterior_ig[g]
    # is correct as written — NO additional sigma factor is needed.
    # (Verified by finite-difference check against all three cases;
    #  see tests/test_edge_cases.py::test_gradient_matches_finite_difference.)
    k = len(orders)

    n_subjects = len(subj_breaks) - 1
    grad_subj = np.zeros((n_subjects, len(params)))

    # V3.0 layout: [Γ: (k-1)*n_mix] [β/δ/γ_drop/tail: everything else]. The
    # suffix starting right after Γ is exactly a "solo" single-outcome
    # parameter vector (V5.0 refactor) — see calc_single_outcome_group_ll_jit's
    # module-level layout note. Sliced once here since it doesn't vary by subject.
    outcome_beta_start = (k - 1) * n_mix
    params_outcome = params[outcome_beta_start:]

    total_ll = 0.0

    for i in range(n_subjects):
        start = subj_breaks[i]
        end = subj_breaks[i+1]

        # ── PER-SUBJECT MIXING PROBABILITIES (V3.0) ─────────────────────────────
        # theta_g(x_i) = Gamma_g . x_i for g>0; theta_0(x_i) ≡ 0 (reference group).
        # With n_mix==1 and baseline_X[:, 0]==1 this reduces exactly to the
        # V1.5.0 subject-invariant theta_g (MATH.md §2a).
        thetas = np.zeros(k)
        for g in range(1, k):
            gamma_row_start = (g - 1) * n_mix
            acc = 0.0
            for p in range(n_mix):
                acc += params[gamma_row_start + p] * baseline_X[i, p]
            thetas[g] = acc

        max_theta = np.max(thetas)
        sum_exp_theta = 0.0
        for g in range(k): sum_exp_theta += np.exp(thetas[g] - max_theta)
        log_pis = thetas - (max_theta + np.log(sum_exp_theta))

        pis = np.empty(k)
        pis_safe = np.empty(k)
        for g in range(k):
            p_val = np.exp(log_pis[g])
            pis[g] = p_val
            pis_safe[g] = 1e-15 if p_val < 1e-15 else p_val

        # ── PER-GROUP LOG-LIKELIHOOD (single-outcome subroutine; V5.0 refactor) ──
        # This is the sole outcome's per-group conditional log-likelihood and
        # score computation, extracted into calc_single_outcome_group_ll_jit so
        # it can be reused unchanged by the V5.0 joint dual-trajectory kernel.
        L_ig_log, err_mu_ig, err_aux_ig = calc_single_outcome_group_ll_jit(
            params_outcome, times, outcomes, dropouts, start, end, orders,
            tvc_Z, n_tvc, use_dropout, dist_code, cnorm_min, cnorm_max
        )

        # ── POSTERIOR PROBABILITY AND TOTAL LL ─────────────────────────────────
        # numerator_log[g] = log(π_g) + L_{ig}  →  log of un-normalised posterior
        numerator_log = np.zeros(k)
        for g in range(k): numerator_log[g] = np.log(pis_safe[g]) + L_ig_log[g]
        # Log-sum-exp trick for numerical stability: log Σ_g exp(numerator_log[g])
        post_max = np.max(numerator_log)
        post_sum_exp = 0.0
        for g in range(k): post_sum_exp += np.exp(numerator_log[g] - post_max)

        # Normalised posterior: P(g | i) = exp(numerator_log[g]) / Σ_g' exp(numerator_log[g'])
        posterior_ig = np.zeros(k)
        for g in range(k):
            posterior_ig[g] = np.exp(numerator_log[g] - (post_max + np.log(post_sum_exp)))
            if g > 0:
                # Mixing-covariate gradient (MATH.md §4a): ∂ℓ_i/∂Γ_{g,p} = [P(g|i) - π_g(x_i)]·x_{i,p}
                # for g > 0 (reference group fixed at 0). NLL sign: store as -(P(g|i) - π_g)·x_{i,p}.
                # With n_mix==1, baseline_X[i,0]==1 this reduces exactly to the V1.5.0 theta gradient.
                diff = -1.0 * (posterior_ig[g] - pis[g])
                gamma_row_start = (g - 1) * n_mix
                for p in range(n_mix):
                    grad_subj[i, gamma_row_start + p] = diff * baseline_X[i, p]

        # Add log-marginal-likelihood for subject i to running total, scaled by
        # the subject's survey/sampling weight (V4.0; weights[i]==1.0 by default)
        total_ll += weights[i] * (post_max + np.log(post_sum_exp))

        # ── OUTCOME GRADIENT ACCUMULATION (single-outcome subroutine; V5.0 refactor) ──
        # Accumulates beta/delta/dropout-gamma/CNORM-sigma/ZIP-zeta gradients
        # into the outcome's slice of grad_subj[i,:], weighted by posterior_ig.
        accumulate_single_outcome_gradient_jit(
            grad_subj[i, outcome_beta_start:], params_outcome, times, outcomes, dropouts,
            start, end, orders, tvc_Z, n_tvc, use_dropout, dist_code,
            err_mu_ig, err_aux_ig, posterior_ig
        )

        # V4.0: scale subject i's ENTIRE gradient row by its survey/sampling weight,
        # once, after all blocks (Gamma, beta, delta, aux, dropout) have been
        # accumulated above. Scaling once here (rather than threading weights[i]
        # into every accumulation site) is deliberate: the Gamma-block line
        # (main.py ~758) uses assignment (=) not +=, so a single post-hoc
        # row-scale is the only safe way to guarantee every block is covered.
        for j in range(len(params)):
            grad_subj[i, j] *= weights[i]

    # Sum per-subject gradients into the flat Jacobian vector
    grad_flat = np.zeros(len(params))
    for i in range(grad_subj.shape[0]):
        for j in range(grad_subj.shape[1]):
            grad_flat[j] += grad_subj[i, j]

    # Return NLL (positive scalar for minimisation) and both gradient forms
    return -1.0 * total_ll, grad_flat, grad_subj


# --- JOINT DUAL-TRAJECTORY ENGINE (V5.0) ---

@njit(cache=True, nogil=True)
def calc_joint_dual_outcome_gradients_jit(
    params,
    times_y, outcomes_y, dropouts_y, subj_breaks_y, orders_y, use_dropout_y, dist_code_y, cnorm_min_y, cnorm_max_y,
    times_z, outcomes_z, dropouts_z, subj_breaks_z, orders_z, use_dropout_z, dist_code_z, cnorm_min_z, cnorm_max_z,
):
    """Joint dual-trajectory NLL + gradient kernel (V5.0, Nagin-style).

    Two outcomes Y and Z, each with its own independent GBTM structure (own
    group count, polynomial orders, distribution, dropout toggle), linked by
    a joint latent-class probability matrix pi_gh (K_Y x K_Z) instead of
    assuming independence. Given class (g,h), Y and Z are conditionally
    independent: P(y_i,z_i|g,h) = P(y_i|g)*P(z_i|h), each factor computed by
    calc_single_outcome_group_ll_jit — the SAME single-outcome subroutine the
    single-outcome kernel (calc_universal_subject_gradients_jit) uses for its
    one outcome. See MATH.md §9 for the full derivation, including the proof
    that each outcome's beta/dropout/tail gradient is the single-outcome
    formula with that outcome's MARGINAL posterior substituted for the
    single-outcome posterior.

    Parameter vector layout (MATH.md §9):
        theta = [ Theta_joint (K_Y*K_Z - 1) | Y-BLOCK | Z-BLOCK ]
        Y-BLOCK = [ beta_Y | gamma_drop_Y (3*K_Y, if use_dropout_y) | tail_Y ]
        Z-BLOCK = [ beta_Z | gamma_drop_Z (3*K_Z, if use_dropout_z) | tail_Z ]
    Theta_joint is the K_Y x K_Z joint-class grid flattened row-major (g
    outer, h inner), skipping the reference cell (0,0) which is implicitly 0.
    Y-BLOCK/Z-BLOCK are each exactly a "solo" single-outcome parameter vector
    (see calc_single_outcome_group_ll_jit's layout note) — Y's tail sits
    immediately before Z's block starts (not at the absolute end of theta),
    which is what makes params[y_beta_start:z_beta_start] and
    params[z_beta_start:] each look like a standalone single-outcome vector
    with its own tail "at the end" of that slice.

    V5.0 scope: does not compose with V3.0 mixing-covariates/TVC (no Gamma/
    delta blocks) or V4.0 survey weights — both are deferred future
    extensions (MATH.md §9 scope note), not attempted here.

    Args:
        params: (p,) flat joint parameter vector, layout above.
        times_y, outcomes_y, dropouts_y, subj_breaks_y: outcome Y's flat arrays
            (same convention as extract_flat_arrays / calc_universal_subject_gradients_jit).
        orders_y: (K_Y,) int32 per-group polynomial orders for Y.
        use_dropout_y, dist_code_y, cnorm_min_y, cnorm_max_y: outcome Y's own
            dropout toggle / distribution / CNORM bounds.
        times_z, ..., cnorm_max_z: the same, for outcome Z. subj_breaks_y and
            subj_breaks_z must describe the SAME N subjects in the SAME
            subject order (see extract_joint_flat_arrays), but may have
            independent per-subject time grids/observation counts.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]:
            nll:       negative total joint log-likelihood.
            grad_flat: (p,) gradient of nll w.r.t. params.
            grad_subj: (N_subjects, p) per-subject gradient matrix.
    """
    k_y = len(orders_y)
    k_z = len(orders_z)
    n_subjects = len(subj_breaks_y) - 1

    num_betas_y = 0
    for g in range(k_y): num_betas_y += orders_y[g] + 1
    num_betas_z = 0
    for g in range(k_z): num_betas_z += orders_z[g] + 1

    n_theta = k_y * k_z - 1
    y_beta_start = n_theta

    y_block_width = num_betas_y
    if use_dropout_y: y_block_width += 3 * k_y
    if dist_code_y == 1: y_block_width += 1
    elif dist_code_y == 3: y_block_width += k_y

    z_beta_start = y_beta_start + y_block_width

    params_y = params[y_beta_start:z_beta_start]
    params_z = params[z_beta_start:]

    # No TVCs in V5.0 (scope note above) — pass empty (N_obs, 0) arrays so the
    # single-outcome subroutines' no-TVC code path is exercised, matching the
    # V3.0-equivalent default used elsewhere.
    tvc_y = np.zeros((len(times_y), 0))
    tvc_z = np.zeros((len(times_z), 0))

    grad_subj = np.zeros((n_subjects, len(params)))
    total_ll = 0.0

    for i in range(n_subjects):
        start_y = subj_breaks_y[i]
        end_y = subj_breaks_y[i + 1]
        start_z = subj_breaks_z[i]
        end_z = subj_breaks_z[i + 1]

        # Per-outcome per-group log-likelihoods (single-outcome subroutine, §1).
        L_y, err_mu_y, err_aux_y = calc_single_outcome_group_ll_jit(
            params_y, times_y, outcomes_y, dropouts_y, start_y, end_y, orders_y,
            tvc_y, 0, use_dropout_y, dist_code_y, cnorm_min_y, cnorm_max_y
        )
        L_z, err_mu_z, err_aux_z = calc_single_outcome_group_ll_jit(
            params_z, times_z, outcomes_z, dropouts_z, start_z, end_z, orders_z,
            tvc_z, 0, use_dropout_z, dist_code_z, cnorm_min_z, cnorm_max_z
        )

        # ── JOINT MIXING PROPORTIONS pi_gh (stable softmax over Theta_joint) ──
        theta_grid = np.zeros((k_y, k_z))
        idx = 0
        for g in range(k_y):
            for h in range(k_z):
                if g == 0 and h == 0:
                    continue
                theta_grid[g, h] = params[idx]
                idx += 1

        max_theta = theta_grid[0, 0]
        for g in range(k_y):
            for h in range(k_z):
                if theta_grid[g, h] > max_theta: max_theta = theta_grid[g, h]
        sum_exp_theta = 0.0
        for g in range(k_y):
            for h in range(k_z):
                sum_exp_theta += np.exp(theta_grid[g, h] - max_theta)
        log_theta_norm = max_theta + np.log(sum_exp_theta)

        log_pi_gh = np.zeros((k_y, k_z))
        pi_gh = np.zeros((k_y, k_z))
        for g in range(k_y):
            for h in range(k_z):
                log_pi_gh[g, h] = theta_grid[g, h] - log_theta_norm
                pi_gh[g, h] = np.exp(log_pi_gh[g, h])

        # ── JOINT LOG-LIKELIHOOD (2-D log-sum-exp) ────────────────────────────
        numerator_log = np.zeros((k_y, k_z))
        for g in range(k_y):
            for h in range(k_z):
                numerator_log[g, h] = log_pi_gh[g, h] + L_y[g] + L_z[h]

        post_max = numerator_log[0, 0]
        for g in range(k_y):
            for h in range(k_z):
                if numerator_log[g, h] > post_max: post_max = numerator_log[g, h]
        post_sum_exp = 0.0
        for g in range(k_y):
            for h in range(k_z):
                post_sum_exp += np.exp(numerator_log[g, h] - post_max)
        log_norm = post_max + np.log(post_sum_exp)
        total_ll += log_norm

        # Joint posterior P(g,h|i) and marginals P(g|i), P(h|i).
        posterior_gh = np.zeros((k_y, k_z))
        for g in range(k_y):
            for h in range(k_z):
                posterior_gh[g, h] = np.exp(numerator_log[g, h] - log_norm)

        posterior_g = np.zeros(k_y)  # P(g|i) = sum_h P(g,h|i) — Y's beta gradient weight
        for g in range(k_y):
            acc = 0.0
            for h in range(k_z): acc += posterior_gh[g, h]
            posterior_g[g] = acc

        posterior_h = np.zeros(k_z)  # P(h|i) = sum_g P(g,h|i) — Z's beta gradient weight
        for h in range(k_z):
            acc = 0.0
            for g in range(k_y): acc += posterior_gh[g, h]
            posterior_h[h] = acc

        # ── Theta_joint GRADIENT: d ell_i / d theta_gh = P(g,h|i) - pi_gh ─────
        idx = 0
        for g in range(k_y):
            for h in range(k_z):
                if g == 0 and h == 0:
                    continue
                grad_subj[i, idx] = -1.0 * (posterior_gh[g, h] - pi_gh[g, h])
                idx += 1

        # ── Y-OUTCOME GRADIENT, weighted by the MARGINAL posterior_g ──────────
        accumulate_single_outcome_gradient_jit(
            grad_subj[i, y_beta_start:z_beta_start], params_y, times_y, outcomes_y, dropouts_y,
            start_y, end_y, orders_y, tvc_y, 0, use_dropout_y, dist_code_y,
            err_mu_y, err_aux_y, posterior_g
        )
        # ── Z-OUTCOME GRADIENT, weighted by the MARGINAL posterior_h ──────────
        accumulate_single_outcome_gradient_jit(
            grad_subj[i, z_beta_start:], params_z, times_z, outcomes_z, dropouts_z,
            start_z, end_z, orders_z, tvc_z, 0, use_dropout_z, dist_code_z,
            err_mu_z, err_aux_z, posterior_h
        )

    grad_flat = np.zeros(len(params))
    for i in range(grad_subj.shape[0]):
        for j in range(grad_subj.shape[1]):
            grad_flat[j] += grad_subj[i, j]

    return -1.0 * total_ll, grad_flat, grad_subj


def calc_joint_nll_wrapper(params, *args):
    """NLL-only callable for SciPy minimise, joint dual-trajectory model (V5.0).

    Args mirror calc_joint_dual_outcome_gradients_jit exactly (see that
    function's docstring) — this wrapper has the signature scipy.optimize.minimize
    expects when passed as ``fun``.
    """
    nll, _, _ = calc_joint_dual_outcome_gradients_jit(params, *args)
    return nll


def calc_joint_jac_wrapper(params, *args):
    """Jacobian-only callable for SciPy minimise, joint dual-trajectory model (V5.0)."""
    _, grad_flat, _ = calc_joint_dual_outcome_gradients_jit(params, *args)
    return grad_flat


def calc_joint_nll_jac_wrapper(params, *args):
    """Combined NLL+Jacobian callable for scipy.optimize.minimize(..., jac=True) --
    see calc_nll_jac_wrapper's docstring for why this avoids a redundant second
    full kernel pass per BFGS evaluation point. Used by _run_multistart only;
    the separate calc_joint_jac_wrapper remains in use for the finite-difference
    Hessian pass (process_joint_optimization_result), which needs many distinct
    perturbed points, not the same x twice."""
    nll, grad_flat, _ = calc_joint_dual_outcome_gradients_jit(params, *args)
    return nll, grad_flat


def calc_joint_grad_subj_wrapper(params, *args):
    """Per-subject-gradient-only callable (Huber-White sandwich G matrix), joint model (V5.0)."""
    _, _, grad_subj = calc_joint_dual_outcome_gradients_jit(params, *args)
    return grad_subj


def _joint_layout(k_y, k_z, orders_y, orders_z, use_dropout_y, dist_y, use_dropout_z, dist_z):
    """Compute the joint parameter vector's block boundaries (MATH.md §9).

    Returns (n_theta, y_beta_start, z_beta_start, num_betas_y, num_betas_z,
    num_params) — the handful of indices every joint fitting/post-processing
    function needs, computed once in one place to avoid drift between them.
    """
    n_theta = k_y * k_z - 1
    num_betas_y = sum(o + 1 for o in orders_y)
    num_betas_z = sum(o + 1 for o in orders_z)
    y_beta_start = n_theta

    y_block_width = num_betas_y
    if use_dropout_y: y_block_width += 3 * k_y
    if dist_y == 'CNORM': y_block_width += 1
    elif dist_y == 'ZIP': y_block_width += k_y

    z_beta_start = y_beta_start + y_block_width

    z_block_width = num_betas_z
    if use_dropout_z: z_block_width += 3 * k_z
    if dist_z == 'CNORM': z_block_width += 1
    elif dist_z == 'ZIP': z_block_width += k_z

    num_params = z_beta_start + z_block_width
    return n_theta, y_beta_start, z_beta_start, num_betas_y, num_betas_z, num_params


def process_joint_optimization_result(result, num_params, k_y, k_z, orders_y, orders_z,
                                       times_y, outcomes_y, dropouts_y, subj_breaks_y, use_dropout_y, dist_y, cnorm_min_y, cnorm_max_y, scale_factor_y,
                                       times_z, outcomes_z, dropouts_z, subj_breaks_z, use_dropout_z, dist_z, cnorm_min_z, cnorm_max_z, scale_factor_z):
    """Post-process a joint dual-trajectory OptimizeResult (V5.0): SEs, BIC/AIC,
    and the joint mixing-probability matrix pi_gh.

    Mirrors process_optimization_result (MATH.md §5) — same finite-difference
    Hessian / model-based / Huber-White sandwich recipe — but for the wider
    joint parameter vector, with TWO independent time-unscaling factors (each
    outcome rescales its own betas by its own scale_factor; Theta_joint stays
    dimensionless, D=1, same convention as Gamma/delta in V3.0).

    Returns:
        Tuple of 12 elements (same shape as process_optimization_result):
            is_valid, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard,
            se_model, se_robust, pis_joint (K_Y x K_Z ndarray), cond_num, V_model_unscaled
    """
    n_subjects = len(subj_breaks_y) - 1
    n_obs = len(times_y) + len(times_z)
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'POISSON': 2, 'ZIP': 3}
    dist_code_y = dist_map.get(dist_y, 0)
    dist_code_z = dist_map.get(dist_z, 0)

    n_theta, y_beta_start, z_beta_start, num_betas_y, num_betas_z, _ = _joint_layout(
        k_y, k_z, orders_y, orders_z, use_dropout_y, dist_y, use_dropout_z, dist_z
    )

    if not (result.success or result.status == 2):
        return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, None, None, np.nan, None

    D_diag = np.ones(num_params)
    idx = y_beta_start
    for g in range(k_y):
        for p in range(orders_y[g] + 1):
            D_diag[idx + p] = 1.0 / (scale_factor_y ** p)
        idx += orders_y[g] + 1
    if use_dropout_y:
        for g in range(k_y):
            D_diag[idx + 1] = 1.0 / scale_factor_y
            idx += 3
    # Y's tail (if any) stays D=1 — already the default (dimensionless, MATH.md §5b).

    idx = z_beta_start
    for g in range(k_z):
        for p in range(orders_z[g] + 1):
            D_diag[idx + p] = 1.0 / (scale_factor_z ** p)
        idx += orders_z[g] + 1
    if use_dropout_z:
        for g in range(k_z):
            D_diag[idx + 1] = 1.0 / scale_factor_z
            idx += 3

    D = np.diag(D_diag)

    try:
        times_y_scaled = times_y / scale_factor_y
        times_z_scaled = times_z / scale_factor_z
        orders_y_arr = np.array(orders_y, dtype=np.int32)
        orders_z_arr = np.array(orders_z, dtype=np.int32)
        args = (times_y_scaled, outcomes_y, dropouts_y, subj_breaks_y, orders_y_arr, use_dropout_y, dist_code_y, float(cnorm_min_y), float(cnorm_max_y),
                times_z_scaled, outcomes_z, dropouts_z, subj_breaks_z, orders_z_arr, use_dropout_z, dist_code_z, float(cnorm_min_z), float(cnorm_max_z))

        H_scaled = np.zeros((num_params, num_params))
        for i in range(num_params):
            eps_i = 1e-5 * max(1.0, abs(result.x[i]))
            if eps_i < 1e-8: eps_i = 1e-8
            p_plus = np.copy(result.x)
            p_minus = np.copy(result.x)
            p_plus[i] += eps_i
            p_minus[i] -= eps_i
            g_plus = calc_joint_jac_wrapper(p_plus, *args)
            g_minus = calc_joint_jac_wrapper(p_minus, *args)
            H_scaled[i, :] = (g_plus - g_minus) / (2.0 * eps_i)

        H_scaled = (H_scaled + H_scaled.T) / 2.0

        try:
            cond_num = np.linalg.cond(H_scaled)
        except np.linalg.LinAlgError:
            cond_num = np.inf

        H_inv_scaled = np.linalg.pinv(H_scaled, rcond=1e-10)

        _, _, grad_subj_scaled = calc_joint_dual_outcome_gradients_jit(result.x, *args)
        G_scaled = grad_subj_scaled.T @ grad_subj_scaled
        V_robust_scaled = H_inv_scaled @ G_scaled @ H_inv_scaled
    except Exception:
        H_inv_scaled = np.eye(num_params)
        V_robust_scaled = np.eye(num_params)
        cond_num = np.inf

    params_unscaled = D @ result.x
    V_model_unscaled = D @ H_inv_scaled @ D
    V_robust_unscaled = D @ V_robust_scaled @ D

    se_model = np.sqrt(np.abs(np.diag(V_model_unscaled)))
    se_robust = np.sqrt(np.abs(np.diag(V_robust_unscaled)))

    ll = -1.0 * result.fun
    aic_nagin = ll - num_params
    bic_nagin = ll - 0.5 * num_params * np.log(n_subjects)
    bic_obs = ll - 0.5 * num_params * np.log(n_obs)
    aic_standard = -2.0 * ll + 2.0 * num_params
    bic_standard = -2.0 * ll + num_params * np.log(n_subjects)

    # Joint mixing proportions pi_gh, recovered from Theta_joint via softmax
    # over the flattened (k_y, k_z) grid (reference cell (0,0) implicitly 0).
    theta_grid = np.zeros((k_y, k_z))
    idx = 0
    for g in range(k_y):
        for h in range(k_z):
            if g == 0 and h == 0: continue
            theta_grid[g, h] = params_unscaled[idx]
            idx += 1
    max_theta = np.max(theta_grid)
    pis_joint = np.exp(theta_grid - max_theta)
    pis_joint /= pis_joint.sum()

    result.x = params_unscaled
    return True, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis_joint, cond_num, V_model_unscaled


def _permute_solo_outcome_blocks(params_slice, se_m_slice, se_r_slice, orders_list, use_dropout, dist, sorted_idx):
    """Permute ONE outcome's solo parameter slice (beta/dropout/tail — the
    same layout calc_single_outcome_group_ll_jit consumes) by sorted_idx, so
    calc_universal_subject_gradients_jit's sort_groups_by_intercept and V5.0's
    sort_joint_groups_by_intercept can share this logic instead of each
    re-implementing beta/dropout/zeta block permutation independently.

    Returns (new_params, new_se_model, new_se_robust, new_orders_list) —
    does not mutate its inputs.
    """
    k = len(orders_list)
    if k == 1:
        return params_slice.copy(), se_m_slice.copy(), se_r_slice.copy(), orders_list

    beta_starts = []
    idx = 0
    for g in range(k):
        beta_starts.append(idx)
        idx += orders_list[g] + 1
    gamma_start = idx

    new_params = params_slice.copy()
    new_se_m = se_m_slice.copy()
    new_se_r = se_r_slice.copy()

    new_orders = [orders_list[sorted_idx[g]] for g in range(k)]
    write_idx = 0
    for new_g in range(k):
        old_g = sorted_idx[new_g]
        n_betas = orders_list[old_g] + 1
        src = beta_starts[old_g]
        new_params[write_idx:write_idx + n_betas] = params_slice[src:src + n_betas]
        new_se_m[write_idx:write_idx + n_betas] = se_m_slice[src:src + n_betas]
        new_se_r[write_idx:write_idx + n_betas] = se_r_slice[src:src + n_betas]
        write_idx += n_betas

    if use_dropout:
        for new_g in range(k):
            old_g = sorted_idx[new_g]
            src = gamma_start + 3 * old_g
            dst = gamma_start + 3 * new_g
            new_params[dst:dst + 3] = params_slice[src:src + 3]
            new_se_m[dst:dst + 3] = se_m_slice[src:src + 3]
            new_se_r[dst:dst + 3] = se_r_slice[src:src + 3]

    if dist == 'ZIP':
        zeta_start = len(params_slice) - k
        for new_g in range(k):
            old_g = sorted_idx[new_g]
            new_params[zeta_start + new_g] = params_slice[zeta_start + old_g]
            new_se_m[zeta_start + new_g] = se_m_slice[zeta_start + old_g]
            new_se_r[zeta_start + new_g] = se_r_slice[zeta_start + old_g]
    # CNORM raw_sigma is a scalar tail — not group-specific, left untouched.

    return new_params, new_se_m, new_se_r, new_orders


def sort_joint_groups_by_intercept(result, k_y, k_z, orders_y, orders_z, se_model, se_robust,
                                    use_dropout_y, dist_y, use_dropout_z, dist_z):
    """Sort BOTH outcomes' groups independently by ascending intercept, and
    consistently re-permute the joint mixing matrix pi_gh across both axes
    (V5.0's two-dimensional generalization of sort_groups_by_intercept).

    Theta_joint's raw logits CANNOT simply be permuted like a beta block —
    they are only meaningful relative to the OLD reference cell (0,0). Instead:
    (1) reconstruct the full pi_gh matrix via softmax (uniquely defined, no
    reference ambiguity), (2) permute BOTH axes simultaneously via
    pi_new = pi_old[np.ix_(sorted_idx_y, sorted_idx_z)], (3) re-derive
    theta'_gh = log(pi_new[g,h]) - log(pi_new[0,0]). This recompute runs
    whenever EITHER axis needs resorting, since the reference cell can change
    even if only one axis actually permutes.

    Theta_joint SE carryover (mapping each new cell back to its old cell) is
    an approximation, same caveat as sort_groups_by_intercept's Gamma SEs.
    Y-BLOCK/Z-BLOCK beta/dropout/tail SEs are exact (mechanical permutation).

    Returns:
        new_orders_y, new_orders_z, new_se_model, new_se_robust, new_pis_joint (K_Y x K_Z)
        result.x is mutated in place.
    """
    n_theta, y_beta_start, z_beta_start, num_betas_y, num_betas_z, _ = _joint_layout(
        k_y, k_z, orders_y, orders_z, use_dropout_y, dist_y, use_dropout_z, dist_z
    )

    params = result.x.copy()
    se_m = se_model.copy()
    se_r = se_robust.copy()

    # Recover the full pi_gh matrix from Theta_joint (reference cell (0,0) = 0).
    theta_grid = np.zeros((k_y, k_z))
    idx = 0
    for g in range(k_y):
        for h in range(k_z):
            if g == 0 and h == 0: continue
            theta_grid[g, h] = params[idx]
            idx += 1
    max_theta = np.max(theta_grid)
    pi_gh = np.exp(theta_grid - max_theta)
    pi_gh /= pi_gh.sum()

    y_beta_starts = []
    idx = y_beta_start
    for g in range(k_y):
        y_beta_starts.append(idx)
        idx += orders_y[g] + 1
    y_intercepts = np.array([params[y_beta_starts[g]] for g in range(k_y)])
    sorted_idx_y = np.argsort(y_intercepts)

    z_beta_starts = []
    idx = z_beta_start
    for g in range(k_z):
        z_beta_starts.append(idx)
        idx += orders_z[g] + 1
    z_intercepts = np.array([params[z_beta_starts[g]] for g in range(k_z)])
    sorted_idx_z = np.argsort(z_intercepts)

    if np.all(sorted_idx_y == np.arange(k_y)) and np.all(sorted_idx_z == np.arange(k_z)):
        return orders_y, orders_z, se_model, se_robust, pi_gh  # already sorted

    new_params = params.copy()
    new_se_m = se_m.copy()
    new_se_r = se_r.copy()

    # --- Permute pi_gh across BOTH axes simultaneously, re-derive Theta_joint ---
    pi_new = pi_gh[np.ix_(sorted_idx_y, sorted_idx_z)]
    theta_new = np.log(np.maximum(pi_new, 1e-300)) - np.log(max(pi_new[0, 0], 1e-300))
    idx = 0
    for g in range(k_y):
        for h in range(k_z):
            if g == 0 and h == 0: continue
            new_params[idx] = theta_new[g, h]
            idx += 1

    # Approximate SE carryover: new cell (g',h') <- old cell (sorted_idx_y[g'], sorted_idx_z[h']).
    old_theta_se_m = np.zeros((k_y, k_z))
    old_theta_se_r = np.zeros((k_y, k_z))
    idx = 0
    for g in range(k_y):
        for h in range(k_z):
            if g == 0 and h == 0: continue
            old_theta_se_m[g, h] = se_m[idx]
            old_theta_se_r[g, h] = se_r[idx]
            idx += 1
    new_theta_se_m = old_theta_se_m[np.ix_(sorted_idx_y, sorted_idx_z)]
    new_theta_se_r = old_theta_se_r[np.ix_(sorted_idx_y, sorted_idx_z)]
    idx = 0
    for g in range(k_y):
        for h in range(k_z):
            if g == 0 and h == 0: continue
            new_se_m[idx] = new_theta_se_m[g, h]
            new_se_r[idx] = new_theta_se_r[g, h]
            idx += 1

    # --- Permute Y-BLOCK and Z-BLOCK independently (shared solo-outcome logic) ---
    new_y_params, new_y_se_m, new_y_se_r, new_orders_y = _permute_solo_outcome_blocks(
        params[y_beta_start:z_beta_start], se_m[y_beta_start:z_beta_start], se_r[y_beta_start:z_beta_start],
        orders_y, use_dropout_y, dist_y, sorted_idx_y
    )
    new_params[y_beta_start:z_beta_start] = new_y_params
    new_se_m[y_beta_start:z_beta_start] = new_y_se_m
    new_se_r[y_beta_start:z_beta_start] = new_y_se_r

    new_z_params, new_z_se_m, new_z_se_r, new_orders_z = _permute_solo_outcome_blocks(
        params[z_beta_start:], se_m[z_beta_start:], se_r[z_beta_start:],
        orders_z, use_dropout_z, dist_z, sorted_idx_z
    )
    new_params[z_beta_start:] = new_z_params
    new_se_m[z_beta_start:] = new_z_se_m
    new_se_r[z_beta_start:] = new_z_se_r

    result.x = new_params
    return new_orders_y, new_orders_z, new_se_m, new_se_r, pi_new


def _resolve_covariate_arrays(n_subjects, n_obs, baseline_X, tvc_Z):
    """Resolve optional baseline_X/tvc_Z to their no-covariate (V1.5.0-equivalent) defaults.

    This is the single source of truth for V3.0's backward-compatibility guarantee
    (MATH.md §2): baseline_X defaults to an intercept-only ones((n_subjects,1))
    column and tvc_Z defaults to an empty zeros((n_obs,0)) array, so every caller
    that omits these arguments gets numerically identical behaviour to V1.5.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: C-contiguous float64 (baseline_X, tvc_Z).
    """
    if baseline_X is None:
        baseline_X = np.ones((n_subjects, 1), dtype=np.float64)
    else:
        baseline_X = np.ascontiguousarray(baseline_X, dtype=np.float64)
    if tvc_Z is None:
        tvc_Z = np.zeros((n_obs, 0), dtype=np.float64)
    else:
        tvc_Z = np.ascontiguousarray(tvc_Z, dtype=np.float64)
    return baseline_X, tvc_Z


def _resolve_weights_array(n_subjects, weights):
    """Resolve an optional per-subject weight array to its unweighted (V3.0-equivalent) default.

    This is the single source of truth for V4.0's backward-compatibility guarantee:
    omitting weights (or passing None) defaults to ones(n_subjects), so every caller
    that doesn't supply weights gets numerically identical behaviour to V3.0.

    Returns:
        np.ndarray: (n_subjects,) C-contiguous float64 weight array.
    """
    if weights is None:
        return np.ones(n_subjects, dtype=np.float64)
    return np.ascontiguousarray(weights, dtype=np.float64)


def calc_nll_wrapper(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max, baseline_X=None, tvc_Z=None, weights=None):
    """NLL-only callable for SciPy minimise (discards gradient and per-subject matrix).

    This wrapper has the exact signature expected by scipy.optimize.minimize
    when passed as the ``fun`` argument (without ``jac``).

    Args:
        params:      (p,) parameter vector.
        times:       (N_obs,) pre-scaled time values.
        outcomes:    (N_obs,) outcome values.
        dropouts:    (N_obs,) dropout indicator array.
        subj_breaks: (N_subjects+1,) subject boundary indices.
        orders:      (K,) int32 polynomial order array.
        zip_iorder:  Unused legacy parameter.
        use_dropout: bool — include dropout model.
        dist_code:   int  — distribution selector (0–3).
        cnorm_min:   float — CNORM lower censoring bound.
        cnorm_max:   float — CNORM upper censoring bound.
        baseline_X:  optional (N_subjects, n_mix) mixing-covariate design matrix
                     (V3.0). Defaults to intercept-only (no covariates).
        tvc_Z:       optional (N_obs, n_tvc) time-varying covariate matrix (V3.0).
                     Defaults to empty (no TVCs).
        weights:     optional (N_subjects,) survey/sampling weight array (V4.0).
                     Defaults to ones (unweighted).

    Returns:
        float: Negative log-likelihood (scalar to minimise).
    """
    n_subjects = len(subj_breaks) - 1
    baseline_X, tvc_Z = _resolve_covariate_arrays(n_subjects, len(times), baseline_X, tvc_Z)
    weights = _resolve_weights_array(n_subjects, weights)
    nll, _, _ = calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max, baseline_X, tvc_Z, baseline_X.shape[1], tvc_Z.shape[1], weights)
    return nll

def calc_jac_wrapper(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max, baseline_X=None, tvc_Z=None, weights=None):
    """Jacobian-only callable for SciPy minimise (discards NLL and per-subject matrix).

    Passed as the ``jac`` argument to scipy.optimize.minimize so that BFGS
    uses the analytical gradient rather than finite-difference approximation.
    See calc_nll_wrapper for the meaning of baseline_X/tvc_Z (V3.0) and weights (V4.0).

    Returns:
        np.ndarray: (p,) gradient of the NLL with respect to params.
    """
    n_subjects = len(subj_breaks) - 1
    baseline_X, tvc_Z = _resolve_covariate_arrays(n_subjects, len(times), baseline_X, tvc_Z)
    weights = _resolve_weights_array(n_subjects, weights)
    _, grad_flat, _ = calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max, baseline_X, tvc_Z, baseline_X.shape[1], tvc_Z.shape[1], weights)
    return grad_flat

def calc_nll_jac_wrapper(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max, baseline_X=None, tvc_Z=None, weights=None):
    """Combined NLL+Jacobian callable for scipy.optimize.minimize(..., jac=True).

    scipy's BFGS calls ``fun(x)`` and ``jac(x)`` as two independent black-box
    functions, at the same x, once per line-search evaluation. Since
    calc_nll_wrapper and calc_jac_wrapper each independently re-run the exact
    same kernel pass (calc_universal_subject_gradients_jit already computes
    both the NLL and the gradient together, then one wrapper discards the
    gradient and the other discards the NLL), calling them separately means
    the entire per-subject likelihood/gradient loop runs TWICE per BFGS
    evaluation point for no reason. This wrapper returns both from a single
    kernel call, used via ``jac=True`` in _run_multistart -- halving the
    number of full kernel passes needed during optimization (does not affect
    the separate finite-difference Hessian pass, which legitimately needs
    calc_jac_wrapper at many distinct perturbed points, not the same x twice).
    """
    n_subjects = len(subj_breaks) - 1
    baseline_X, tvc_Z = _resolve_covariate_arrays(n_subjects, len(times), baseline_X, tvc_Z)
    weights = _resolve_weights_array(n_subjects, weights)
    nll, grad_flat, _ = calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max, baseline_X, tvc_Z, baseline_X.shape[1], tvc_Z.shape[1], weights)
    return nll, grad_flat

def calc_grad_subj_wrapper(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max, baseline_X=None, tvc_Z=None, weights=None):
    """Per-subject-gradient-only callable (used to build the Huber-White sandwich G matrix).

    See calc_nll_wrapper for the meaning of baseline_X/tvc_Z (V3.0) and weights (V4.0).

    Returns:
        np.ndarray: (N_subjects, p) per-subject gradient matrix.
    """
    n_subjects = len(subj_breaks) - 1
    baseline_X, tvc_Z = _resolve_covariate_arrays(n_subjects, len(times), baseline_X, tvc_Z)
    weights = _resolve_weights_array(n_subjects, weights)
    _, _, grad_subj = calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max, baseline_X, tvc_Z, baseline_X.shape[1], tvc_Z.shape[1], weights)
    return grad_subj

# --- DISTRIBUTION-SPECIFIC PUBLIC ALIASES ---
# The universal engine (dist_code dispatch) handles all distributions.
# These thin wrappers expose the expected function names for external consumers
# (e.g. verification scripts, notebooks) without duplicating any math.

def calc_poisson_dynamic_nll_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, cnorm_min=0.0, cnorm_max=0.0, baseline_X=None, tvc_Z=None, weights=None):
    """NLL for Poisson trajectories — delegates to universal engine (dist_code=2)."""
    n_subjects = len(subj_breaks) - 1
    baseline_X, tvc_Z = _resolve_covariate_arrays(n_subjects, len(times), baseline_X, tvc_Z)
    weights = _resolve_weights_array(n_subjects, weights)
    nll, _, _ = calc_universal_subject_gradients_jit(
        params, times, outcomes, dropouts, subj_breaks, orders,
        int(zip_iorder), use_dropout, 2, float(cnorm_min), float(cnorm_max),
        baseline_X, tvc_Z, baseline_X.shape[1], tvc_Z.shape[1], weights
    )
    return nll

def calc_poisson_dynamic_jacobian_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, cnorm_min=0.0, cnorm_max=0.0, baseline_X=None, tvc_Z=None, weights=None):
    """Gradient for Poisson trajectories — delegates to universal engine (dist_code=2)."""
    n_subjects = len(subj_breaks) - 1
    baseline_X, tvc_Z = _resolve_covariate_arrays(n_subjects, len(times), baseline_X, tvc_Z)
    weights = _resolve_weights_array(n_subjects, weights)
    _, grad, _ = calc_universal_subject_gradients_jit(
        params, times, outcomes, dropouts, subj_breaks, orders,
        int(zip_iorder), use_dropout, 2, float(cnorm_min), float(cnorm_max),
        baseline_X, tvc_Z, baseline_X.shape[1], tvc_Z.shape[1], weights
    )
    return grad

def calc_zip_dynamic_nll_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, cnorm_min=0.0, cnorm_max=0.0, baseline_X=None, tvc_Z=None, weights=None):
    """NLL for ZIP trajectories — delegates to universal engine (dist_code=3)."""
    n_subjects = len(subj_breaks) - 1
    baseline_X, tvc_Z = _resolve_covariate_arrays(n_subjects, len(times), baseline_X, tvc_Z)
    weights = _resolve_weights_array(n_subjects, weights)
    nll, _, _ = calc_universal_subject_gradients_jit(
        params, times, outcomes, dropouts, subj_breaks, orders,
        int(zip_iorder), use_dropout, 3, float(cnorm_min), float(cnorm_max),
        baseline_X, tvc_Z, baseline_X.shape[1], tvc_Z.shape[1], weights
    )
    return nll

def calc_zip_dynamic_jacobian_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, cnorm_min=0.0, cnorm_max=0.0, baseline_X=None, tvc_Z=None, weights=None):
    """Gradient for ZIP trajectories — delegates to universal engine (dist_code=3)."""
    n_subjects = len(subj_breaks) - 1
    baseline_X, tvc_Z = _resolve_covariate_arrays(n_subjects, len(times), baseline_X, tvc_Z)
    weights = _resolve_weights_array(n_subjects, weights)
    _, grad, _ = calc_universal_subject_gradients_jit(
        params, times, outcomes, dropouts, subj_breaks, orders,
        int(zip_iorder), use_dropout, 3, float(cnorm_min), float(cnorm_max),
        baseline_X, tvc_Z, baseline_X.shape[1], tvc_Z.shape[1], weights
    )
    return grad

# --- ENGINE WRAPPERS ---

def process_optimization_result(result, num_params, times, outcomes, dropouts, subj_breaks, orders_list, zip_iorder, use_dropout, scale_factor, dist, cnorm_min, cnorm_max, baseline_X=None, tvc_Z=None, weights=None):
    """Post-process a SciPy OptimizeResult: compute SEs, BIC/AIC, and mixture weights.

    This function is called immediately after each SciPy BFGS optimisation.
    It performs the following steps:

    1. Early-exit if the optimiser did not converge (result.success is False
       AND result.status != 2).
    2. Build the time-scale unscaling matrix D so that reported betas are in
       original time units (not scaled-time units used during optimisation).
    3. Compute the numerical Hessian H by central finite-differences on the
       analytical Jacobian.  Step size is adaptive: ε = max(1e-5·|θ_j|, 1e-8).
    4. Compute model-based covariance:   V_model  = D · pinv(H) · D
    5. Compute Huber-White sandwich:     V_robust = D · H⁻¹ · G · H⁻¹ · D
       where G = Σ_i gᵢ gᵢᵀ (outer products of per-subject analytical gradients).
    6. Compute SEs as sqrt(|diag(V)|) — absolute value guards against small
       negative diagonals caused by numerical noise.
    7. Compute LL, BIC/AIC (both Nagin and standard conventions), mixing weights,
       and the condition number of H (used as an identifiability proxy).

    Args:
        result:       scipy.optimize.OptimizeResult from BFGS.
        num_params:   Total number of free parameters p.
        times:        (N_obs,) original (unscaled) observation times.
        outcomes:     (N_obs,) outcome values.
        dropouts:     (N_obs,) dropout indicator array.
        subj_breaks:  (N_subjects+1,) boundary indices.
        orders_list:  List of polynomial orders per group.
        zip_iorder:   Legacy parameter (unused).
        use_dropout:  bool — whether the dropout model was fitted.
        scale_factor: max|t|; used to build D and to re-scale times for gradient calls.
        dist:         Distribution string 'LOGIT' | 'CNORM' | 'POISSON' | 'ZIP'.
        cnorm_min:    float — CNORM lower censoring bound.
        cnorm_max:    float — CNORM upper censoring bound.
        baseline_X:   optional (N_subjects, n_mix) mixing-covariate design matrix
                      (V3.0). Defaults to intercept-only (no covariates).
        tvc_Z:        optional (N_obs, n_tvc) time-varying covariate matrix (V3.0).
                      Defaults to empty (no TVCs).
        weights:      optional (N_subjects,) survey/sampling weight array (V4.0).
                      Defaults to ones (unweighted).

    Returns:
        Tuple of 12 elements:
            is_valid    : bool — True if the optimiser converged.
            ll          : float — Log-likelihood (NaN if not converged).
            aic_nagin   : float — Nagin AIC = ℓ - p.
            bic_nagin   : float — Nagin BIC = ℓ - ½·p·log(N).
            bic_obs     : float — BIC using N_obs instead of N_subjects.
            aic_standard: float — Standard AIC = -2ℓ + 2p.
            bic_standard: float — Standard BIC = -2ℓ + p·log(N).
            se_model    : (p,) array of model-based SEs (None if not converged).
            se_robust   : (p,) array of robust sandwich SEs (None if not converged).
            pis         : (K,) mixing weight array (None if not converged).
            cond_num    : float — condition number of H; >1e10 flags near-singularity.
            V_model_unscaled: (p,p) model-based covariance matrix (None if not converged).
    """
    n_subjects = len(subj_breaks) - 1
    n_obs = len(times)
    orders_arr = np.array(orders_list, dtype=np.int32)
    k = len(orders_list)
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'POISSON': 2, 'ZIP': 3}
    dist_code = dist_map.get(dist, 0)
    baseline_X, tvc_Z = _resolve_covariate_arrays(n_subjects, n_obs, baseline_X, tvc_Z)
    weights = _resolve_weights_array(n_subjects, weights)
    n_mix = baseline_X.shape[1]
    n_tvc = tvc_Z.shape[1]
    num_betas = sum(order + 1 for order in orders_list)

    if not (result.success or result.status == 2):
        return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, None, None, np.nan, None

    D_diag = np.ones(num_params)
    current_beta_idx = (k - 1) * n_mix
    for g in range(k):
        for p in range(orders_list[g] + 1):
            D_diag[current_beta_idx + p] = 1.0 / (scale_factor ** p)
        current_beta_idx += orders_list[g] + 1
    # Gamma (mixing covariate) and delta (TVC) blocks are dimensionless — D=1.0
    # (already the default), since they multiply arbitrary covariates, not
    # powers of rescaled time (MATH.md §5b).

    if use_dropout:
        current_gamma_idx = current_beta_idx + k * n_tvc
        for g in range(k):
            D_diag[current_gamma_idx + 1] = 1.0 / scale_factor
            current_gamma_idx += 3

    if dist == 'CNORM':
        D_diag[-1] = 1.0
    # ZIP zeta params are dimensionless logits — no time-unit scaling needed

    D = np.diag(D_diag)

    try:
        times_scaled = times / scale_factor
        args = (times_scaled, outcomes, dropouts, subj_breaks, orders_arr, int(zip_iorder), use_dropout, dist_code, float(cnorm_min), float(cnorm_max), baseline_X, tvc_Z)

        H_scaled = np.zeros((num_params, num_params))
        for i in range(num_params):
            eps_i = 1e-5 * max(1.0, abs(result.x[i]))
            if eps_i < 1e-8: eps_i = 1e-8

            p_plus = np.copy(result.x)
            p_minus = np.copy(result.x)
            p_plus[i] += eps_i
            p_minus[i] -= eps_i
            g_plus = calc_jac_wrapper(p_plus, *args, weights=weights)
            g_minus = calc_jac_wrapper(p_minus, *args, weights=weights)
            H_scaled[i, :] = (g_plus - g_minus) / (2.0 * eps_i)

        H_scaled = (H_scaled + H_scaled.T) / 2.0

        try:
            cond_num = np.linalg.cond(H_scaled)
        except np.linalg.LinAlgError:
            cond_num = np.inf

        H_inv_scaled = np.linalg.pinv(H_scaled, rcond=1e-10)

        _, _, grad_subj_scaled = calc_universal_subject_gradients_jit(result.x, *args, n_mix, n_tvc, weights)
        G_scaled = grad_subj_scaled.T @ grad_subj_scaled
        V_robust_scaled = H_inv_scaled @ G_scaled @ H_inv_scaled
    except Exception:
        H_inv_scaled = np.eye(num_params)
        V_robust_scaled = np.eye(num_params)
        cond_num = np.inf

    params_unscaled = D @ result.x
    V_model_unscaled = D @ H_inv_scaled @ D
    V_robust_unscaled = D @ V_robust_scaled @ D

    se_model = np.sqrt(np.abs(np.diag(V_model_unscaled)))
    se_robust = np.sqrt(np.abs(np.diag(V_robust_unscaled)))

    ll = -1.0 * result.fun
    aic_nagin = ll - num_params
    bic_nagin = ll - 0.5 * num_params * np.log(n_subjects)
    bic_obs = ll - 0.5 * num_params * np.log(n_obs)
    aic_standard = -2.0 * ll + 2.0 * num_params
    bic_standard = -2.0 * ll + num_params * np.log(n_subjects)

    # Mixing proportions reported at the sample-mean covariate profile (MATH.md
    # §7 OCC note). With n_mix==1 (intercept-only), x_bar==[1.0] and this
    # reduces exactly to the V1.5.0 formula.
    x_bar = baseline_X.mean(axis=0)
    thetas = np.zeros(k)
    for g in range(1, k):
        thetas[g] = np.dot(params_unscaled[(g - 1) * n_mix : g * n_mix], x_bar)
    pis = np.exp(thetas - logsumexp(thetas))

    result.x = params_unscaled
    return True, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis, cond_num, V_model_unscaled

def sort_groups_by_intercept(result, orders_list, se_model, se_robust, pis, use_dropout, dist, n_mix=1, n_tvc=0):
    """
    Sort groups by ascending intercept (beta_0) to eliminate label switching.

    After optimization the labelling of Group 1 / Group 2 / … is arbitrary —
    whichever local optimum the solver found determines which label goes where.
    This function reorders all group-specific blocks in result.x (Gamma, betas,
    delta, gammas) so that Group 1 always has the lowest intercept.

    Args:
        ... (existing) ...
        n_mix: Mixing-covariate block width per group (P+1, incl. intercept;
               V3.0). Default 1 = intercept-only (V1.5.0-equivalent).
        n_tvc: Number of time-varying covariates (V3.0). Default 0 = none.

    Returns
    -------
    new_orders_list : list  — polynomial orders in the new group ordering
    new_se_model    : ndarray
    new_se_robust   : ndarray
    new_pis         : ndarray
    result.x is mutated in place.

    Notes
    -----
    • The Gamma re-normalisation (subtracting the new reference group's full
      Gamma vector) is exact for the likelihood: θ_g(x) - θ_r(x) is invariant
      to the choice of reference group for any x, so subtracting the same
      vector from every group's Gamma preserves softmax(θ(x)) identically.
      The Gamma SEs are rearranged but NOT re-derived — approximate after a
      change of reference group, same caveat as the V1.5.0 theta SEs.
      Beta / delta / gamma_drop SEs are exact.
    • CNORM log-sigma and ZIP alpha params sit at the tail and are not
      group-specific, so they are left untouched.
    """
    k = len(orders_list)
    if k == 1:
        return orders_list, se_model, se_robust, pis

    params = result.x.copy()
    se_m   = se_model.copy()
    se_r   = se_robust.copy()

    # Recover full (k, n_mix) Gamma matrix (row 0 = 0 is the implicit reference)
    gammas = np.zeros((k, n_mix))
    for g in range(1, k):
        gammas[g] = params[(g - 1) * n_mix : g * n_mix]

    # Locate the start index of each group's beta block
    beta_starts = []
    idx = (k - 1) * n_mix
    for g in range(k):
        beta_starts.append(idx)
        idx += orders_list[g] + 1
    delta_start = idx           # first index of the delta (TVC) block
    gamma_start = delta_start + k * n_tvc  # first index of gamma block (used only when use_dropout)

    # Intercepts are the first beta of each group
    intercepts   = np.array([params[beta_starts[g]] for g in range(k)])
    sorted_idx   = np.argsort(intercepts)

    if np.all(sorted_idx == np.arange(k)):
        return orders_list, se_model, se_robust, pis  # already sorted

    new_params = params.copy()
    new_se_m   = se_m.copy()
    new_se_r   = se_r.copy()

    # --- Rearrange Gamma blocks ---
    new_gammas = gammas[sorted_idx]
    new_gammas = new_gammas - new_gammas[0]   # re-reference: new group 0 becomes implicit zero
    for g in range(1, k):
        new_params[(g - 1) * n_mix : g * n_mix] = new_gammas[g]

    # Approximate SE rearrangement for Gamma (row 0 has no stored SE; treat as 0)
    old_gse_m = np.zeros((k, n_mix))
    old_gse_r = np.zeros((k, n_mix))
    if k > 1:
        old_gse_m[1:] = se_m[0:(k - 1) * n_mix].reshape(k - 1, n_mix)
        old_gse_r[1:] = se_r[0:(k - 1) * n_mix].reshape(k - 1, n_mix)
    new_gse_m = old_gse_m[sorted_idx]
    new_gse_r = old_gse_r[sorted_idx]
    for g in range(1, k):
        new_se_m[(g - 1) * n_mix : g * n_mix] = new_gse_m[g]
        new_se_r[(g - 1) * n_mix : g * n_mix] = new_gse_r[g]

    # --- Rearrange beta blocks ---
    new_orders = [orders_list[sorted_idx[g]] for g in range(k)]
    write_idx  = (k - 1) * n_mix
    for new_g in range(k):
        old_g   = sorted_idx[new_g]
        n_betas = orders_list[old_g] + 1
        src     = beta_starts[old_g]
        new_params[write_idx:write_idx + n_betas] = params[src:src + n_betas]
        new_se_m[write_idx:write_idx + n_betas]   = se_m[src:src + n_betas]
        new_se_r[write_idx:write_idx + n_betas]   = se_r[src:src + n_betas]
        write_idx += n_betas

    # --- Rearrange delta (TVC) blocks (n_tvc params per group, always) ---
    if n_tvc > 0:
        for new_g in range(k):
            old_g = sorted_idx[new_g]
            src   = delta_start + old_g * n_tvc
            dst   = delta_start + new_g * n_tvc
            new_params[dst:dst + n_tvc] = params[src:src + n_tvc]
            new_se_m[dst:dst + n_tvc]   = se_m[src:src + n_tvc]
            new_se_r[dst:dst + n_tvc]   = se_r[src:src + n_tvc]

    # --- Rearrange gamma blocks (3 params per group, always) ---
    if use_dropout:
        for new_g in range(k):
            old_g = sorted_idx[new_g]
            src   = gamma_start + 3 * old_g
            dst   = gamma_start + 3 * new_g
            new_params[dst:dst + 3] = params[src:src + 3]
            new_se_m[dst:dst + 3]   = se_m[src:src + 3]
            new_se_r[dst:dst + 3]   = se_r[src:src + 3]

    # CNORM log-sigma is a scalar tail param — not group-specific, left untouched.
    # ZIP zeta params (last k entries) ARE group-specific — rearrange them.
    if dist == 'ZIP':
        zeta_start = len(params) - k
        for new_g in range(k):
            old_g = sorted_idx[new_g]
            new_params[zeta_start + new_g] = params[zeta_start + old_g]
            new_se_m[zeta_start + new_g]   = se_m[zeta_start + old_g]
            new_se_r[zeta_start + new_g]   = se_r[zeta_start + old_g]

    result.x = new_params
    return new_orders, new_se_m, new_se_r, pis[sorted_idx]


def generate_initial_params(k, orders_list, zip_iorder, use_dropout, dist, outcomes, n_starts=10, n_mix=1, n_tvc=0):
    """Generate n_starts starting points for multi-start BFGS optimisation.

    The first starting point (index 0) is deterministic: intercepts are
    staggered across groups to avoid identical starts, slopes initialised at
    zero.  Points 1..n_starts-1 add seeded Gaussian perturbations to the
    deterministic base, giving the multi-start procedure coverage of the
    parameter space without requiring a global random state.

    Initialisation strategy by parameter block:
      - Gamma (mixing covariates): intercepts equally spaced logit-quantiles
        (e.g. for k=3: logit(0.25), logit(0.50), logit(0.75)) to start with
        dispersed mixing weights; covariate-slope entries (p>0) at 0. With
        n_mix=1 this is identical to the V1.5.0 theta initialisation.
      - Betas:  intercepts staggered; slopes at 0.
      - Delta (TVC): all at 0 (no prior on deflection direction).
      - Gammas (dropout): intercepts at −2 (≈ 12% dropout baseline); slopes at 0.
      - raw_sigma (CNORM): log(std(outcomes)) as a sensible starting scale.
      - zeta (ZIP): −1.0 per group (≈ 27% structural zeros baseline).

    Args:
        k:           Number of groups.
        orders_list: List of polynomial orders per group.
        zip_iorder:  Legacy parameter (unused).
        use_dropout: bool — include dropout gamma parameters.
        dist:        Distribution string 'LOGIT'|'CNORM'|'POISSON'|'ZIP'.
        outcomes:    (N_obs,) outcome array — used to set Poisson/CNORM baseline.
        n_starts:    Number of starting vectors to generate (default 10).
        n_mix:       Mixing-covariate block width per group (P+1, incl. intercept;
                     V3.0). Default 1 = intercept-only (V1.5.0-equivalent).
        n_tvc:       Number of time-varying covariates (V3.0). Default 0 = none.

    Returns:
        List[np.ndarray]: List of n_starts parameter vectors, each of length p.
    """
    num_betas = sum([order + 1 for order in orders_list])
    num_params = (k - 1) * n_mix + num_betas + k * n_tvc
    if use_dropout: num_params += (3 * k)
    if dist == 'CNORM': num_params += 1
    if dist == 'ZIP': num_params += k  # one zeta per group

    delta_start_idx = (k - 1) * n_mix + num_betas

    # --- deterministic base ---
    base = np.zeros(num_params)

    if dist == 'POISSON':
        # For Poisson (log-link): initialise intercepts near log(mean_outcome),
        # staggered across groups so each group starts at a distinct value.
        mean_out = np.mean(outcomes[outcomes > 0]) if np.any(outcomes > 0) else 1.0
        log_mean = np.log(mean_out)
        offsets = np.linspace(-0.5 * (k - 1), 0.5 * (k - 1), k)
        staggered_intercepts = log_mean + offsets * 0.5
    else:
        p_init = np.linspace(1.0 / (k + 1.0), k * 1.0 / (k + 1.0), k) if k > 1 else [0.5]
        staggered_intercepts = np.log(p_init / (1.0 - np.array(p_init)))

    # Gamma (mixing covariate) block stays at 0 in the deterministic base start
    # (equal a-priori mixing weights) — identical to the base model's theta
    # initialisation. Only the perturbed starts below add noise to it.

    current_beta_idx = (k - 1) * n_mix
    for g in range(k):
        base[current_beta_idx] = staggered_intercepts[g]
        current_beta_idx += orders_list[g] + 1

    # delta (TVC) block stays at 0 — no prior on deflection direction.

    if use_dropout:
        current_gamma_idx = delta_start_idx + k * n_tvc
        for g in range(k):
            base[current_gamma_idx] = -2.0
            current_gamma_idx += 3

    if dist == 'CNORM':
        sd_guess = np.std(outcomes)
        base[-1] = np.log(sd_guess) if sd_guess > 0 else np.log(1.0)
    elif dist == 'ZIP':
        base[num_params - k:] = -1.0  # k per-group zeta params (logit ~= -1 => ~27% ZI)

    starts = [base.copy()]

    # --- perturbed starts ---
    for s in range(1, n_starts):
        np.random.seed(42 + s)
        perturbed = base.copy()

        # Gamma (mixing covariate) params: intercepts get the same noise scale
        # as the old theta perturbation; covariate slopes get smaller noise
        # (no strong prior on their direction/magnitude).
        if k > 1:
            for g in range(1, k):
                row_start = (g - 1) * n_mix
                perturbed[row_start] += np.random.normal(0, 0.5)
                if n_mix > 1:
                    perturbed[row_start + 1:row_start + n_mix] += np.random.normal(0, 0.3, n_mix - 1)

        # beta (trajectory) params
        cb_idx = (k - 1) * n_mix
        for g in range(k):
            n_betas = orders_list[g] + 1
            perturbed[cb_idx:cb_idx + n_betas] += np.random.normal(0, 0.3, n_betas)
            cb_idx += n_betas

        # delta (TVC) params
        if n_tvc > 0:
            perturbed[delta_start_idx:delta_start_idx + k * n_tvc] += np.random.normal(0, 0.3, k * n_tvc)

        # gamma (dropout) params
        if use_dropout:
            cg_idx = delta_start_idx + k * n_tvc
            for g in range(k):
                perturbed[cg_idx:cg_idx + 3] += np.random.normal(0, 0.2, 3)
                cg_idx += 3

        # log-sigma (CNORM)
        if dist == 'CNORM':
            perturbed[-1] += np.random.normal(0, 0.2)

        # zeta (ZIP per-group zero-inflation logits)
        if dist == 'ZIP':
            perturbed[-k:] += np.random.normal(0, 0.3, k)

        starts.append(perturbed)

    return starts


def _run_multistart(nll_jac_fn, starts, args, max_workers=None):
    """Run BFGS from every start in ``starts`` and return the best result.

    ``nll_jac_fn`` must return ``(nll, grad)`` in a single call (passed to
    scipy as ``jac=True``) rather than being split into separate fun/jac
    callables -- scipy calls fun(x) and jac(x) independently at the same x
    per line-search evaluation, and since the underlying kernel already
    computes both together, splitting them means the entire per-subject
    likelihood/gradient loop would otherwise run twice per evaluation point
    for no reason (see calc_nll_jac_wrapper's docstring).

    Restarts are independent (each ``minimize`` call only touches its own
    local state and the shared, read-only ``args`` tuple), so they run
    concurrently via a thread pool. Real parallelism (not just concurrency)
    requires the underlying JIT kernel to release the GIL during the call —
    ``calc_universal_subject_gradients_jit``/``calc_joint_dual_outcome_gradients_jit``
    are decorated ``nogil=True`` for exactly this reason. Falls back to
    ``os.cpu_count()`` workers (capped at ``len(starts)``) when max_workers
    is None.

    Returns:
        Tuple[OptimizeResult, float, int]: (best_result, best_nll, best_start_idx).
        best_result is the last-attempted result if no start converged
        (mirrors the previous sequential fallback behavior).
    """
    def _one(s_idx):
        return s_idx, minimize(
            nll_jac_fn, starts[s_idx], args=args,
            method='BFGS', jac=True, options={'maxiter': 3000, 'gtol': 1e-6}
        )

    n_workers = max_workers or min(len(starts), os.cpu_count() or 1)
    best_result, best_nll, best_start_idx = None, np.inf, 0
    last_result = None

    if n_workers <= 1 or len(starts) <= 1:
        for s_idx in range(len(starts)):
            _, res = _one(s_idx)
            last_result = res
            if (res.success or res.status == 2) and res.fun < best_nll:
                best_nll, best_result, best_start_idx = res.fun, res, s_idx
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for s_idx, res in pool.map(_one, range(len(starts))):
                last_result = res
                if (res.success or res.status == 2) and res.fun < best_nll:
                    best_nll, best_result, best_start_idx = res.fun, res, s_idx

    if best_result is None:
        best_result = last_result  # fallback: last attempted result

    return best_result, best_nll, best_start_idx


def run_single_model(df, orders_list, zip_iorder=0, use_dropout=False, dist='LOGIT', cnorm_min=0.0, cnorm_max=0.0, n_starts=5, baseline_cov_cols=None, tvc_cols=None, weight_col=None, log_callback=None):
    """Fit a single GBTM model with a fixed group count and polynomial order specification.

    Runs n_starts independent BFGS optimisations from different starting
    points and returns the result with the lowest NLL.  The returned dict
    contains everything needed for reporting and visualisation.

    Args:
        df:          Long-format DataFrame with columns ID, Time, Outcome.
        orders_list: List of polynomial orders, one per group.
                     e.g. [2, 2] fits a 2-group quadratic model.
        zip_iorder:  Legacy parameter (unused; kept for backward compatibility).
        use_dropout: If True, augment the model with per-group informative
                     dropout gammas (γ₀, γ₁, γ₂).
        dist:        Distribution family: 'LOGIT' | 'CNORM' | 'POISSON' | 'ZIP'.
        cnorm_min:   Lower censoring bound for CNORM (auto-set to min(y) if NaN).
        cnorm_max:   Upper censoring bound for CNORM (auto-set to max(y) if NaN).
        n_starts:    Number of multi-start random restarts (default 5).
        baseline_cov_cols: optional list of column names in df to use as
                     time-invariant baseline covariates for group membership
                     (V3.0). None/[] = intercept-only (V1.5.0-equivalent).
        tvc_cols:    optional list of column names in df to use as time-varying
                     covariates in the trajectory equation (V3.0). None/[] =
                     no TVCs (V1.5.0-equivalent).
        weight_col:  optional column name in df giving a per-subject survey/
                     sampling weight (V4.0). None = unweighted (V3.0-equivalent).
                     Robust (Huber-White) SEs are the valid inference basis
                     once weights are used; model-based SEs are reference only.
        log_callback: optional callable(str) invoked with progress messages
                     (e.g. which multi-start restart won) instead of print().

    Returns:
        dict with keys:
            'bic' / 'bic_nagin'  : Nagin BIC (higher = better).
            'bic_obs'            : BIC computed using N_obs (not N_subjects).
            'bic_standard'       : Standard BIC (lower = better).
            'aic' / 'aic_nagin'  : Nagin AIC.
            'aic_standard'       : Standard AIC.
            'll'                 : Log-likelihood.
            'orders'             : orders_list (possibly re-sorted).
            'zip_iorder'         : 0 (legacy).
            'result'             : scipy OptimizeResult with .x in original time units.
            'min_pct'            : Smallest group proportion (%) — NaN if not converged.
            'pis'                : (K,) mixing weight array.
            'use_dropout'        : bool.
            'se_model'           : (p,) model-based SEs.
            'se_robust'          : (p,) Huber-White robust SEs.
            'dof'                : Degrees of freedom = N_obs - p.
            'cond_num'           : Hessian condition number (>1e10 → near-singular).
            'dist'               : distribution string.
            'cnorm_min/max'      : censoring bounds.
            'v_model'            : (p,p) model-based covariance matrix.
            'baseline_cov_cols'  : list — mixing-covariate column names used (V3.0).
            'tvc_cols'           : list — TVC column names used (V3.0).
            'n_mix'              : int — mixing-covariate block width (P+1).
            'n_tvc'              : int — number of TVCs (Q).
            'weight_col'         : str or None — survey weight column used (V4.0).
    """
    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    n_subjects = len(subj_breaks) - 1
    n_obs = len(times)
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'POISSON': 2, 'ZIP': 3}
    dist_code = dist_map.get(dist, 0)

    baseline_cov_cols = list(baseline_cov_cols) if baseline_cov_cols else []
    tvc_cols = list(tvc_cols) if tvc_cols else []
    baseline_X = build_baseline_covariate_matrix(df, baseline_cov_cols)
    tvc_Z = extract_tvc_array(df, tvc_cols)
    weights = extract_weights_array(df, weight_col)
    n_mix = baseline_X.shape[1]
    n_tvc = tvc_Z.shape[1]

    if dist == 'CNORM':
        if cnorm_min is None or np.isnan(cnorm_min): cnorm_min = np.min(outcomes)
        if cnorm_max is None or np.isnan(cnorm_max): cnorm_max = np.max(outcomes)

    max_t = np.max(np.abs(times))
    scale_factor = max_t if max_t > 0 else 1.0
    times_scaled = times / scale_factor

    orders_arr = np.array(orders_list, dtype=np.int32)
    k = len(orders_list)

    args = (times_scaled, outcomes, dropouts, subj_breaks, orders_arr, int(zip_iorder), use_dropout, dist_code, float(cnorm_min), float(cnorm_max), baseline_X, tvc_Z, weights)
    num_betas = sum(order + 1 for order in orders_list)
    num_params = (k - 1) * n_mix + num_betas + k * n_tvc
    if use_dropout: num_params += (3 * k)
    if dist == 'CNORM': num_params += 1
    if dist == 'ZIP': num_params += k  # one zeta per group

    starts = generate_initial_params(k, orders_list, zip_iorder, use_dropout, dist, outcomes, n_starts=n_starts, n_mix=n_mix, n_tvc=n_tvc)

    best_result, best_nll, best_start_idx = _run_multistart(calc_nll_jac_wrapper, starts, args)

    msg = None
    if best_start_idx > 0:
        msg = f"  [multi-start] single model {orders_list}: best on start {best_start_idx + 1}/{n_starts} (NLL={best_nll:.4f})"
    if log_callback and msg: log_callback(msg)
    elif msg: print(msg)

    result = best_result
    is_valid, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis, cond_num, v_model = process_optimization_result(
        result, num_params, times, outcomes, dropouts, subj_breaks, orders_list, zip_iorder, use_dropout, scale_factor, dist, cnorm_min, cnorm_max,
        baseline_X, tvc_Z, weights
    )

    if is_valid:
        orders_list, se_model, se_robust, pis = sort_groups_by_intercept(
            result, orders_list, se_model, se_robust, pis, use_dropout, dist, n_mix=n_mix, n_tvc=n_tvc
        )

    min_group_size = np.min(pis) * 100 if is_valid else np.nan
    return {
        'bic': bic_nagin, 'bic_nagin': bic_nagin, 'bic_obs': bic_obs, 'bic_standard': bic_standard,
        'aic': aic_nagin, 'aic_nagin': aic_nagin, 'aic_standard': aic_standard, 'll': ll,
        'orders': orders_list, 'zip_iorder': zip_iorder, 'result': result, 'min_pct': min_group_size,
        'pis': pis, 'use_dropout': use_dropout, 'se_model': se_model, 'se_robust': se_robust,
        'dof': n_obs - num_params, 'cond_num': cond_num, 'dist': dist, 'cnorm_min': cnorm_min, 'cnorm_max': cnorm_max,
        'v_model': v_model, 'baseline_cov_cols': baseline_cov_cols, 'tvc_cols': tvc_cols,
        'n_mix': n_mix, 'n_tvc': n_tvc, 'weight_col': weight_col,
    }

def run_autotraj(df, min_groups=1, max_groups=3, min_order=0, max_order=3, min_group_pct=5.0, p_val_thresh=0.05, use_dropout=False, dist='LOGIT', cnorm_min=0.0, cnorm_max=0.0, zip_iorder=0, n_starts=3, baseline_cov_cols=None, tvc_cols=None, weight_col=None, progress_callback=None, log_callback=None):
    """Exhaustive automated search over all (k, orders) combinations.

    Evaluates every combination of group count and polynomial orders within
    the specified ranges, applying a cascade of heuristic filters to select
    well-specified models.  Models are ranked by Nagin BIC (higher = better).

    Search space
    ------------
    All K in [min_groups, max_groups] and all per-group order combinations
    in [min_order, max_order]^K are evaluated.  For k=3 and max_order=3 this
    is 4³ = 64 combinations; total may be large for wide ranges.

    Heuristic rejection filters (applied in order after each fit)
    --------------------------------------------------------------
    1. Convergence check:  result.success or result.status == 2 required.
    2. Singularity check:  Hessian condition number ≤ 1e10.
    3. SE sanity check:    All model SEs in [0.001, 50].
    4. Group size check:   All groups ≥ min_group_pct % of sample.
    5. Significance check: The highest-order polynomial coefficient of every
       group must have |T| / SE > critical value at p_val_thresh (two-tailed
       t-test against zero).  This follows the Nagin & Jones (2005) guideline
       that superfluous polynomial terms should be dropped.

    Models passing all five filters are added to valid_models and sorted by
    Nagin BIC descending.  All evaluated models (including rejected ones) are
    returned in all_evaluated_models for diagnostic inspection.

    Args:
        df:            Long-format DataFrame with columns ID, Time, Outcome.
        min_groups:    Minimum K to evaluate (default 1).
        max_groups:    Maximum K to evaluate (default 3).
        min_order:     Minimum polynomial order per group (default 0).
        max_order:     Maximum polynomial order per group (default 3).
        min_group_pct: Minimum group size as % of N (default 5.0).
        p_val_thresh:  Maximum p-value for highest-order coefficient (default 0.05).
        use_dropout:   If True, fit the informative-dropout augmentation.
        dist:          Distribution family: 'LOGIT'|'CNORM'|'POISSON'|'ZIP'.
        cnorm_min:     CNORM lower censoring bound.
        cnorm_max:     CNORM upper censoring bound.
        zip_iorder:    Legacy parameter (unused).
        n_starts:      Multi-start restarts per model (default 3).
        baseline_cov_cols: optional list of column names in df to use as
                       time-invariant baseline covariates for group membership
                       (V3.0). None/[] = intercept-only (V1.5.0-equivalent).
        tvc_cols:      optional list of column names in df to use as
                       time-varying covariates in the trajectory equation
                       (V3.0). None/[] = no TVCs (V1.5.0-equivalent).
        weight_col:    optional column name in df giving a per-subject survey/
                       sampling weight (V4.0). None = unweighted
                       (V3.0-equivalent).
        progress_callback: optional callable(current, total, orders_list)
                       invoked after each (k, orders) combination is
                       evaluated, for driving a UI progress bar.
        log_callback:  optional callable(str) invoked with progress messages
                       instead of print().

    Returns:
        Tuple[List[dict], List[dict]]:
            valid_models:        List of model dicts passing all filters,
                                 sorted by bic_nagin descending.
            all_evaluated_models: List of summary dicts for every evaluated
                                 model including rejection reason in 'Status'.
    """
    valid_models = []
    all_evaluated_models = []
    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    n_subjects = len(subj_breaks) - 1
    n_obs = len(times)
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'POISSON': 2, 'ZIP': 3}
    dist_code = dist_map.get(dist, 0)

    baseline_cov_cols = list(baseline_cov_cols) if baseline_cov_cols else []
    tvc_cols = list(tvc_cols) if tvc_cols else []
    baseline_X = build_baseline_covariate_matrix(df, baseline_cov_cols)
    tvc_Z = extract_tvc_array(df, tvc_cols)
    weights = extract_weights_array(df, weight_col)
    n_mix = baseline_X.shape[1]
    n_tvc = tvc_Z.shape[1]

    if dist == 'CNORM':
        if cnorm_min is None or np.isnan(cnorm_min): cnorm_min = np.min(outcomes)
        if cnorm_max is None or np.isnan(cnorm_max): cnorm_max = np.max(outcomes)

    max_t = np.max(np.abs(times))
    scale_factor = max_t if max_t > 0 else 1.0
    times_scaled = times / scale_factor

    all_combinations = []
    for k in range(min_groups, max_groups + 1):
        order_combinations = list(itertools.product(range(min_order, max_order + 1), repeat=k))
        all_combinations.extend([list(orders) for orders in order_combinations])

    for i, orders_list in enumerate(all_combinations):
        orders_arr = np.array(orders_list, dtype=np.int32)
        k = len(orders_list)

        num_betas = sum(order + 1 for order in orders_list)
        num_params = (k - 1) * n_mix + num_betas + k * n_tvc
        if use_dropout: num_params += (3 * k)
        if dist == 'CNORM': num_params += 1
        if dist == 'ZIP': num_params += k  # one zeta per group

        args = (times_scaled, outcomes, dropouts, subj_breaks, orders_arr, int(zip_iorder), use_dropout, dist_code, float(cnorm_min), float(cnorm_max), baseline_X, tvc_Z, weights)
        starts = generate_initial_params(k, orders_list, zip_iorder, use_dropout, dist, outcomes, n_starts=n_starts, n_mix=n_mix, n_tvc=n_tvc)

        best_result, best_nll, best_start_idx = _run_multistart(calc_nll_jac_wrapper, starts, args)

        if best_start_idx > 0:
            msg = f"  [multi-start] autotraj {orders_list}: best on start {best_start_idx + 1}/{n_starts} (NLL={best_nll:.4f})"
            if log_callback: log_callback(msg)
            else: print(msg)

        if progress_callback: progress_callback(i + 1, len(all_combinations), orders_list)

        result = best_result
        is_converged, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis, cond_num, v_model = process_optimization_result(
            result, num_params, times, outcomes, dropouts, subj_breaks, orders_list, zip_iorder, use_dropout, scale_factor, dist, cnorm_min, cnorm_max,
            baseline_X, tvc_Z, weights
        )

        if is_converged:
            orders_list, se_model, se_robust, pis = sort_groups_by_intercept(
                result, orders_list, se_model, se_robust, pis, use_dropout, dist, n_mix=n_mix, n_tvc=n_tvc
            )
            min_group_size = np.min(pis) * 100
            status = ""
            is_valid = True
            dof = n_obs - num_params

            if cond_num > 1e10:
                status = "Rejected (Singular Matrix / Unidentifiable)"
                is_valid = False
            elif np.any(se_model < 1e-3) or np.any(se_model > 50):
                status = "Rejected (Degenerate SE / Flat Likelihood)"
                is_valid = False
            elif min_group_size < min_group_pct:
                status = f"Rejected (Group Size < {min_group_pct}%)"
                is_valid = False
            else:
                all_significant = True
                current_beta_idx = (k - 1) * n_mix
                for g in range(k):
                    n_betas = orders_list[g] + 1
                    highest_est = result.x[current_beta_idx + n_betas - 1]
                    highest_se = se_model[current_beta_idx + n_betas - 1]
                    
                    t_stat = highest_est / highest_se if highest_se > 0 else 0
                    p_value_t = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
                    
                    if p_value_t >= p_val_thresh: all_significant = False
                    current_beta_idx += n_betas
                        
                if not all_significant:
                    status = f"Rejected (P-Value > {p_val_thresh})"
                    is_valid = False
                else:
                    status = "Valid"
            
            all_evaluated_models.append({
                'Groups': k, 'Orders': str(orders_list), 'Status': status,
                'BIC (Nagin)': bic_nagin, 'BIC (Standard)': bic_standard,
                'AIC (Nagin)': aic_nagin, 'AIC (Standard)': aic_standard,
                'LL': ll, 'Min_Group_%': min_group_size
            })

            if is_valid:
                valid_models.append({
                    'bic': bic_nagin, 'bic_nagin': bic_nagin, 'bic_obs': bic_obs, 'bic_standard': bic_standard,
                    'aic': aic_nagin, 'aic_nagin': aic_nagin, 'aic_standard': aic_standard, 'll': ll,
                    'orders': orders_list, 'zip_iorder': zip_iorder, 'result': result, 'min_pct': min_group_size,
                    'pis': pis, 'use_dropout': use_dropout, 'se_model': se_model, 'se_robust': se_robust, 'dof': dof, 'cond_num': cond_num, 'dist': dist, 'cnorm_min': cnorm_min, 'cnorm_max': cnorm_max,
                    'v_model': v_model, 'baseline_cov_cols': baseline_cov_cols, 'tvc_cols': tvc_cols,
                    'n_mix': n_mix, 'n_tvc': n_tvc, 'weight_col': weight_col,
                })
        else:
            all_evaluated_models.append({
                'Groups': k, 'Orders': str(orders_list), 'Status': "Failed Convergence",
                'BIC (Nagin)': np.nan, 'BIC (Standard)': np.nan,
                'AIC (Nagin)': np.nan, 'AIC (Standard)': np.nan,
                'LL': np.nan, 'Min_Group_%': np.nan
            })

    valid_models = sorted(valid_models, key=lambda x: x['bic_nagin'], reverse=True)
    all_evaluated_models = sorted(all_evaluated_models, key=lambda x: x['BIC (Nagin)'] if pd.notnull(x['BIC (Nagin)']) else -np.inf, reverse=True)
    return valid_models, all_evaluated_models


# --- JOINT DUAL-TRAJECTORY FITTING PIPELINE (V5.0) ---

def generate_joint_initial_params(k_y, k_z, orders_y, orders_z, use_dropout_y, dist_y, outcomes_y,
                                   use_dropout_z, dist_z, outcomes_z, n_starts=10):
    """Generate n_starts starting points for the V5.0 joint dual-trajectory model.

    Mirrors generate_initial_params's initialisation strategy independently
    for each outcome's own block (staggered intercepts; dropout intercepts at
    -2; CNORM log-sigma from std(outcomes); ZIP zeta at -1), plus an
    all-zero (uniform joint mixture) Theta_joint block for the deterministic
    base start — analogous to the base model's theta staying at 0.
    """
    n_theta, y_beta_start, z_beta_start, num_betas_y, num_betas_z, num_params = _joint_layout(
        k_y, k_z, orders_y, orders_z, use_dropout_y, dist_y, use_dropout_z, dist_z
    )
    y_block_width = z_beta_start - y_beta_start
    z_block_width = num_params - z_beta_start

    def _staggered_intercepts(k, dist, outcomes):
        if dist == 'POISSON':
            mean_out = np.mean(outcomes[outcomes > 0]) if np.any(outcomes > 0) else 1.0
            log_mean = np.log(mean_out)
            offsets = np.linspace(-0.5 * (k - 1), 0.5 * (k - 1), k)
            return log_mean + offsets * 0.5
        elif dist == 'CNORM':
            qs = np.linspace(1.0 / (k + 1.0), k * 1.0 / (k + 1.0), k) if k > 1 else [0.5]
            return np.quantile(outcomes, qs)
        else:
            p_init = np.linspace(1.0 / (k + 1.0), k * 1.0 / (k + 1.0), k) if k > 1 else [0.5]
            return np.log(p_init / (1.0 - np.array(p_init)))

    base = np.zeros(num_params)

    y_intercepts = _staggered_intercepts(k_y, dist_y, outcomes_y)
    idx = y_beta_start
    for g in range(k_y):
        base[idx] = y_intercepts[g]
        idx += orders_y[g] + 1
    if use_dropout_y:
        for g in range(k_y):
            base[idx] = -2.0
            idx += 3
    if dist_y == 'CNORM':
        sd_guess = np.std(outcomes_y)
        base[z_beta_start - 1] = np.log(sd_guess) if sd_guess > 0 else 0.0
    elif dist_y == 'ZIP':
        base[z_beta_start - k_y:z_beta_start] = -1.0

    z_intercepts = _staggered_intercepts(k_z, dist_z, outcomes_z)
    idx = z_beta_start
    for g in range(k_z):
        base[idx] = z_intercepts[g]
        idx += orders_z[g] + 1
    if use_dropout_z:
        for g in range(k_z):
            base[idx] = -2.0
            idx += 3
    if dist_z == 'CNORM':
        sd_guess = np.std(outcomes_z)
        base[num_params - 1] = np.log(sd_guess) if sd_guess > 0 else 0.0
    elif dist_z == 'ZIP':
        base[num_params - k_z:num_params] = -1.0

    starts = [base.copy()]

    for s in range(1, n_starts):
        np.random.seed(42 + s)
        perturbed = base.copy()

        if n_theta > 0:
            perturbed[:n_theta] += np.random.normal(0, 0.5, n_theta)

        idx = y_beta_start
        for g in range(k_y):
            n_betas = orders_y[g] + 1
            perturbed[idx:idx + n_betas] += np.random.normal(0, 0.3, n_betas)
            idx += n_betas
        if use_dropout_y:
            for g in range(k_y):
                perturbed[idx:idx + 3] += np.random.normal(0, 0.2, 3)
                idx += 3
        if dist_y == 'CNORM':
            perturbed[z_beta_start - 1] += np.random.normal(0, 0.2)
        elif dist_y == 'ZIP':
            perturbed[z_beta_start - k_y:z_beta_start] += np.random.normal(0, 0.3, k_y)

        idx = z_beta_start
        for g in range(k_z):
            n_betas = orders_z[g] + 1
            perturbed[idx:idx + n_betas] += np.random.normal(0, 0.3, n_betas)
            idx += n_betas
        if use_dropout_z:
            for g in range(k_z):
                perturbed[idx:idx + 3] += np.random.normal(0, 0.2, 3)
                idx += 3
        if dist_z == 'CNORM':
            perturbed[num_params - 1] += np.random.normal(0, 0.2)
        elif dist_z == 'ZIP':
            perturbed[num_params - k_z:num_params] += np.random.normal(0, 0.3, k_z)

        starts.append(perturbed)

    return starts


def run_joint_dual_trajectory_model(df_y, df_z, orders_y, orders_z, dist_y='LOGIT', dist_z='LOGIT',
                                     use_dropout_y=False, use_dropout_z=False,
                                     cnorm_min_y=0.0, cnorm_max_y=0.0, cnorm_min_z=0.0, cnorm_max_z=0.0,
                                     n_starts=5, log_callback=None):
    """Fit a V5.0 Nagin-style joint dual-trajectory model.

    Two outcomes Y and Z, each with its own independent GBTM structure (own
    group count/orders/distribution/dropout), linked by a joint latent-class
    probability matrix pi_gh instead of assuming independence. See MATH.md §9.

    Single Model Mode only — fixed (K_Y, orders_y) and (K_Z, orders_z), no
    AutoTraj-style combinatorial search over both outcomes' grids
    simultaneously (explicit V5.0 scope boundary; a future extension).
    Does not compose with V3.0 mixing-covariates/TVC or V4.0 survey weights
    in this pass (also explicit scope boundaries).

    Args:
        df_y, df_z:  Long-format DataFrames (ID, Time, Outcome) for outcomes Y
                     and Z — must share the identical subject-ID set
                     (extract_joint_flat_arrays raises a clear error otherwise).
        orders_y, orders_z: list of per-group polynomial orders for Y/Z.
        dist_y, dist_z: distribution family per outcome, chosen independently.
        use_dropout_y, use_dropout_z: independent MNAR dropout toggles.
        cnorm_min_y/max_y, cnorm_min_z/max_z: independent CNORM bounds
                     (auto-set to observed min/max if NaN, same as
                     run_single_model).
        n_starts:    multi-start random restarts (default 5).
        log_callback: optional callable(str) invoked with progress messages
                     instead of print().

    Returns:
        dict with keys:
            'll', 'bic'/'bic_nagin', 'bic_standard', 'aic'/'aic_nagin', 'aic_standard',
            'orders_y', 'orders_z', 'k_y', 'k_z', 'result' (scipy OptimizeResult,
            .x in original time units), 'pis_joint' (K_Y x K_Z ndarray, None if
            not converged), 'use_dropout_y', 'use_dropout_z', 'dist_y', 'dist_z',
            'cnorm_min_y'/'cnorm_max_y', 'cnorm_min_z'/'cnorm_max_z',
            'se_model', 'se_robust', 'cond_num', 'dof', 'v_model'.
    """
    (times_y, outcomes_y, dropouts_y, subj_breaks_y), \
        (times_z, outcomes_z, dropouts_z, subj_breaks_z), \
        canonical_ids = extract_joint_flat_arrays(df_y, df_z)

    n_obs = len(times_y) + len(times_z)

    dist_map = {'LOGIT': 0, 'CNORM': 1, 'POISSON': 2, 'ZIP': 3}
    dist_code_y = dist_map.get(dist_y, 0)
    dist_code_z = dist_map.get(dist_z, 0)

    if dist_y == 'CNORM':
        if cnorm_min_y is None or np.isnan(cnorm_min_y): cnorm_min_y = np.min(outcomes_y)
        if cnorm_max_y is None or np.isnan(cnorm_max_y): cnorm_max_y = np.max(outcomes_y)
    if dist_z == 'CNORM':
        if cnorm_min_z is None or np.isnan(cnorm_min_z): cnorm_min_z = np.min(outcomes_z)
        if cnorm_max_z is None or np.isnan(cnorm_max_z): cnorm_max_z = np.max(outcomes_z)

    max_t_y = np.max(np.abs(times_y)); scale_factor_y = max_t_y if max_t_y > 0 else 1.0
    max_t_z = np.max(np.abs(times_z)); scale_factor_z = max_t_z if max_t_z > 0 else 1.0
    times_y_scaled = times_y / scale_factor_y
    times_z_scaled = times_z / scale_factor_z

    k_y = len(orders_y)
    k_z = len(orders_z)
    orders_y_arr = np.array(orders_y, dtype=np.int32)
    orders_z_arr = np.array(orders_z, dtype=np.int32)

    _, _, _, _, _, num_params = _joint_layout(
        k_y, k_z, orders_y, orders_z, use_dropout_y, dist_y, use_dropout_z, dist_z
    )

    args = (times_y_scaled, outcomes_y, dropouts_y, subj_breaks_y, orders_y_arr, use_dropout_y, dist_code_y, float(cnorm_min_y), float(cnorm_max_y),
            times_z_scaled, outcomes_z, dropouts_z, subj_breaks_z, orders_z_arr, use_dropout_z, dist_code_z, float(cnorm_min_z), float(cnorm_max_z))

    starts = generate_joint_initial_params(k_y, k_z, orders_y, orders_z, use_dropout_y, dist_y, outcomes_y,
                                            use_dropout_z, dist_z, outcomes_z, n_starts=n_starts)

    best_result, best_nll, best_start_idx = _run_multistart(calc_joint_nll_jac_wrapper, starts, args)

    if best_start_idx > 0:
        msg = f"  [multi-start] joint model Y{orders_y}/Z{orders_z}: best on start {best_start_idx + 1}/{n_starts} (NLL={best_nll:.4f})"
        if log_callback: log_callback(msg)
        else: print(msg)

    result = best_result
    is_valid, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis_joint, cond_num, v_model = process_joint_optimization_result(
        result, num_params, k_y, k_z, orders_y, orders_z,
        times_y, outcomes_y, dropouts_y, subj_breaks_y, use_dropout_y, dist_y, cnorm_min_y, cnorm_max_y, scale_factor_y,
        times_z, outcomes_z, dropouts_z, subj_breaks_z, use_dropout_z, dist_z, cnorm_min_z, cnorm_max_z, scale_factor_z,
    )

    if is_valid:
        orders_y, orders_z, se_model, se_robust, pis_joint = sort_joint_groups_by_intercept(
            result, k_y, k_z, orders_y, orders_z, se_model, se_robust, use_dropout_y, dist_y, use_dropout_z, dist_z
        )

    return {
        'bic': bic_nagin, 'bic_nagin': bic_nagin, 'bic_standard': bic_standard,
        'aic': aic_nagin, 'aic_nagin': aic_nagin, 'aic_standard': aic_standard, 'll': ll,
        'orders_y': orders_y, 'orders_z': orders_z, 'k_y': k_y, 'k_z': k_z,
        'result': result, 'pis_joint': pis_joint,
        'use_dropout_y': use_dropout_y, 'use_dropout_z': use_dropout_z,
        'dist_y': dist_y, 'dist_z': dist_z,
        'cnorm_min_y': cnorm_min_y, 'cnorm_max_y': cnorm_max_y,
        'cnorm_min_z': cnorm_min_z, 'cnorm_max_z': cnorm_max_z,
        'se_model': se_model, 'se_robust': se_robust, 'cond_num': cond_num,
        'dof': n_obs - num_params, 'v_model': v_model,
    }


def run_joint_autotraj(df_y, df_z,
                        min_groups_y=1, max_groups_y=3, min_order_y=0, max_order_y=3,
                        min_groups_z=1, max_groups_z=3, min_order_z=0, max_order_z=3,
                        min_group_pct=5.0, p_val_thresh=0.05,
                        dist_y='LOGIT', dist_z='LOGIT',
                        use_dropout_y=False, use_dropout_z=False,
                        cnorm_min_y=0.0, cnorm_max_y=0.0, cnorm_min_z=0.0, cnorm_max_z=0.0,
                        n_starts=5, progress_callback=None, log_callback=None):
    """Exhaustive automated search over BOTH outcomes' (K, orders) grids for
    the joint dual-trajectory model — the joint analogue of run_autotraj.

    Builds every (orders_y, orders_z) combination (Cartesian product of each
    outcome's own combinatorial grid, generated the same way run_autotraj
    builds its single-outcome grid), fits each via the existing
    run_joint_dual_trajectory_model (reused as the per-combo worker rather
    than re-inlining the fit/extract/scale/multistart/post-process logic a
    second time), and applies a rejection cascade mirroring run_autotraj's
    (main.py ~2440-2471) but doubled per-outcome where relevant:
        1. Convergence      : pis_joint is not None.
        2. Singularity      : cond_num > 1e10 -> reject (not previously
           checked anywhere in the joint path -- new here, matching the
           single-outcome threshold).
        3. SE sanity        : any(se_model < 1e-3) or any(se_model > 50).
        4. Group size       : BOTH outcomes' marginal min group % must clear
           min_group_pct (pis_joint.sum(axis=1)/.sum(axis=0) give the Y/Z
           marginals directly -- no extra fitting needed).
        5. Significance     : BOTH outcomes' highest-order beta per group
           must be significant at p_val_thresh (t-test, dof from the
           model's own 'dof' key) -- walks each outcome's beta blocks via
           _joint_layout's y_beta_start/z_beta_start offsets.

    The outer loop over combos is sequential (not thread-pooled) -- each
    inner run_joint_dual_trajectory_model call already parallelises its own
    n_starts restarts via _run_multistart's thread pool; parallelising
    across combos too would oversubscribe CPU cores.

    Args:
        df_y, df_z: Long-format DataFrames for outcomes Y and Z (same
            requirements as run_joint_dual_trajectory_model).
        min_groups_y/max_groups_y/min_order_y/max_order_y: Y's search grid.
        min_groups_z/max_groups_z/min_order_z/max_order_z: Z's search grid.
        min_group_pct: minimum acceptable marginal group % for EITHER outcome.
        p_val_thresh: significance threshold for the highest-order term.
        dist_y, dist_z, use_dropout_y, use_dropout_z, cnorm_min/max_y/z,
            n_starts: passed straight through to every combo's fit.
        progress_callback: optional callable(current, total, (orders_y, orders_z)).
        log_callback: optional callable(str), forwarded to every inner fit.

    Returns:
        Tuple[List[dict], List[dict]]:
            valid_models: full model dicts (same shape as
                run_joint_dual_trajectory_model's return), one per combo
                that passed every filter, sorted by 'bic' descending.
            all_evaluated_models: lightweight dicts for every combo tried --
                keys 'Groups_Y', 'Orders_Y', 'Groups_Z', 'Orders_Z', 'Status',
                'BIC (Nagin)', 'BIC (Standard)', 'AIC (Nagin)', 'LL',
                'Min_Group_%_Y', 'Min_Group_%_Z' -- sorted the same way,
                NaN-safe.
    """
    def _build_combos(min_groups, max_groups, min_order, max_order):
        combos = []
        for k in range(min_groups, max_groups + 1):
            for orders in itertools.product(range(min_order, max_order + 1), repeat=k):
                combos.append(list(orders))
        return combos

    y_combos = _build_combos(min_groups_y, max_groups_y, min_order_y, max_order_y)
    z_combos = _build_combos(min_groups_z, max_groups_z, min_order_z, max_order_z)
    all_combinations = list(itertools.product(y_combos, z_combos))

    def _outcome_significant(result_x, se_model, beta_start, orders_list, k, dof):
        idx = beta_start
        for g in range(k):
            n_betas = orders_list[g] + 1
            est, se = result_x[idx + n_betas - 1], se_model[idx + n_betas - 1]
            t_stat = est / se if se > 0 else 0
            p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
            if p_val >= p_val_thresh:
                return False
            idx += n_betas
        return True

    valid_models = []
    all_evaluated_models = []

    for i, (orders_y, orders_z) in enumerate(all_combinations):
        model = run_joint_dual_trajectory_model(
            df_y, df_z, orders_y=orders_y, orders_z=orders_z,
            dist_y=dist_y, dist_z=dist_z,
            use_dropout_y=use_dropout_y, use_dropout_z=use_dropout_z,
            cnorm_min_y=cnorm_min_y, cnorm_max_y=cnorm_max_y,
            cnorm_min_z=cnorm_min_z, cnorm_max_z=cnorm_max_z,
            n_starts=n_starts, log_callback=log_callback,
        )

        k_y, k_z = model['k_y'], model['k_z']
        status = None
        min_pct_y = min_pct_z = np.nan

        if model['pis_joint'] is None:
            status = "Failed Convergence"
        else:
            pis_joint = model['pis_joint']
            min_pct_y = float(pis_joint.sum(axis=1).min() * 100)
            min_pct_z = float(pis_joint.sum(axis=0).min() * 100)

            if model['cond_num'] > 1e10:
                status = "Rejected (Singular Matrix / Unidentifiable)"
            elif np.any(model['se_model'] < 1e-3) or np.any(model['se_model'] > 50):
                status = "Rejected (Degenerate SE / Flat Likelihood)"
            elif min_pct_y < min_group_pct:
                status = f"Rejected (Group Size < {min_group_pct}% — Y)"
            elif min_pct_z < min_group_pct:
                status = f"Rejected (Group Size < {min_group_pct}% — Z)"
            else:
                n_theta, y_beta_start, z_beta_start, _, _, _ = _joint_layout(
                    k_y, k_z, model['orders_y'], model['orders_z'],
                    use_dropout_y, dist_y, use_dropout_z, dist_z,
                )
                y_ok = _outcome_significant(model['result'].x, model['se_model'], y_beta_start, model['orders_y'], k_y, model['dof'])
                z_ok = _outcome_significant(model['result'].x, model['se_model'], z_beta_start, model['orders_z'], k_z, model['dof'])
                if not (y_ok and z_ok):
                    status = f"Rejected (P-Value > {p_val_thresh})"
                else:
                    status = "Valid"

        all_evaluated_models.append({
            'Groups_Y': k_y, 'Orders_Y': str(orders_y), 'Groups_Z': k_z, 'Orders_Z': str(orders_z),
            'Status': status,
            'BIC (Nagin)': model.get('bic', np.nan), 'BIC (Standard)': model.get('bic_standard', np.nan),
            'AIC (Nagin)': model.get('aic', np.nan), 'LL': model.get('ll', np.nan),
            'Min_Group_%_Y': min_pct_y, 'Min_Group_%_Z': min_pct_z,
        })

        if status == "Valid":
            valid_models.append(model)

        if progress_callback:
            progress_callback(i + 1, len(all_combinations), (orders_y, orders_z))

    valid_models.sort(key=lambda m: m['bic'], reverse=True)
    all_evaluated_models.sort(
        key=lambda m: m['BIC (Nagin)'] if pd.notnull(m['BIC (Nagin)']) else -np.inf, reverse=True
    )

    return valid_models, all_evaluated_models


def get_subject_assignments(model_dict, df):
    """Compute posterior group probabilities and hard assignments for every subject.

    For each subject, evaluates the group-conditional log-likelihood
    L_{ig} = Σ_t log P(y_{it} | g, t) under the fitted model, then computes
    the normalised posterior P(g | i) ∝ π_g(x_i) · exp(L_{ig}).  The hard
    assignment is argmax_g P(g | i).

    Args:
        model_dict: Model dict returned by run_single_model or run_autotraj
                    (must have keys 'orders', 'result', 'use_dropout', 'pis',
                    'dist', 'cnorm_min', 'cnorm_max'; optionally 'baseline_cov_cols',
                    'tvc_cols', 'n_mix', 'n_tvc' for V3.0 — older model dicts
                    without these keys default to the no-covariate case).
        df:         Long-format DataFrame with columns ID, Time, Outcome (plus
                    any baseline-covariate/TVC columns named in model_dict).

    Returns:
        pd.DataFrame: One row per subject with columns:
            'ID'                    : Subject identifier.
            'Assigned_Group'        : Hard assignment (1-based group number).
            'Group_1_Prob', …, 'Group_K_Prob': Posterior probability for each group.
    """
    orders = model_dict['orders']
    zip_iorder = model_dict.get('zip_iorder', 0)
    use_dropout = model_dict['use_dropout']
    params = model_dict['result'].x
    dist = model_dict.get('dist', 'LOGIT')
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'POISSON': 2, 'ZIP': 3}
    dist_code = dist_map.get(dist, 0)
    min_val = float(model_dict.get('cnorm_min', 0.0))
    max_val = float(model_dict.get('cnorm_max', 0.0))
    baseline_cov_cols = model_dict.get('baseline_cov_cols') or []
    tvc_cols = model_dict.get('tvc_cols') or []

    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    ids = df['ID'].values
    subject_ids_unique = ids[subj_breaks[:-1]]

    baseline_X = build_baseline_covariate_matrix(df, baseline_cov_cols)
    tvc_Z = extract_tvc_array(df, tvc_cols)
    n_mix = model_dict.get('n_mix', baseline_X.shape[1])
    n_tvc = model_dict.get('n_tvc', tvc_Z.shape[1])

    k = len(orders)

    num_betas = 0
    for g in range(k): num_betas += orders[g] + 1
    delta_start_idx = (k - 1) * n_mix + num_betas
    gamma_start_idx = delta_start_idx + k * n_tvc

    if dist == 'CNORM':
        raw_sigma = params[-1]
        sigma = np.exp(raw_sigma) if raw_sigma < 20 else np.exp(20)
    elif dist == 'ZIP':
        zeta_start = len(params) - k

    assignments = []
    n_subjects = len(subj_breaks) - 1

    for i in range(n_subjects):
        start, end = subj_breaks[i], subj_breaks[i+1]
        n_obs = end - start

        # Per-subject mixing probabilities (V3.0): theta_g(x_i) = Gamma_g . x_i
        # for g > 0; theta_0(x_i) ≡ 0. Reduces to the fixed base-model pis when
        # n_mix == 1 (baseline_X[i, 0] == 1).
        thetas = np.zeros(k)
        for g in range(1, k):
            thetas[g] = np.dot(params[(g - 1) * n_mix : g * n_mix], baseline_X[i, :])
        pis = np.exp(thetas - logsumexp(thetas))
        pis_safe = np.clip(pis, 1e-15, 1.0)

        L_ig_log = np.zeros(k)
        current_beta_idx = (k - 1) * n_mix
        current_gamma_idx = gamma_start_idx

        for g in range(k):
            n_betas = orders[g] + 1
            group_betas = params[current_beta_idx : current_beta_idx + n_betas]
            current_beta_idx += n_betas
            group_delta = params[delta_start_idx + g * n_tvc : delta_start_idx + (g + 1) * n_tvc]

            if use_dropout:
                gamma_0 = params[current_gamma_idx]
                gamma_1 = params[current_gamma_idx + 1]
                gamma_2 = params[current_gamma_idx + 2]
                current_gamma_idx += 3

            zeta_g_zip = params[zeta_start + g] if dist_code == 3 else 0.0

            ll_g = 0.0

            for obs in range(n_obs):
                idx = start + obs
                t_val = times[idx]
                y_val = outcomes[idx]

                mu = sum(group_betas[p] * (t_val ** p) for p in range(orders[g] + 1))
                mu += sum(group_delta[q] * tvc_Z[idx, q] for q in range(n_tvc))

                if dist_code == 2: # POISSON
                    eta = mu
                    if eta > 20.0: eta = 20.0
                    if eta < -20.0: eta = -20.0
                    exp_eta = np.exp(eta)
                    ll_g += y_val * eta - exp_eta - math.lgamma(y_val + 1.0)
                elif dist_code == 1: # CNORM
                    if y_val <= min_val:
                        z = (min_val - mu) / sigma
                        ll_g += fast_norm_logcdf(z)
                    elif y_val >= max_val:
                        z = (max_val - mu) / sigma
                        ll_g += fast_norm_logsf(z)
                    else:
                        ll_g += fast_norm_logpdf(y_val, mu, sigma)
                elif dist_code == 3: # ZIP (per-group zeta)
                    ll_val, _, _ = fast_zip_logpmf_grad(y_val, mu, zeta_g_zip)
                    ll_g += ll_val
                else: # LOGIT
                    z = mu
                    if z > 25.0: z = 25.0
                    if z < -25.0: z = -25.0
                    prob = 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))
                    prob = max(1e-12, min(1.0 - 1e-12, prob))
                    ll_g += y_val * np.log(prob) + (1.0 - y_val) * np.log(1.0 - prob)
                
                if use_dropout and obs > 0:
                    y_prev = outcomes[idx - 1]
                    z_drop = gamma_0 + (gamma_1 * t_val) + (gamma_2 * y_prev)
                    if z_drop > 25.0: z_drop = 25.0
                    if z_drop < -25.0: z_drop = -25.0
                    p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                    p_drop = max(1e-12, min(1.0 - 1e-12, p_drop))
                    ll_g += np.log(1.0 - p_drop)
                    
            if use_dropout:
                last_idx = end - 1
                if dropouts[last_idx] == 1.0:
                    t_last = times[last_idx]
                    y_last = outcomes[last_idx]
                    z_drop = gamma_0 + (gamma_1 * t_last) + (gamma_2 * y_last)
                    if z_drop > 25.0: z_drop = 25.0
                    if z_drop < -25.0: z_drop = -25.0
                    p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                    p_drop = max(1e-12, min(1.0 - 1e-12, p_drop))
                    ll_g += np.log(p_drop)
                
            L_ig_log[g] = ll_g
            
        numerator_log = np.log(pis_safe) + L_ig_log
        post_max = np.max(numerator_log)
        post_sum_exp = np.sum(np.exp(numerator_log - post_max))
        posterior_ig = np.exp(numerator_log - (post_max + np.log(post_sum_exp)))
        
        row = {'ID': subject_ids_unique[i], 'Assigned_Group': np.argmax(posterior_ig) + 1}
        for g in range(k): row[f'Group_{g+1}_Prob'] = posterior_ig[g]
        assignments.append(row)

    return pd.DataFrame(assignments)


def get_joint_subject_assignments(model_dict, df_y, df_z):
    """Compute joint and marginal posterior group probabilities and hard
    assignments for every subject, for a fitted V5.0 joint dual-trajectory
    model (mirrors get_subject_assignments, generalized to the (K_Y,K_Z)
    joint posterior grid).

    Reuses calc_single_outcome_group_ll_jit directly (the same JIT subroutine
    the joint kernel itself calls) rather than a third hand-duplicated copy
    of the forward-pass math.

    Args:
        model_dict: Model dict returned by run_joint_dual_trajectory_model
                    (keys: 'orders_y', 'orders_z', 'result', 'use_dropout_y',
                    'use_dropout_z', 'dist_y', 'dist_z', 'cnorm_min_y/max_y',
                    'cnorm_min_z/max_z').
        df_y, df_z: Long-format DataFrames for outcomes Y and Z — must share
                    the identical subject-ID set (extract_joint_flat_arrays).

    Returns:
        pd.DataFrame: One row per subject with columns:
            'ID', 'Assigned_Group_Y', 'Assigned_Group_Z' (1-based hard
            assignments from the MARGINAL posteriors), 'Joint_G{g}_H{h}_Prob'
            for every joint cell, 'Y_Group_{g}_Prob' / 'Z_Group_{h}_Prob'
            marginal posteriors.
    """
    orders_y = model_dict['orders_y']
    orders_z = model_dict['orders_z']
    use_dropout_y = model_dict['use_dropout_y']
    use_dropout_z = model_dict['use_dropout_z']
    dist_y = model_dict.get('dist_y', 'LOGIT')
    dist_z = model_dict.get('dist_z', 'LOGIT')
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'POISSON': 2, 'ZIP': 3}
    dist_code_y = dist_map.get(dist_y, 0)
    dist_code_z = dist_map.get(dist_z, 0)
    cnorm_min_y = float(model_dict.get('cnorm_min_y', 0.0))
    cnorm_max_y = float(model_dict.get('cnorm_max_y', 0.0))
    cnorm_min_z = float(model_dict.get('cnorm_min_z', 0.0))
    cnorm_max_z = float(model_dict.get('cnorm_max_z', 0.0))
    params = model_dict['result'].x

    k_y = len(orders_y)
    k_z = len(orders_z)
    orders_y_arr = np.array(orders_y, dtype=np.int32)
    orders_z_arr = np.array(orders_z, dtype=np.int32)

    (times_y, outcomes_y, dropouts_y, subj_breaks_y), \
        (times_z, outcomes_z, dropouts_z, subj_breaks_z), \
        canonical_ids = extract_joint_flat_arrays(df_y, df_z)

    _, y_beta_start, z_beta_start, _, _, _ = _joint_layout(
        k_y, k_z, orders_y, orders_z, use_dropout_y, dist_y, use_dropout_z, dist_z
    )
    params_y = params[y_beta_start:z_beta_start]
    params_z = params[z_beta_start:]

    tvc_y = np.zeros((len(times_y), 0))
    tvc_z = np.zeros((len(times_z), 0))

    n_subjects = len(subj_breaks_y) - 1
    rows = []

    for i in range(n_subjects):
        start_y, end_y = subj_breaks_y[i], subj_breaks_y[i + 1]
        start_z, end_z = subj_breaks_z[i], subj_breaks_z[i + 1]

        L_y, _, _ = calc_single_outcome_group_ll_jit(
            params_y, times_y, outcomes_y, dropouts_y, start_y, end_y, orders_y_arr,
            tvc_y, 0, use_dropout_y, dist_code_y, cnorm_min_y, cnorm_max_y
        )
        L_z, _, _ = calc_single_outcome_group_ll_jit(
            params_z, times_z, outcomes_z, dropouts_z, start_z, end_z, orders_z_arr,
            tvc_z, 0, use_dropout_z, dist_code_z, cnorm_min_z, cnorm_max_z
        )

        theta_grid = np.zeros((k_y, k_z))
        idx = 0
        for g in range(k_y):
            for h in range(k_z):
                if g == 0 and h == 0: continue
                theta_grid[g, h] = params[idx]
                idx += 1
        max_theta = np.max(theta_grid)
        log_theta_norm = max_theta + np.log(np.sum(np.exp(theta_grid - max_theta)))
        log_pi_gh = theta_grid - log_theta_norm

        numerator_log = log_pi_gh + np.asarray(L_y).reshape(-1, 1) + np.asarray(L_z).reshape(1, -1)
        post_max = np.max(numerator_log)
        log_norm = post_max + np.log(np.sum(np.exp(numerator_log - post_max)))
        posterior_gh = np.exp(numerator_log - log_norm)

        posterior_g = posterior_gh.sum(axis=1)
        posterior_h = posterior_gh.sum(axis=0)

        row = {
            'ID': canonical_ids[i],
            'Assigned_Group_Y': int(np.argmax(posterior_g)) + 1,
            'Assigned_Group_Z': int(np.argmax(posterior_h)) + 1,
        }
        for g in range(k_y):
            for h in range(k_z):
                row[f'Joint_G{g+1}_H{h+1}_Prob'] = posterior_gh[g, h]
        for g in range(k_y):
            row[f'Y_Group_{g+1}_Prob'] = posterior_g[g]
        for h in range(k_z):
            row[f'Z_Group_{h+1}_Prob'] = posterior_h[h]
        rows.append(row)

    return pd.DataFrame(rows)

def calc_model_adequacy(assignments_df, pis, group_names):
    """Compute Nagin (2005) adequacy metrics: AvePP, OCC, and relative entropy.

    Three standard diagnostics for assessing how well the estimated model
    separates subjects into distinct latent groups:

    AvePP (average posterior probability)
        For subjects hard-assigned to group g, the mean of their posterior
        P(g | i).  Threshold ≥ 0.70 (Nagin 2005).

    OCC (odds of correct classification)
        OCC_g = [AvePP_g / (1 - AvePP_g)] / [π_g / (1 - π_g)]
        Compares classification accuracy to a chance baseline.
        Threshold ≥ 5.0 (Nagin 2005).

    Relative entropy
        H_rel = 1 + (1 / (N · log K)) · Σ_i Σ_g P(g|i) · log P(g|i)
        Ranges from 0 (uniform posteriors — no group separation) to 1
        (perfectly crisp assignments).  Threshold ≥ 0.50.

    Args:
        assignments_df: DataFrame from get_subject_assignments (columns
                        'Assigned_Group', 'Group_1_Prob', …).
        pis:            (K,) mixing weight array from the model dict.
        group_names:    List of K display names for the output DataFrame.

    Returns:
        Tuple[pd.DataFrame, float]:
            adequacy_df:     One row per group with columns Group, Assigned N,
                             Estimated Pi (%), AvePP, OCC.
            relative_entropy: Overall H_rel scalar.
    """
    k = len(pis)
    adequacy_data = []

    if k > 1:
        prob_cols = [col for col in assignments_df.columns if 'Prob' in col]
        probs = assignments_df[prob_cols].values
        # H_rel = 1 + (1/(N·log K)) · Σ_{i,g} P(g|i)·log P(g|i)
        entropy_sum = np.sum(probs * np.log(np.clip(probs, 1e-15, 1.0)))
        relative_entropy = 1.0 + (entropy_sum / (len(assignments_df) * np.log(k)))
    else:
        relative_entropy = 1.0
    
    for g in range(k):
        group_num = g + 1
        group_data = assignments_df[assignments_df['Assigned_Group'] == group_num]
        n_assigned = len(group_data)
        
        if n_assigned > 0:
            ave_pp = group_data[f'Group_{group_num}_Prob'].mean()
        else:
            ave_pp = np.nan
            
        pi_g = pis[g]
        
        if pd.notnull(ave_pp) and ave_pp < 1.0 and pi_g < 1.0 and pi_g > 0:
            occ = (ave_pp / (1.0 - ave_pp)) / (pi_g / (1.0 - pi_g))
        else:
            occ = np.nan
            
        adequacy_data.append({
            "Group": group_names[g],
            "Assigned N": n_assigned,
            "Estimated Pi (%)": round(pi_g * 100, 2),
            "AvePP": round(ave_pp, 4) if pd.notnull(ave_pp) else "N/A",
            "OCC": round(occ, 2) if pd.notnull(occ) else "N/A"
        })

    return pd.DataFrame(adequacy_data), relative_entropy


def calc_joint_model_adequacy(assignments_df, pis_joint, group_names_y, group_names_z):
    """Joint AND per-outcome-marginal Nagin (2005) adequacy metrics (AvePP,
    OCC, relative entropy) for a fitted V5.0 joint dual-trajectory model.

    calc_model_adequacy is already generic over the number of groups/columns,
    so no new adequacy-metric math is needed for V5.0 — this function just
    builds three column-adapted views of get_joint_subject_assignments's
    output (joint cells flattened into a single group index, plus each
    outcome's own marginal) and calls calc_model_adequacy unchanged on each.

    Args:
        assignments_df: DataFrame from get_joint_subject_assignments.
        pis_joint:      (K_Y, K_Z) joint mixing-probability matrix.
        group_names_y, group_names_z: display names for Y's/Z's groups.

    Returns:
        Tuple[pd.DataFrame, float, pd.DataFrame, float, pd.DataFrame, float]:
            joint_adq_df, joint_rel_entropy, y_adq_df, y_rel_entropy, z_adq_df, z_rel_entropy
    """
    k_y, k_z = pis_joint.shape

    # --- Joint: flatten (g,h) cells into a single 1..K_Y*K_Z group index ---
    joint_group_names = [f"{gn}/{hn}" for gn in group_names_y for hn in group_names_z]
    joint_prob_cols = [f'Joint_G{g+1}_H{h+1}_Prob' for g in range(k_y) for h in range(k_z)]
    joint_df = assignments_df[['ID'] + joint_prob_cols].copy()
    joint_df.columns = ['ID'] + [f'Group_{idx+1}_Prob' for idx in range(k_y * k_z)]
    joint_df['Assigned_Group'] = (
        (assignments_df['Assigned_Group_Y'] - 1) * k_z + (assignments_df['Assigned_Group_Z'] - 1) + 1
    )
    joint_adq_df, joint_rel_entropy = calc_model_adequacy(joint_df, pis_joint.flatten(), joint_group_names)

    # --- Y-marginal ---
    y_prob_cols = [f'Y_Group_{g+1}_Prob' for g in range(k_y)]
    y_df = assignments_df[['ID'] + y_prob_cols].copy()
    y_df.columns = ['ID'] + [f'Group_{g+1}_Prob' for g in range(k_y)]
    y_df['Assigned_Group'] = assignments_df['Assigned_Group_Y']
    y_adq_df, y_rel_entropy = calc_model_adequacy(y_df, pis_joint.sum(axis=1), group_names_y)

    # --- Z-marginal ---
    z_prob_cols = [f'Z_Group_{h+1}_Prob' for h in range(k_z)]
    z_df = assignments_df[['ID'] + z_prob_cols].copy()
    z_df.columns = ['ID'] + [f'Group_{h+1}_Prob' for h in range(k_z)]
    z_df['Assigned_Group'] = assignments_df['Assigned_Group_Z']
    z_adq_df, z_rel_entropy = calc_model_adequacy(z_df, pis_joint.sum(axis=0), group_names_z)

    return joint_adq_df, joint_rel_entropy, y_adq_df, y_rel_entropy, z_adq_df, z_rel_entropy