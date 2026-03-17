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

# --- CORE LIKELIHOOD/GRADIENT ENGINE (UNIVERSAL) ---

@njit(cache=True)
def calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max):
    """Compute total NLL, flat gradient, and per-subject gradient matrix in one pass.

    This is the single performance-critical kernel that drives every model fit.
    It supports all four distributions (LOGIT / CNORM / Poisson / ZIP) and the
    optional informative-dropout augmentation through a dist_code dispatch.

    Algorithm overview
    ------------------
    For each subject i:
      1. For each group g, compute the conditional log-likelihood L_{ig} =
         Σ_t log P(y_{it} | g, t) + (optional dropout terms).
      2. Compute the posterior probability P(g | i) ∝ π_g · exp(L_{ig}).
      3. Accumulate the total log-likelihood:  ℓ += log Σ_g π_g · exp(L_{ig}).
      4. Compute the per-subject gradient contributions for all parameter blocks.

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
    # (Verified by finite-difference check against all three cases.)
    k = len(orders)
    thetas = np.zeros(k)
    if k > 1: thetas[1:] = params[0 : k-1]
        
    max_theta = np.max(thetas)
    sum_exp_theta = 0.0
    for i in range(k): sum_exp_theta += np.exp(thetas[i] - max_theta)
    log_pis = thetas - (max_theta + np.log(sum_exp_theta))
    
    pis = np.empty(k)
    pis_safe = np.empty(k)
    for i in range(k):
        p_val = np.exp(log_pis[i])
        pis[i] = p_val
        pis_safe[i] = 1e-15 if p_val < 1e-15 else p_val
        
    n_subjects = len(subj_breaks) - 1
    grad_subj = np.zeros((n_subjects, len(params)))
    
    num_betas = 0
    for g in range(k): num_betas += orders[g] + 1
    gamma_start_idx = (k - 1) + num_betas
    
    sigma = 1.0
    var = 1.0
    sigma_idx = -1
    zeta_start_idx = -1  # first index of per-group zeta block when dist_code == 3

    if dist_code == 1: # CNORM
        sigma_idx = len(params) - 1
        raw_sigma = params[sigma_idx]
        sigma = np.exp(raw_sigma) if raw_sigma < 20 else np.exp(20)
        var = sigma ** 2
    elif dist_code == 3: # ZIP — per-group zeta (logit of zero-inflation), last k params
        zeta_start_idx = len(params) - k

    zeta_g = 0.0  # per-group zero-inflation logit; set inside group loop when dist_code == 3
    
    total_ll = 0.0
    
    for i in range(n_subjects):
        start = subj_breaks[i]
        end = subj_breaks[i+1]
        n_obs = end - start
        
        # L_ig_log[g] = log P(y_i | group=g) accumulated over all time points
        L_ig_log = np.zeros(k)
        # err_mu_ig[g, obs]  = ∂ℓ_{g,obs}/∂μ  (distribution-specific score residual)
        err_mu_ig = np.zeros((k, n_obs))
        # err_aux_ig[g, obs] = ∂ℓ_{g,obs}/∂raw_σ (CNORM) or ∂ℓ_{g,obs}/∂ζ_g (ZIP)
        err_aux_ig = np.zeros((k, n_obs))

        # Pointer to the first beta of each group in params (advances inside loop)
        current_beta_idx = k - 1
        current_gamma_idx = gamma_start_idx

        for g in range(k):
            order = orders[g]
            n_betas = order + 1
            group_betas = params[current_beta_idx : current_beta_idx + n_betas]
            current_beta_idx += n_betas

            if use_dropout:
                # Three dropout coefficients per group: γ₀ (intercept), γ₁ (time), γ₂ (lag-y)
                gamma_0 = params[current_gamma_idx]
                gamma_1 = params[current_gamma_idx + 1]
                gamma_2 = params[current_gamma_idx + 2]
                current_gamma_idx += 3

            if dist_code == 3:  # ZIP: extract per-group zeta constant (ζ_g)
                zeta_g = params[zeta_start_idx + g]

            ll_g = 0.0   # accumulates log P(y_i | group=g) over all observations

            for obs in range(n_obs):
                idx = start + obs
                t_val = times[idx]
                y_val = outcomes[idx]

                # Evaluate polynomial: η = β₀ + β₁·t + β₂·t² + …
                mu = 0.0
                for p in range(order + 1): mu += group_betas[p] * (t_val ** p)

                if dist_code == 0: # LOGIT — binary outcomes
                    # Clamp linear predictor to prevent exp() overflow
                    if mu > 25.0: mu = 25.0
                    if mu < -25.0: mu = -25.0
                    # Numerically stable logistic: two branches avoid large exp()
                    prob = 1.0 / (1.0 + np.exp(-mu)) if mu >= 0 else np.exp(mu) / (1.0 + np.exp(mu))
                    prob = max(1e-12, min(1.0 - 1e-12, prob))   # hard numerical floor
                    # log P(y|η): log-sum-exp stable form of y·η - log(1+e^η)
                    if mu >= 0: ll_g += y_val * mu - (mu + np.log(1.0 + np.exp(-mu)))
                    else: ll_g += y_val * mu - np.log(1.0 + np.exp(mu))
                    # Score residual for beta gradient: ∂ℓ/∂η = y - P(y=1)
                    err_mu_ig[g, obs] = y_val - prob

                elif dist_code == 2: # POISSON (log-link: eta=X@beta, mu=exp(eta))
                    # Clamp linear predictor to avoid overflow in exp()
                    if mu > 20.0: mu = 20.0
                    if mu < -20.0: mu = -20.0
                    exp_eta = np.exp(mu)
                    # log P(y|μ) = y·η - exp(η) - log(y!)  [canonical log-link]
                    ll_g += y_val * mu - exp_eta - math.lgamma(y_val + 1.0)
                    # ∂ℓ/∂η = y - exp(η)  [canonical link, clean score residual]
                    err_mu_ig[g, obs] = y_val - exp_eta

                elif dist_code == 1: # CNORM — censored normal (Tobit)
                    if y_val <= cnorm_min:
                        # LEFT-CENSORED: y is at or below detection limit
                        # LL = log Φ(z_min) where z_min = (y_min - μ)/σ
                        z = (cnorm_min - mu) / sigma
                        cdf_val = max(1e-15, 0.5 * math.erfc(-z / math.sqrt(2.0)))
                        imr = fast_norm_pdf(z) / cdf_val   # inverse Mills ratio φ(z)/Φ(z)
                        ll_g += np.log(cdf_val)
                        # ∂ℓ/∂μ = -IMR/σ  (from dΦ/dμ via chain rule through z)
                        err_mu_ig[g, obs] = -(1.0 / sigma) * imr
                        # ∂ℓ/∂raw_σ = -z·IMR  (chain rule already applied; see module docstring)
                        err_aux_ig[g, obs] = -z * imr
                    elif y_val >= cnorm_max:
                        # RIGHT-CENSORED: y is at or above upper limit
                        # LL = log(1 - Φ(z_max)) where z_max = (y_max - μ)/σ
                        z = (cnorm_max - mu) / sigma
                        sf_val = max(1e-15, 0.5 * math.erfc(z / math.sqrt(2.0)))
                        imr = fast_norm_pdf(z) / sf_val   # inverse Mills ratio φ(z)/(1-Φ(z))
                        ll_g += np.log(sf_val)
                        err_mu_ig[g, obs] = (1.0 / sigma) * imr
                        err_aux_ig[g, obs] = z * imr
                    else:
                        z = (y_val - mu) / sigma
                        # INTERIOR (uncensored): standard normal log-PDF
                        ll_g += fast_norm_logpdf(y_val, mu, sigma)
                        # ∂ℓ/∂μ = (y - μ)/σ²
                        err_mu_ig[g, obs] = (y_val - mu) / var
                        # ∂ℓ/∂raw_σ = -1 + z²  (chain rule absorbed; see module docstring)
                        err_aux_ig[g, obs] = -1.0 + (z ** 2)

                elif dist_code == 3: # ZIP (per-group zero-inflation zeta)
                    # Clamp log-rate predictor before passing to ZIP helper
                    if mu > 20.0: mu = 20.0
                    if mu < -20.0: mu = -20.0
                    # fast_zip_logpmf_grad returns (ll, ∂ll/∂η, ∂ll/∂ζ_g)
                    ll_val, err_m, err_t = fast_zip_logpmf_grad(y_val, mu, zeta_g)
                    ll_g += ll_val
                    err_mu_ig[g, obs] = err_m    # ∂ll/∂η for beta gradient
                    err_aux_ig[g, obs] = err_t   # ∂ll/∂ζ_g for zeta gradient

                if use_dropout and obs > 0:
                    # DROPOUT LIKELIHOOD — not-dropped contribution at each non-first obs
                    # P(not drop | t, y_prev) = 1 - σ(z_drop)  →  log = -log(1+e^{z_drop})
                    y_prev = outcomes[idx - 1]
                    z_drop = gamma_0 + (gamma_1 * t_val) + (gamma_2 * y_prev)
                    if z_drop > 25.0: z_drop = 25.0
                    if z_drop < -25.0: z_drop = -25.0
                    # Numerically stable log(1 - σ(z)) = -softplus(z)
                    if z_drop >= 0: ll_g += -z_drop - np.log(1.0 + np.exp(-z_drop))
                    else: ll_g += -np.log(1.0 + np.exp(z_drop))

            if use_dropout:
                last_idx = end - 1
                if dropouts[last_idx] == 1.0:
                    # DROPOUT LIKELIHOOD — dropped contribution at last obs
                    # P(drop | t_last, y_last) = σ(z_drop)  →  log = log σ(z_drop)
                    t_last = times[last_idx]
                    y_last = outcomes[last_idx]
                    z_drop = gamma_0 + (gamma_1 * t_last) + (gamma_2 * y_last)
                    if z_drop > 25.0: z_drop = 25.0
                    if z_drop < -25.0: z_drop = -25.0
                    # Numerically stable log σ(z)
                    if z_drop >= 0: ll_g += -np.log(1.0 + np.exp(-z_drop))
                    else: ll_g += z_drop - np.log(1.0 + np.exp(z_drop))

            L_ig_log[g] = ll_g   # store P(y_i | group=g) for posterior computation

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
            if k > 1 and g > 0:
                # Theta gradient: ∂ℓ_i/∂θ_g = P(g|i) - π_g  for g > 0 (reference group fixed at 0)
                # NLL sign: store as -(P(g|i) - π_g)
                grad_subj[i, g - 1] = -1.0 * (posterior_ig[g] - pis[g])

        # Add log-marginal-likelihood for subject i to running total
        total_ll += (post_max + np.log(post_sum_exp))

        # ── BETA / GAMMA / AUX GRADIENT ACCUMULATION ───────────────────────────
        # Second pass over groups: accumulate weighted score gradients now that
        # posterior_ig is available.
        current_beta_idx = k - 1
        current_gamma_idx = gamma_start_idx

        for g in range(k):
            order = orders[g]
            n_betas = order + 1
            if use_dropout:
                gamma_0 = params[current_gamma_idx]
                gamma_1 = params[current_gamma_idx + 1]
                gamma_2 = params[current_gamma_idx + 2]

            for obs in range(n_obs):
                idx = start + obs
                t_val = times[idx]

                # Beta gradient: ∂ℓ_i/∂β_{gp} = Σ_t P(g|i) · err_μ_{g,t} · t^p
                # NLL sign: negate and weight by posterior
                weighted_err_mu = err_mu_ig[g, obs] * posterior_ig[g]
                for p in range(order + 1):
                    grad_subj[i, current_beta_idx + p] += -1.0 * weighted_err_mu * (t_val ** p)

                if dist_code == 1: # CNORM: accumulate raw_sigma gradient
                    # ∂NLL_i/∂raw_σ = -Σ_{g,t} P(g|i) · err_aux_{g,t}
                    grad_subj[i, sigma_idx] += -1.0 * err_aux_ig[g, obs] * posterior_ig[g]
                elif dist_code == 3: # ZIP: accumulate per-group zeta gradient
                    # ∂NLL_i/∂ζ_g = -Σ_t P(g|i) · err_aux_{g,t}
                    grad_subj[i, zeta_start_idx + g] += -1.0 * err_aux_ig[g, obs] * posterior_ig[g]
                    
                if use_dropout and obs > 0:
                    y_prev = outcomes[idx - 1]
                    z_drop = gamma_0 + (gamma_1 * t_val) + (gamma_2 * y_prev)
                    p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                    err_drop = (0.0 - p_drop) * posterior_ig[g] 
                    grad_subj[i, current_gamma_idx] += -1.0 * err_drop * 1.0
                    grad_subj[i, current_gamma_idx + 1] += -1.0 * err_drop * t_val
                    grad_subj[i, current_gamma_idx + 2] += -1.0 * err_drop * y_prev
                    
            if use_dropout:
                last_idx = end - 1
                if dropouts[last_idx] == 1.0:
                    t_last = times[last_idx]
                    y_last = outcomes[last_idx]
                    z_drop = gamma_0 + (gamma_1 * t_last) + (gamma_2 * y_last)
                    p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                    err_drop = (1.0 - p_drop) * posterior_ig[g] 
                    grad_subj[i, current_gamma_idx] += -1.0 * err_drop * 1.0
                    grad_subj[i, current_gamma_idx + 1] += -1.0 * err_drop * t_last
                    grad_subj[i, current_gamma_idx + 2] += -1.0 * err_drop * y_last
                current_gamma_idx += 3
            current_beta_idx += n_betas
            
    # Sum per-subject gradients into the flat Jacobian vector
    grad_flat = np.zeros(len(params))
    for i in range(grad_subj.shape[0]):
        for j in range(grad_subj.shape[1]):
            grad_flat[j] += grad_subj[i, j]

    # Return NLL (positive scalar for minimisation) and both gradient forms
    return -1.0 * total_ll, grad_flat, grad_subj


def calc_nll_wrapper(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max):
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

    Returns:
        float: Negative log-likelihood (scalar to minimise).
    """
    nll, _, _ = calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max)
    return nll

def calc_jac_wrapper(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max):
    """Jacobian-only callable for SciPy minimise (discards NLL and per-subject matrix).

    Passed as the ``jac`` argument to scipy.optimize.minimize so that BFGS
    uses the analytical gradient rather than finite-difference approximation.

    Returns:
        np.ndarray: (p,) gradient of the NLL with respect to params.
    """
    _, grad_flat, _ = calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max)
    return grad_flat

# --- DISTRIBUTION-SPECIFIC PUBLIC ALIASES ---
# The universal engine (dist_code dispatch) handles all distributions.
# These thin wrappers expose the expected function names for external consumers
# (e.g. verification scripts, notebooks) without duplicating any math.

def calc_poisson_dynamic_nll_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, cnorm_min=0.0, cnorm_max=0.0):
    """NLL for Poisson trajectories — delegates to universal engine (dist_code=2)."""
    nll, _, _ = calc_universal_subject_gradients_jit(
        params, times, outcomes, dropouts, subj_breaks, orders,
        int(zip_iorder), use_dropout, 2, float(cnorm_min), float(cnorm_max)
    )
    return nll

def calc_poisson_dynamic_jacobian_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, cnorm_min=0.0, cnorm_max=0.0):
    """Gradient for Poisson trajectories — delegates to universal engine (dist_code=2)."""
    _, grad, _ = calc_universal_subject_gradients_jit(
        params, times, outcomes, dropouts, subj_breaks, orders,
        int(zip_iorder), use_dropout, 2, float(cnorm_min), float(cnorm_max)
    )
    return grad

def calc_zip_dynamic_nll_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, cnorm_min=0.0, cnorm_max=0.0):
    """NLL for ZIP trajectories — delegates to universal engine (dist_code=3)."""
    nll, _, _ = calc_universal_subject_gradients_jit(
        params, times, outcomes, dropouts, subj_breaks, orders,
        int(zip_iorder), use_dropout, 3, float(cnorm_min), float(cnorm_max)
    )
    return nll

def calc_zip_dynamic_jacobian_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, cnorm_min=0.0, cnorm_max=0.0):
    """Gradient for ZIP trajectories — delegates to universal engine (dist_code=3)."""
    _, grad, _ = calc_universal_subject_gradients_jit(
        params, times, outcomes, dropouts, subj_breaks, orders,
        int(zip_iorder), use_dropout, 3, float(cnorm_min), float(cnorm_max)
    )
    return grad

# --- ENGINE WRAPPERS ---

def process_optimization_result(result, num_params, times, outcomes, dropouts, subj_breaks, orders_list, zip_iorder, use_dropout, scale_factor, dist, cnorm_min, cnorm_max):
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
    
    if not (result.success or result.status == 2):
        return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, None, None, np.nan, None
        
    D_diag = np.ones(num_params)
    current_beta_idx = k - 1
    for g in range(k):
        for p in range(orders_list[g] + 1):
            D_diag[current_beta_idx + p] = 1.0 / (scale_factor ** p)
        current_beta_idx += orders_list[g] + 1
        
    if use_dropout:
        current_gamma_idx = current_beta_idx
        for g in range(k):
            D_diag[current_gamma_idx + 1] = 1.0 / scale_factor
            current_gamma_idx += 3
            
    if dist == 'CNORM':
        D_diag[-1] = 1.0
    # ZIP zeta params are dimensionless logits — no time-unit scaling needed

    D = np.diag(D_diag)
    
    try:
        times_scaled = times / scale_factor
        args = (times_scaled, outcomes, dropouts, subj_breaks, orders_arr, int(zip_iorder), use_dropout, dist_code, float(cnorm_min), float(cnorm_max))
        
        H_scaled = np.zeros((num_params, num_params))
        for i in range(num_params):
            eps_i = 1e-5 * max(1.0, abs(result.x[i]))
            if eps_i < 1e-8: eps_i = 1e-8
            
            p_plus = np.copy(result.x)
            p_minus = np.copy(result.x)
            p_plus[i] += eps_i
            p_minus[i] -= eps_i
            g_plus = calc_jac_wrapper(p_plus, *args)
            g_minus = calc_jac_wrapper(p_minus, *args)
            H_scaled[i, :] = (g_plus - g_minus) / (2.0 * eps_i)
            
        H_scaled = (H_scaled + H_scaled.T) / 2.0 
        
        try:
            cond_num = np.linalg.cond(H_scaled)
        except np.linalg.LinAlgError:
            cond_num = np.inf
            
        H_inv_scaled = np.linalg.pinv(H_scaled, rcond=1e-10) 
        
        _, _, grad_subj_scaled = calc_universal_subject_gradients_jit(result.x, *args)
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

    thetas = np.zeros(k)
    if k > 1: thetas[1:] = params_unscaled[0 : k-1]
    pis = np.exp(thetas - logsumexp(thetas))

    result.x = params_unscaled
    return True, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis, cond_num, V_model_unscaled

def sort_groups_by_intercept(result, orders_list, se_model, se_robust, pis, use_dropout, dist):
    """
    Sort groups by ascending intercept (beta_0) to eliminate label switching.

    After optimization the labelling of Group 1 / Group 2 / … is arbitrary —
    whichever local optimum the solver found determines which label goes where.
    This function reorders all group-specific blocks in result.x (thetas, betas,
    gammas) so that Group 1 always has the lowest intercept.

    Returns
    -------
    new_orders_list : list  — polynomial orders in the new group ordering
    new_se_model    : ndarray
    new_se_robust   : ndarray
    new_pis         : ndarray
    result.x is mutated in place.

    Notes
    -----
    • The theta re-normalisation (subtracting new_thetas[0]) is exact for the
      likelihood; the theta SEs are rearranged but NOT re-derived — they are
      approximate after a change of reference group.  Beta / gamma SEs are exact.
    • CNORM log-sigma and ZIP alpha params sit at the tail and are not
      group-specific, so they are left untouched.
    """
    k = len(orders_list)
    if k == 1:
        return orders_list, se_model, se_robust, pis

    params = result.x.copy()
    se_m   = se_model.copy()
    se_r   = se_robust.copy()

    # Recover full k-length theta vector (theta[0] = 0 is the implicit reference)
    thetas = np.zeros(k)
    thetas[1:] = params[0:k - 1]

    # Locate the start index of each group's beta block
    beta_starts = []
    idx = k - 1
    for g in range(k):
        beta_starts.append(idx)
        idx += orders_list[g] + 1
    gamma_start = idx  # first index of gamma block (used only when use_dropout)

    # Intercepts are the first beta of each group
    intercepts   = np.array([params[beta_starts[g]] for g in range(k)])
    sorted_idx   = np.argsort(intercepts)

    if np.all(sorted_idx == np.arange(k)):
        return orders_list, se_model, se_robust, pis  # already sorted

    new_params = params.copy()
    new_se_m   = se_m.copy()
    new_se_r   = se_r.copy()

    # --- Rearrange theta params (indices 0..k-2) ---
    new_thetas = thetas[sorted_idx]
    new_thetas -= new_thetas[0]           # re-normalise: new group 0 becomes reference
    new_params[0:k - 1] = new_thetas[1:]

    # Approximate SE rearrangement for thetas
    old_tse_m = np.concatenate(([0.0], se_m[0:k - 1]))
    old_tse_r = np.concatenate(([0.0], se_r[0:k - 1]))
    new_params_tse_m = old_tse_m[sorted_idx]
    new_params_tse_r = old_tse_r[sorted_idx]
    new_se_m[0:k - 1] = new_params_tse_m[1:]
    new_se_r[0:k - 1] = new_params_tse_r[1:]

    # --- Rearrange beta blocks ---
    new_orders = [orders_list[sorted_idx[g]] for g in range(k)]
    write_idx  = k - 1
    for new_g in range(k):
        old_g   = sorted_idx[new_g]
        n_betas = orders_list[old_g] + 1
        src     = beta_starts[old_g]
        new_params[write_idx:write_idx + n_betas] = params[src:src + n_betas]
        new_se_m[write_idx:write_idx + n_betas]   = se_m[src:src + n_betas]
        new_se_r[write_idx:write_idx + n_betas]   = se_r[src:src + n_betas]
        write_idx += n_betas

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


def generate_initial_params(k, orders_list, zip_iorder, use_dropout, dist, outcomes, n_starts=10):
    """Generate n_starts starting points for multi-start BFGS optimisation.

    The first starting point (index 0) is deterministic: intercepts are
    staggered across groups to avoid identical starts, slopes initialised at
    zero.  Points 1..n_starts-1 add seeded Gaussian perturbations to the
    deterministic base, giving the multi-start procedure coverage of the
    parameter space without requiring a global random state.

    Initialisation strategy by parameter block:
      - Thetas: equally spaced logit-quantiles (e.g. for k=3: logit(0.25),
        logit(0.50), logit(0.75)) to start with dispersed mixing weights.
      - Betas:  intercepts staggered; slopes at 0.
      - Gammas: intercepts at −2 (≈ 12% dropout baseline); slopes at 0.
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

    Returns:
        List[np.ndarray]: List of n_starts parameter vectors, each of length p.
    """
    num_params = (k - 1) + sum([order + 1 for order in orders_list])
    if use_dropout: num_params += (3 * k)
    if dist == 'CNORM': num_params += 1
    if dist == 'ZIP': num_params += k  # one zeta per group

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

    current_beta_idx = k - 1
    for g in range(k):
        base[current_beta_idx] = staggered_intercepts[g]
        current_beta_idx += orders_list[g] + 1

    if use_dropout:
        current_gamma_idx = current_beta_idx
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

        # theta (group membership) params: indices 0..k-2
        if k > 1:
            perturbed[:k - 1] += np.random.normal(0, 0.5, k - 1)

        # beta (trajectory) params
        cb_idx = k - 1
        for g in range(k):
            n_betas = orders_list[g] + 1
            perturbed[cb_idx:cb_idx + n_betas] += np.random.normal(0, 0.3, n_betas)
            cb_idx += n_betas

        # gamma (dropout) params
        if use_dropout:
            cg_idx = cb_idx
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


def run_single_model(df, orders_list, zip_iorder=0, use_dropout=False, dist='LOGIT', cnorm_min=0.0, cnorm_max=0.0, n_starts=5):
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
    """
    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    n_subjects = len(subj_breaks) - 1
    n_obs = len(times)
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'POISSON': 2, 'ZIP': 3}
    dist_code = dist_map.get(dist, 0)

    if dist == 'CNORM':
        if cnorm_min is None or np.isnan(cnorm_min): cnorm_min = np.min(outcomes)
        if cnorm_max is None or np.isnan(cnorm_max): cnorm_max = np.max(outcomes)

    max_t = np.max(np.abs(times))
    scale_factor = max_t if max_t > 0 else 1.0
    times_scaled = times / scale_factor

    orders_arr = np.array(orders_list, dtype=np.int32)
    k = len(orders_list)

    args = (times_scaled, outcomes, dropouts, subj_breaks, orders_arr, int(zip_iorder), use_dropout, dist_code, float(cnorm_min), float(cnorm_max))
    num_params = (k - 1) + sum([order + 1 for order in orders_list])
    if use_dropout: num_params += (3 * k)
    if dist == 'CNORM': num_params += 1
    if dist == 'ZIP': num_params += k  # one zeta per group

    starts = generate_initial_params(k, orders_list, zip_iorder, use_dropout, dist, outcomes, n_starts=n_starts)

    best_result = None
    best_nll = np.inf
    best_start_idx = 0

    for s_idx, initial_guess in enumerate(starts):
        res = minimize(
            calc_nll_wrapper, initial_guess, args=args,
            method='BFGS', jac=calc_jac_wrapper, options={'maxiter': 3000, 'gtol': 1e-6}
        )
        if (res.success or res.status == 2) and res.fun < best_nll:
            best_nll = res.fun
            best_result = res
            best_start_idx = s_idx

    if best_result is None:
        best_result = res  # fallback: last attempted result

    if best_start_idx > 0:
        print(f"  [multi-start] single model {orders_list}: best on start {best_start_idx + 1}/{n_starts} (NLL={best_nll:.4f})")

    result = best_result
    is_valid, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis, cond_num, v_model = process_optimization_result(
        result, num_params, times, outcomes, dropouts, subj_breaks, orders_list, zip_iorder, use_dropout, scale_factor, dist, cnorm_min, cnorm_max
    )

    if is_valid:
        orders_list, se_model, se_robust, pis = sort_groups_by_intercept(
            result, orders_list, se_model, se_robust, pis, use_dropout, dist
        )

    min_group_size = np.min(pis) * 100 if is_valid else np.nan
    return {
        'bic': bic_nagin, 'bic_nagin': bic_nagin, 'bic_obs': bic_obs, 'bic_standard': bic_standard,
        'aic': aic_nagin, 'aic_nagin': aic_nagin, 'aic_standard': aic_standard, 'll': ll,
        'orders': orders_list, 'zip_iorder': zip_iorder, 'result': result, 'min_pct': min_group_size,
        'pis': pis, 'use_dropout': use_dropout, 'se_model': se_model, 'se_robust': se_robust,
        'dof': n_obs - num_params, 'cond_num': cond_num, 'dist': dist, 'cnorm_min': cnorm_min, 'cnorm_max': cnorm_max,
        'v_model': v_model
    }

def run_autotraj(df, min_groups=1, max_groups=3, min_order=0, max_order=3, min_group_pct=5.0, p_val_thresh=0.05, use_dropout=False, dist='LOGIT', cnorm_min=0.0, cnorm_max=0.0, zip_iorder=0, n_starts=3):
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

        num_params = (k - 1) + sum([order + 1 for order in orders_list])
        if use_dropout: num_params += (3 * k)
        if dist == 'CNORM': num_params += 1
        if dist == 'ZIP': num_params += k  # one zeta per group

        args = (times_scaled, outcomes, dropouts, subj_breaks, orders_arr, int(zip_iorder), use_dropout, dist_code, float(cnorm_min), float(cnorm_max))
        starts = generate_initial_params(k, orders_list, zip_iorder, use_dropout, dist, outcomes, n_starts=n_starts)

        best_result = None
        best_nll = np.inf
        best_start_idx = 0

        for s_idx, initial_guess in enumerate(starts):
            res = minimize(
                calc_nll_wrapper, initial_guess, args=args,
                method='BFGS', jac=calc_jac_wrapper, options={'maxiter': 3000, 'gtol': 1e-6}
            )
            if (res.success or res.status == 2) and res.fun < best_nll:
                best_nll = res.fun
                best_result = res
                best_start_idx = s_idx

        if best_result is None:
            best_result = res  # fallback: last attempted result

        if best_start_idx > 0:
            print(f"  [multi-start] autotraj {orders_list}: best on start {best_start_idx + 1}/{n_starts} (NLL={best_nll:.4f})")

        result = best_result
        is_converged, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis, cond_num, v_model = process_optimization_result(
            result, num_params, times, outcomes, dropouts, subj_breaks, orders_list, zip_iorder, use_dropout, scale_factor, dist, cnorm_min, cnorm_max
        )

        if is_converged:
            orders_list, se_model, se_robust, pis = sort_groups_by_intercept(
                result, orders_list, se_model, se_robust, pis, use_dropout, dist
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
                current_beta_idx = k - 1
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
                    'v_model': v_model
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

def get_subject_assignments(model_dict, df):
    """Compute posterior group probabilities and hard assignments for every subject.

    For each subject, evaluates the group-conditional log-likelihood
    L_{ig} = Σ_t log P(y_{it} | g, t) under the fitted model, then computes
    the normalised posterior P(g | i) ∝ π_g · exp(L_{ig}).  The hard
    assignment is argmax_g P(g | i).

    Args:
        model_dict: Model dict returned by run_single_model or run_autotraj
                    (must have keys 'orders', 'result', 'use_dropout', 'pis',
                    'dist', 'cnorm_min', 'cnorm_max').
        df:         Long-format DataFrame with columns ID, Time, Outcome.

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
    
    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    ids = df['ID'].values
    subject_ids_unique = ids[subj_breaks[:-1]]
    
    k = len(orders)
    thetas = np.zeros(k)
    if k > 1: thetas[1:] = params[0 : k-1]
    pis = np.exp(thetas - logsumexp(thetas))
    pis_safe = np.clip(pis, 1e-15, 1.0)
    
    num_betas = 0
    for g in range(k): num_betas += orders[g] + 1
    gamma_start_idx = (k - 1) + num_betas
    
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
        L_ig_log = np.zeros(k)
        current_beta_idx = k - 1
        current_gamma_idx = gamma_start_idx
        
        for g in range(k):
            n_betas = orders[g] + 1
            group_betas = params[current_beta_idx : current_beta_idx + n_betas]
            current_beta_idx += n_betas
            
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
        max_val = np.max(numerator_log)
        sum_exp = np.sum(np.exp(numerator_log - max_val))
        posterior_ig = np.exp(numerator_log - (max_val + np.log(sum_exp)))
        
        row = {'ID': subject_ids_unique[i], 'Assigned_Group': np.argmax(posterior_ig) + 1}
        for g in range(k): row[f'Group_{g+1}_Prob'] = posterior_ig[g]
        assignments.append(row)
        
    return pd.DataFrame(assignments)

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