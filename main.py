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
    variance = sigma ** 2
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - ((x - mu) ** 2) / (2 * variance)

@njit(cache=True)
def fast_norm_pdf(x):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x ** 2))

@njit(cache=True)
def fast_norm_logcdf(x):
    cdf_val = 0.5 * math.erfc(-x / math.sqrt(2.0))
    if cdf_val < 1e-15: cdf_val = 1e-15
    return np.log(cdf_val)

@njit(cache=True)
def fast_norm_logsf(x):
    sf_val = 0.5 * math.erfc(x / math.sqrt(2.0))
    if sf_val < 1e-15: sf_val = 1e-15
    return np.log(sf_val)

# --- C-LEVEL MATH HELPERS FOR ZIP ---
@njit(cache=True)
def fast_zip_logpmf_grad(y, z, tau):
    if z > 50.0: z = 50.0
    if z < -50.0: z = -50.0
    lam = np.exp(z)
    
    if tau > 25.0: tau = 25.0
    if tau < -25.0: tau = -25.0
    rho = 1.0 / (1.0 + np.exp(-tau))
    
    if y == 0.0:
        exp_neg_lam = np.exp(-lam)
        p0 = rho + (1.0 - rho) * exp_neg_lam
        p0 = max(1e-15, p0)
        ll = np.log(p0)
        
        dLL_dlam = -(1.0 - rho) * exp_neg_lam / p0
        err_mu = dLL_dlam * lam
        
        dLL_drho = (1.0 - exp_neg_lam) / p0
        err_tau = dLL_drho * rho * (1.0 - rho)
    else:
        one_minus_rho = max(1e-15, 1.0 - rho)
        ll = np.log(one_minus_rho) + y * z - lam - math.lgamma(y + 1.0)
        
        err_mu = y - lam
        err_tau = -rho
        
    return ll, err_mu, err_tau

# --- DATA PREP ---
def load_cambridge_data():
    df = pd.read_csv("cambridge.txt", sep=r'\s+')
    return df

def prep_trajectory_data(df, id_col='ID', outcome_prefix='C', time_prefix='T'):
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    id_col = id_col.strip()
    outcome_prefix = outcome_prefix.strip()
    time_prefix = time_prefix.strip()
    long_df = pd.wide_to_long(df, stubnames=[outcome_prefix, time_prefix], i=id_col, j='Measurement_Period', suffix=r'\d+').reset_index()
    long_df = long_df.rename(columns={outcome_prefix: 'Outcome', time_prefix: 'Time', id_col: 'ID'})
    long_df = long_df.sort_values(by=['ID', 'Measurement_Period'])
    return long_df

def extract_flat_arrays(df):
    ids = df['ID'].values
    times = df['Time'].values.astype(np.float64)
    outcomes = df['Outcome'].values.astype(np.float64)
    max_study_time = np.max(times)
    changes = np.where(ids[:-1] != ids[1:])[0] + 1
    subj_breaks = np.concatenate(([0], changes, [len(df)])).astype(np.int64)
    dropouts = np.zeros(len(df), dtype=np.float64)
    n_subjects = len(subj_breaks) - 1
    for i in range(n_subjects):
        end_idx = subj_breaks[i+1] - 1 
        if times[end_idx] < max_study_time:
            dropouts[end_idx] = 1.0 
    return times, outcomes, dropouts, subj_breaks

@njit(cache=True)
def create_design_matrix_jit(times, order):
    n = len(times)
    X = np.empty((n, order + 1))
    for i in range(n):
        for p in range(order + 1): X[i, p] = times[i] ** p
    return X

@njit(cache=True)
def calc_logit_prob_jit(betas, X):
    z = X @ betas 
    probs = np.empty_like(z)
    for i in range(len(z)):
        if z[i] > 25.0: z[i] = 25.0
        if z[i] < -25.0: z[i] = -25.0
        if z[i] >= 0: probs[i] = 1.0 / (1.0 + np.exp(-z[i]))
        else:
            exp_z = np.exp(z[i])
            probs[i] = exp_z / (1.0 + exp_z)
    return probs

@njit(cache=True)
def logsumexp_jit(a):
    max_val = np.max(a)
    sum_exp = 0.0
    for i in range(len(a)): sum_exp += np.exp(a[i] - max_val)
    return max_val + np.log(sum_exp)

# --- CORE LIKELIHOOD/GRADIENT ENGINE (UNIVERSAL) ---
@njit(cache=True)
def calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max):
    # dist_code: 0=LOGIT, 1=CNORM, 3=ZIP
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
    tau_start_idx = -1
    
    if dist_code == 1: # CNORM
        sigma_idx = len(params) - 1
        raw_sigma = params[sigma_idx]
        sigma = np.exp(raw_sigma) if raw_sigma < 20 else np.exp(20)
        var = sigma ** 2
    elif dist_code == 3: # ZIP
        tau_start_idx = len(params) - (zip_iorder + 1)
        group_alphas = params[tau_start_idx : tau_start_idx + zip_iorder + 1]
    
    total_ll = 0.0
    
    for i in range(n_subjects):
        start = subj_breaks[i]
        end = subj_breaks[i+1]
        n_obs = end - start
        
        L_ig_log = np.zeros(k)
        err_mu_ig = np.zeros((k, n_obs))
        err_aux_ig = np.zeros((k, n_obs)) # Used for sigma in CNORM or tau in ZIP
        
        current_beta_idx = k - 1
        current_gamma_idx = gamma_start_idx
        
        for g in range(k):
            order = orders[g]
            n_betas = order + 1
            group_betas = params[current_beta_idx : current_beta_idx + n_betas]
            current_beta_idx += n_betas
            
            if use_dropout:
                gamma_0 = params[current_gamma_idx]
                gamma_1 = params[current_gamma_idx + 1]
                gamma_2 = params[current_gamma_idx + 2]
                current_gamma_idx += 3
            
            ll_g = 0.0
            
            for obs in range(n_obs):
                idx = start + obs
                t_val = times[idx]
                y_val = outcomes[idx]
                
                mu = 0.0
                for p in range(order + 1): mu += group_betas[p] * (t_val ** p)
                
                if dist_code == 0: # LOGIT
                    if mu > 25.0: mu = 25.0
                    if mu < -25.0: mu = -25.0
                    prob = 1.0 / (1.0 + np.exp(-mu)) if mu >= 0 else np.exp(mu) / (1.0 + np.exp(mu))
                    prob = max(1e-12, min(1.0 - 1e-12, prob))
                    if mu >= 0: ll_g += y_val * mu - (mu + np.log(1.0 + np.exp(-mu)))
                    else: ll_g += y_val * mu - np.log(1.0 + np.exp(mu))
                    err_mu_ig[g, obs] = y_val - prob
                    
                elif dist_code == 1: # CNORM
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
                        
                elif dist_code == 3: # ZIP
                    tau_t = 0.0
                    for p in range(zip_iorder + 1): 
                        tau_t += group_alphas[p] * (t_val ** p)
                    ll_val, err_m, err_t = fast_zip_logpmf_grad(y_val, mu, tau_t)
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
                
            L_ig_log[g] = ll_g
            
        numerator_log = np.zeros(k)
        for g in range(k): numerator_log[g] = np.log(pis_safe[g]) + L_ig_log[g]
        post_max = np.max(numerator_log)
        post_sum_exp = 0.0
        for g in range(k): post_sum_exp += np.exp(numerator_log[g] - post_max)
        
        posterior_ig = np.zeros(k)
        for g in range(k):
            posterior_ig[g] = np.exp(numerator_log[g] - (post_max + np.log(post_sum_exp)))
            if k > 1 and g > 0:
                grad_subj[i, g - 1] = -1.0 * (posterior_ig[g] - pis[g])
                
        total_ll += (post_max + np.log(post_sum_exp))
            
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
                
                weighted_err_mu = err_mu_ig[g, obs] * posterior_ig[g]
                for p in range(order + 1):
                    grad_subj[i, current_beta_idx + p] += -1.0 * weighted_err_mu * (t_val ** p)
                
                if dist_code == 1: # CNORM Sigma Grad
                    grad_subj[i, sigma_idx] += -1.0 * err_aux_ig[g, obs] * posterior_ig[g]
                elif dist_code == 3: # ZIP Global Alpha Grad
                    weighted_err_tau = err_aux_ig[g, obs] * posterior_ig[g]
                    for p in range(zip_iorder + 1):
                        grad_subj[i, tau_start_idx + p] += -1.0 * weighted_err_tau * (t_val ** p)
                    
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
            
    grad_flat = np.zeros(len(params))
    for i in range(grad_subj.shape[0]):
        for j in range(grad_subj.shape[1]):
            grad_flat[j] += grad_subj[i, j]
            
    return -1.0 * total_ll, grad_flat, grad_subj

def calc_nll_wrapper(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max):
    nll, _, _ = calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max)
    return nll

def calc_jac_wrapper(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max):
    _, grad_flat, _ = calc_universal_subject_gradients_jit(params, times, outcomes, dropouts, subj_breaks, orders, zip_iorder, use_dropout, dist_code, cnorm_min, cnorm_max)
    return grad_flat

# --- ENGINE WRAPPERS ---
def process_optimization_result(result, num_params, times, outcomes, dropouts, subj_breaks, orders_list, zip_iorder, use_dropout, scale_factor, dist, cnorm_min, cnorm_max):
    n_subjects = len(subj_breaks) - 1
    n_obs = len(times)
    orders_arr = np.array(orders_list, dtype=np.int32)
    k = len(orders_list)
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'ZIP': 3}
    dist_code = dist_map.get(dist, 0)
    
    if not (result.success or result.status == 2):
        return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, None, None, np.nan
        
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
    elif dist == 'ZIP':
        tau_start_idx = num_params - (zip_iorder + 1)
        for p in range(zip_iorder + 1):
            D_diag[tau_start_idx + p] = 1.0 / (scale_factor ** p)
        
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
    return True, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis, cond_num

def generate_initial_params(k, orders_list, zip_iorder, use_dropout, dist, outcomes, n_starts=10):
    """Return n_starts initial parameter vectors. Index 0 is deterministic; the rest add seeded random perturbations."""
    num_params = (k - 1) + sum([order + 1 for order in orders_list])
    if use_dropout: num_params += (3 * k)
    if dist == 'CNORM': num_params += 1
    if dist == 'ZIP': num_params += (zip_iorder + 1)

    # --- deterministic base ---
    base = np.zeros(num_params)
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
        tau_start_idx = num_params - (zip_iorder + 1)
        base[tau_start_idx] = -2.2

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

        starts.append(perturbed)

    return starts


def run_single_model(df, orders_list, zip_iorder=0, use_dropout=False, dist='LOGIT', cnorm_min=0.0, cnorm_max=0.0, n_starts=5):
    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    n_subjects = len(subj_breaks) - 1
    n_obs = len(times)
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'ZIP': 3}
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
    if dist == 'ZIP': num_params += (zip_iorder + 1)

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
    is_valid, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis, cond_num = process_optimization_result(
        result, num_params, times, outcomes, dropouts, subj_breaks, orders_list, zip_iorder, use_dropout, scale_factor, dist, cnorm_min, cnorm_max
    )

    min_group_size = np.min(pis) * 100 if is_valid else np.nan
    return {
        'bic': bic_nagin, 'bic_nagin': bic_nagin, 'bic_obs': bic_obs, 'bic_standard': bic_standard,
        'aic': aic_nagin, 'aic_nagin': aic_nagin, 'aic_standard': aic_standard, 'll': ll,
        'orders': orders_list, 'zip_iorder': zip_iorder, 'result': result, 'min_pct': min_group_size,
        'pis': pis, 'use_dropout': use_dropout, 'se_model': se_model, 'se_robust': se_robust,
        'dof': n_obs - num_params, 'cond_num': cond_num, 'dist': dist, 'cnorm_min': cnorm_min, 'cnorm_max': cnorm_max
    }

def run_autotraj(df, min_groups=1, max_groups=3, min_order=0, max_order=3, min_group_pct=5.0, p_val_thresh=0.05, use_dropout=False, dist='LOGIT', cnorm_min=0.0, cnorm_max=0.0, zip_iorder=0, n_starts=3):
    valid_models = []
    all_evaluated_models = []
    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    n_subjects = len(subj_breaks) - 1
    n_obs = len(times)
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'ZIP': 3}
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
        if dist == 'ZIP': num_params += (zip_iorder + 1)

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
        is_converged, ll, aic_nagin, bic_nagin, bic_obs, aic_standard, bic_standard, se_model, se_robust, pis, cond_num = process_optimization_result(
            result, num_params, times, outcomes, dropouts, subj_breaks, orders_list, zip_iorder, use_dropout, scale_factor, dist, cnorm_min, cnorm_max
        )
        
        if is_converged:
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
                    'pis': pis, 'use_dropout': use_dropout, 'se_model': se_model, 'se_robust': se_robust, 'dof': dof, 'cond_num': cond_num, 'dist': dist, 'cnorm_min': cnorm_min, 'cnorm_max': cnorm_max
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
    orders = model_dict['orders']
    zip_iorder = model_dict.get('zip_iorder', 0)
    use_dropout = model_dict['use_dropout']
    params = model_dict['result'].x
    dist = model_dict.get('dist', 'LOGIT')
    dist_map = {'LOGIT': 0, 'CNORM': 1, 'ZIP': 3}
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
        tau_start_idx = len(params) - (zip_iorder + 1)
        group_alphas = params[tau_start_idx : tau_start_idx + zip_iorder + 1]
    
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
            
            ll_g = 0.0
            
            for obs in range(n_obs):
                idx = start + obs
                t_val = times[idx]
                y_val = outcomes[idx]
                
                mu = sum(group_betas[p] * (t_val ** p) for p in range(orders[g] + 1))
                
                tau_t = 0.0
                if dist_code == 3:
                    for p in range(zip_iorder + 1): tau_t += group_alphas[p] * (t_val ** p)
                
                if dist_code == 1: # CNORM
                    if y_val <= min_val:
                        z = (min_val - mu) / sigma
                        ll_g += fast_norm_logcdf(z)
                    elif y_val >= max_val:
                        z = (max_val - mu) / sigma
                        ll_g += fast_norm_logsf(z)
                    else:
                        ll_g += fast_norm_logpdf(y_val, mu, sigma)
                elif dist_code == 3: # ZIP
                    ll_val, _, _ = fast_zip_logpmf_grad(y_val, mu, tau_t)
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
    k = len(pis)
    adequacy_data = []
    
    if k > 1:
        prob_cols = [col for col in assignments_df.columns if 'Prob' in col]
        probs = assignments_df[prob_cols].values
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