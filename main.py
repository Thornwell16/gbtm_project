import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
import itertools
import argparse
from numba import njit

def load_cambridge_data():
    """Loads the Cambridge dataset."""
    df = pd.read_csv("cambridge.txt", sep=r'\s+')
    return df

def prep_trajectory_data(df, id_col='ID', outcome_prefix='C', time_prefix='T'):
    """Pivots wide-format subject data into long-format observation data."""
    long_df = pd.wide_to_long(df, stubnames=[outcome_prefix, time_prefix], i=id_col, j='Measurement_Period').reset_index()
    long_df = long_df.rename(columns={outcome_prefix: 'Outcome', time_prefix: 'Time'})
    long_df = long_df.sort_values(by=[id_col, 'Measurement_Period'])
    return long_df

def extract_flat_arrays(df):
    """Flattens Pandas DataFrame and generates binary flags for FMM Informative Dropout."""
    ids = df['ID'].values
    times = df['Time'].values.astype(np.float64)
    outcomes = df['Outcome'].values.astype(np.float64)
    
    max_study_time = np.max(times)
    changes = np.where(ids[:-1] != ids[1:])[0] + 1
    subj_breaks = np.concatenate(([0], changes, [len(df)])).astype(np.int64)
    
    dropouts = np.zeros(len(df), dtype=np.float64)
    n_subjects = len(subj_breaks) - 1
    
    for i in range(n_subjects):
        start_idx = subj_breaks[i]
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

@njit(cache=True)
def calc_dynamic_nll_jit(params, times, outcomes, dropouts, subj_breaks, orders):
    k = len(orders)
    thetas = np.zeros(k)
    if k > 1: thetas[1:] = params[0 : k-1]
        
    max_theta = np.max(thetas)
    sum_exp_theta = 0.0
    for i in range(k): sum_exp_theta += np.exp(thetas[i] - max_theta)
    log_pis = thetas - (max_theta + np.log(sum_exp_theta))
    
    num_betas = 0
    for g in range(k): num_betas += orders[g] + 1
    gamma_start_idx = (k - 1) + num_betas
    
    total_ll = 0.0
    n_subjects = len(subj_breaks) - 1
    
    for i in range(n_subjects):
        start = subj_breaks[i]
        end = subj_breaks[i+1]
        n_obs = end - start
        
        subject_group_lls = np.zeros(k)
        current_beta_idx = k - 1
        current_gamma_idx = gamma_start_idx
        
        for g in range(k):
            order = orders[g]
            n_betas = order + 1
            group_betas = params[current_beta_idx : current_beta_idx + n_betas]
            current_beta_idx += n_betas
            
            gamma_0 = params[current_gamma_idx]
            gamma_1 = params[current_gamma_idx + 1]
            gamma_2 = params[current_gamma_idx + 2]
            current_gamma_idx += 3
            
            ll_g = 0.0
            for obs in range(n_obs):
                idx = start + obs
                t_val = times[idx]
                y_val = outcomes[idx]
                
                z = 0.0
                for p in range(order + 1): z += group_betas[p] * (t_val ** p)
                prob = 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))
                prob = max(1e-10, min(1.0 - 1e-10, prob))
                ll_g += (y_val * np.log(prob)) + ((1.0 - y_val) * np.log(1.0 - prob))
                
                if obs > 0:
                    y_prev = outcomes[idx - 1]
                    z_drop = gamma_0 + (gamma_1 * t_val) + (gamma_2 * y_prev)
                    p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                    p_drop = max(1e-10, min(1.0 - 1e-10, p_drop))
                    ll_g += np.log(1.0 - p_drop) 
                    
            last_idx = end - 1
            if dropouts[last_idx] == 1.0:
                t_last = times[last_idx]
                y_last = outcomes[last_idx]
                z_drop = gamma_0 + (gamma_1 * t_last) + (gamma_2 * y_last)
                p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                p_drop = max(1e-10, min(1.0 - 1e-10, p_drop))
                ll_g += np.log(p_drop) 
                
            subject_group_lls[g] = log_pis[g] + ll_g
            
        total_ll += logsumexp_jit(subject_group_lls)
        
    return -1.0 * total_ll

@njit(cache=True)
def calc_dynamic_jacobian_jit(params, times, outcomes, dropouts, subj_breaks, orders):
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
        
    grad_thetas = np.zeros(k)
    grad_flat = np.zeros(len(params))
    n_subjects = len(subj_breaks) - 1
    
    num_betas = 0
    for g in range(k): num_betas += orders[g] + 1
    gamma_start_idx = (k - 1) + num_betas
    
    for i in range(n_subjects):
        start = subj_breaks[i]
        end = subj_breaks[i+1]
        n_obs = end - start
        
        L_ig_log = np.zeros(k)
        p_ig = np.zeros((k, n_obs))
        
        current_beta_idx = k - 1
        current_gamma_idx = gamma_start_idx
        
        for g in range(k):
            order = orders[g]
            n_betas = order + 1
            group_betas = params[current_beta_idx : current_beta_idx + n_betas]
            current_beta_idx += n_betas
            
            gamma_0 = params[current_gamma_idx]
            gamma_1 = params[current_gamma_idx + 1]
            gamma_2 = params[current_gamma_idx + 2]
            current_gamma_idx += 3
            
            ll_g = 0.0
            for obs in range(n_obs):
                idx = start + obs
                t_val = times[idx]
                y_val = outcomes[idx]
                
                z = 0.0
                for p in range(order + 1): z += group_betas[p] * (t_val ** p)
                prob = 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))
                prob = max(1e-10, min(1.0 - 1e-10, prob))
                
                p_ig[g, obs] = prob
                ll_g += (y_val * np.log(prob)) + ((1.0 - y_val) * np.log(1.0 - prob))
                
                if obs > 0:
                    y_prev = outcomes[idx - 1]
                    z_drop = gamma_0 + (gamma_1 * t_val) + (gamma_2 * y_prev)
                    p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                    p_drop = max(1e-10, min(1.0 - 1e-10, p_drop))
                    ll_g += np.log(1.0 - p_drop)
                    
            last_idx = end - 1
            if dropouts[last_idx] == 1.0:
                t_last = times[last_idx]
                y_last = outcomes[last_idx]
                z_drop = gamma_0 + (gamma_1 * t_last) + (gamma_2 * y_last)
                p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                p_drop = max(1e-10, min(1.0 - 1e-10, p_drop))
                ll_g += np.log(p_drop)
                
            L_ig_log[g] = ll_g
            
        numerator_log = np.zeros(k)
        for g in range(k): numerator_log[g] = np.log(pis_safe[g]) + L_ig_log[g]
            
        post_max = np.max(numerator_log)
        post_sum_exp = 0.0
        for g in range(k): post_sum_exp += np.exp(numerator_log[g] - post_max)
        
        posterior_ig = np.zeros(k)
        for g in range(k):
            posterior_ig[g] = np.exp(numerator_log[g] - (post_max + np.log(post_sum_exp)))
            grad_thetas[g] += (posterior_ig[g] - pis[g])
            
        current_beta_idx = k - 1
        current_gamma_idx = gamma_start_idx
        
        for g in range(k):
            order = orders[g]
            n_betas = order + 1
            gamma_0 = params[current_gamma_idx]
            gamma_1 = params[current_gamma_idx + 1]
            gamma_2 = params[current_gamma_idx + 2]
            
            for obs in range(n_obs):
                idx = start + obs
                t_val = times[idx]
                
                # Beta Gradients
                error_y = (outcomes[idx] - p_ig[g, obs]) * posterior_ig[g]
                for p in range(order + 1):
                    grad_flat[current_beta_idx + p] += error_y * (t_val ** p)
                    
                # Survival Gradients
                if obs > 0:
                    y_prev = outcomes[idx - 1]
                    z_drop = gamma_0 + (gamma_1 * t_val) + (gamma_2 * y_prev)
                    p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                    err_drop = (0.0 - p_drop) * posterior_ig[g] 
                    grad_flat[current_gamma_idx] += err_drop * 1.0
                    grad_flat[current_gamma_idx + 1] += err_drop * t_val
                    grad_flat[current_gamma_idx + 2] += err_drop * y_prev
                    
            # Dropout Gradients
            last_idx = end - 1
            if dropouts[last_idx] == 1.0:
                t_last = times[last_idx]
                y_last = outcomes[last_idx]
                z_drop = gamma_0 + (gamma_1 * t_last) + (gamma_2 * y_last)
                p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                err_drop = (1.0 - p_drop) * posterior_ig[g] 
                grad_flat[current_gamma_idx] += err_drop * 1.0
                grad_flat[current_gamma_idx + 1] += err_drop * t_last
                grad_flat[current_gamma_idx + 2] += err_drop * y_last
                
            current_beta_idx += n_betas
            current_gamma_idx += 3
            
    for i in range(len(grad_flat)): grad_flat[i] = -1.0 * grad_flat[i]
    if k > 1:
        for i in range(1, k): grad_flat[i - 1] = -1.0 * grad_thetas[i]
            
    return grad_flat

def run_single_model(df, orders_list):
    """Executes a single model, breaking symmetry perfectly for PROC TRAJ equivalence."""
    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    n_subjects = len(subj_breaks) - 1
    
    _ = create_design_matrix_jit(np.array([1.0]), 1)
    
    orders_arr = np.array(orders_list, dtype=np.int32)
    k = len(orders_list)
    num_params = (k - 1) + sum([order + 1 for order in orders_list]) + (3 * k)
    initial_guess = np.zeros(num_params)
    
    # 1. Symmetry Breaker (Stagger the intercepts so identical groups split apart)
    current_beta_idx = k - 1
    staggered_intercepts = np.linspace(-2.0, 2.0, k) if k > 1 else [0.0]
    for g in range(k):
        initial_guess[current_beta_idx] = staggered_intercepts[g]
        current_beta_idx += orders_list[g] + 1
        
    # 2. Dropout Baseline Fix (Start with realistic 11% dropout instead of 50%)
    current_gamma_idx = (k - 1) + sum([order + 1 for order in orders_list])
    for g in range(k):
        initial_guess[current_gamma_idx] = -2.0
        current_gamma_idx += 3
    
    result = minimize(
        calc_dynamic_nll_jit, initial_guess, args=(times, outcomes, dropouts, subj_breaks, orders_arr),
        method='BFGS', jac=calc_dynamic_jacobian_jit
    )
    
    max_ll = -1 * result.fun if result.success or result.status == 2 else np.nan
    bic = (-2 * max_ll) + (num_params * np.log(n_subjects)) if not np.isnan(max_ll) else np.nan
    
    thetas = np.zeros(k)
    if k > 1: thetas[1:] = result.x[0 : k-1]
    pis = np.exp(thetas - logsumexp(thetas))
    min_group_size = np.min(pis) * 100
    
    return {
        'bic': bic, 'orders': orders_list, 'result': result, 
        'min_pct': min_group_size, 'pis': pis
    }

def run_autotraj(df, min_groups=1, max_groups=3, min_order=0, max_order=3, min_group_pct=5.0, p_val_thresh=0.05):
    valid_models = []
    n_subjects = df['ID'].nunique()
    
    print("\n" + "*"*80)
    print("STARTING C-COMPILED MNAR SEARCH (Informative Dropout Active)")
    print("*"*80)
    
    _ = create_design_matrix_jit(np.array([1.0]), 1)
    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    
    all_combinations = []
    for k in range(min_groups, max_groups + 1):
        order_combinations = list(itertools.product(range(min_order, max_order + 1), repeat=k))
        all_combinations.extend([list(orders) for orders in order_combinations])
        
    total_models = len(all_combinations)
    
    for i, orders_list in enumerate(all_combinations):
        orders_arr = np.array(orders_list, dtype=np.int32)
        print(f"Testing Model {i+1}/{total_models}: {len(orders_list)}-Group {orders_list}...".ljust(40), end=" ", flush=True)
        
        k = len(orders_list)
        num_params = (k - 1) + sum([order + 1 for order in orders_list]) + (3 * k)
        initial_guess = np.zeros(num_params)
        
        # 1. Symmetry Breaker
        current_beta_idx = k - 1
        staggered_intercepts = np.linspace(-2.0, 2.0, k) if k > 1 else [0.0]
        for g in range(k):
            initial_guess[current_beta_idx] = staggered_intercepts[g]
            current_beta_idx += orders_list[g] + 1
            
        # 2. Dropout Baseline Fix
        current_gamma_idx = (k - 1) + sum([order + 1 for order in orders_list])
        for g in range(k):
            initial_guess[current_gamma_idx] = -2.0
            current_gamma_idx += 3
        
        result = minimize(
            calc_dynamic_nll_jit, initial_guess, args=(times, outcomes, dropouts, subj_breaks, orders_arr),
            method='BFGS', jac=calc_dynamic_jacobian_jit
        )
        
        if result.success or result.status == 2:
            max_ll = -1 * result.fun
            bic = (-2 * max_ll) + (num_params * np.log(n_subjects))
            
            thetas = np.zeros(k)
            if k > 1: thetas[1:] = result.x[0 : k-1]
            pis = np.exp(thetas - logsumexp(thetas))
            min_group_size = np.min(pis) * 100
            
            if min_group_size < min_group_pct:
                print(f"REJECTED (Size: {min_group_size:.1f}%)")
                continue
                
            try: se = np.sqrt(np.abs(np.diag(result.hess_inv))) # Prevents NaN Warning
            except: se = np.full(num_params, np.nan)
                
            all_significant = True
            current_beta_idx = k - 1
            for g in range(k):
                n_betas = orders_list[g] + 1
                highest_est = result.x[current_beta_idx + n_betas - 1]
                highest_se = se[current_beta_idx + n_betas - 1]
                z_score = highest_est / highest_se if highest_se > 0 else 0
                if 2 * (1 - norm.cdf(abs(z_score))) >= p_val_thresh: all_significant = False
                current_beta_idx += n_betas
                    
            if all_significant:
                print(f"VALID! BIC: {bic:.2f}")
                valid_models.append({
                    'bic': bic, 'orders': orders_list, 'result': result, 
                    'min_pct': min_group_size, 'pis': pis
                })
            else:
                print("REJECTED (P-Value)")
        else:
            print("FAILED (No Convergence)")
                
    valid_models = sorted(valid_models, key=lambda x: x['bic'])
    return valid_models[:5]

def get_subject_assignments(result, df, orders):
    times, outcomes, dropouts, subj_breaks = extract_flat_arrays(df)
    ids = df['ID'].values
    subject_ids_unique = ids[subj_breaks[:-1]]
    
    k = len(orders)
    params = result.x
    thetas = np.zeros(k)
    if k > 1: thetas[1:] = params[0 : k-1]
    pis = np.exp(thetas - logsumexp(thetas))
    pis_safe = np.clip(pis, 1e-15, 1.0)
    
    num_betas = 0
    for g in range(k): num_betas += orders[g] + 1
    gamma_start_idx = (k - 1) + num_betas
    
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
            
            gamma_0 = params[current_gamma_idx]
            gamma_1 = params[current_gamma_idx + 1]
            gamma_2 = params[current_gamma_idx + 2]
            current_gamma_idx += 3
            
            ll_g = 0.0
            for obs in range(n_obs):
                idx = start + obs
                t_val = times[idx]
                y_val = outcomes[idx]
                
                z = sum(group_betas[p] * (t_val ** p) for p in range(orders[g] + 1))
                prob = 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))
                prob = max(1e-10, min(1.0 - 1e-10, prob))
                ll_g += y_val * np.log(prob) + (1.0 - y_val) * np.log(1.0 - prob)
                
                if obs > 0:
                    y_prev = outcomes[idx - 1]
                    z_drop = gamma_0 + (gamma_1 * t_val) + (gamma_2 * y_prev)
                    p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                    p_drop = max(1e-10, min(1.0 - 1e-10, p_drop))
                    ll_g += np.log(1.0 - p_drop)
                    
            last_idx = end - 1
            if dropouts[last_idx] == 1.0:
                t_last = times[last_idx]
                y_last = outcomes[last_idx]
                z_drop = gamma_0 + (gamma_1 * t_last) + (gamma_2 * y_last)
                p_drop = 1.0 / (1.0 + np.exp(-z_drop)) if z_drop >= 0 else np.exp(z_drop) / (1.0 + np.exp(z_drop))
                p_drop = max(1e-10, min(1.0 - 1e-10, p_drop))
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

def get_parameter_estimates(result, orders, group_names=None):
    k = len(orders)
    if group_names is None or len(group_names) != k:
        group_names = [f"Group {g+1}" for g in range(k)]
        
    params = result.x
    try: se = np.sqrt(np.abs(np.diag(result.hess_inv)))
    except: se = np.full(len(params), np.nan)
        
    data = []
    current_beta_idx = k - 1
    current_gamma_idx = (k - 1) + sum([o + 1 for o in orders])
    
    labels = ["Intercept", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
    gamma_labels = ["Dropout: Intercept", "Dropout: Time", "Dropout: Prev Outcome"]
    
    for g in range(k):
        n_betas = orders[g] + 1
        
        # 1. Trajectory Betas
        for b_idx in range(n_betas):
            est = params[current_beta_idx + b_idx]
            err = se[current_beta_idx + b_idx]
            z_score = est / err if err > 0 else 0
            p_val = 2 * (1 - norm.cdf(abs(z_score)))
            data.append({
                "Component": "Trajectory", "Group": group_names[g], "Parameter": labels[b_idx],
                "Estimate": round(est, 4), "Std Error": round(err, 4),
                "P-Value": round(p_val, 4) if p_val >= 0.0001 else "< 0.0001"
            })
        current_beta_idx += n_betas
        
        # 2. Dropout Gammas
        for gam_idx in range(3):
            est = params[current_gamma_idx + gam_idx]
            err = se[current_gamma_idx + gam_idx]
            z_score = est / err if err > 0 else 0
            p_val = 2 * (1 - norm.cdf(abs(z_score)))
            data.append({
                "Component": "Dropout", "Group": group_names[g], "Parameter": gamma_labels[gam_idx],
                "Estimate": round(est, 4), "Std Error": round(err, 4),
                "P-Value": round(p_val, 4) if p_val >= 0.0001 else "< 0.0001"
            })
        current_gamma_idx += 3
        
    return pd.DataFrame(data)

def calc_model_adequacy(assignments_df, pis, group_names=None):
    if group_names is None or len(group_names) != len(pis):
        group_names = [f"Group {g+1}" for g in range(len(pis))]
        
    adequacy_data = []
    for g in range(1, len(pis) + 1):
        group_subjects = assignments_df[assignments_df['Assigned_Group'] == g]
        if len(group_subjects) == 0: continue
            
        ave_pp = np.clip(group_subjects[f'Group_{g}_Prob'].mean(), 0.0001, 0.9999)
        pi_safe = np.clip(pis[g-1], 0.0001, 0.9999)
        occ = (ave_pp / (1 - ave_pp)) / (pi_safe / (1 - pi_safe))
        
        adequacy_data.append({
            "Group": group_names[g-1], "N Assigned": len(group_subjects),
            "Est. Population %": f"{pis[g-1] * 100:.1f}%",
            "AvePP": round(ave_pp, 3), "OCC": round(occ, 2)
        })
    return pd.DataFrame(adequacy_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GBTM Python Engine")
    parser.add_argument('--mode', choices=['single', 'auto'], default='single')
    args = parser.parse_args()
    if args.mode == 'auto':
        long_data = prep_trajectory_data(load_cambridge_data()).dropna(subset=['Time', 'Outcome'])
        run_autotraj(long_data, min_groups=1, max_groups=3, min_order=0, max_order=2)