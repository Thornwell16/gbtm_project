import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import norm
import itertools
import argparse

def load_cambridge_data():
    """
    Loads the Cambridge dataset used for testing logit trajectory models.
    Reference: https://www.andrew.cmu.edu/user/bjj/traj/
    """
    df = pd.read_csv("cambridge.txt", sep=r'\s+')
    return df

def prep_trajectory_data(df, id_col='ID', outcome_prefix='C', time_prefix='T'):
    """Pivots wide-format subject data into long-format observation data."""
    long_df = pd.wide_to_long(df, stubnames=[outcome_prefix, time_prefix], i=id_col, j='Measurement_Period').reset_index()
    long_df = long_df.rename(columns={outcome_prefix: 'Outcome', time_prefix: 'Time'})
    long_df = long_df.sort_values(by=[id_col, 'Measurement_Period'])
    return long_df

def create_design_matrix(times, order):
    """Creates a dynamic design matrix (X) for any polynomial order."""
    return np.column_stack([times**p for p in range(order + 1)])

def calc_logit_prob_matrix(betas, X):
    """Calculates probabilities for ALL observations safely."""
    z = np.dot(X, betas)
    return expit(z)  # This handles the extreme numbers without crashing RAM!

def calc_log_likelihood_matrix(betas, X, actual_outcomes):
    """Calculates the Bernoulli Log-Likelihood for an array of observations."""
    p = np.clip(calc_logit_prob_matrix(betas, X), 1e-10, 1 - 1e-10)
    ll_array = (actual_outcomes * np.log(p)) + ((1 - actual_outcomes) * np.log(1 - p))
    return np.sum(ll_array)

def calc_dynamic_nll(params, df, orders):
    """Calculates the Negative Log-Likelihood for ANY number of groups/orders."""
    k = len(orders)
    
    # Parse Thetas
    thetas = np.zeros(k)
    if k > 1:
        thetas[1:] = params[0 : k-1]
    log_pis = thetas - logsumexp(thetas)
    
    # Parse Betas
    group_betas = []
    current_idx = k - 1
    for order in orders:
        n_betas = order + 1
        group_betas.append(params[current_idx : current_idx + n_betas])
        current_idx += n_betas

    total_ll = 0
    grouped = df.groupby('ID')
    
    for subject_id, subject_data in grouped:
        times = subject_data['Time'].values
        outcomes = subject_data['Outcome'].values
        subject_group_lls = []
        
        for g in range(k):
            X = create_design_matrix(times, orders[g])
            ll_g = calc_log_likelihood_matrix(group_betas[g], X, outcomes)
            subject_group_lls.append(log_pis[g] + ll_g)
            
        total_ll += logsumexp(subject_group_lls)
        
    return -1 * total_ll

def calc_dynamic_jacobian(params, df, orders):
    """
    Calculates the exact analytical gradient (Jacobian) of the Negative Log-Likelihood.
    This replaces SciPy's slow numerical guessing, making the optimization lightning fast.
    """
    k = len(orders)
    
    # 1. Parse Thetas
    thetas = np.zeros(k)
    if k > 1:
        thetas[1:] = params[0 : k-1]
    pis = np.exp(thetas - logsumexp(thetas))
    
    # 2. Parse Betas
    group_betas = []
    current_idx = k - 1
    for order in orders:
        n_betas = order + 1
        group_betas.append(params[current_idx : current_idx + n_betas])
        current_idx += n_betas

    # Initialize empty arrays to hold our calculated slopes
    grad_thetas = np.zeros(k)
    grad_betas = [np.zeros(len(b)) for b in group_betas]
    
    grouped = df.groupby('ID')
    
    for subject_id, subject_data in grouped:
        times = subject_data['Time'].values
        outcomes = subject_data['Outcome'].values
        
        # Calculate likelihoods for this subject
        L_ig_log = np.zeros(k)
        p_ig = [] 
        X_ig = [] 
        
        for g in range(k):
            X = create_design_matrix(times, orders[g])
            X_ig.append(X)
            
            p = np.clip(calc_logit_prob_matrix(group_betas[g], X), 1e-10, 1 - 1e-10)
            p_ig.append(p)
            
            ll_array = (outcomes * np.log(p)) + ((1 - outcomes) * np.log(1 - p))
            L_ig_log[g] = np.sum(ll_array)
            
        # Calculate Posterior Probability (hat{pi}_{ig})
        numerator_log = np.log(pis) + L_ig_log
        posterior_ig = np.exp(numerator_log - logsumexp(numerator_log))
        
        # Add to Gradients
        for g in range(k):
            # Gradient for group size
            grad_thetas[g] += (posterior_ig[g] - pis[g])
            
            # Gradient for trajectory shape: X^T * (y - p) * posterior
            error = outcomes - p_ig[g]
            grad_betas[g] += np.dot(X_ig[g].T, error) * posterior_ig[g]

    # Because we are MINIMIZING the negative log-likelihood, flip the signs
    grad_thetas = -1 * grad_thetas
    grad_betas = [-1 * gb for gb in grad_betas]
    
    # Flatten everything back into a single 1D array to hand back to SciPy
    flat_gradient = []
    if k > 1:
        flat_gradient.extend(grad_thetas[1:]) 
    for gb in grad_betas:
        flat_gradient.extend(gb)
        
    return np.array(flat_gradient)


def extract_and_print_metrics(result, df, orders):
    """
    Translates raw output into AIC, BICs, Group Sizes, and P-Values.
    """
    k = len(orders)
    params = result.x
    max_ll = -1 * result.fun
    
    print("\n" + "="*60)
    print(f"MODEL METRICS: {k}-GROUP ({orders})")
    print("="*60)
    
    # 1. Information Criteria (The Deciders)
    n_subjects = df['ID'].nunique()
    n_obs = len(df)
    p_params = len(params)
    
    aic = (-2 * max_ll) + (2 * p_params)
    bic_subj = (-2 * max_ll) + (p_params * np.log(n_subjects))
    bic_obs = (-2 * max_ll) + (p_params * np.log(n_obs))
    
    print(f"Maximum Log-Likelihood: {max_ll:.4f}")
    print(f"AIC:                    {aic:.4f}")
    print(f"BIC (N={n_subjects} subjects):  {bic_subj:.4f}")
    print(f"BIC (N={n_obs} obs):       {bic_obs:.4f}")
    print("-" * 60)
    
    # 2. Group Sizes
    thetas = np.zeros(k)
    if k > 1:
        thetas[1:] = params[0 : k-1]
    pis = np.exp(thetas - logsumexp(thetas))
    
    # 3. Standard Errors and P-values
    # SciPy's BFGS method approximates the inverse Hessian (variance-covariance matrix)
    try:
        se = np.sqrt(np.diag(result.hess_inv))
    except:
        # Fallback if the matrix is singular/fails
        se = np.full(p_params, np.nan) 
        
    current_idx = k - 1
    for g in range(k):
        n_betas = orders[g] + 1
        group_betas = params[current_idx : current_idx + n_betas]
        group_se = se[current_idx : current_idx + n_betas]
        current_idx += n_betas
        
        print(f"GROUP {g+1} (Size: {pis[g] * 100:.1f}%)")
        print(f"{'Parameter':<12} | {'Estimate':<10} | {'Std Err':<10} | {'P-Value':<10}")
        print("-" * 55)
        
        beta_labels = ["Intercept", "Linear", "Quadratic", "Cubic", "Quartic"]
        
        for b_idx in range(n_betas):
            estimate = group_betas[b_idx]
            error = group_se[b_idx]
            
            # Calculate Z-score and two-tailed P-value
            z_score = estimate / error if error > 0 else 0
            p_val = 2 * (1 - norm.cdf(abs(z_score)))
            
            print(f"{beta_labels[b_idx]:<12} | {estimate:>10.4f} | {error:>10.4f} | {p_val:>10.4f}")
        print()
        
def run_autotraj(df, min_groups=1, max_groups=3, min_order=0, max_order=3, min_group_pct=5.0, p_val_thresh=0.05):
    """
    Exhaustively searches for the mathematically optimal trajectory model.
    Enforces Group Size and Highest-Order Significance rules dynamically.
    """
    best_bic = np.inf
    best_orders = None
    best_result = None
    
    n_subjects = df['ID'].nunique()
    
    print("\n" + "*"*80)
    print("STARTING AUTOTRAJ EXHAUSTIVE SEARCH")
    print("*"*80)
    
    # 1. Outer Loop: Number of Groups
    for k in range(min_groups, max_groups + 1):
        
        # 2. Inner Loop: Every polynomial combination (e.g., [0,1], [1,1], [2,1]...)
        order_combinations = list(itertools.product(range(min_order, max_order + 1), repeat=k))
        
        for orders in order_combinations:
            orders_list = list(orders)
            
            # Visual feedback in the terminal
            print(f"Testing {k}-Group {orders_list}...".ljust(30), end=" ", flush=True)
            
            # Calculate total parameters: (k-1 thetas) + (betas for each group)
            num_params = (k - 1) + sum([order + 1 for order in orders_list])
            initial_guess = np.zeros(num_params)
            
            # 3. Optimization using the Analytical Jacobian (Speed Upgrade)
            result = minimize(
                calc_dynamic_nll, 
                initial_guess, 
                args=(df, orders_list),
                method='BFGS',
                jac=calc_dynamic_jacobian
            )
            
            if result.success or result.status == 2:
                # Calculate BIC (Subject-level N)
                max_ll = -1 * result.fun
                bic = (-2 * max_ll) + (num_params * np.log(n_subjects))
                
                # Check Group Sizes (The 5% Rule)
                thetas = np.zeros(k)
                if k > 1:
                    thetas[1:] = result.x[0 : k-1]
                pis = np.exp(thetas - logsumexp(thetas))
                min_group_size = np.min(pis) * 100
                
                # Check P-values for highest orders
                try:
                    # Use the Hessian to get Standard Errors
                    se = np.sqrt(np.diag(result.hess_inv))
                except:
                    se = np.full(num_params, np.nan)
                    
                highest_order_pvals = []
                all_significant = True
                current_idx = k - 1
                
                for g in range(k):
                    n_betas = orders_list[g] + 1
                    group_betas = result.x[current_idx : current_idx + n_betas]
                    group_se = se[current_idx : current_idx + n_betas]
                    current_idx += n_betas
                    
                    # Identify the p-value of the highest order term (the last beta)
                    highest_est = group_betas[-1]
                    highest_se = group_se[-1]
                    z_score = highest_est / highest_se if highest_se > 0 else 0
                    p_val = 2 * (1 - norm.cdf(abs(z_score)))
                    highest_order_pvals.append(p_val)
                    
                    # Flag if the highest term fails the user-defined threshold
                    if p_val >= p_val_thresh:
                        all_significant = False
                
                # Logic Gate: Model must meet size and significance requirements to be considered
                if min_group_size < min_group_pct:
                    print(f"REJECTED: Small Group ({min_group_size:.1f}%)")
                elif not all_significant:
                    p_str = "[" + ", ".join([f"{p:.3f}" for p in highest_order_pvals]) + "]"
                    print(f"REJECTED: Non-Sig Order {p_str}")
                else:
                    print(f"VALID! BIC: {bic:.2f} | Min %: {min_group_size:.1f}%")
                    
                    # Update the global winner if this is the lowest BIC found so far
                    if bic < best_bic:
                        best_bic = bic
                        best_orders = orders_list
                        best_result = result
            else:
                print("FAILED: Did not converge.")
                
    return best_orders, best_result



# ==========================================
# EXECUTION BLOCK & CLI
# ==========================================
if __name__ == "__main__":
    # 1. Set up the terminal parser
    parser = argparse.ArgumentParser(description="GBTM Python Engine")
    parser.add_argument('--mode', choices=['single', 'auto'], default='single', 
                        help="Run a 'single' model or 'auto' exhaustive search.")
    parser.add_argument('--orders', type=int, nargs='+', default=[1, 1], 
                        help="Polynomial orders for a single model (e.g., --orders 1 2)")
    
    args = parser.parse_args()

    # 2. Load and prep data
    raw_wide = load_cambridge_data()
    long_data = prep_trajectory_data(raw_wide).dropna(subset=['Time', 'Outcome'])

    # 3. Route the command
    if args.mode == 'auto':
        # Run the exhaustive loop
        winning_orders, winning_result = run_autotraj(long_data, max_groups=3, max_order=2)
        print("\n" + "*"*50)
        print(f"THE WINNER IS: {len(winning_orders)}-Group Model with orders {winning_orders}")
        print("*"*50)
        if winning_result:
            extract_and_print_metrics(winning_result, long_data, winning_orders)

    elif args.mode == 'single':
        # Run just the specific model requested
        model_orders = args.orders
        k = len(model_orders)
        num_params = (k - 1) + sum([order + 1 for order in model_orders])
        initial_guess = np.zeros(num_params)

        print(f"Running SINGLE {k}-Group model with orders {model_orders}...")
        
        result = minimize(
            calc_dynamic_nll, 
            initial_guess, 
            args=(long_data, model_orders),
            method='BFGS',
            jac=calc_dynamic_jacobian  # <--- THE SPEED UPGRADE
        )
        
        if result.success or result.status == 2:
            extract_and_print_metrics(result, long_data, model_orders)
        else:
            print("Failed to converge.")

# ==========================================
    # CLI COMMAND REFERENCE
    # ==========================================
    # To run a single specific model (e.g., 2 groups, Linear and Linear):
    # python main.py --mode single --orders 1 1
    #
    # To run the automated exhaustive search loop:
    # python main.py --mode auto