import pandas as pd
import numpy as np
from scipy.optimize import minimize

def load_cambridge_data():
    """
    Loads the Cambridge dataset used for testing logit trajectory models.
    
    Data Source Attribution:
    Dataset provided by Dr. Bobby L. Jones. 
    Originally utilized in the development and demonstration of the SAS PROC TRAJ package.
    Reference: https://www.andrew.cmu.edu/user/bjj/traj/
    """
    df = pd.read_csv("cambridge.txt", sep=r'\s+')
    
    return df
cambridge_df = load_cambridge_data()


def calc_logit_prob(betas, time):
    """
    Calculates the probability of an event using a quadratic logit model.
    betas: A list of 3 numbers [intercept, linear_slope, quadratic_slope]
    time: The time point (T)
    """
    beta0, beta1, beta2 = betas
    
    # 1. Calculate the core equation (z)
    z = beta0 + (beta1 * time) + (beta2 * (time**2))
    
    # 2. Apply the logistic function to turn 'z' into a probability between 0 and 1
    probability = 1 / (1 + np.exp(-z))
    
    return probability

def calc_log_likelihood(betas, actual_times, actual_outcomes):
    """
    Calculates the Bernoulli Log-Likelihood for a set of data points.
    Higher values (closer to 0, since they are negative) mean a better fit.
    """
    total_log_likelihood = 0
    
    # Loop through every single observation in our data
    for i in range(len(actual_times)):
        time = actual_times[i]
        y = actual_outcomes[i]
        
        # 1. Use our previous function to guess the probability
        p = calc_logit_prob(betas, time)
        
        # 2. Prevent the math engine from crashing
        # The natural log of 0 is undefined. If our probability gets too close 
        # to 0 or 1, we nudge it slightly so the computer doesn't crash.
        p = np.clip(p, 1e-10, 1 - 1e-10)
        
        # 3. The Bernoulli Log-Likelihood Calculus
        ll = (y * np.log(p)) + ((1 - y) * np.log(1 - p))
        total_log_likelihood += ll
        
    return total_log_likelihood

def prep_trajectory_data(df):
    """
    Takes a wide-format dataset (one row per subject) and pivots it 
    into a long-format dataset (one row per observation) for the math engine.
    """
    # We use wide_to_long. 
    # stubnames: 'C' is the outcome column prefix, 'T' is the time column prefix
    # i: the unique subject identifier
    # j: the name we want to give to the new "measurement period" column (1 through 23)
    long_df = pd.wide_to_long(
        df, 
        stubnames=['C', 'T'], 
        i='ID', 
        j='Measurement_Period'
    )
    
    # wide_to_long creates a complex multi-index. We reset it to a standard flat table.
    long_df = long_df.reset_index()
    
    # Clean up: Rename 'C' and 'T' to be more descriptive for our math engine
    long_df = long_df.rename(columns={'C': 'Outcome', 'T': 'Time'})
    
    # Sort by ID and Period so it reads chronologically per subject
    long_df = long_df.sort_values(by=['ID', 'Measurement_Period'])
    
    return long_df

# --- Let's test the data pipeline! ---

# 1. Load the raw wide data (from Step 3)
raw_wide_data = load_cambridge_data()

# 2. Pass it through our new pivot function
clean_long_data = prep_trajectory_data(raw_wide_data)

# 3. Print the results to verify
print("\n--- Data Pipeline Test ---")
print("First 10 rows of the mathematical 'Long' dataset:")
print("-" * 50)
print(clean_long_data[['ID', 'Measurement_Period', 'Time', 'Outcome']].head(10))

def negative_log_likelihood(betas, actual_times, actual_outcomes):
    """
    Wrapper function for SciPy. 
    It calculates the log-likelihood and flips the sign so the optimizer can minimize it.
    """
    ll = calc_log_likelihood(betas, actual_times, actual_outcomes)
    return -1 * ll

# --- Let's run the model! ---

# 1. Load and prep the data
raw_wide = load_cambridge_data()
long_data = prep_trajectory_data(raw_wide)

# Standard practice: Drop any rows where the Outcome or Time is missing (NaN)
long_data = long_data.dropna(subset=['Time', 'Outcome'])

# Extract just the columns of numbers we need for the math engine
# .values turns the Pandas column into a flat, fast Numpy array
actual_times = long_data['Time'].values
actual_outcomes = long_data['Outcome'].values

# 2. Set an initial guess for the optimizer (Intercept, Linear, Quadratic)
# Starting at 0.0 is standard practice
initial_guess = [0.0, 0.0, 0.0]

print("Running optimization engine... (this might take a few seconds)...")

# 3. Ask SciPy to find the best betas
# method='BFGS' is a very standard, robust algorithm for solving unconstrained MLE
result = minimize(
    negative_log_likelihood, 
    initial_guess, 
    args=(actual_times, actual_outcomes),
    method='BFGS' 
)

# 4. Print the final results!
print("\n--- 1-Group Quadratic Model Results ---")
print(f"Optimization Success: {result.success}")
print(f"Maximum Log-Likelihood: {-1 * result.fun:.4f}")
print(f"Optimized B0 (Intercept): {result.x[0]:.4f}")
print(f"Optimized B1 (Linear):    {result.x[1]:.4f}")
print(f"Optimized B2 (Quadratic): {result.x[2]:.4f}")