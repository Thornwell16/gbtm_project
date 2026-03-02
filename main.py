import pandas as pd
import numpy as np

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
print("Dataset successfully loaded!")
print(f"The dataset has {cambridge_df.shape[0]} rows and {cambridge_df.shape[1]} columns.\n")
print(cambridge_df.head())