"""AutoTraj: Group-Based Trajectory Modeling engine, importable as a library.

This package is a thin, stable public API over ``main.py`` (the engine
module also used directly by the Streamlit UI and test suite). Importing
``autotraj`` pulls in no Streamlit dependency — it's safe to use in scripts,
notebooks, or other applications.

Example:
    >>> import autotraj
    >>> long_df = autotraj.prep_trajectory_data(wide_df)
    >>> model = autotraj.run_single_model(long_df, orders_list=[1, 1])
    >>> assignments = autotraj.get_subject_assignments(model, long_df)

Full mathematical documentation: MATH.md in the repository root.
"""

from main import (
    # Data prep
    prep_trajectory_data,
    extract_flat_arrays,
    extract_joint_flat_arrays,
    load_cambridge_data,
    build_baseline_covariate_matrix,
    extract_tvc_array,
    extract_weights_array,
    # Single-outcome fitting
    run_single_model,
    run_autotraj,
    get_subject_assignments,
    calc_model_adequacy,
    # Joint dual-trajectory fitting
    run_joint_dual_trajectory_model,
    get_joint_subject_assignments,
    calc_joint_model_adequacy,
)

__version__ = "1.0.0"

__all__ = [
    "prep_trajectory_data",
    "extract_flat_arrays",
    "extract_joint_flat_arrays",
    "load_cambridge_data",
    "build_baseline_covariate_matrix",
    "extract_tvc_array",
    "extract_weights_array",
    "run_single_model",
    "run_autotraj",
    "get_subject_assignments",
    "calc_model_adequacy",
    "run_joint_dual_trajectory_model",
    "get_joint_subject_assignments",
    "calc_joint_model_adequacy",
]
