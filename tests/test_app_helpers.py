"""Regression tests for app.py's report-generation helpers.

These are pure functions (no Streamlit runtime required) that were only
smoke-tested ad hoc during development — this file gives them permanent
CI coverage so future edits to app.py can't silently break report
generation without a test failing.
"""
import ast

import numpy as np
import pandas as pd
import pytest

from app import (
    _generate_plain_language_summary,
    _generate_joint_plain_language_summary,
    _build_html_report,
    _build_pdf_report,
    _build_reproducible_script,
    _build_joint_reproducible_script,
    _build_joint_html_report,
    _build_joint_pdf_report,
    _make_model_summary_txt,
    get_parameter_estimates_for_ui,
    get_joint_parameter_estimates_for_ui,
    _obs_vs_est_figure,
    _suggest_distribution,
)
from main import (
    run_single_model,
    run_joint_dual_trajectory_model,
    load_cambridge_data,
    prep_trajectory_data,
    get_subject_assignments,
    get_joint_subject_assignments,
    calc_model_adequacy,
    calc_joint_model_adequacy,
    _joint_layout,
)
from tests.simulate import simulate_joint_two_outcome_trajectories


@pytest.fixture(scope="module")
def fitted_model():
    """A real 2-group LOGIT fit on the Cambridge dataset, shared across tests
    in this file (fitting is the expensive part; report generation is not)."""
    df = load_cambridge_data()
    long_df = prep_trajectory_data(df)
    model = run_single_model(long_df, orders_list=[1, 2], n_starts=3)
    assert model['result'].success or model['result'].status == 2
    group_names = ['Low', 'High']
    assignments = get_subject_assignments(model, long_df)
    adq_df, rel_entropy = calc_model_adequacy(assignments, model['pis'], group_names)
    return {
        'model': model, 'long_df': long_df, 'group_names': group_names,
        'assignments': assignments, 'adq_df': adq_df, 'rel_entropy': rel_entropy,
    }


@pytest.fixture(scope="module")
def fitted_joint_model():
    """A real joint dual-trajectory fit (2x2, LOGIT+CNORM) with a deliberately
    non-independent, concordant pi_gh (high-Y paired with high-Z more than
    chance) -- shared across tests in this file."""
    group_params_y = [{'betas': [-1.5]}, {'betas': [1.0, -0.3]}]
    group_params_z = [
        {'betas': [2.0], 'sigma': 1.0, 'cnorm_min': 0.0, 'cnorm_max': 10.0},
        {'betas': [7.0, -0.3], 'sigma': 1.0, 'cnorm_min': 0.0, 'cnorm_max': 10.0},
    ]
    pi_gh_concordant = np.array([[0.45, 0.05], [0.05, 0.45]])
    df_y, df_z, truth = simulate_joint_two_outcome_trajectories(
        n_subjects=400, time_points_y=np.linspace(-1, 1, 8), time_points_z=np.linspace(-1, 1, 8),
        group_params_y=group_params_y, group_params_z=group_params_z,
        pi_gh=pi_gh_concordant, dist_y='LOGIT', dist_z='CNORM', seed=42,
    )
    model_j = run_joint_dual_trajectory_model(
        df_y, df_z, orders_y=[0, 1], orders_z=[0, 1], dist_y='LOGIT', dist_z='CNORM',
        cnorm_min_z=0.0, cnorm_max_z=10.0, n_starts=3,
    )
    assert model_j['pis_joint'] is not None
    k_y, k_z = model_j['k_y'], model_j['k_z']
    group_names_y = [f'Y-Group {g+1}' for g in range(k_y)]
    group_names_z = [f'Z-Group {h+1}' for h in range(k_z)]
    assignments_df_j = get_joint_subject_assignments(model_j, df_y, df_z)
    joint_adq_df, joint_rel_entropy, y_adq_df, y_rel_entropy, z_adq_df, z_rel_entropy = calc_joint_model_adequacy(
        assignments_df_j, model_j['pis_joint'], group_names_y, group_names_z
    )
    param_df_j = get_joint_parameter_estimates_for_ui(model_j)
    return {
        'model': model_j, 'df_y': df_y, 'df_z': df_z,
        'group_names_y': group_names_y, 'group_names_z': group_names_z,
        'assignments': assignments_df_j, 'param_df': param_df_j,
        'joint_adq_df': joint_adq_df, 'joint_rel_entropy': joint_rel_entropy,
        'y_adq_df': y_adq_df, 'y_rel_entropy': y_rel_entropy,
        'z_adq_df': z_adq_df, 'z_rel_entropy': z_rel_entropy,
    }


def test_plain_language_summary_mentions_every_group(fitted_model):
    summary = _generate_plain_language_summary(
        fitted_model['model'], fitted_model['group_names'], fitted_model['long_df'],
        fitted_model['adq_df'], fitted_model['rel_entropy'], 'LOGIT',
    )
    for name in fitted_model['group_names']:
        assert name in summary
    assert "relative entropy" in summary.lower()
    assert any(d in summary for d in ("stable", "increasing", "decreasing"))


def test_plain_language_summary_handles_single_group(fitted_model):
    """K=1 has no other group to rank against — must not divide by zero or crash."""
    df_long = fitted_model['long_df']
    model = run_single_model(df_long, orders_list=[1], n_starts=2)
    assert model['result'].success or model['result'].status == 2
    assignments = get_subject_assignments(model, df_long)
    adq_df, rel_entropy = calc_model_adequacy(assignments, model['pis'], ['Everyone'])
    summary = _generate_plain_language_summary(model, ['Everyone'], df_long, adq_df, rel_entropy, 'LOGIT')
    assert "Everyone" in summary
    assert "1 distinct trajectory group" in summary


def test_html_report_is_well_formed_and_includes_all_sections(fitted_model):
    model, group_names = fitted_model['model'], fitted_model['group_names']
    estimates_df = get_parameter_estimates_for_ui(model, group_names)
    summary_txt = _make_model_summary_txt(model, group_names, fitted_model['rel_entropy'])
    plain_summary = _generate_plain_language_summary(
        model, group_names, fitted_model['long_df'], fitted_model['adq_df'],
        fitted_model['rel_entropy'], 'LOGIT',
    )
    html = _build_html_report(
        model, group_names, estimates_df, fitted_model['adq_df'], fitted_model['rel_entropy'],
        summary_txt, ['y = 1.0 + 2.0t'], png_bytes=None, plain_summary=plain_summary,
    )
    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    assert "AutoTraj Model Report" in html
    assert "Plain-Language Summary" in html
    assert "Parameter Estimates" in html
    assert "Model Adequacy" in html
    # The plain-language summary's markdown must have been converted, not left raw.
    assert "**" not in html.split("<h2>Model Summary</h2>")[0]


def test_pdf_report_produces_valid_pdf_bytes(fitted_model):
    model, group_names = fitted_model['model'], fitted_model['group_names']
    estimates_df = get_parameter_estimates_for_ui(model, group_names)
    summary_txt = _make_model_summary_txt(model, group_names, fitted_model['rel_entropy'])
    plain_summary = _generate_plain_language_summary(
        model, group_names, fitted_model['long_df'], fitted_model['adq_df'],
        fitted_model['rel_entropy'], 'LOGIT',
    )
    pdf_bytes = _build_pdf_report(
        model, group_names, estimates_df, fitted_model['adq_df'], fitted_model['rel_entropy'],
        summary_txt, [r'\text{Group 1}: y = 1.0 + 2.0t'], png_bytes=None, plain_summary=plain_summary,
    )
    assert pdf_bytes[:5] == b'%PDF-'
    assert len(pdf_bytes) > 500  # sanity check against a truncated/empty document


def test_pdf_report_embeds_a_plot_image(fitted_model):
    """A non-trivial PNG payload should measurably increase PDF size."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io

    model, group_names = fitted_model['model'], fitted_model['group_names']
    estimates_df = get_parameter_estimates_for_ui(model, group_names)
    summary_txt = _make_model_summary_txt(model, group_names, fitted_model['rel_entropy'])

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 2, 3])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)

    pdf_without = _build_pdf_report(
        model, group_names, estimates_df, fitted_model['adq_df'], fitted_model['rel_entropy'],
        summary_txt, [], png_bytes=None,
    )
    pdf_with = _build_pdf_report(
        model, group_names, estimates_df, fitted_model['adq_df'], fitted_model['rel_entropy'],
        summary_txt, [], png_bytes=buf.getvalue(),
    )
    assert pdf_with[:5] == b'%PDF-'
    assert len(pdf_with) > len(pdf_without)


# ---------------------------------------------------------------------------
# _suggest_distribution
# ---------------------------------------------------------------------------

def test_suggest_distribution_binary():
    s = pd.Series([0, 1, 0, 1, 1, 0, 0, 1] * 20)
    result = _suggest_distribution(s)
    assert result['suggestion'] == 'LOGIT'
    assert result['confidence'] == 'high'


def test_suggest_distribution_count_no_excess_zeros():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.poisson(lam=4.0, size=2000))
    result = _suggest_distribution(s)
    assert result['suggestion'] == 'POISSON'


def test_suggest_distribution_zero_inflated_count():
    rng = np.random.default_rng(2)
    base = rng.poisson(lam=3.0, size=2000)
    extra_zeros = np.zeros(800, dtype=int)
    s = pd.Series(np.concatenate([base, extra_zeros]))
    result = _suggest_distribution(s)
    assert result['suggestion'] == 'ZIP'
    assert result['stats']['p0_observed'] > result['stats']['p0_poisson_implied']


def test_suggest_distribution_continuous():
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(loc=5.0, scale=2.0, size=1000))
    result = _suggest_distribution(s)
    assert result['suggestion'] == 'CNORM'


def test_suggest_distribution_continuous_with_floor_ceiling_spikes():
    # scale=5 vs. a clip range of width 10 forces a substantial fraction of
    # mass to pile up at each boundary (~15-16% here), reliably clearing the
    # 5% floor/ceiling-spike threshold regardless of the exact random draw.
    rng = np.random.default_rng(4)
    s = pd.Series(np.clip(rng.normal(loc=5.0, scale=5.0, size=1000), 0, 10))
    result = _suggest_distribution(s)
    assert result['suggestion'] == 'CNORM'
    assert result['stats']['pct_at_min'] > 5.0 and result['stats']['pct_at_max'] > 5.0
    assert result['confidence'] == 'high'


def test_suggest_distribution_matches_cambridge_binary_outcome():
    """Real-data sanity check: Cambridge's binary conviction outcome should
    suggest LOGIT, matching how the dataset is actually modeled."""
    df = load_cambridge_data()
    long_df = prep_trajectory_data(df)
    result = _suggest_distribution(long_df['Outcome'])
    assert result['suggestion'] == 'LOGIT'


# ---------------------------------------------------------------------------
# Reproducible script export
# ---------------------------------------------------------------------------

def test_reproducible_script_is_valid_python_and_reflects_the_model(fitted_model):
    script = _build_reproducible_script(fitted_model['model'])
    ast.parse(script)  # raises SyntaxError if malformed
    assert "import autotraj" in script
    assert "run_single_model" in script
    assert repr(fitted_model['model']['orders']) in script
    assert repr(fitted_model['model'].get('dist', 'LOGIT')) in script


def test_joint_reproducible_script_is_valid_python_and_reflects_the_model(fitted_joint_model):
    model_j = fitted_joint_model['model']
    script = _build_joint_reproducible_script(model_j)
    ast.parse(script)
    assert "import autotraj" in script
    assert "run_joint_dual_trajectory_model" in script
    assert repr(model_j['orders_y']) in script
    assert repr(model_j['orders_z']) in script
    assert repr(model_j['dist_z']) in script  # CNORM
    assert "cnorm_min_z" in script  # CNORM bounds must be threaded through


# ---------------------------------------------------------------------------
# Joint plain-language summary
# ---------------------------------------------------------------------------

def test_joint_plain_language_summary_mentions_both_outcomes_and_comorbidity(fitted_joint_model):
    fj = fitted_joint_model
    summary = _generate_joint_plain_language_summary(
        fj['model'], fj['group_names_y'], fj['group_names_z'], fj['df_y'], fj['df_z'],
        fj['y_adq_df'], fj['y_rel_entropy'], fj['z_adq_df'], fj['z_rel_entropy'],
    )
    assert "Outcome Y" in summary and "Outcome Z" in summary
    assert "Comorbidity" in summary
    for name in fj['group_names_y'] + fj['group_names_z']:
        assert name in summary


def test_joint_plain_language_summary_identifies_the_true_concordant_association(fitted_joint_model):
    """The fixture's true pi_gh is concordant (Y-Group2/Z-Group2 co-occur far
    more than chance) -- the summary's identified strongest positive
    association must be a genuinely over-represented cell, not an
    arbitrary/incorrect one."""
    fj = fitted_joint_model
    pis_joint = fj['model']['pis_joint']
    marginal_y, marginal_z = pis_joint.sum(axis=1), pis_joint.sum(axis=0)
    independent = np.outer(marginal_y, marginal_z)
    ratio = pis_joint / independent
    g_max, h_max = np.unravel_index(np.argmax(ratio), ratio.shape)
    assert ratio[g_max, h_max] > 1.0, "The identified 'strongest positive association' cell must actually be over-represented relative to independence"

    summary = _generate_joint_plain_language_summary(
        fj['model'], fj['group_names_y'], fj['group_names_z'], fj['df_y'], fj['df_z'],
        fj['y_adq_df'], fj['y_rel_entropy'], fj['z_adq_df'], fj['z_rel_entropy'],
    )
    assert fj['group_names_y'][g_max] in summary
    assert fj['group_names_z'][h_max] in summary


# ---------------------------------------------------------------------------
# Joint HTML/PDF reports
# ---------------------------------------------------------------------------

def test_joint_html_report_is_well_formed_and_includes_all_sections(fitted_joint_model):
    fj = fitted_joint_model
    model_j = fj['model']
    k_y, k_z = model_j['k_y'], model_j['k_z']
    _, y_beta_start, z_beta_start, _, _, _ = _joint_layout(
        k_y, k_z, model_j['orders_y'], model_j['orders_z'],
        model_j['use_dropout_y'], model_j['dist_y'], model_j['use_dropout_z'], model_j['dist_z'],
    )
    from types import SimpleNamespace
    result_x = model_j['result'].x
    model_y_view = {'orders': model_j['orders_y'], 'n_mix': 1,
                     'result': SimpleNamespace(x=np.concatenate([np.zeros(k_y - 1), result_x[y_beta_start:z_beta_start]]))}
    model_z_view = {'orders': model_j['orders_z'], 'n_mix': 1,
                     'result': SimpleNamespace(x=np.concatenate([np.zeros(k_z - 1), result_x[z_beta_start:]]))}
    assignments_y_view = fj['assignments'].rename(columns={
        **{f'Y_Group_{g+1}_Prob': f'Group_{g+1}_Prob' for g in range(k_y)}, 'Assigned_Group_Y': 'Assigned_Group'})
    assignments_z_view = fj['assignments'].rename(columns={
        **{f'Z_Group_{h+1}_Prob': f'Group_{h+1}_Prob' for h in range(k_z)}, 'Assigned_Group_Z': 'Assigned_Group'})
    fig_y = _obs_vs_est_figure(fj['df_y'], assignments_y_view, model_y_view, fj['group_names_y'], model_j['dist_y'])
    fig_z = _obs_vs_est_figure(fj['df_z'], assignments_z_view, model_z_view, fj['group_names_z'], model_j['dist_z'])

    plain_summary = _generate_joint_plain_language_summary(
        model_j, fj['group_names_y'], fj['group_names_z'], fj['df_y'], fj['df_z'],
        fj['y_adq_df'], fj['y_rel_entropy'], fj['z_adq_df'], fj['z_rel_entropy'],
    )
    html = _build_joint_html_report(
        model_j, fj['group_names_y'], fj['group_names_z'], model_j['pis_joint'], fj['param_df'],
        fj['joint_adq_df'], fj['joint_rel_entropy'], fj['y_adq_df'], fj['y_rel_entropy'],
        fj['z_adq_df'], fj['z_rel_entropy'], plain_summary, fig_y, fig_z,
    )
    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    assert "Joint Dual-Trajectory Model Report" in html
    assert "Plain-Language Summary" in html
    assert "Fitted Trajectories" in html
    assert "Parameter Estimates" in html
    assert "Model Adequacy" in html


def test_joint_pdf_report_produces_valid_pdf_bytes(fitted_joint_model):
    fj = fitted_joint_model
    plain_summary = _generate_joint_plain_language_summary(
        fj['model'], fj['group_names_y'], fj['group_names_z'], fj['df_y'], fj['df_z'],
        fj['y_adq_df'], fj['y_rel_entropy'], fj['z_adq_df'], fj['z_rel_entropy'],
    )
    pdf_bytes = _build_joint_pdf_report(
        fj['model'], fj['group_names_y'], fj['group_names_z'], fj['model']['pis_joint'], fj['param_df'],
        fj['joint_adq_df'], fj['joint_rel_entropy'], fj['y_adq_df'], fj['y_rel_entropy'],
        fj['z_adq_df'], fj['z_rel_entropy'], plain_summary,
    )
    assert pdf_bytes[:5] == b'%PDF-'
    assert len(pdf_bytes) > 500
