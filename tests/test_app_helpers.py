"""Regression tests for app.py's report-generation helpers.

These are pure functions (no Streamlit runtime required) that were only
smoke-tested ad hoc during development — this file gives them permanent
CI coverage so future edits to app.py can't silently break report
generation without a test failing.
"""
import numpy as np
import pytest

from app import (
    _generate_plain_language_summary,
    _build_html_report,
    _build_pdf_report,
    _make_model_summary_txt,
    get_parameter_estimates_for_ui,
)
from main import (
    run_single_model,
    load_cambridge_data,
    prep_trajectory_data,
    get_subject_assignments,
    calc_model_adequacy,
)


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
