"""Subject-level group comparisons and paired condition effects."""

import numpy as np
import pandas as pd
from scipy import stats

from .motor_metrics import PRIMARY_METRICS
from .session import EXPERIMENT_TYPES

CONTRASTS = {
    "blind_minus_normal_dom": ("blind_dom", "normal_dom"),
    "blind_minus_normal_ndom": ("blind_ndom", "normal_ndom"),
    "ndom_minus_dom_normal": ("normal_ndom", "normal_dom"),
    "ndom_minus_dom_blind": ("blind_ndom", "blind_dom"),
}


def subject_summary(trials, metrics=PRIMARY_METRICS):
    """Average repeat trials per subject/condition; never pool frames as subjects."""
    columns = ["subject_name", "group", "experiment_type", *metrics, "trial_count"]
    eligible = trials.loc[trials.analysis_eligible & trials.group.isin(["control", "disease"])]
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    result = eligible.groupby(["subject_name", "group", "experiment_type"])[list(metrics)].mean()
    result["trial_count"] = eligible.groupby(["subject_name", "group", "experiment_type"]).size()
    return result.reset_index()


def descriptive_statistics(subjects, metrics=PRIMARY_METRICS):
    columns = ["experiment_type", "group", "metric", "n_subjects", "mean", "sd", "median", "q1", "q3"]
    rows = []
    for (experiment, group), sample in subjects.groupby(["experiment_type", "group"]):
        for metric in metrics:
            values = sample[metric].dropna()
            rows.append(dict(experiment_type=experiment, group=group, metric=metric,
                             n_subjects=len(values), mean=values.mean(), sd=values.std(ddof=1),
                             median=values.median(), q1=values.quantile(.25), q3=values.quantile(.75)))
    return pd.DataFrame(rows, columns=columns)


def holm_adjust(pvalues):
    """Holm family-wise correction, preserving unavailable tests as NaN."""
    values = np.asarray(pvalues, dtype=float)
    result = np.full(len(values), np.nan)
    indices = np.flatnonzero(np.isfinite(values))
    order = indices[np.argsort(values[indices])]
    if len(order):
        result[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(order), 0, -1)))
    return result


def _effectively_constant(values):
    tolerance = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
    return np.ptp(values) <= tolerance


def _welch(control, disease):
    control = np.asarray(control, dtype=float)
    disease = np.asarray(disease, dtype=float)
    control = control[np.isfinite(control)]
    disease = disease[np.isfinite(disease)]
    result = dict(n_control=len(control), n_disease=len(disease),
                  difference_disease_minus_control=np.nan, ci_low=np.nan, ci_high=np.nan,
                  hedges_g=np.nan, p_welch=np.nan, p_mannwhitney=np.nan,
                  mannwhitney_method="", note="")
    if len(control) and len(disease):
        result["difference_disease_minus_control"] = float(disease.mean() - control.mean())
    if min(len(control), len(disease)) < 2:
        result["note"] = "Need at least two independent subjects in each group"
        return result
    v_control, v_disease = control.var(ddof=1), disease.var(ddof=1)
    if _effectively_constant(control) and _effectively_constant(disease):
        result["note"] = "Both groups have effectively zero variance; inferential statistics unavailable"
        return result
    # Welch-Satterthwaite calculation also handles a constant sample without
    # catastrophic-cancellation warnings from a third-moment implementation.
    a, b = v_disease / len(disease), v_control / len(control)
    standard_error = np.sqrt(a + b)
    degrees_of_freedom = (a + b) ** 2 / (a ** 2 / (len(disease) - 1) + b ** 2 / (len(control) - 1))
    difference = disease.mean() - control.mean()
    margin = stats.t.ppf(.975, degrees_of_freedom) * standard_error
    dof = len(control) + len(disease) - 2
    pooled = np.sqrt(((len(control) - 1) * v_control + (len(disease) - 1) * v_disease) / dof)
    combined = np.r_[disease, control]
    has_ties = len(np.unique(combined)) < len(combined)
    if has_ties and min(len(control), len(disease)) < 10:
        rank_method = stats.PermutationMethod(n_resamples=9999, rng=np.random.default_rng(42))
        rank_method_name = "permutation (up to 9999, seed 42)"
    else:
        rank_method = "auto"
        rank_method_name = "exact" if min(len(control), len(disease)) <= 8 and not has_ties else "asymptotic"
    result.update(ci_low=float(difference - margin), ci_high=float(difference + margin),
                  hedges_g=float((1 - 3 / (4 * dof - 1)) * (disease.mean() - control.mean()) / pooled),
                  p_welch=float(2 * stats.t.sf(abs(difference / standard_error), degrees_of_freedom)),
                  p_mannwhitney=float(stats.mannwhitneyu(disease, control, alternative="two-sided", method=rank_method).pvalue),
                  mannwhitney_method=rank_method_name)
    if min(len(control), len(disease)) < 5:
        result["note"] = "Small sample; results are exploratory"
    return result


def compare_groups(subjects, metrics=PRIMARY_METRICS, *, conditions=EXPERIMENT_TYPES):
    rows = []
    for experiment in conditions:
        sample = subjects.loc[subjects.experiment_type == experiment]
        for metric in metrics:
            row = dict(experiment_type=experiment, metric=metric)
            row.update(_welch(sample.loc[sample.group == "control", metric],
                              sample.loc[sample.group == "disease", metric]))
            rows.append(row)
    result = pd.DataFrame(rows)
    # Each test family spans all requested metrics and conditions.
    result["p_welch_holm"] = holm_adjust(result.p_welch)
    result["p_mannwhitney_holm"] = holm_adjust(result.p_mannwhitney)
    return result


def balanced_overall_summary(subjects, metrics=PRIMARY_METRICS):
    """Average all four condition means with equal weight per subject.

    Exclude subjects missing a condition; leave a metric unavailable unless all
    four condition values exist. Repeated trials already average within condition.
    """
    rows = []
    for (name, group), sample in subjects.groupby(["subject_name", "group"]):
        sample = sample.loc[sample.experiment_type.isin(EXPERIMENT_TYPES)]
        if set(sample.experiment_type) != set(EXPERIMENT_TYPES):
            continue
        if sample.experiment_type.duplicated().any():
            raise ValueError("Overall summaries require one subject-level row per condition")
        row = dict(subject_name=name, group=group, experiment_type="all_conditions",
                   trial_count=int(sample.trial_count.sum()))
        for metric in metrics:
            values = sample[metric].dropna()
            row[metric] = float(values.mean()) if len(values) == 4 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=["subject_name", "group", "experiment_type", *metrics, "trial_count"])


def paired_condition_effects(subjects, metrics=PRIMARY_METRICS):
    """Complete pairs only; also compare subject-level changes between groups."""
    paired_rows, interaction_rows = [], []
    for metric in metrics:
        wide = subjects.pivot(index=["subject_name", "group"], columns="experiment_type", values=metric)
        for contrast, (positive, negative) in CONTRASTS.items():
            if positive in wide and negative in wide:
                differences = (wide[positive] - wide[negative]).dropna().reset_index(name="difference")
            else:
                differences = pd.DataFrame(columns=["subject_name", "group", "difference"])
            for group in ("control", "disease"):
                values = differences.loc[differences.group == group, "difference"].to_numpy(dtype=float)
                row = dict(metric=metric, contrast=contrast, group=group, n_pairs=len(values),
                           mean_change=float(values.mean()) if len(values) else np.nan,
                           ci_low=np.nan, ci_high=np.nan, p_paired=np.nan, note="")
                if len(values) < 2:
                    row["note"] = "Need at least two complete subject pairs"
                elif _effectively_constant(values):
                    row["note"] = "Zero variance in paired differences"
                else:
                    test = stats.ttest_1samp(values, 0)
                    interval = test.confidence_interval(.95)
                    row.update(ci_low=float(interval.low), ci_high=float(interval.high), p_paired=float(test.pvalue))
                paired_rows.append(row)
            interaction = dict(metric=metric, contrast=contrast)
            interaction.update(_welch(differences.loc[differences.group == "control", "difference"],
                                      differences.loc[differences.group == "disease", "difference"]))
            interaction_rows.append(interaction)
    paired = pd.DataFrame(paired_rows)
    interactions = pd.DataFrame(interaction_rows)
    paired["p_paired_holm"] = holm_adjust(paired.p_paired)
    interactions["p_welch_holm"] = holm_adjust(interactions.p_welch)
    return paired, interactions
