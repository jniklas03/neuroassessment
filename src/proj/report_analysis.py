"""PDF-guided pilot-study report, keeping timing and landmark eligibility separate."""
import numpy as np
import pandas as pd
from scipy import stats, signal
from matplotlib.figure import Figure
from .motor_metrics import movement_series
from .review_data import load_trial, load_metadata
from .study_statistics import holm_adjust, balanced_overall_summary

REPORT_METRICS = ['recording_duration_seconds', 'time_to_key_seconds',
                  'post_key_duration_seconds', 'peak_velocity', 'movement_units']
LABELS = {'recording_duration_seconds': 'Total duration proxy (s)',
          'time_to_key_seconds': 'Time to insertion (s)',
          'post_key_duration_seconds': 'After insertion (s)',
          'peak_velocity': 'Peak relative fingertip speed (palm lengths/s)',
          'movement_units': 'Velocity peaks >10% task maximum (count)'}


def velocity_outcomes(series):
    """Count strict local maxima in contiguous valid speed runs; no gap bridging."""
    speed = series.tip_speed.to_numpy(float)
    finite = np.isfinite(speed)
    if not finite.any():
        return np.nan, np.nan
    maximum = float(speed[finite].max())
    peaks = 0
    for run in np.split(np.flatnonzero(finite), np.flatnonzero(np.diff(np.flatnonzero(finite)) != 1) + 1):
        positions, _ = signal.find_peaks(speed[run])
        peaks += int(np.sum(speed[run][positions] > .1 * maximum))
    return maximum, float(peaks)


def report_trials(trials, hand_selections, *, excluded_subjects=(), subject_aliases=None,
                  min_confidence=.6, max_gap_seconds=.25):
    result = trials.copy()
    result['source_subject_name'] = result.subject_name
    result['subject_name'] = result.subject_name.replace(subject_aliases or {})
    result['report_cohort_eligible'] = (~result.source_subject_name.isin(excluded_subjects)
        & ~result.manually_excluded & result.status.eq('completed')
        & result.error.eq('') & result.group.isin(['control', 'disease']))
    if result.groupby('subject_name').group.nunique().gt(1).any():
        raise ValueError('Subject aliases merge conflicting group assignments')
    result['timing_eligible'] = (result.report_cohort_eligible
        & np.isfinite(result.recording_duration_seconds) & result.recording_duration_seconds.gt(0))
    result['motion_eligible'] = result.report_cohort_eligible & result.analysis_eligible
    result['peak_velocity'] = np.nan
    result['movement_units'] = np.nan
    result['velocity_note'] = ''
    for i, row in result.loc[result.motion_eligible].iterrows():
        try:
            series, _ = movement_series(load_trial(row.csv_path), load_metadata(row),
                hand_selections.get(row.trial_id), min_confidence=min_confidence,
                max_gap_seconds=max_gap_seconds)
            result.loc[i, ['peak_velocity', 'movement_units']] = velocity_outcomes(series)
        except (ValueError, OSError, KeyError, TypeError) as error:
            result.loc[i, 'velocity_note'] = str(error)
    return result


def aggregate_report(trials, metrics=REPORT_METRICS, *, required_trials=None, selected_ids=None):
    """Use explicit selection if given; never choose repeats from outcome values."""
    sample = trials.copy()
    if selected_ids is not None:
        unknown = set(selected_ids) - set(sample.trial_id)
        if unknown:
            raise ValueError(f'Unknown selected trial IDs: {unknown}')
        sample = sample.loc[sample.trial_id.isin(selected_ids)].copy()
    rows = []
    for (name, group, condition), part in sample.groupby(['subject_name', 'group', 'experiment_type']):
        if group not in ['control', 'disease']:
            continue
        row = dict(subject_name=name, group=group, experiment_type=condition,
                   trial_count=int(part.timing_eligible.sum()))
        for metric in metrics:
            eligible = part.motion_eligible if metric in ['peak_velocity', 'movement_units'] else part.timing_eligible
            values = part.loc[eligible, metric].dropna()
            row[metric + '_n_trials'] = len(values)
            row[metric] = values.mean() if len(values) and (required_trials is None or len(values) == required_trials) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=['subject_name','group','experiment_type','trial_count',
        *[key for metric in metrics for key in (metric, metric + '_n_trials')]])


def _bootstrap(values, statistic, seed=42, repetitions=2000):
    rng = np.random.default_rng(seed)
    if any(len(v) < 2 for v in values):
        return np.nan, np.nan
    draws = [statistic(*[rng.choice(v, len(v), replace=True) for v in values]) for _ in range(repetitions)]
    return tuple(np.quantile(draws, [.025, .975]))


def _rank_effect(disease, control):
    # Equivalent to 2U/(n1*n2)-1, including half-credit for ties.
    differences = np.asarray(disease)[:, None] - np.asarray(control)[None, :]
    return float(np.sign(differences).mean())


def rank_comparison(control, disease):
    control, disease = [np.asarray(v, float) for v in (control, disease)]
    control, disease = [v[np.isfinite(v)] for v in (control, disease)]
    row = dict(n_control=len(control), n_disease=len(disease), u=np.nan, p=np.nan,
               rank_biserial=np.nan, effect_ci_low=np.nan, effect_ci_high=np.nan,
               median_difference=np.nan, method='', note='')
    if not len(control) or not len(disease):
        row['note'] = 'Missing group'; return row
    row['median_difference'] = float(np.median(disease)-np.median(control))
    row['rank_biserial'] = _rank_effect(disease, control)
    if min(len(control), len(disease)) < 2:
        row['note'] = 'Fewer than two subjects per group'; return row
    ties = len(np.unique(np.r_[control,disease])) < len(control)+len(disease)
    method = stats.PermutationMethod(n_resamples=9999, rng=np.random.default_rng(42)) if ties and min(len(control),len(disease)) <= 8 else 'auto'
    test = stats.mannwhitneyu(disease, control, alternative='two-sided', method=method)
    row.update(u=float(test.statistic), p=float(test.pvalue),
        method='permutation, seed 42' if not isinstance(method,str) else ('exact' if not ties and min(len(control),len(disease)) <= 8 else 'asymptotic'))
    row['effect_ci_low'], row['effect_ci_high'] = _bootstrap([disease,control], _rank_effect)
    return row


def group_tests(subjects, metrics=REPORT_METRICS):
    rows=[]
    for condition, part in subjects.groupby('experiment_type'):
        for metric in metrics:
            rows.append(dict(experiment_type=condition, metric=metric,
                **rank_comparison(part.loc[part.group.eq('control'),metric], part.loc[part.group.eq('disease'),metric])))
    result=pd.DataFrame(rows)
    result['p_holm']=holm_adjust(result.p) if len(result) else []
    return result


def condition_changes(subjects, metrics=REPORT_METRICS):
    rows=[]
    for metric in metrics:
        wide=subjects.pivot(index=['subject_name','group'],columns='experiment_type',values=metric)
        contrasts={'delta_vision_dom': ('blind_dom','normal_dom'),
                   'delta_vision_ndom': ('blind_ndom','normal_ndom'),
                   'delta_hand_open': ('normal_ndom','normal_dom'),
                   'delta_hand_closed': ('blind_ndom','blind_dom')}
        for (name,group), part in wide.iterrows():
            for contrast,(positive,negative) in contrasts.items():
                delta=part.get(positive,np.nan)-part.get(negative,np.nan)
                if np.isfinite(delta):
                    rows.append(dict(subject_name=name,group=group,metric=metric,contrast=contrast,difference=delta))
            if all(np.isfinite(part.get(c,np.nan)) for c in ['normal_dom','normal_ndom','blind_dom','blind_ndom']):
                for contrast,delta in [('delta_vision',(part.blind_dom+part.blind_ndom-part.normal_dom-part.normal_ndom)/2),
                                       ('delta_hand',(part.normal_ndom+part.blind_ndom-part.normal_dom-part.blind_dom)/2)]:
                    rows.append(dict(subject_name=name,group=group,metric=metric,contrast=contrast,difference=delta))
    return pd.DataFrame(rows,columns=['subject_name','group','metric','contrast','difference'])


def paired_tests(changes):
    rows=[]
    for (metric,contrast), part in changes.groupby(['metric','contrast']):
        for group in ['control','disease','pooled']:
            differences=part.loc[part.group.eq(group) if group!='pooled' else np.ones(len(part),bool),'difference'].to_numpy(float)
            nonzero=differences[differences!=0]
            row=dict(metric=metric,contrast=contrast,group=group,n_pairs=len(differences),
                n_nonzero=len(nonzero), median_change=np.median(differences) if len(differences) else np.nan,
                statistic=np.nan,p=np.nan,rank_biserial=np.nan,change_ci_low=np.nan,change_ci_high=np.nan,method='',note='')
            if len(differences)>=2:
                row['change_ci_low'],row['change_ci_high']=_bootstrap([differences],np.median)
                if len(nonzero):
                    ranks=stats.rankdata(abs(nonzero)); row['rank_biserial']=float(np.sum(np.sign(nonzero)*ranks)/ranks.sum())
                    method=stats.PermutationMethod(n_resamples=9999,rng=np.random.default_rng(42)) if len(differences)<=20 else 'asymptotic'
                    test=stats.wilcoxon(differences,zero_method='wilcox',alternative='two-sided',method=method)
                    row.update(statistic=float(test.statistic),p=float(test.pvalue),method='sign permutations, seed 42' if not isinstance(method,str) else method)
                else:
                    row.update(statistic=0.,p=1.,rank_biserial=0.,method='all differences zero')
            else:
                row['note']='Fewer than two complete pairs'
            rows.append(row)
    result=pd.DataFrame(rows)
    result['p_holm']=holm_adjust(result.p) if len(result) else []
    return result


def change_tests(changes):
    rows=[]
    for (metric,contrast),part in changes.groupby(['metric','contrast']):
        rows.append(dict(metric=metric,contrast=contrast,**rank_comparison(
            part.loc[part.group.eq('control'),'difference'],part.loc[part.group.eq('disease'),'difference'])))
    result=pd.DataFrame(rows); result['p_holm']=holm_adjust(result.p) if len(result) else []
    return result


def report_figures(subjects, overall, changes, metrics=REPORT_METRICS):
    figures={}
    conditions=['normal_dom','normal_ndom','blind_dom','blind_ndom']
    colors=['#2563eb','#e76f51']
    for metric in metrics:
        fig=Figure(figsize=(13,4)); axes=fig.subplots(1,3)
        for x,group in enumerate(['control','disease']):
            values=overall.loc[overall.group.eq(group),metric].dropna().to_numpy()
            if len(values):
                axes[0].boxplot([values],positions=[x],widths=.4,showfliers=False)
                axes[0].scatter(x+np.random.default_rng(42).uniform(-.07,.07,len(values)),values,color=colors[x])
            wide=subjects.loc[subjects.group.eq(group)].pivot(index='subject_name',columns='experiment_type',values=metric)
            for _,part in wide.iterrows():
                axes[x+1].plot(range(4),[part.get(c,np.nan) for c in conditions],'o-',alpha=.55,color=colors[x])
            axes[x+1].set_xticks(range(4),['Open D','Open ND','Closed D','Closed ND'],rotation=25)
            axes[x+1].set_title(['Healthy controls','MS'][x])
        axes[0].set_xticks([0,1],['Healthy controls','MS']); axes[0].set_title('Equal-weight four-condition mean')
        for ax in axes: ax.set_ylabel(LABELS[metric]); ax.grid(axis='y',alpha=.2)
        fig.tight_layout(); figures[metric]=fig
    fig=Figure(figsize=(12,4)); axes=fig.subplots(1,2)
    for ax,contrast in zip(axes,['delta_vision','delta_hand']):
        part=changes.loc[changes.metric.eq('recording_duration_seconds') & changes.contrast.eq(contrast)]
        for x,group in enumerate(['control','disease']):
            values=part.loc[part.group.eq(group),'difference'].to_numpy()
            ax.scatter(x+np.random.default_rng(42).uniform(-.07,.07,len(values)),values,color=colors[x])
            if len(values): ax.plot([x-.15,x+.15],[np.median(values)]*2,color='black')
        ax.axhline(0,color='gray',ls='--'); ax.set_xticks([0,1],['Healthy controls','MS'])
        ax.set_title(contrast); ax.set_ylabel('Within-person duration change (s)')
    fig.tight_layout(); figures['duration_changes']=fig
    fig=Figure(figsize=(12,7)); axes=fig.subplots(2,2)
    for row,group in enumerate(['control','disease']):
        wide=subjects.loc[subjects.group.eq(group)].pivot(index='subject_name',columns='experiment_type',values='recording_duration_seconds')
        for col,hand in enumerate(['dom','ndom']):
            ax=axes[row,col]
            for _,part in wide.iterrows():
                values=[part.get('normal_'+hand,np.nan),part.get('blind_'+hand,np.nan)]
                if np.isfinite(values).all(): ax.plot([0,1],values,'o-',alpha=.55,color=colors[row])
            ax.set_xticks([0,1],['Eyes open','Eyes closed'])
            ax.set_title(('Healthy controls' if group=='control' else 'MS')+' / '+('dominant' if hand=='dom' else 'nondominant'))
            ax.set_ylabel('Total duration proxy (s)'); ax.grid(axis='y',alpha=.2)
    fig.tight_layout(); figures['paired_vision_duration']=fig
    return figures
