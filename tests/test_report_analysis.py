import unittest
import numpy as np
import pandas as pd
from scipy import stats
from proj.report_analysis import (velocity_outcomes, aggregate_report,
    rank_comparison, condition_changes, paired_tests)


class ReportAnalysisTests(unittest.TestCase):
    def test_velocity_threshold_and_gaps(self):
        # A gap must not create a peak at a segment endpoint.
        series = pd.DataFrame({'tip_speed': [np.nan, 0, 1, 0, .05, 0, np.nan, 2, 0]})
        self.assertEqual(velocity_outcomes(series), (2., 1.))

    def test_outcome_specific_trial_counts(self):
        rows = [dict(subject_name='A',group='control',experiment_type='normal_dom',
                     trial_id=str(i),timing_eligible=True,motion_eligible=i==0,
                     recording_duration_seconds=value,peak_velocity=3.)
                for i,value in enumerate([2.,4.])]
        result=aggregate_report(pd.DataFrame(rows),metrics=['recording_duration_seconds','peak_velocity']).iloc[0]
        self.assertEqual(result.recording_duration_seconds,3.)
        self.assertEqual(result.recording_duration_seconds_n_trials,2)
        self.assertEqual(result.peak_velocity_n_trials,1)
        self.assertEqual(result.peak_velocity,3.)

    def test_mannwhitney_direction(self):
        result=rank_comparison([1,2],[3,4])
        self.assertEqual(result['u'],4.)
        self.assertEqual(result['rank_biserial'],1.)
        self.assertAlmostEqual(result['p'],stats.mannwhitneyu([3,4],[1,2],method='exact').pvalue)

    def test_balanced_changes_and_missing_condition(self):
        subjects=pd.DataFrame([dict(subject_name='A',group='control',experiment_type=c,
                                     duration=v) for c,v in zip(
            ['normal_dom','normal_ndom','blind_dom','blind_ndom'],[1,3,5,9])])
        changes=condition_changes(subjects,metrics=['duration'])
        self.assertEqual(changes.set_index('contrast').loc['delta_vision','difference'],5.)
        self.assertEqual(changes.set_index('contrast').loc['delta_hand','difference'],3.)
        missing=condition_changes(subjects.iloc[:3],metrics=['duration'])
        self.assertNotIn('delta_vision',missing.contrast.values)

    def test_wilcoxon_all_zero_and_nonzero(self):
        changes=pd.DataFrame(dict(subject_name=['A','B','C'],group=['control']*3,
                                  metric=['duration']*3,contrast=['delta_vision']*3,difference=[0.,0.,0.]))
        result=paired_tests(changes).query("group=='control'").iloc[0]
        self.assertEqual(result.p,1.)
        self.assertEqual(result.n_nonzero,0)
        changes['difference']=[1.,2.,3.]
        result=paired_tests(changes).query("group=='control'").iloc[0]
        self.assertEqual(result.rank_biserial,1.)
        self.assertEqual(result.p,.25)

if __name__=='__main__': unittest.main()
