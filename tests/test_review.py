"""Known-value and synthetic-cohort checks; no camera or clinical data required."""

import json
import itertools
import importlib
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy import stats

from proj.landmark_replay import replay_figure
from proj.motor_metrics import movement_series, extract_metrics, PRIMARY_METRICS
from proj.participants import get_subject_group, read_groups
from proj.review_data import (
    COORDINATES, discover_trials, load_trial, prepare_group_file, apply_recording_exclusions,
)
from proj.study_statistics import (
    compare_groups, holm_adjust, paired_condition_effects, subject_summary,
)


def landmarks(tip_x, frames=None, times=None):
    count = len(tip_x)
    points = np.zeros((count, 21, 3))
    points[:, 9, 0] = 1  # Fixed unit palm reference.
    points[:, 8, 0] = tip_x
    data = pd.DataFrame(points.reshape(count, -1), columns=COORDINATES)
    data['timestamp_ms'] = np.arange(count) * 100 if times is None else times
    data['frame'] = np.arange(count) + 1 if frames is None else frames
    data['hand'] = 0
    data['handedness'] = 'Right'
    data['handedness_score'] = .99
    return data


def save_trial(root, subject='Alice', group='control', experiment='normal_dom',
               data=None, status='completed', name='trial'):
    directory = Path(root) / subject / name
    directory.mkdir(parents=True)
    if data is None:
        data = landmarks(np.linspace(0, 1, 60))
    data.to_csv(directory / 'hand_data.csv', index=False)
    metadata = dict(subject_name=subject, group=group, experiment_type=experiment,
                    frames_recorded=int(data.frame.max()) if len(data) else 0,
                    duration_seconds=float(data.timestamp_ms.max() / 1000) if len(data) else 0,
                    status=status, image_width=640, image_height=480)
    (directory / 'metadata.json').write_text(json.dumps(metadata))
    return directory


class ReviewTests(unittest.TestCase):
    def test_manual_recording_exclusions_include_timing_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            save_trial(directory, name='keep')
            save_trial(directory, name='exclude')
            original = extract_metrics(discover_trials(directory))
            metrics = apply_recording_exclusions(original, {'Alice/exclude': 'Wrong task'})
            self.assertTrue(original.analysis_eligible.all())
            self.assertEqual(len(metrics), 2)
            excluded = metrics.loc[metrics.trial_id == 'Alice/exclude'].iloc[0]
            self.assertFalse(excluded.analysis_eligible)
            self.assertIn('Wrong task', excluded.exclusion_reason)
            self.assertEqual(subject_summary(metrics).trial_count.tolist(), [1])

            # Run the notebook's eligibility logic: usable event timing must not
            # re-enable a recording that was manually excluded.
            metrics['annotation_status'] = 'annotated'
            metrics['task_outcome'] = 'success'
            metrics['key_in_lock_seconds'] = [1., 2.]
            notebook = json.loads((Path(__file__).parents[1] / 'src/proj/review.ipynb').read_text())
            cell = ''.join(notebook['cells'][9]['source']).split('subjects = subject_summary')[0]
            context = dict(trial_metrics=metrics, OUTCOMES=['key_in_lock_seconds'],
                           TASK_OUTCOMES=['key_in_lock_seconds'], SUCCESSFUL_TASKS_ONLY=True)
            exec(cell, context)
            analysis = context['analysis_trials']
            self.assertFalse(analysis.loc[analysis.trial_id == 'Alice/exclude', 'analysis_eligible'].iloc[0])
            self.assertTrue(analysis.loc[analysis.trial_id == 'Alice/keep', 'analysis_eligible'].iloc[0])
            with self.assertRaisesRegex(ValueError, 'Unknown excluded trial IDs'):
                apply_recording_exclusions(original, {'Alice/typo': 'Wrong task'})
            with self.assertRaisesRegex(ValueError, 'nonempty reason'):
                apply_recording_exclusions(original, {'Alice/exclude': ' '})

    def test_excluded_recording_is_not_loaded(self):
        from proj.task_analysis import add_task_metrics, outcome_counts
        with tempfile.TemporaryDirectory() as directory:
            save_trial(directory)
            inventory = discover_trials(directory).assign(analysis_eligible=True, exclusion_reason='')
            inventory = apply_recording_exclusions(inventory, {'Alice/trial': 'Incomplete task'})
            inventory = inventory.drop(columns=['analysis_eligible', 'exclusion_reason'])
            with patch('proj.motor_metrics.load_trial', side_effect=AssertionError('Loaded excluded CSV')):
                metrics = extract_metrics(inventory)
            with patch('proj.task_analysis.load_annotations', side_effect=AssertionError('Loaded excluded annotations')):
                metrics = add_task_metrics(metrics)
            self.assertFalse(metrics.analysis_eligible.iloc[0])
            self.assertEqual(metrics.annotation_status.iloc[0], 'manually excluded')
            self.assertTrue(outcome_counts(metrics).empty)

    def test_trial_sets_legacy_and_partial_resume(self):
        from proj.session import EXPERIMENT_TYPES, create_session, latest_trial_set, remaining_experiments, save_metadata
        with tempfile.TemporaryDirectory() as directory:
            for experiment in EXPERIMENT_TYPES:
                save_trial(directory, experiment=experiment, name=experiment)
            self.assertEqual(latest_trial_set('Alice', directory), 1)
            self.assertEqual(remaining_experiments('Alice', directory), [])
            trial, metadata = create_session('Alice', directory, trial_set=2)
            metadata['status'] = 'completed'
            save_metadata(trial, metadata)
            self.assertEqual(latest_trial_set('Alice', directory), 2)
            remaining = remaining_experiments('Alice', directory)
            self.assertEqual(len(remaining), 3)
            self.assertNotIn(metadata['experiment_type'], remaining)
            interrupted, partial = create_session('Alice', directory)
            partial['status'] = 'interrupted'
            save_metadata(interrupted, partial)
            self.assertEqual(partial['trial_set'], 2)
            self.assertEqual(remaining_experiments('Alice', directory), remaining)

    def test_metrics_known_values_and_gaps(self):
        data = landmarks([0, 1, 2])
        data.handedness = data.handedness.str.lower()
        series, hand = movement_series(data, {'image_width': 640, 'image_height': 480})
        self.assertEqual(hand, 'right')
        np.testing.assert_allclose(series.pinch, [0, 1, 2])
        np.testing.assert_allclose(series.tip_speed.iloc[1:], [10, 10])
        data.frame = [1, 2, 4]
        series, _ = movement_series(data, {'image_width': 640, 'image_height': 480})
        self.assertTrue(np.isnan(series.tip_speed.iloc[2]))
        self.assertEqual(series.valid_step_length.sum(), 1)
        with tempfile.TemporaryDirectory() as directory:
            save_trial(directory, data=landmarks([0, 1, 2]))
            metrics = extract_metrics(discover_trials(directory), min_frames=3,
                                      min_duration_seconds=.1)
            self.assertAlmostEqual(metrics.iloc[0].tip_dispersion_rms, np.sqrt(2 / 3))
            self.assertAlmostEqual(metrics.iloc[0].pinch_sd, 1)
            self.assertAlmostEqual(metrics.iloc[0].tip_speed_mean, 10)
            self.assertTrue(metrics.iloc[0].analysis_eligible)

    def test_aspect_ratio_and_hand_identity(self):
        data = landmarks([0, 0, 0]); data.handedness = 'right'
        data['y8'] = [0, 1, 2]
        series, _ = movement_series(data, {'image_width': 640, 'image_height': 480})
        np.testing.assert_allclose(series.relative_tip_y, [0, .75, 1.5])
        second = data.copy(); second.hand = 1; second.handedness = 'left'
        both = pd.concat([data, second])
        with self.assertRaises(ValueError):
            movement_series(both, {'image_width':640, 'image_height':480})
        chosen, _ = movement_series(both, {'image_width':640, 'image_height':480}, 'right')
        self.assertEqual(len(chosen), 3)
        data.handedness = 'unknown'
        with self.assertRaises(ValueError):
            movement_series(data, {'image_width':640, 'image_height':480})
        movement_series(data, {'image_width':640, 'image_height':480}, 0, allow_legacy=True)

    def test_inventory_qc_and_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            good = save_trial(directory)
            save_trial(directory, subject='Bob', group='', status='cancelled')
            broken = Path(directory) / 'Broken' / 'trial'
            broken.mkdir(parents=True); (broken / 'metadata.json').write_text('{')
            inventory = discover_trials(directory)
            self.assertEqual(len(inventory), 3)
            metrics = extract_metrics(inventory)
            self.assertEqual(int(metrics.analysis_eligible.sum()), 1)
            self.assertTrue(metrics.loc[metrics.subject_name == 'Bob', 'exclusion_reason'].str.contains('cancelled').all())
            prepare_group_file(inventory, directory)
            with patch('builtins.input', return_value='disease'):
                self.assertEqual(get_subject_group('Bob',directory), 'disease')
            self.assertEqual(read_groups(directory)['Bob'], 'disease')
            data = pd.read_csv(good / 'hand_data.csv')
            duplicate = pd.concat([data, data.iloc[[0]]])
            duplicate.to_csv(good / 'hand_data.csv',index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                load_trial(good / 'hand_data.csv')

    def test_subject_not_frame_or_trial_is_unit(self):
        rows = [dict(subject_name=name, group=group, experiment_type='normal_dom',
                     tip_dispersion_rms=value, pinch_sd=value, analysis_eligible=True)
                for name, group, value in [('A','control',1),('A','control',3),
                                           ('B','control',4),('C','disease',6),('D','disease',8)]]
        subjects = subject_summary(pd.DataFrame(rows))
        self.assertEqual(len(subjects),4)
        self.assertEqual(subjects.loc[subjects.subject_name=='A','trial_count'].iloc[0],2)
        comparisons = compare_groups(subjects)
        row = comparisons.iloc[0]
        self.assertEqual(row.n_control,2);self.assertEqual(row.n_disease,2)
        self.assertEqual(row.difference_disease_minus_control,4)
        expected=stats.ttest_ind([6,8],[2,4],equal_var=False)
        self.assertAlmostEqual(row.p_welch,expected.pvalue)
        self.assertAlmostEqual(row.ci_low,expected.confidence_interval().low)
        self.assertTrue(np.isnan(comparisons.loc[comparisons.experiment_type=='blind_dom','p_welch']).all())
        np.testing.assert_allclose(holm_adjust([.01,.04,.03,np.nan]),[.03,.06,.06,np.nan])

    def test_paired_complete_subjects_only(self):
        rows=[]
        for name,group,base,change in [('A','control',1,1),('B','control',2,2),('C','disease',3,3),('D','disease',4,5)]:
            for experiment,value in [('normal_dom',base),('blind_dom',base+change)]:
                rows.append(dict(subject_name=name,group=group,experiment_type=experiment,
                                 tip_dispersion_rms=value,pinch_sd=value,trial_count=1))
        rows.append(dict(subject_name='Unpaired',group='control',experiment_type='normal_dom',
                         tip_dispersion_rms=100,pinch_sd=100,trial_count=1))
        paired,changes=paired_condition_effects(pd.DataFrame(rows))
        selected=paired.loc[(paired.metric=='tip_dispersion_rms') & (paired.contrast=='blind_minus_normal_dom')]
        self.assertEqual(selected.n_pairs.tolist(),[2,2])
        self.assertEqual(selected.mean_change.tolist(),[1.5,4])
        difference=changes.loc[(changes.metric=='tip_dispersion_rms') & (changes.contrast=='blind_minus_normal_dom')].iloc[0]
        self.assertEqual(difference.difference_disease_minus_control,2.5)

    def test_replay_clock_and_missing_detections(self):
        data=landmarks([.1,.2],frames=[1,3],times=[0,200]);data.handedness='right'
        figure=replay_figure(data,fps=10,max_gap_seconds=.05)
        self.assertEqual(len(figure.frames),3)
        self.assertTrue(np.isnan(figure.frames[1].data[0].x).all())
        self.assertEqual(figure.layout.sliders[0].steps[-1].label,'0.20')
        self.assertEqual(figure.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'],100)
        html=figure.to_html(include_plotlyjs=True,auto_play=False)
        self.assertNotIn('<script src="https://cdn.plot.ly/',html)

    def test_recording_saves_group_dimensions_and_handedness(self):
        # Exercise the real writer and recording loop with camera/MediaPipe mocked.
        cv = ModuleType('cv2')
        cv.COLOR_BGR2RGB = 1; cv.WND_PROP_VISIBLE = 1
        for name in ['VideoCapture', 'destroyAllWindows', 'flip', 'cvtColor',
                     'imshow', 'waitKey', 'getWindowProperty']:
            setattr(cv, name, Mock())
        cv.flip.side_effect = lambda frame, _: frame
        cv.cvtColor.side_effect = lambda frame, _: frame
        cap = Mock(); cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        cv.VideoCapture.return_value = cap
        cv.getWindowProperty.return_value = 1
        cv.waitKey.side_effect = [32, -1, 32, -1, 32] * 8 + [ord('q')]
        modules = {'cv2': cv}
        for name in ['mediapipe', 'mediapipe.tasks', 'mediapipe.tasks.python',
                     'mediapipe.tasks.python.vision']:
            modules[name] = ModuleType(name)
        saved = {name: sys.modules.get(name) for name in ('proj.recording', 'proj.tracking')}
        for name in saved:
            sys.modules.pop(name, None)
        try:
            with patch.dict(sys.modules, modules), tempfile.TemporaryDirectory() as directory:
                recording = importlib.import_module('proj.recording')
                points = [SimpleNamespace(x=.1, y=.2, z=.3) for _ in range(21)]
                labels = [[SimpleNamespace(category_name='Right', score=.99)]]
                with patch.object(recording, 'create_landmarker', return_value=Mock()), \
                     patch.object(recording, 'draw_status'), patch.object(recording, 'draw_landmarks'), \
                     patch.object(recording, 'play_countdown_sound'), \
                     patch.object(recording, 'detect_hands', return_value=([points], labels)), \
                     patch.object(recording.time, 'perf_counter', side_effect=itertools.count()):
                    paths = recording.record(subject_name='Alice', group='control', dominant_hand='right',
                                             data_dir=directory, camera_index=2, countdown_seconds=1,
                                             trial_sets=2)
                self.assertEqual(len(paths), 8)
                from proj.session import EXPERIMENT_TYPES, remaining_experiments
                for trial_set in (1, 2):
                    set_metadata = [json.loads(path.with_name('metadata.json').read_text()) for path in paths]
                    self.assertCountEqual([row['experiment_type'] for row in set_metadata
                                           if row['trial_set'] == trial_set], EXPERIMENT_TYPES)
                for path in paths:
                    metadata = json.loads(path.with_name('metadata.json').read_text())
                    self.assertEqual(metadata['group'], 'control')
                    self.assertEqual(metadata['camera_index'], 2)
                    self.assertEqual(metadata['image_width'], 640)
                    self.assertEqual(metadata['image_height'], 480)
                    self.assertEqual(metadata['status'], 'completed')
                    data = load_trial(path)
                    self.assertEqual(len(data), 4)
                    self.assertTrue(data.handedness.eq('right').all())
                    annotation = json.loads(path.with_name('task_annotations.json').read_text())
                    self.assertEqual(annotation['outcome'], 'success')
                    self.assertTrue(annotation['phase_log_complete'])
                    self.assertEqual(annotation['key_in_lock_seconds'], 1)
                    self.assertEqual(metadata['dominant_hand'], 'right')
                    self.assertEqual(metadata['task_hand'], 'left' if metadata['experiment_type'].endswith('_ndom') else 'right')
                    self.assertTrue(data.handedness_score.eq(.99).all())
                    self.assertEqual(data.timestamp_ms.iloc[0], 0)
                cv.VideoCapture.assert_called_once_with(2)
                cap.release.assert_called_once()
                original_metadata = [path.with_name('metadata.json').read_bytes() for path in paths]
                with patch('builtins.input', return_value='no') as prompt:
                    self.assertEqual(recording.record(subject_name='Alice', data_dir=directory), [])
                    prompt.assert_called_once()
                cv.VideoCapture.assert_called_once_with(2)
                cv.waitKey.side_effect = [32, -1, 32, -1, 32] * 4 + [ord('q')]
                with patch('builtins.input', side_effect=['invalid', 'yes']) as prompt, \
                     patch.object(recording, 'create_landmarker', return_value=Mock()), \
                     patch.object(recording, 'draw_status'), patch.object(recording, 'draw_landmarks'), \
                     patch.object(recording, 'play_countdown_sound'), \
                     patch.object(recording, 'detect_hands', return_value=([points], labels)), \
                     patch.object(recording.time, 'perf_counter', side_effect=itertools.count()):
                    appended = recording.record(subject_name='Alice', data_dir=directory, countdown_seconds=1)
                self.assertEqual(prompt.call_count, 2)
                self.assertEqual(len(appended), 4)
                self.assertTrue(set(paths).isdisjoint(appended))
                self.assertTrue(all(json.loads(path.with_name('metadata.json').read_text())['trial_set'] == 3
                                    for path in appended))
                self.assertEqual(original_metadata, [path.with_name('metadata.json').read_bytes() for path in paths])
                self.assertEqual(remaining_experiments('Alice', directory), [])
                for invalid in (0, -1, True, 1.5):
                    with self.assertRaisesRegex(ValueError, 'trial_sets'):
                        recording.record(trial_sets=invalid)
                # Q mid-task retains the partial phase log without consuming a condition.
                cv.waitKey.side_effect = [32, -1, ord('q')]
                with patch.object(recording, 'create_landmarker', return_value=Mock()), \
                     patch.object(recording, 'draw_status'), patch.object(recording, 'draw_landmarks'), \
                     patch.object(recording, 'play_countdown_sound'), \
                     patch.object(recording, 'detect_hands', return_value=([points], labels)), \
                     patch.object(recording.time, 'perf_counter', side_effect=itertools.count()):
                    partial = recording.record(subject_name='Bob', group='disease', dominant_hand='left',
                                               data_dir=directory, countdown_seconds=1)
                self.assertEqual(len(partial), 1)
                metadata = json.loads(partial[0].with_name('metadata.json').read_text())
                self.assertEqual(metadata['status'], 'interrupted')
                self.assertFalse(metadata['phase_log_complete'])
                from proj.session import remaining_experiments
                self.assertEqual(len(remaining_experiments('Bob', directory)), 4)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_lock_and_key_annotations(self):
        from proj.task_analysis import add_task_metrics, save_annotations, validate_annotations, TASK_OUTCOMES
        with tempfile.TemporaryDirectory() as directory:
            trial = save_trial(directory)
            metrics = extract_metrics(discover_trials(directory))
            missing = add_task_metrics(metrics)
            self.assertTrue(missing[['time_to_key_seconds', 'post_key_duration_seconds']].isna().all().all())
            annotation = dict(pickup_start_seconds=1.5, unlock_start_seconds=2,
                              unlock_end_seconds=3, return_end_seconds=4,
                              outcome='success', notes='Observer confirmed opening')
            save_annotations(trial / 'hand_data.csv', annotation, 5.9)
            augmented = add_task_metrics(metrics)
            row = augmented.iloc[0]
            self.assertEqual(row.task_duration_seconds, 2.5)
            self.assertEqual(row.unlock_duration_seconds, 1)
            self.assertEqual(row.task_success, 1)
            self.assertEqual(row.initial_rest_duration_seconds, 1.5)
            self.assertAlmostEqual(row.final_rest_duration_seconds, 1.9)
            self.assertEqual(row.annotation_status, 'annotated')
            self.assertTrue(np.isfinite(row.initial_rest_tip_dispersion_rms))
            self.assertTrue(np.isfinite(row.final_rest_tip_dispersion_rms))
            for invalid in (dict(annotation, unlock_end_seconds=1),
                            dict(annotation, return_end_seconds=7),
                            dict(annotation, outcome='inferred')):
                with self.assertRaises(ValueError):
                    validate_annotations(invalid, 5.9)

    def test_nearly_constant_changes_are_not_spurious_tests(self):
        from proj.study_statistics import _welch
        result = _welch([1, 1 + 1e-15], [2, 2 + 1e-15])
        self.assertTrue(np.isnan(result['p_welch']))
        self.assertIn('zero variance', result['note'])

    def test_single_spacebar_event_and_assumed_success(self):
        from proj.trial_phases import PhaseLog
        log = PhaseLog()
        self.assertFalse(log.advance(2))
        self.assertEqual(log.events, {'key_in_lock_seconds': 2})
        self.assertTrue(log.complete)
        self.assertEqual(log.outcome, 'success')
        self.assertTrue(log.advance(9))
        self.assertEqual(len(log.metadata_events()), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hand_data.csv'
            log.save(path, 9)
            annotation = json.loads(path.with_name('task_annotations.json').read_text())
            self.assertTrue(annotation['success_assumed'])
            self.assertNotIn('unlock_start_seconds', annotation)

    def test_balanced_overall_summary(self):
        from proj.study_statistics import balanced_overall_summary
        conditions = ['normal_dom', 'normal_ndom', 'blind_dom', 'blind_ndom']
        rows = [dict(subject_name='Alice', group='control', experiment_type=condition,
                     tip_dispersion_rms=value, trial_count=count)
                for condition, value, count in zip(conditions, [1, 2, 3, 4], [1, 5, 1, 1])]
        rows.append(dict(subject_name='Incomplete', group='disease', experiment_type='normal_dom',
                         tip_dispersion_rms=100, trial_count=1))
        summary = balanced_overall_summary(pd.DataFrame(rows), ['tip_dispersion_rms'])
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0].tip_dispersion_rms, 2.5)
        self.assertEqual(summary.iloc[0].trial_count, 8)
        result = compare_groups(summary, metrics=['tip_dispersion_rms'], conditions=['all_conditions'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].n_control, 1)
        rows[0]['tip_dispersion_rms'] = np.nan
        summary = balanced_overall_summary(pd.DataFrame(rows), ['tip_dispersion_rms'])
        self.assertTrue(np.isnan(summary.iloc[0].tip_dispersion_rms))

    def test_dominant_hand_registry_and_event_rendering(self):
        from proj.participants import get_dominant_hand, get_subject_group, read_dominant_hands
        from proj.task_analysis import save_annotations, add_task_metrics
        from proj.review_plots import plot_trial_series
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            get_subject_group('Alice', directory, 'control')
            with patch('builtins.input', return_value='left') as prompt:
                self.assertEqual(get_dominant_hand('Alice', directory), 'left')
                prompt.assert_called_once()
            get_subject_group('Bob', directory, 'disease')
            self.assertEqual(read_dominant_hands(directory)['Alice'], 'left')
            with patch('builtins.input', side_effect=AssertionError('Unexpected prompt')):
                self.assertEqual(get_dominant_hand('Alice', directory), 'left')
            trial = save_trial(directory)
            save_annotations(trial / 'hand_data.csv', {'key_in_lock_seconds': 2}, 5.9)
            metrics = add_task_metrics(extract_metrics(discover_trials(directory)))
            row = metrics.iloc[0]
            self.assertEqual(row.time_to_key_seconds, 2)
            self.assertAlmostEqual(row.post_key_duration_seconds, 3.9)
            self.assertEqual(row.task_outcome, 'success')
            data = load_trial(trial / 'hand_data.csv')
            figure = replay_figure(data, key_in_lock_seconds=2)
            event = next(frame for frame in figure.frames if 'KEY IN LOCK' in frame.layout.annotations[0].text)
            self.assertEqual(event.data[0].marker.color, '#d97706')
            self.assertEqual(figure.layout.updatemenus[0].buttons[-1].label, 'Key in lock')
            series, _ = movement_series(data, {'image_width':640,'image_height':480})
            plot = plot_trial_series(series, key_in_lock_seconds=2)
            self.assertTrue(any(line.get_label() == 'Key in lock' for line in plot.axes[1].lines))
            self.assertTrue(any(collection.get_label() == 'Key in lock' for collection in plot.axes[0].collections))
            plt.close(plot)

    def test_widget_renders_one_replay_and_one_plot_per_click(self):
        import matplotlib.pyplot as plt
        from IPython.display import HTML, Image
        from proj.landmark_replay import review_widget, replay_html
        from proj.review_plots import plot_trial_series, display_figure_once
        data = landmarks([0, 1, 2]); data.handedness = 'right'
        series, _ = movement_series(data, {'image_width':640, 'image_height':480})
        existing_figures = plt.get_fignums()
        with plt.ion():
            figure = plot_trial_series(series)
            self.assertEqual(plt.get_fignums(), existing_figures)
            with patch('IPython.display.display') as publish:
                display_figure_once(figure)
                publish.assert_called_once()
                self.assertIsInstance(publish.call_args.args[0], Image)
        html = replay_html(data)
        self.assertTrue(html.startswith('<iframe'))
        self.assertIn('srcdoc=', html)
        self.assertIn('background:#111827', html)
        with tempfile.TemporaryDirectory() as directory:
            save_trial(directory)
            save_trial(directory, experiment='blind_ndom', name='trial-b')
            with patch('IPython.display.display') as publish:
                widget = review_widget(discover_trials(directory))
                button = widget.children[1]
                body = widget.children[2]
                slots = None
                for _ in range(3):
                    button.click()
                    self.assertEqual(len(body.children), 2)
                    replay, image = body.children
                    self.assertEqual(replay.value.count('<iframe'), 1)
                    self.assertTrue(bytes(image.value).startswith(b'\x89PNG'))
                    if slots is not None:
                        self.assertEqual((replay.model_id, image.model_id), slots)
                    slots = (replay.model_id, image.model_id)
                    self.assertEqual(plt.get_fignums(), existing_figures)
                publish.assert_not_called()  # Widget callbacks never broadcast kernel display messages.
                dropdown = widget.children[0]
                dropdown.value = next(value for _, value in dropdown.options if 'trial-b' in value)
                self.assertEqual(len(body.children), 0)
                button.click()
                self.assertEqual(len(body.children), 2)
                self.assertIn('blind_ndom', body.children[0].value)
                self.assertNotIn('normal_dom', body.children[0].value)
                self.assertEqual((body.children[0].model_id, body.children[1].model_id), slots)
                publish.assert_not_called()

    def test_empty_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory=discover_trials(directory)
            metrics=extract_metrics(inventory)
            subjects=subject_summary(metrics)
            self.assertTrue(subjects.empty)
            self.assertEqual(len(compare_groups(subjects)),8)
            paired,changes=paired_condition_effects(subjects)
            self.assertTrue(paired.p_paired.isna().all())
            self.assertTrue(changes.p_welch.isna().all())


if __name__ == '__main__':
    unittest.main()
