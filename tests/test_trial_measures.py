"""Tests for trial_measures.py: marker track, RT / MT / onset / end, flags, summaries (synthetic reaches).

    venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trial_measures as tm  # noqa: E402

FPS = 60.0


def min_jerk(t, t0, duration, start, end):
    """Minimum-jerk reach from `start` to `end` between t0 and t0 + duration (positions at times t)."""
    tau = np.clip((np.asarray(t, float) - t0) / duration, 0, 1)[:, None]
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return np.asarray(start, float) + s * (np.asarray(end, float) - np.asarray(start, float))


def reach(stim_s=1.0, onset_s=1.25, duration=0.5, total_s=2.5, start=(100, 400), end=(400, 400)):
    t = np.arange(int(total_s * FPS)) / FPS
    return t, min_jerk(t, onset_s, duration, start, end)


class MarkerTrack(unittest.TestCase):
    def test_visibility_and_glitches(self):
        n = 40
        pose = np.zeros((n, 33, 3))
        pose[:, 16, 0] = np.linspace(100, 140, n)     # right wrist moves slowly
        pose[:, 16, 1] = 300
        pose[:, 16, 2] = 0.9
        pose[5, 16, 2] = 0.1                           # not sure -> NaN
        pose[20, 16, :2] = [900, 900]                  # one-frame jump -> NaN (a glitch)
        xy = tm.marker_track(pose, 'upper', 'right', image_size=[1920, 1080])
        self.assertTrue(np.isnan(xy[5]).all())
        self.assertTrue(np.isnan(xy[20]).all())
        self.assertEqual(int(np.isnan(xy[:, 0]).sum()), 2)
        np.testing.assert_allclose(xy[10], pose[10, 16, :2])
        # Without an image size there is no glitch test
        xy = tm.marker_track(pose, 'upper', 'right')
        np.testing.assert_allclose(xy[20], [900, 900])
        # Lower limb uses the ankle (27 left / 28 right)
        pose[:, 27, :] = [50, 60, 0.8]
        np.testing.assert_allclose(tm.marker_track(pose, 'lower', 'left')[0], [50, 60])


class Kinematics(unittest.TestCase):
    def test_too_little_data(self):
        t, xy = reach()
        bad = xy.copy()
        bad[::2] = np.nan
        bad[1::4] = np.nan                          # < 50% seen
        self.assertIsNone(tm.trial_kinematics(t, bad, 1.0))
        self.assertIsNone(tm.trial_kinematics(t[:4], xy[:4], 0.0))

    def test_resampled_to_100hz(self):
        t, xy = reach()
        k = tm.trial_kinematics(t, xy, 1.0)
        self.assertAlmostEqual(k['time'][1] - k['time'][0], 0.01)
        self.assertAlmostEqual(k['time'][0], t[0])
        self.assertLessEqual(k['time'][-1], t[-1] + 1e-9)
        self.assertEqual(k['xyz'].shape, (len(k['time']), 2))
        self.assertEqual(k['stim'], 1.0)
        # Peak speed of a minimum-jerk reach: 1.875 * distance / duration
        self.assertAlmostEqual(k['tangvel'].max(), 1.875 * 300 / 0.5, delta=0.05 * 1125)


class Measures(unittest.TestCase):
    def test_reach(self):
        """Light at 1.0 s, a 0.5 s reach starting at 1.25 s: RT ~0.28 s (5% speed threshold), MT ~0.44 s."""
        t, xy = reach()
        k = tm.trial_kinematics(t, xy, 1.0)
        m = tm.calculate_measures(k, np.array([400, 400]), home_pos=np.array([100, 400]))
        self.assertTrue(0.25 <= m['responsetime'] <= 0.31, m['responsetime'])
        self.assertAlmostEqual(m['movementtime'], 0.44, delta=0.05)
        self.assertAlmostEqual(m['onset_s'], 1.0 + m['responsetime'])
        self.assertAlmostEqual(m['end_s'], m['onset_s'] + m['movementtime'])
        self.assertAlmostEqual(m['peak_s'], 1.5, delta=0.02)
        self.assertAlmostEqual(m['AT'] + m['DT'], m['movementtime'])
        self.assertAlmostEqual(m['AT_over_DT'], 1.0, delta=0.15)
        self.assertEqual(m['numpeaks'], 1)
        self.assertLess(m['straighlinedeviation'], 1.0)
        self.assertLess(m['endpointdistance'], 15)
        self.assertFalse(m['anticipation'])
        # The exact values of the current method (regression: any change of the method shows here)
        self.assertAlmostEqual(m['responsetime'], 0.28, places=6)
        self.assertAlmostEqual(m['movementtime'], 0.45, places=6)
        self.assertAlmostEqual(m['AT'], 0.23, places=6)
        self.assertAlmostEqual(m['DT'], 0.22, places=6)
        self.assertAlmostEqual(m['endpointdistance'], 0.244978, places=5)

    def test_rt_follows_the_light(self):
        """Same movement, light 0.1 s later -> RT 0.1 s shorter, MT unchanged."""
        t, xy = reach()
        a = tm.calculate_measures(tm.trial_kinematics(t, xy, 1.0), np.array([400, 400]))
        b = tm.calculate_measures(tm.trial_kinematics(t, xy, 1.1), np.array([400, 400]))
        self.assertAlmostEqual(a['responsetime'] - b['responsetime'], 0.1, places=6)
        self.assertAlmostEqual(a['movementtime'], b['movementtime'], places=6)

    def test_two_peaks(self):
        """A reach with a correction (stop half way, then on) has 2 speed peaks; MT ends at the first stop."""
        t = np.arange(int(3 * FPS)) / FPS
        xy = min_jerk(t, 1.2, 0.4, (0, 0), (150, 0)) + min_jerk(t, 1.9, 0.4, (0, 0), (150, 0))
        k = tm.trial_kinematics(t, xy, 1.0)
        m = tm.calculate_measures(k, np.array([300, 0]))
        self.assertEqual(m['numpeaks'], 1)          # counted between onset and end of the first movement
        self.assertLess(m['end_s'], 1.9)
        self.assertAlmostEqual(m['endpointdistance'], 150, delta=10)

    def test_no_movement(self):
        t = np.arange(120) / FPS
        k = tm.trial_kinematics(t, np.tile([[10.0, 10.0]], (120, 1)), 1.0)
        m = tm.calculate_measures(k, np.array([0, 0]))
        self.assertTrue(np.isnan(m['responsetime']))
        self.assertTrue(np.isnan(tm.calculate_measures(None, None)['movementtime']))

    def test_anticipation(self):
        """The reach started 0.1 s before the light, from home, toward the target -> anticipation."""
        t, xy = reach(stim_s=1.0, onset_s=0.9)
        k = tm.trial_kinematics(t, xy, 1.0)
        m = tm.calculate_measures(k, np.array([400, 400]), home_pos=np.array([100, 400]))
        self.assertTrue(m['anticipation'])
        self.assertTrue(70 <= m['anticipation_ms'] <= 100, m['anticipation_ms'])
        # Without a home position it can't be told
        self.assertFalse(tm.calculate_measures(k, np.array([400, 400]))['anticipation'])

    def test_3d(self):
        t = np.arange(int(2.5 * FPS)) / FPS
        xyz = min_jerk(t, 1.25, 0.5, (0, 0, 0), (30, 0, 40))
        m = tm.calculate_measures(tm.trial_kinematics(t, xyz, 1.0), np.array([30, 0, 40]))
        self.assertAlmostEqual(m['responsetime'], 0.28, places=6)
        self.assertLess(m['endpointdistance'], 2)


SWITCHES = np.array([[400, 400], [250, 200], [100, 100], [500, 150], [600, 300]], float)


def trial_row(target, on, off, start, outcome='hit', limb='upper', side='right'):
    return dict(Part='pre_test', Block=1, Sequence=1, Trial=1, Target=target, Limb=limb, Side=side,
                Outcome=outcome, on_frame=on, off_frame=off, window_start=start)


class MeasureTrials(unittest.TestCase):
    def setUp(self):
        t, xy = reach()
        self.t_ms = t * 1000
        self.xy = xy

    def test_one_source(self):
        trials = [trial_row(1, 60, 140, 0), trial_row(1, None, None, 0, 'not detected')]
        src = dict(name='camera A', track={('upper', 'right'): self.xy}, switch_pos=SWITCHES,
                   switch_size=np.full(5, 40.0))
        tm.measure_trials(trials, self.t_ms, [src])
        r = trials[0]
        self.assertEqual(r['Data'], 'camera A')
        self.assertTrue(r['touse'])
        self.assertAlmostEqual(r['responsetime'], 0.28, places=6)
        self.assertFalse(trials[1]['touse'])
        self.assertTrue(np.isnan(trials[1]['responsetime']))

    def test_fallback_source(self):
        """No marker in the first source -> the second one is used, its distances are dropped (other units)."""
        empty = np.full_like(self.xy, np.nan)
        s3d = dict(name='3D', track={('upper', 'right'): empty}, switch_pos=SWITCHES, switch_size=np.ones(5))
        sa = dict(name='camera A', track={('upper', 'right'): self.xy}, switch_pos=SWITCHES,
                  switch_size=np.full(5, 40.0))
        trials = [trial_row(1, 60, 140, 0)]
        tm.measure_trials(trials, self.t_ms, [s3d, sa])
        self.assertEqual(trials[0]['Data'], 'camera A')
        self.assertTrue(np.isnan(trials[0]['endpointdistance']))
        self.assertAlmostEqual(trials[0]['responsetime'], 0.28, places=6)

    def test_miss_not_used(self):
        src = dict(name='camera A', track={('upper', 'right'): self.xy}, switch_pos=SWITCHES,
                   switch_size=np.full(5, 40.0))
        trials = [trial_row(1, 60, 140, 0, outcome='miss')]
        tm.measure_trials(trials, self.t_ms, [src])
        self.assertFalse(trials[0]['touse'])
        self.assertFalse(np.isnan(trials[0]['responsetime']))

    def test_corrected(self):
        """First reach to switch 5, then on to switch 1 (the target) -> corrected from 5."""
        t = np.arange(int(3 * FPS)) / FPS
        xy = min_jerk(t, 1.2, 0.4, (100, 400), SWITCHES[4]) + min_jerk(t, 1.9, 0.4, (0, 0), SWITCHES[0] - SWITCHES[4])
        src = dict(name='camera A', track={('upper', 'right'): xy}, switch_pos=SWITCHES, switch_size=np.full(5, 40.0))
        trials = [trial_row(1, 60, int(2.4 * FPS), 0)]
        tm.measure_trials(trials, t * 1000, [src])
        self.assertEqual(trials[0]['corrected_from'], 5)


class Summaries(unittest.TestCase):
    def test_tables(self):
        rows = []
        for i, (part, rt, use) in enumerate([('pre_test', 0.2, True), ('pre_test', 0.4, True), ('pre_test', 0.3, True),
                                             ('pre_test', 9.0, False), ('Familiarization', 5.0, True),
                                             ('post_test', 0.25, True)]):
            r = {m: np.nan for m in tm.MEASURES}
            r.update(Part=part, Block=1 if i < 2 else 2, Sequence=1, touse=use, responsetime=rt)
            rows.append(r)
        t = tm.summary_tables({'S01': rows})
        self.assertEqual(len(t), 12)
        mean = t['mean_bypart'][0]
        self.assertAlmostEqual(mean['mean_pre_test_responsetime'], 0.3)
        self.assertAlmostEqual(mean['mean_post_test_responsetime'], 0.25)
        self.assertNotIn('mean_Familiarization_responsetime', mean)
        self.assertAlmostEqual(t['sd_bypart'][0]['sd_pre_test_responsetime'], 0.1)
        self.assertAlmostEqual(t['median_byblock'][0]['median_pre_test_block1_responsetime'], 0.3)
        self.assertAlmostEqual(t['iqr_bypart'][0]['iqr_pre_test_responsetime'], 0.15)  # MATLAB iqr of .2 .3 .4
        self.assertTrue(np.isnan(t['sd_bypart'][0]['sd_post_test_responsetime']))       # one value

    def test_cells(self):
        self.assertEqual(tm._cell(np.nan), '')
        self.assertEqual(tm._cell(None), '')
        self.assertEqual(tm._cell(True), 1)
        self.assertEqual(tm._cell(np.float64(0.123456)), 0.1235)
        self.assertEqual(tm._cell(np.int64(3)), 3)
        self.assertEqual(tm._cell('hit'), 'hit')


if __name__ == '__main__':
    unittest.main()
