"""Tests for negative reaction times and the not-from-home (shortcut) mark.

    venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trial_measures as tm  # noqa: E402

FPS = 60.0
HOME = np.array([0.0, 0.0])
SWITCHES = np.array([[-300.0, -200.0], [-150.0, -300.0], [0.0, -350.0], [150.0, -300.0], [300.0, -200.0]])


def reach(p0, p1, t0, duration=0.35):
    """Smooth (minimum-jerk) movement from p0 to p1 starting at t0 s."""
    def pos(t):
        tau = np.clip((t - t0) / duration, 0, 1)
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        return p0 + (p1 - p0) * s[:, None]
    return pos


def trial(stim, segments, t_end=None):
    """Positions at 60 fps from 0 to t_end for a list of (start time, from, to, duration)."""
    t = np.arange(0, t_end or stim + 1.2, 1 / FPS)
    p = np.repeat(np.asarray(segments[0][1], float)[None], len(t), 0)
    for t0, a, b, dur in segments:
        f = reach(np.asarray(a, float), np.asarray(b, float), t0, dur)(t)
        p = np.where((t >= t0)[:, None], f, p)
    return t, p


def measure(t, p, stim, target):
    k = tm.trial_kinematics(t, p, stim)
    m = tm.calculate_measures(k, SWITCHES[target - 1], HOME)
    return m, tm.start_place(m['start_pos'], HOME, SWITCHES[target - 1], SWITCHES)


class NegativeReactionTime(unittest.TestCase):
    def test_reach_before_the_light(self):
        stim = 1.0
        t, p = trial(stim, [(0.0, HOME, HOME, 0.1), (stim - 0.15, HOME, SWITCHES[2], 0.4)])
        m, (where, _) = measure(t, p, stim, 3)
        self.assertTrue(m['anticipation'])
        self.assertLess(m['responsetime'], 0)
        self.assertAlmostEqual(m['responsetime'], -0.15, delta=0.06)
        self.assertAlmostEqual(m['anticipation_ms'], -1000 * m['responsetime'], delta=1)
        self.assertGreater(m['movementtime'], 0.3)        # MT from the (earlier) onset
        self.assertEqual(where, 'home')

    def test_reach_after_the_light(self):
        stim = 1.0
        t, p = trial(stim, [(0.0, HOME, HOME, 0.1), (stim + 0.25, HOME, SWITCHES[0], 0.4)])
        m, (where, _) = measure(t, p, stim, 1)
        self.assertFalse(m['anticipation'])
        self.assertAlmostEqual(m['responsetime'], 0.25, delta=0.06)
        self.assertEqual(where, 'home')

    def test_return_home_is_not_anticipation(self):
        """Still returning home from the previous switch when the light comes on: not a negative RT."""
        stim = 1.0
        t, p = trial(stim, [(0.0, SWITCHES[0], SWITCHES[0], 0.1), (stim - 0.3, SWITCHES[0], HOME, 0.35),
                            (stim + 0.2, HOME, SWITCHES[4], 0.4)])
        m, _ = measure(t, p, stim, 5)
        self.assertFalse(m['anticipation'])
        self.assertGreaterEqual(m['responsetime'], 0)

    def test_slow_drift_before_the_reach(self):
        """The resting hand drifts slowly (above 5% of the peak speed) until the reach starts 0.3 s after the light:
        calculateMeasures.m found no sample < 5% and gave RT ~ 0; the onset is now where the reach starts."""
        stim = 1.0
        t = np.arange(0, stim + 1.2, 1 / FPS)
        drift = np.column_stack([150.0 * t, np.zeros_like(t)])          # 150 px/s sideways (~9% of the peak)
        go = reach(np.zeros(2), SWITCHES[2], stim + 0.3, 0.4)(t)
        p = drift + go
        m, _ = measure(t, p, stim, 3)
        self.assertFalse(m['anticipation'])
        self.assertAlmostEqual(m['responsetime'], 0.3, delta=0.07)


class NotFromHome(unittest.TestCase):
    def test_shortcut_from_the_previous_switch(self):
        """Hit switch 4, didn't go back home, went straight to switch 5."""
        stim = 1.0
        t, p = trial(stim, [(0.0, SWITCHES[3], SWITCHES[3], 0.1), (stim + 0.2, SWITCHES[3], SWITCHES[4], 0.3)])
        m, (where, pct) = measure(t, p, stim, 5)
        self.assertEqual(where, 'switch 4')
        self.assertGreater(pct, 50)

    def test_start_somewhere_else(self):
        stim = 1.0
        mid = np.array([0.0, -200.0])           # halfway out, not at a switch
        t, p = trial(stim, [(0.0, mid, mid, 0.1), (stim + 0.2, mid, SWITCHES[0], 0.3)])
        _, (where, _) = measure(t, p, stim, 1)
        self.assertEqual(where, 'elsewhere')

    def test_small_shift_of_the_resting_position_is_home(self):
        stim = 1.0
        rest = np.array([40.0, -40.0])          # ~15% of the reach away from the usual home
        t, p = trial(stim, [(0.0, rest, rest, 0.1), (stim + 0.2, rest, SWITCHES[2], 0.4)])
        _, (where, _) = measure(t, p, stim, 3)
        self.assertEqual(where, 'home')


if __name__ == '__main__':
    unittest.main()
