"""Tests for the switch lights in extract_kinematics.py: light_energy (one frame) and find_light_events (a signal).

    venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extract_kinematics as ek  # noqa: E402

FPS = 60.0
PINK = (200, 180, 255)        # BGR of a lit LED: red near saturation, green/blue high, blue >= green
DEEP_RED = (30, 20, 220)      # the mushroom button
WHITE = (240, 240, 240)       # the table
SKIN = (120, 170, 230)        # orange: blue well below green


class LightEnergy(unittest.TestCase):
    def frame(self):
        return np.full((200, 300, 3), 60, np.uint8)

    def test_pink_only(self):
        f = self.frame()
        box = np.float32([[100, 50], [200, 50], [200, 150], [100, 150]])
        self.assertEqual(ek.light_energy(f, box), 0.0)
        f[60:70, 110:130] = PINK                         # 200 px, excess = 255 - 200 = 55
        self.assertAlmostEqual(ek.light_energy(f, box), 200 * 55 * 100 / (100 * 100))
        for colour in (DEEP_RED, WHITE, SKIN):
            g = self.frame()
            g[60:90, 110:150] = colour
            self.assertEqual(ek.light_energy(g, box), 0.0, colour)

    def test_bounding_box_and_clipping(self):
        f = self.frame()
        f[0:10, 0:10] = PINK
        # A rotated quad: its bounding box is measured
        quad = np.float32([[5, -20], [40, 5], [5, 30], [-30, 5]])
        x0, y0, x1, y1 = 0, 0, 40, 30                    # clipped to the frame
        self.assertAlmostEqual(ek.light_energy(f, quad), 100 * 55 * 100 / ((x1 - x0) * (y1 - y0)))
        # Outside the frame / too small -> 0
        self.assertEqual(ek.light_energy(f, np.float32([[400, 400], [500, 400], [500, 500], [400, 500]])), 0.0)
        self.assertEqual(ek.light_energy(f, np.float32([[2, 2], [3, 2], [3, 3], [2, 3]])), 0.0)

    def test_pink_mask_same_colour_test(self):
        f = self.frame()
        f[10:20, 10:20] = PINK
        f[30:40, 30:40] = DEEP_RED
        m = ek.pink_mask(f)
        self.assertEqual(int(m.sum()), 100)
        self.assertTrue(m[15, 15])

    def test_box_corners(self):
        c = ek.box_corners([0.1, 0.2, 0.3, 0.6], 1000, 500, 0.0)
        np.testing.assert_allclose(c, [[100, 100], [300, 100], [300, 300], [100, 300]])
        c = ek.box_corners([0.1, 0.2, 0.3, 0.6], 1000, 500, 0.5)     # enlarged by 50% of its size
        np.testing.assert_allclose(c[0], [50, 50])
        np.testing.assert_allclose(c[2], [350, 350])


def signal(n=3000, pulses=(), level=40.0, noise=0.5, drift=0.0, seed=0):
    """Light level per frame: noise + slow drift + square pulses [(on, off)]."""
    rng = np.random.default_rng(seed)
    s = 5 + rng.normal(0, noise, n) + drift * np.sin(np.arange(n) / n * 2 * np.pi)
    for a, b in pulses:
        s[a:b] += level
    return s


class FindLightEvents(unittest.TestCase):
    def test_pulses(self):
        pulses = [(200 + 200 * k, 230 + 200 * k + 3 * k) for k in range(12)]
        ev, why = ek.find_light_events(signal(pulses=pulses), FPS)
        self.assertEqual(why, '')
        self.assertEqual(ev, pulses)

    def test_slow_drift_removed(self):
        pulses = [(300 + 200 * k, 330 + 200 * k) for k in range(10)]
        ev, _ = ek.find_light_events(signal(pulses=pulses, drift=30.0), FPS)
        self.assertEqual(ev, pulses)

    def test_on_at_start_and_end(self):
        n = 3000
        pulses = [(0, 25)] + [(200 + 150 * k, 230 + 150 * k) for k in range(10)] + [(n - 20, n)]
        ev, _ = ek.find_light_events(signal(n, pulses), FPS)
        self.assertEqual(ev[0], (0, 25))                 # on 0 = already on when the video started
        self.assertEqual(ev[-1], (n - 20, None))         # None = still on when it ended
        self.assertEqual(len(ev), 12)

    def test_short_blips_and_gaps(self):
        pulses = [(200 + 150 * k, 240 + 150 * k) for k in range(10)]
        s = signal(pulses=pulses)
        s[1000:1003] += 40                               # 3 frames (< 0.07 s): noise, not an event
        s[200 + 150 * 3 + 15:200 + 150 * 3 + 19] -= 40   # a hand hides the LED for 4 frames: still one event
        ev, _ = ek.find_light_events(s, FPS)
        self.assertEqual(ev, pulses)

    def test_partly_hidden_still_on(self):
        """A light dimmed by the hand (above 20% of its level) counts as on; a weak bump alone isn't an event."""
        pulses = [(200 + 150 * k, 240 + 150 * k) for k in range(10)]
        s = signal(pulses=pulses)
        s[200 + 150 * 2 + 10:200 + 150 * 2 + 30] -= 25   # down to 15 of 40
        s[2000:2030] += 12                               # an arm passing: 30% of the level, never full
        ev, _ = ek.find_light_events(s, FPS)
        self.assertEqual(ev, pulses)

    def test_no_events(self):
        self.assertEqual(ek.find_light_events(np.full(500, 3.0), FPS), ([], 'light never changed'))
        self.assertEqual(ek.find_light_events([1.0, 2.0], FPS)[0], [])
        ev, why = ek.find_light_events(signal(noise=1.0), FPS)
        self.assertEqual(ev, [])
        self.assertIn('no clear', why)

    def test_always_lit(self):
        """Lit the whole time (brighter now and then, e.g. a reflection) -> no events."""
        n = 3000
        s = signal(n, [(0, n)] + [(200 + 250 * k, 230 + 250 * k) for k in range(10)])
        ev, why = ek.find_light_events(s, FPS)
        self.assertEqual(ev, [])
        self.assertIn('on the whole time', why)

    def test_regression(self):
        """The exact result for a noisy signal with uneven pulses (any change of the method shows here)."""
        rng = np.random.default_rng(7)
        n = 4000
        pulses, t = [], 150
        while t < n - 200:
            d = int(rng.integers(8, 60))
            pulses.append((t, t + d))
            t += d + int(rng.integers(40, 200))
        s = signal(n, pulses, level=30.0, noise=0.6, drift=10.0, seed=3)
        s *= 1 + 0.02 * rng.random(n)
        ev, _ = ek.find_light_events(s, FPS)
        self.assertEqual(ev, pulses)


if __name__ == '__main__':
    unittest.main()
