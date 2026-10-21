"""Tests for the helpers shared by extract_kinematics.py, analyze_two_views.py, session_engine.py and
session_analysis.py: switch geometry, touch distance, camera movement, merging two views' light events.

    venv/bin/python -m unittest discover -s tests -v
"""
import csv
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_two_views as atv  # noqa: E402
import experiment_design as ed  # noqa: E402
import extract_kinematics as ek  # noqa: E402


class Geometry(unittest.TestCase):
    def test_switch_geometry(self):
        boxes = [[0.1, 0.2, 0.3, 0.6], [0.5, 0.5, 0.6, 0.7]]
        light, touch, sizes = ek.switch_geometry(boxes, 1000, 500)
        np.testing.assert_allclose(touch[0], ek.box_corners(boxes[0], 1000, 500, 0.0))
        np.testing.assert_allclose(light[1], ek.box_corners(boxes[1], 1000, 500, ek.LIGHT_BOX_MARGIN))
        np.testing.assert_allclose(sizes, [np.sqrt(200 * 200), np.sqrt(100 * 100)])

    def test_touch_distance(self):
        poly = np.float32([[100, 100], [200, 100], [200, 200], [100, 200]])
        self.assertAlmostEqual(ek.touch_distance(poly, [(250, 150), (230, 150)], 100.0), 0.3)   # closest point
        self.assertLess(ek.touch_distance(poly, [(150, 150)], 100.0), 0)                        # inside
        self.assertEqual(ek.touch_distance(poly, [], 100.0), np.inf)

    def test_camera_movement(self):
        a = 0.05
        M = np.float32([[1.02 * np.cos(a), -1.02 * np.sin(a), 30], [1.02 * np.sin(a), 1.02 * np.cos(a), -12]])
        pts = np.array([[100.0, 200.0], [3000.0, 1500.0]])
        np.testing.assert_allclose(ek.to_first_frame(ek.to_frame(pts, M), M), pts, atol=1e-6)
        # One matrix per frame, one or several points per frame
        Ms = np.stack([M, np.float32([[1, 0, 5], [0, 1, 7]])])
        np.testing.assert_allclose(ek.to_first_frame(pts, Ms), [ek.to_first_frame(pts[0], M), pts[1] - [5, 7]],
                                   atol=1e-6)
        many = np.stack([pts, pts + 1])
        np.testing.assert_allclose(ek.to_first_frame(many, Ms)[1], pts + 1 - [5, 7], atol=1e-6)


class SwitchEvents(unittest.TestCase):
    def test_touch_search_end(self):
        fps = 60
        events = [(10, 40), (100, None)]
        self.assertEqual(ek.touch_search_end(events, 0, 1000, fps), 100)        # not into the next event
        self.assertEqual(ek.touch_search_end([(10, 40)], 0, 1000, fps), 100)    # 1 s after the light off
        self.assertEqual(ek.touch_search_end([(10, 40)], 0, 70, fps), 70)       # video end
        self.assertEqual(ek.touch_search_end(events, 1, 1000, fps), 1000)       # still on at the end

    def test_csv(self):
        times = np.arange(200) * 1000 / 60
        rows = [ek.event_row(1, 60, 70, 90, times), ek.event_row(0, 120, None, None, times),
                ek.event_row(0, 0, 5, 20, times), [5, '', '', '']]
        path = os.path.join(tempfile.mkdtemp(), 'e.csv')
        ek.write_events_csv(path, rows)
        with open(path) as f:
            got = list(csv.reader(f))
        self.assertEqual(got[0], ['Switch', 'Start_ms', 'Touching_ms', 'Turn_off_ms'])
        self.assertEqual(got[1], ['1', '', '83.3', '333.3'])                    # light already on: no start
        self.assertEqual(got[2], ['1', '2000.0', '', ''])
        self.assertEqual(got[3], ['2', '1000.0', '1166.7', '1500.0'])
        self.assertEqual(got[4], ['5', '', '', ''])

    def test_two_view_merge(self):
        """Synced videos end together: a light still on at the end in one view stays 'still on'."""
        a = [(10, 40), (100, None)]
        b = [(12, 45), (101, 150), (300, 330)]
        self.assertEqual(atv.merge_events(a, b, 60), [(10, 45), (100, None), (300, 330)])
        # The session timeline (cameras may stop at different times) uses the other camera's off
        m = ed.merge_camera_events([(1,) + e for e in a], [(1,) + e for e in b], 60)
        self.assertEqual(m, [(1, 10, 45), (1, 100, 150), (1, 300, 330)])


if __name__ == '__main__':
    unittest.main()
