"""Tests for experiment_design.py: light events -> blocks -> parts -> trials (a synthetic session).

    venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiment_design as ed  # noqa: E402

FPS = 60.0
PATTERN = [1, 4, 3, 5, 4, 2]


def make_session(n_blocks=18, rt_s=0.5, miss_every=7, delay_s=1.0, gap_s=30.0, lead_s=5.0, drop=(), extra=(),
                 test_lights=0):
    """Light events of a synthetic session with the 6-switch pattern: each trial the light is on for rt_s (a hit:
    pressing turns it off) or 1 s (every miss_every-th trial: a miss); the next light comes delay_s later; gap_s
    between blocks. drop: (block, slot) lights no camera saw; extra: (block, after slot, switch) stray lights.
    Returns events [(switch, on, off)], frame times (ms), touch {'upper', 'lower'} (switches, frames), truth."""
    f = lambda s: int(round(s * FPS))
    events, truth = [], []
    t = lead_s
    for k in range(test_lights):                 # testing the lights before the session
        events.append((1 + k % 5, f(t), f(t + 0.3)))
        t += 1.0
    t += gap_s if test_lights else 0
    for b in range(n_blocks):
        for slot in range(30):
            target = PATTERN[slot % 6]
            miss = miss_every and (b * 30 + slot) % miss_every == miss_every - 1
            dur = 1.0 if miss else rt_s
            on, off = f(t), f(t + dur)
            truth.append(dict(block=b, slot=slot, target=target, on=on, off=off, miss=miss))
            if (b, slot) not in drop:
                events.append((target, on, off))
            for bb, ss, sw in extra:
                if (bb, ss) == (b, slot):
                    events.append((sw, f(t + dur + 0.3), f(t + dur + 0.5)))
            t += dur + delay_s
        t += gap_s
    n = f(t + 5)
    times = np.arange(n) * 1000 / FPS
    touch = {'upper': np.full((5, n), np.inf), 'lower': np.full((5, n), np.inf)}
    for r in truth:
        if not r['miss']:
            touch['upper'][r['target'] - 1, r['off'] - 5:r['off'] + 1] = 0.0    # hand on the switch at the press
    return sorted(events, key=lambda e: e[1]), times, touch, truth


class Template(unittest.TestCase):
    def test_parts(self):
        t = ed.session_template('right')
        self.assertEqual([(p.name, p.blocks) for p in t],
                         [('Familiarization', 2), ('pre_test', 3), ('practice', 10), ('post_test', 3)])
        t = ed.session_template('left', contralateral=True, practice_limb='lower')
        self.assertEqual([p.name for p in t], ['Familiarization', 'pre_test', 'pre_test_contralateral', 'practice',
                                               'post_test', 'post_test_contralateral'])
        self.assertEqual((t[2].side, t[3].limb, t[3].side), ('right', 'lower', 'left'))
        self.assertEqual([p.name for p in ed.session_template(retest=True, contralateral=True)],
                         ['re_test', 're_test_contralateral'])


class Pattern(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(ed.parse_pattern('1,4,3,5,4,2'), PATTERN)
        self.assertEqual(ed.parse_pattern('1 4 3 5 4 2'), PATTERN)
        self.assertEqual(ed.parse_pattern('143542'), PATTERN)
        self.assertEqual(ed.parse_pattern('1;4;3'), [1, 4, 3])
        self.assertEqual(ed.parse_pattern(''), [])

    def test_problem(self):
        s = ed.Settings()
        self.assertIsNone(ed.pattern_problem(PATTERN, s))
        self.assertIsNone(ed.pattern_problem([], s))
        self.assertIn('1-5', ed.pattern_problem([1, 4, 3, 6, 4, 2], s))
        self.assertIn('5 switches', ed.pattern_problem([1, 4, 3, 5, 4], s))

    def test_align(self):
        exp = PATTERN * 5
        m = ed.align_block(exp, PATTERN, 5)
        self.assertEqual(m, {i: i for i in range(30)})
        # A missing light leaves its slot empty, the rest are not shifted
        obs = exp[:7] + exp[8:]
        m = ed.align_block(obs, PATTERN, 5)
        self.assertNotIn(7, m)
        self.assertEqual(m[8], 7)
        self.assertEqual(m[29], 28)
        # An extra light is left out
        obs = exp[:10] + [3] + exp[10:]
        m = ed.align_block(obs, PATTERN, 5)
        self.assertEqual(len(m), 30)
        self.assertNotIn(10, m.values())
        # A partial block (starts at the 4th light of a sequence; a start at a sequence boundary can't be told
        # from the block's start, the pattern repeats)
        m = ed.align_block(exp[15:], PATTERN, 5)
        self.assertEqual(min(m), 3)
        self.assertEqual(len(m), 15)
        # A hidden light (0) fits any slot
        obs = exp[:4] + [0] + exp[5:]
        self.assertEqual(ed.align_block(obs, PATTERN, 5)[4], 4)


class Events(unittest.TestCase):
    def test_light_events(self):
        n = 2000
        light = np.zeros((5, n))
        rng = np.random.default_rng(0)
        light += rng.normal(0, 0.1, light.shape)
        for k in range(10):
            light[2, 100 + 150 * k:130 + 150 * k] += 50
            light[0, 160 + 150 * k:190 + 150 * k] += 40
        ev = ed.light_events(light, FPS)
        self.assertEqual(len(ev), 20)
        self.assertEqual([e[1] for e in ev], sorted(e[1] for e in ev))
        self.assertEqual(ev[0], (3, 100, 130))
        self.assertEqual(ev[1], (1, 160, 190))

    def test_merge_cameras(self):
        a = [(1, 100, 130), (2, 300, None), (3, 500, 520)]
        b = [(1, 102, 135), (2, 298, 340), (4, 700, 720), (3, 0, 510)]
        m = ed.merge_camera_events(a, b, FPS)
        self.assertIn((1, 100, 135), m)                  # earliest on, latest off
        self.assertIn((2, 298, 340), m)                  # off unknown in A: B knows it
        self.assertIn((4, 700, 720), m)                  # only B saw it
        self.assertIn((3, 500, 520), m)                  # B's light 3 is far in time: not the same event
        self.assertIn((3, 0, 510), m)
        self.assertEqual(len(m), 5)
        self.assertEqual([e[1] for e in m], sorted(e[1] for e in m))

    def test_delay(self):
        ev, times, _, _ = make_session(n_blocks=1)
        self.assertAlmostEqual(ed.inter_trial_delay(ev, times), 1.0, delta=1 / FPS)
        self.assertIsNone(ed.inter_trial_delay(ev[:3], times))


class Session(unittest.TestCase):
    def test_full_session(self):
        ev, times, touch, truth = make_session(test_lights=4)
        tpl = ed.session_template('right')
        res = ed.analyze_session(ev, times, FPS, touch, tpl, ed.Settings())
        self.assertEqual(res['pattern'], PATTERN)
        self.assertEqual(len(res['blocks']), 19)
        self.assertEqual(res['parts'][0], (ed.IGNORE, 0))          # the 4 test lights
        self.assertEqual(res['parts'][1:4], [('Familiarization', 1), ('Familiarization', 2), ('pre_test', 1)])
        self.assertEqual(res['parts'][-1], ('post_test', 3))
        self.assertEqual(len(res['trials']), 18 * 30)
        self.assertTrue(any('short group' in w for w in res['warnings']))
        self.assertFalse(any('trials found instead' in w for w in res['warnings']))
        self.assertAlmostEqual(res['delay'], 1.0, delta=1 / FPS)
        for r, tr in zip(res['trials'], truth):
            self.assertEqual(r['Target'], tr['target'])
            self.assertEqual(r['on_frame'], tr['on'])
            self.assertEqual(r['off_frame'], tr['off'])
            self.assertEqual(r['Outcome'], 'miss' if tr['miss'] else 'hit')
            self.assertEqual(r['Light_source'], 'video')
            self.assertAlmostEqual(r['Light_on_ms'], times[tr['on']], places=1)
            if tr['miss']:
                self.assertIsNone(r['Hit_ms'])
                self.assertIsNone(r['Touch_ms'])
            else:
                self.assertAlmostEqual(r['Hit_ms'], times[tr['off']], places=1)
                self.assertAlmostEqual(r['Touch_ms'], times[tr['off'] - 5], places=1)
        first = [r for r in res['trials'] if r['block_index'] == 1]
        self.assertEqual(first[0]['window_start'], first[0]['on_frame'] - int(2 * FPS))
        self.assertEqual(first[1]['window_start'], first[0]['off_frame'])
        self.assertEqual((first[0]['Part'], first[0]['Sequence'], first[0]['Trial']), ('Familiarization', 1, 1))
        self.assertEqual((first[-1]['Sequence'], first[-1]['Trial']), (5, 6))
        self.assertEqual(first[0]['Limb'], 'upper')

    def test_parts_override(self):
        ev, times, touch, _ = make_session(n_blocks=3)
        tpl = ed.session_template('right', retest=True)
        res = ed.analyze_session(ev, times, FPS, touch, tpl, ed.Settings(),
                                 parts_override=[('re_test', 1), (ed.IGNORE, 0), ('re_test', 2)])
        self.assertEqual({r['Block'] for r in res['trials']}, {1, 2})
        self.assertEqual(len(res['trials']), 60)

    def test_hidden_light_from_timing(self):
        """A light no camera saw is found from the 1 s timing; its switch from where the hand goes."""
        ev, times, touch, truth = make_session(n_blocks=2, miss_every=0, drop=[(0, 9)])
        tpl = ed.session_template('right', retest=True)
        res = ed.analyze_session(ev, times, FPS, touch, tpl, ed.Settings())
        r = res['trials'][9]
        self.assertEqual(r['Light_source'], 'timing')
        self.assertEqual(r['Target'], truth[9]['target'])
        self.assertLessEqual(abs(r['on_frame'] - truth[9]['on']), 1)
        self.assertLessEqual(abs(r['off_frame'] - truth[9]['off']), 1)
        self.assertEqual(r['Outcome'], 'hit')
        # Switched off: the slot stays empty
        res = ed.analyze_session(ev, times, FPS, touch, tpl, ed.Settings(fill_hidden_lights=False))
        self.assertEqual(res['trials'][9]['Outcome'], 'not detected')
        self.assertEqual(res['trials'][10]['on_frame'], truth[10]['on'])     # the rest is not shifted
        self.assertTrue(any('29 trials found' in w for w in res['warnings']))

    def test_hidden_last_light(self):
        """The last light of a block not seen: placed 1 s after the previous trial, ended by the touch."""
        ev, times, touch, truth = make_session(n_blocks=2, miss_every=0, drop=[(0, 29)])
        res = ed.analyze_session(ev, times, FPS, touch, ed.session_template('right', retest=True), ed.Settings())
        r = res['trials'][29]
        self.assertEqual(r['Light_source'], 'timing')
        self.assertLessEqual(abs(r['on_frame'] - truth[29]['on']), 1)
        self.assertEqual(r['off_frame'], truth[29]['off'] - 5)        # the first frame the hand is on the switch

    def test_extra_light(self):
        ev, times, touch, truth = make_session(n_blocks=2, miss_every=0, extra=[(1, 12, 5)])
        res = ed.analyze_session(ev, times, FPS, touch, ed.session_template('right', retest=True),
                                 ed.Settings(fill_hidden_lights=False))
        b2 = [r for r in res['trials'] if r['Block'] == 2]
        self.assertEqual(len(b2), 30)
        self.assertEqual([r['on_frame'] for r in b2], [t['on'] for t in truth[30:]])

    def test_typed_pattern(self):
        ev, times, touch, _ = make_session(n_blocks=2)
        tpl = ed.session_template('right', retest=True)
        res = ed.analyze_session(ev, times, FPS, touch, tpl, ed.Settings(pattern=[1, 4, 3]))
        self.assertEqual(res['pattern'], PATTERN)                   # wrong length: ignored, found from the lights
        self.assertTrue(any('is ignored' in w for w in res['warnings']))

    def test_light_still_on(self):
        """A light already on at the start / still on at the end -> outcome unknown."""
        ev, times, touch, _ = make_session(n_blocks=1, miss_every=0)
        ev = [(ev[0][0], 0, ev[0][2])] + ev[1:-1] + [(ev[-1][0], ev[-1][1], None)]
        res = ed.analyze_session(ev, times, FPS, touch, ed.session_template('right', retest=True), ed.Settings())
        self.assertEqual(res['trials'][0]['Outcome'], 'unknown')
        self.assertIsNone(res['trials'][0]['Light_on_ms'])
        self.assertEqual(res['trials'][-1]['Outcome'], 'unknown')


if __name__ == '__main__':
    unittest.main()
