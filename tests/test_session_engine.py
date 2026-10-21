"""Tests for session_engine.process_video on short synthetic videos (made with OpenCV): every frame exactly once,
in order, also where the worker processes' parts meet and across GoPro chapters, and the per-frame light / touch
values equal a plain frame-by-frame computation.

    venv/bin/python -m unittest discover -s tests -v
"""
import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extract_kinematics as ek  # noqa: E402
import session_engine as se  # noqa: E402

FPS = 30.0
SIZE = (640, 360)
N_FRAMES = 960                 # >= 3 x 300: process_video may use 3 parts
BOXES = [[0.08 + 0.18 * s, 0.70, 0.16 + 0.18 * s, 0.84] for s in range(5)]


def synthetic_frame(i, texture):
    """Frame i: a fixed texture (for the camera-motion tracker) and the 5 switch LEDs; LED s is lit when bit s of
    i % 32 is set, its brightness changes every 32 frames - so each frame's light levels tell which frame it is."""
    f = texture.copy()
    w, h = SIZE
    level = 110 + 20 * ((i // 32) % 6)                     # green / blue of the pink: 110..210
    for s, (x0, y0, x1, y1) in enumerate(BOXES):
        if (i % 32) >> s & 1:
            c = (int((x0 + x1) / 2 * w), int((y0 + y1) / 2 * h))
            cv2.circle(f, c, 14, (min(255, level + 10), level, 255), -1)
    return f


def make_texture(seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 100, (SIZE[1] // 8, SIZE[0] // 8, 3)).astype(np.uint8)
    return cv2.GaussianBlur(cv2.resize(t, SIZE, interpolation=cv2.INTER_NEAREST), (3, 3), 0)


def write_video(path, frames):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, SIZE)
    for f in frames:
        w.write(f)
    w.release()


def decoded_frames(paths):
    """All frames of the files one after the other, decoded the plain way (FFmpeg, no jumping)."""
    out = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
        cap.release()
    return out


def reference(frames, d):
    """Light levels and touch distances computed frame by frame with the cache's own camera movement and pose."""
    w, h = SIZE
    light_c = [ek.box_corners(b, w, h, ek.LIGHT_BOX_MARGIN) for b in BOXES]
    touch_c = [ek.box_corners(b, w, h, 0.0) for b in BOXES]
    sizes = [np.sqrt((b[2] - b[0]) * w * (b[3] - b[1]) * h) for b in BOXES]
    light = np.zeros((5, len(frames)))
    touch = {k: np.full((5, len(frames)), np.inf) for k in se.TOUCH_POINTS}
    for i, f in enumerate(frames):
        M = d['motion'][i]
        for s in range(5):
            light[s, i] = ek.light_energy(f, cv2.transform(light_c[s].reshape(-1, 1, 2), M).reshape(-1, 2))
            poly = cv2.transform(touch_c[s].reshape(-1, 1, 2), M).reshape(-1, 2).astype(np.float32)
            for k, ids in se.TOUCH_POINTS.items():
                pts = [p for p in d['pose'][i, ids, :2] if not np.isnan(p[0])]
                if pts:
                    touch[k][s, i] = min(-cv2.pointPolygonTest(poly, (float(x), float(y)), True)
                                         for x, y in pts) / sizes[s]
    return light, touch


class ProcessVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        tex = make_texture()
        cls.frames = [synthetic_frame(i, tex) for i in range(N_FRAMES)]
        cls.single = os.path.join(cls.dir, 'session.mp4')
        write_video(cls.single, cls.frames)
        # The same video as two GoPro chapters; the third part's jump (1 s early) lands in the first chapter
        cls.gopro = os.path.join(cls.dir, 'GX010042.MP4')
        write_video(cls.gopro, cls.frames[:650])
        write_video(os.path.join(cls.dir, 'GX020042.MP4'), cls.frames[650:])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def check(self, video, paths, workers):
        out = tempfile.mkdtemp(dir=self.dir)
        d = se.process_video(video, BOXES, out, workers=workers)
        meta = d['meta']
        self.assertEqual(meta['workers'], workers)
        self.assertTrue(meta['complete'])
        self.assertEqual(meta['seams'], [])                  # the parts meet without repeated or missing frames
        # Every frame once, in order, at its own time (no duplicates / gaps where the parts meet)
        self.assertEqual(len(d['times']), N_FRAMES)
        np.testing.assert_allclose(d['times'], np.arange(N_FRAMES) * 1000 / FPS, atol=0.5)
        self.assertTrue(np.all(np.diff(d['times']) > 0))
        self.assertEqual(d['pose'].shape, (N_FRAMES, 33, 3))
        self.assertEqual(d['motion'].shape, (N_FRAMES, 2, 3))
        self.assertEqual(d['light'].shape, (5, N_FRAMES))
        self.assertEqual(d['touch_upper'].shape, (5, N_FRAMES))
        # Per-frame values = the frame-by-frame computation on the same decoded frames
        frames = decoded_frames(paths)
        self.assertEqual(len(frames), N_FRAMES)
        light, touch = reference(frames, d)
        self.assertGreater((light > 0).mean(), 0.3)          # the LEDs are seen
        np.testing.assert_allclose(d['light'], light, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(d['touch_upper'], touch['upper'], rtol=1e-5)
        np.testing.assert_allclose(d['touch_lower'], touch['lower'], rtol=1e-5)
        # The LED pattern tells the frame: frame i has LED s lit when bit s of i % 32 is set
        lit = light > 0.1 * light.max(axis=1, keepdims=True)
        want = np.array([[(i % 32) >> s & 1 for i in range(N_FRAMES)] for s in range(5)], bool)
        self.assertGreater((lit == want).mean(), 0.99)
        # The camera does not move
        np.testing.assert_allclose(d['motion'][:, :, :2], np.tile(np.eye(2), (N_FRAMES, 1, 1)), atol=0.01)
        np.testing.assert_allclose(d['motion'][:, :, 2], 0, atol=1.0)
        # Saved cache = returned data
        c = se.load_cache(video, out, BOXES)
        self.assertIsNotNone(c)
        np.testing.assert_array_equal(c['light'], d['light'])
        np.testing.assert_array_equal(c['times'], d['times'])
        self.assertIsNone(se.load_cache(video, out, [b[:2] + [b[2] + 0.01, b[3]] for b in BOXES]))  # other boxes
        return d

    def test_one_worker(self):
        self.check(self.single, [self.single], 1)

    def test_three_workers(self):
        d3 = self.check(self.single, [self.single], 3)
        d1 = se.process_video(self.single, BOXES, tempfile.mkdtemp(dir=self.dir), workers=1)
        np.testing.assert_array_equal(d3['times'], d1['times'])
        np.testing.assert_allclose(d3['light'], d1['light'], rtol=1e-5, atol=1e-4)

    def test_gopro_chapters(self):
        paths = [self.gopro, os.path.join(self.dir, 'GX020042.MP4')]
        d = self.check(self.gopro, paths, 3)
        self.assertEqual(d['meta']['chapters'], ['GX010042.MP4', 'GX020042.MP4'])

    def test_progress_and_stop(self):
        calls = []
        out = tempfile.mkdtemp(dir=self.dir)
        d = se.process_video(self.single, BOXES, out, progress=lambda n, tot, left: calls.append((n, tot)),
                             should_stop=lambda: len(calls) >= 2, workers=3)
        self.assertIsNone(d)
        self.assertEqual(calls[0][1], N_FRAMES)
        self.assertIsNone(se.load_cache(self.single, out, BOXES))
        self.assertEqual([f for f in os.listdir(out) if f.endswith('.npz')], [])   # no part files left


class JoinParts(unittest.TestCase):
    """Where two workers' parts meet: a frame read twice is dropped, missing frames are reported."""

    def part(self, frames):
        f = np.asarray(frames)
        return dict(times=f * 1000 / FPS, light=np.stack([f, f], 1).astype(np.float32), pose=np.zeros((len(f), 33, 3)))

    def test_clean(self):
        d, seams = se.join_parts([self.part(range(0, 10)), self.part(range(10, 25)), self.part(range(25, 30))], FPS)
        np.testing.assert_allclose(d['times'], np.arange(30) * 1000 / FPS)
        self.assertEqual(seams, [])

    def test_repeated_frames(self):
        d, seams = se.join_parts([self.part(range(0, 10)), self.part(range(8, 20))], FPS)
        np.testing.assert_allclose(d['times'], np.arange(20) * 1000 / FPS)
        np.testing.assert_allclose(d['light'][:, 0], np.arange(20))      # all arrays cut the same way
        self.assertEqual(d['pose'].shape[0], 20)
        self.assertEqual(seams, [dict(ms=round(9 * 1000 / FPS, 1), repeated=2, missing=0)])

    def test_missing_frames(self):
        d, seams = se.join_parts([self.part(range(0, 10)), self.part(range(13, 20))], FPS)
        self.assertEqual(len(d['times']), 17)
        self.assertEqual(seams[0]['missing'], 3)

    def test_empty_part(self):
        d, seams = se.join_parts([self.part(range(0, 10)), self.part([]), self.part(range(10, 15))], FPS)
        self.assertEqual(len(d['times']), 15)
        self.assertEqual(seams, [])


class OneWorker(unittest.TestCase):
    """workers=1 runs in a thread of this process: Cancel must stop it, and its error must come through."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        tex = make_texture()
        cls.video = os.path.join(cls.dir, 'one.mp4')
        write_video(cls.video, [synthetic_frame(i, tex) for i in range(N_FRAMES)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_stop(self):
        import time
        calls = []
        out = tempfile.mkdtemp(dir=self.dir)
        t = time.time()
        d = se.process_video(self.video, BOXES, out, progress=lambda n, tot, left: calls.append(n),
                             should_stop=lambda: len(calls) >= 2, workers=1)
        self.assertIsNone(d)
        self.assertLess(calls[-1], N_FRAMES // 2)                 # stopped early, not after the whole video
        self.assertLess(time.time() - t, 4)                      # not processing on in the background
        self.assertEqual([f for f in os.listdir(out) if f.endswith('.npz')], [])

    def test_error_comes_through(self):
        from unittest import mock
        with mock.patch.object(se, '_process_range', side_effect=ValueError('broken video')):
            with self.assertRaisesRegex(ValueError, 'broken video'):
                se.process_video(self.video, BOXES, tempfile.mkdtemp(dir=self.dir), workers=1)


if __name__ == '__main__':
    unittest.main()
