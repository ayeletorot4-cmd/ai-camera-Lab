"""Tests for video_source.py: choosing the hardware decoder, falling back to the normal one, decoder threads.

    venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_source as vs  # noqa: E402
from test_gopro import write_numbered_video  # noqa: E402


class FailingCapture:
    """A capture that stops after `n` frames, like a graphics-card decoder giving up in the middle."""

    def __init__(self, cap, n):
        self.cap, self.n = cap, n

    def grab(self):
        if self.n <= 0:
            return False
        self.n -= 1
        return self.cap.grab()

    def __getattr__(self, name):
        return getattr(self.cap, name)


def read_all(src):
    levels, times = [], []
    while True:
        ok, fr = src.read()
        if not ok:
            return levels, times
        levels.append(int(fr[0, 0, 0]))
        times.append(src.get(cv2.CAP_PROP_POS_MSEC))


class HardwareChoice(unittest.TestCase):
    def test_switch(self):
        with mock.patch.dict(os.environ, {'LAB_HW_DECODE': '1'}):
            self.assertTrue(vs.hardware_decoding())
        with mock.patch.dict(os.environ, {'LAB_HW_DECODE': '0'}):
            self.assertFalse(vs.hardware_decoding())
        with mock.patch.dict(os.environ, {'LAB_HW_DECODE': ''}):
            self.assertEqual(vs.hardware_decoding(), sys.platform == 'darwin')

    def test_off_means_plain_ffmpeg(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'v.mp4')
        write_numbered_video(p, 0, 10)
        with mock.patch.dict(os.environ, {'LAB_HW_DECODE': '0'}):
            src = vs.VideoSource(p, hardware=True)
            self.assertEqual(src.decoder, 'ffmpeg')
            self.assertEqual(len(read_all(src)[0]), 10)

    def test_threads(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'v.mp4')
        write_numbered_video(p, 0, 20)
        ref = read_all(vs.VideoSource(p, hardware=False))
        src = vs.VideoSource(p, hardware=False, threads=1)
        self.assertEqual(src.cap.get(cv2.CAP_PROP_N_THREADS), 1)
        self.assertEqual(read_all(src), ref)                     # same frames, same times


class SoftwareFallback(unittest.TestCase):
    def test_decoder_fails_in_the_middle(self):
        """The hardware decoder stops after 20 frames of chapter 1: the normal decoder goes on from frame 20."""
        d = tempfile.mkdtemp()
        write_numbered_video(os.path.join(d, 'GX010077.MP4'), 0, 50)
        write_numbered_video(os.path.join(d, 'GX020077.MP4'), 50, 40)
        first = os.path.join(d, 'GX010077.MP4')
        ref_levels, ref_times = read_all(vs.VideoSource(first, hardware=False))
        real_open = vs._open
        calls = []

        def fake_open(path, hardware=True, threads=0):
            cap, name = real_open(path, False, threads)
            calls.append((os.path.basename(path), hardware))
            if hardware and len(calls) == 1:
                return FailingCapture(cap, 20), 'ffmpeg-hw'
            return cap, name

        with mock.patch.object(vs, '_open', fake_open):
            levels, times = read_all(vs.VideoSource(first, hardware=True))
        self.assertEqual(levels, ref_levels)
        np.testing.assert_allclose(times, ref_times)
        self.assertEqual(calls[1], ('GX010077.MP4', False))        # chapter 1 again, normal decoder
        self.assertEqual(calls[2], ('GX020077.MP4', True))         # the next chapter tries the hardware again

    def _after_jump(self, fail_after):
        """Jump by time into chapter 1 (like a worker), read to the end with a 'hardware' decoder that stops after
        fail_after frames (None: only at the real end). Returns the frame levels and times read."""
        d = tempfile.mkdtemp()
        write_numbered_video(os.path.join(d, 'GX010078.MP4'), 0, 60)
        write_numbered_video(os.path.join(d, 'GX020078.MP4'), 60, 30)
        first = os.path.join(d, 'GX010078.MP4')
        real_open = vs._open
        opened = []

        def fake_open(path, hardware=True, threads=0):
            cap, _ = real_open(path, False, threads)
            opened.append(hardware)
            if hardware and fail_after is not None and len(opened) == 1:
                return FailingCapture(cap, 10 ** 6), 'ffmpeg-hw'
            if hardware and fail_after is not None and len(opened) == 2:
                return FailingCapture(cap, fail_after), 'ffmpeg-hw'
            return cap, 'ffmpeg-hw' if hardware else 'ffmpeg'

        with mock.patch.object(vs, '_open', fake_open):
            src = vs.VideoSource(first, hardware=True)
            src.set(cv2.CAP_PROP_POS_MSEC, 25 * 1000 / 60)
            return read_all(src)

    def test_no_duplicates_after_a_jump(self):
        for fail_after in (None, 7):
            levels, times = self._after_jump(fail_after)
            self.assertTrue(np.all(np.diff(times) > 0), (fail_after, times))       # no frame twice
            self.assertTrue(np.all(np.diff(times) < 1.5 * 1000 / 60), fail_after)  # no frame missing
            self.assertAlmostEqual(times[-1], 89 * 1000 / 60, delta=1)
            self.assertEqual(len(levels), len(times))

    def test_no_fallback_at_the_end(self):
        """The end of the last chapter is the end - not a failure."""
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'v.mp4')
        write_numbered_video(p, 0, 15)
        real_open = vs._open
        calls = []

        def fake_open(path, hardware=True, threads=0):
            calls.append(hardware)
            cap, _ = real_open(path, False, threads)
            return cap, 'ffmpeg-hw' if hardware else 'ffmpeg'

        with mock.patch.object(vs, '_open', fake_open):
            self.assertEqual(len(read_all(vs.VideoSource(p, hardware=True))[0]), 15)
        self.assertEqual(calls, [True])


if __name__ == '__main__':
    unittest.main()
