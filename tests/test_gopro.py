"""Tests for the GoPro workflow: timecodes, offset sign, chapters, board-frame pairing.

    venv/bin/python -m unittest discover -s tests -v
"""
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import timecode as tc  # noqa: E402
import video_source as vs  # noqa: E402


def _box(typ, payload):
    return struct.pack('>I4s', 8 + len(payload), typ.encode()) + payload


def write_tmcd_mp4(path, counter, timescale=60000, frame_duration=1001, n_frames=60, drop=False):
    """A minimal MP4 with only a timecode track (the layout GoPro files use): ftyp, mdat (the frame counter),
    moov/trak/mdia/{hdlr 'tmcd', minf/stbl/{stsd 'tmcd', stco}}."""
    ftyp = _box('ftyp', b'mp41' + b'\0\0\0\0' + b'mp41')
    mdat = _box('mdat', struct.pack('>I', counter))
    counter_offset = len(ftyp) + 8
    hdlr = _box('hdlr', b'\0' * 4 + b'\0' * 4 + b'tmcd' + b'\0' * 12 + b'GoPro TCD\0')
    entry = (b'\0' * 6 + struct.pack('>H', 1) + b'\0' * 4 +
             struct.pack('>IIIBB', 1 if drop else 0, timescale, frame_duration, n_frames, 0))
    stsd = _box('stsd', b'\0' * 4 + struct.pack('>I', 1) + struct.pack('>I4s', 8 + len(entry), b'tmcd') + entry)
    stco = _box('stco', b'\0' * 4 + struct.pack('>II', 1, counter_offset))
    moov = _box('moov', _box('trak', _box('mdia', hdlr + _box('minf', _box('stbl', stsd + stco)))))
    with open(path, 'wb') as f:
        f.write(ftyp + mdat + moov)


class TimecodeParsing(unittest.TestCase):
    def test_non_drop(self):
        self.assertEqual(tc.timecode_to_frames('01:02:03:04', 60), ((1 * 60 + 2) * 60 + 3) * 60 + 4)
        self.assertEqual(tc.timecode_to_frames('00:00:01:00', 59.94), 60)   # nominal 60 per timecode second

    def test_drop_frame_5994(self):
        # 59.94 drop-frame drops 4 frame numbers each minute except every 10th: 10 min = 35964 real frames
        self.assertEqual(tc.timecode_to_frames('00:10:00;00', 59.94), 35964)
        self.assertEqual(tc.timecode_to_frames('00:01:00;04', 59.94), 3600)  # first label after ;00-;03 dropped
        self.assertEqual(tc.timecode_to_frames('00:00:59;59', 59.94), 3599)

    def test_drop_frame_2997(self):
        self.assertEqual(tc.timecode_to_frames('00:10:00;00', 29.97), 17982)
        self.assertEqual(tc.timecode_to_frames('00:01:00;02', 29.97), 1800)

    def test_round_trip(self):
        for fps, drop in ((59.94, True), (29.97, True), (60, False), (59.94, False), (25, False)):
            for n in (0, 1, 3595, 3596, 3600, 35963, 35964, 123456, 5_000_000):
                label = tc.frames_to_timecode(n, fps, drop)
                self.assertEqual(tc.timecode_to_frames(label, fps, drop), n, (fps, drop, n, label))

    def test_seconds_use_real_rate(self):
        # 1 hour of non-drop 59.94 timecode = 216000 frames = 3603.6 real seconds
        self.assertAlmostEqual(tc.timecode_seconds('01:00:00:00', 60000 / 1001), 3603.6, places=6)
        self.assertAlmostEqual(tc.timecode_seconds('01:00:00:00', 60), 3600.0)

    def test_bad_timecode(self):
        with self.assertRaises(tc.TimecodeError):
            tc.timecode_to_frames('01:02:03', 60)
        with self.assertRaises(tc.TimecodeError):
            tc.timecode_to_frames('01:02:03:60', 60)

    def test_parse_ffprobe(self):
        info = {'streams': [
            {'codec_type': 'video', 'codec_name': 'hevc', 'r_frame_rate': '60000/1001', 'avg_frame_rate': '60000/1001',
             'tags': {'timecode': '13:45:10:22'}},
            {'codec_type': 'audio'},
            {'codec_type': 'data', 'codec_tag_string': 'tmcd', 'tags': {'timecode': '13:45:10:22', 'handler_name': 'GoPro TCD'}},
            {'codec_type': 'data', 'codec_tag_string': 'gpmd'}],
            'format': {'tags': {'timecode': '13:45:10:22'}}}
        t, fps = tc.parse_ffprobe(info)
        self.assertEqual(t, '13:45:10:22')
        self.assertAlmostEqual(fps, 59.94006, places=4)
        # Only the format tag
        t, fps = tc.parse_ffprobe({'streams': [{'codec_type': 'video', 'r_frame_rate': '60/1'}],
                                   'format': {'tags': {'timecode': '01:00:00;00'}}})
        self.assertEqual((t, fps), ('01:00:00;00', 60.0))


class TimecodeFromFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_read_tmcd(self):
        p = os.path.join(self.dir, 'GX010001.MP4')
        write_tmcd_mp4(p, counter=2_975_342, drop=False)
        counter, rate, drop = tc.read_tmcd(p)
        self.assertEqual(counter, 2_975_342)
        self.assertAlmostEqual(rate, 59.94006, places=4)
        self.assertFalse(drop)
        write_tmcd_mp4(p, counter=35964, drop=True)
        self.assertEqual(tc.read_tmcd(p)[2], True)

    def test_offset_sign(self):
        """offset_s = seconds into video A at which video B started (B started later -> positive)."""
        a, b = os.path.join(self.dir, 'A.MP4'), os.path.join(self.dir, 'B.MP4')
        start_a = tc.timecode_to_frames('10:00:00:00', 59.94)
        write_tmcd_mp4(a, counter=start_a)
        write_tmcd_mp4(b, counter=start_a + 330)        # 330 frames later
        with mock.patch('shutil.which', return_value=None), \
                mock.patch('cv2.VideoCapture') as cap:   # the test files have no video: frame rate from tmcd
            cap.return_value.get.return_value = 60000 / 1001
            off, ia, ib = tc.timecode_offset(a, b)
            self.assertAlmostEqual(off, 330 / (60000 / 1001), places=6)
            self.assertGreater(off, 0)
            self.assertEqual(ia['timecode'], '10:00:00:00')
            off2, _, _ = tc.timecode_offset(b, a)
            self.assertAlmostEqual(off2, -off)

    def test_midnight(self):
        a, b = os.path.join(self.dir, 'A.MP4'), os.path.join(self.dir, 'B.MP4')
        write_tmcd_mp4(a, counter=tc.timecode_to_frames('23:59:58:00', 60), timescale=60, frame_duration=1)
        write_tmcd_mp4(b, counter=tc.timecode_to_frames('00:00:02:00', 60), timescale=60, frame_duration=1)
        with mock.patch('shutil.which', return_value=None), mock.patch('cv2.VideoCapture') as cap:
            cap.return_value.get.return_value = 60.0
            off, _, _ = tc.timecode_offset(a, b)
        self.assertAlmostEqual(off, 4.0)

    def test_same_offset_convention_as_session_timeline(self):
        """session_analysis puts camera B's frame time t at t + offset_s on the session (camera A) clock."""
        import session_analysis as sa
        s = sa.Session.__new__(sa.Session)
        s.state = {'offset_s': 5.0}
        s.cams = {'a': {'times': np.arange(0, 20000, 1000 / 60), 'meta': {'fps': 60}},
                  'b': {'times': np.arange(0, 30000, 1000 / 60), 'meta': {'fps': 60}}}
        s.A, s.B = s.cams['a'], s.cams['b']
        s._timeline()
        self.assertEqual(s.coverage['b'][0], 5.0)          # B's first frame is 5 s into A
        f = s.to_master('b', 0)
        self.assertAlmostEqual(s.times[f], 5000.0, places=3)
        self.assertEqual(s.idx['a'][f], 300)                 # = camera A's frame at 5 s
        self.assertAlmostEqual(s.times[-1], 35000, delta=20)  # B runs on after A stopped


class Chapters(unittest.TestCase):
    def test_detect_and_order(self):
        d = tempfile.mkdtemp()
        for name in ('GX030123.MP4', 'GX010123.MP4', 'GX020123.MP4', 'GX010124.MP4', 'GH010123.MP4', 'notes.txt'):
            open(os.path.join(d, name), 'w').close()
        want = [os.path.join(d, n) for n in ('GX010123.MP4', 'GX020123.MP4', 'GX030123.MP4')]
        self.assertEqual(vs.chapters(os.path.join(d, 'GX010123.MP4')), want)
        self.assertEqual(vs.chapters(os.path.join(d, 'GX020123.MP4')), want)      # picked a later chapter
        self.assertEqual(vs.first_chapter(os.path.join(d, 'GX030123.MP4')), want[0])
        self.assertEqual(vs.chapters(os.path.join(d, 'GX010124.MP4')), [os.path.join(d, 'GX010124.MP4')])
        self.assertEqual(vs.chapters(os.path.join(d, 'GH010123.MP4')), [os.path.join(d, 'GH010123.MP4')])
        self.assertEqual(vs.chapters('/x/video 1.mov'), ['/x/video 1.mov'])       # not a GoPro name

    def test_lowercase(self):
        d = tempfile.mkdtemp()
        for name in ('gx010007.mp4', 'gx020007.mp4'):
            open(os.path.join(d, name), 'w').close()
        self.assertEqual(len(vs.chapters(os.path.join(d, 'gx010007.mp4'))), 2)


def write_numbered_video(path, first, n, fps=60, size=(320, 240)):
    """A small video whose frame k shows the number first + k (as a gray level, 2 per step)."""
    import cv2
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for k in range(n):
        w.write(np.full((size[1], size[0], 3), (2 * (first + k)) % 256, np.uint8))
    w.release()


class ChapteredVideo(unittest.TestCase):
    def test_continuous_reading(self):
        import cv2
        d = tempfile.mkdtemp()
        write_numbered_video(os.path.join(d, 'GX010099.MP4'), 0, 50)
        write_numbered_video(os.path.join(d, 'GX020099.MP4'), 50, 40)
        for hardware in (False, True):
            src = vs.VideoSource(os.path.join(d, 'GX010099.MP4'), hardware=hardware)
            self.assertEqual(len(src.paths), 2)
            self.assertEqual(src.get(cv2.CAP_PROP_FRAME_COUNT), 90)
            levels, times = [], []
            while True:
                ok, fr = src.read()
                if not ok:
                    break
                levels.append(int(fr[0, 0, 0]))
                times.append(src.get(cv2.CAP_PROP_POS_MSEC))
            self.assertEqual(len(levels), 90, hardware)
            self.assertTrue(np.all(np.diff(levels) >= 0), 'frames in order across the chapter boundary')
            self.assertGreater(levels[50], levels[49] - 1)
            if not hardware:
                ref = levels
            self.assertTrue(np.all(np.diff(times) > 0), 'timestamps run on across the chapters')
            self.assertAlmostEqual(times[50] - times[49], 1000 / 60, delta=2)
            src.release()
        # Exact jump into the second chapter (FFmpeg path)
        src = vs.VideoSource(os.path.join(d, 'GX010099.MP4'), hardware=False)
        src.set(cv2.CAP_PROP_POS_FRAMES, 70)
        ok, fr = src.read()
        self.assertAlmostEqual(int(fr[0, 0, 0]), ref[70], delta=1)
        self.assertNotAlmostEqual(int(fr[0, 0, 0]), ref[60], delta=3)
        self.assertEqual(vs.total_size(os.path.join(d, 'GX010099.MP4')),
                         sum(os.path.getsize(os.path.join(d, f)) for f in ('GX010099.MP4', 'GX020099.MP4')))


if __name__ == '__main__':
    unittest.main()
