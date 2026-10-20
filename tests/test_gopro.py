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


class FakeVideo:
    """Frames are their own frame numbers; timestamps n / fps (ms) - like VideoSource."""

    def __init__(self, n, fps):
        self.n, self.fps, self.i, self.ms = n, fps, -1, 0.0

    def grab(self):
        if self.i + 1 >= self.n:
            return False
        self.i += 1
        self.ms = self.i / self.fps * 1000
        return True

    def retrieve(self):
        return True, self.i

    def get(self, prop):
        return self.ms


class BoardPairing(unittest.TestCase):
    def test_pairs_by_time(self):
        import calibrate_cameras as cc
        fps = 60000 / 1001
        offset = 2.5          # B started 2.5 s into A
        times = cc.sample_times(10.0, 10.0, offset, start_s=0, seconds=None, per_second=4)
        self.assertAlmostEqual(times[0], 2.5)                    # no moment before B started
        self.assertLess(times[-1], 10.0)                           # A ends first (at 10 s)
        pairs = list(cc.paired_frames(FakeVideo(600, fps), FakeVideo(600, fps), offset, times, fps, fps))
        for T, fa, fb in pairs:
            self.assertEqual(fa, round(T * fps))                   # A's frame at T
            self.assertEqual(fb, round((T - offset) * fps))        # B's frame at T - offset (same moment)
        # B started first (negative offset): A's frame at T, B's later in its own file
        times = cc.sample_times(10.0, 12.0, -1.0, per_second=2)
        self.assertAlmostEqual(times[0], 0.0)
        for T, fa, fb in cc.paired_frames(FakeVideo(600, fps), FakeVideo(720, fps), -1.0, times, fps, fps):
            self.assertEqual(fb, round((T + 1.0) * fps))

    def test_window(self):
        import calibrate_cameras as cc
        t = cc.sample_times(100, 100, 0, start_s=10, seconds=5, per_second=4)
        self.assertAlmostEqual(t[0], 10)
        self.assertAlmostEqual(t[-1], 14.75)


def make_board_videos(folder, offset_frames, n_poses=70, fps=30, size=(1280, 720)):
    """Two synthetic videos of the ChArUco board seen by two cameras with a known geometry; camera B starts
    recording offset_frames later than A. Returns the true (R, t_mm)."""
    import cv2
    import calibrate_cameras as cc
    board, _ = cc.make_board()
    bw, bh = 950, 1330
    tex = board.generateImage((bw, bh), marginSize=0)
    mm_per_px = cc.BOARD_SQUARE_MM * cc.BOARD_SQUARES[0] / bw
    K = cc.K_matrix(1000, size)
    Rb = cv2.Rodrigues(np.array([0, 1, 0]) * np.radians(-40))[0]
    tb = -Rb @ np.array([-900, 0, 300.0])
    rng = np.random.default_rng(1)
    poses = []
    for _ in range(n_poses + offset_frames):
        Rw = cv2.Rodrigues(rng.normal(0, 0.3, 3) + np.array([0, -0.35, 0]))[0]
        tw = np.array([rng.uniform(-250, 50), rng.uniform(-200, 0), rng.uniform(1100, 1500)])
        poses.append((Rw, tw))
    S = np.diag([mm_per_px, mm_per_px, 1.0])
    paths = [os.path.join(folder, 'A.mp4'), os.path.join(folder, 'B.mp4')]
    for v, (R0, t0, first) in enumerate(((np.eye(3), np.zeros(3), 0), (Rb, tb, offset_frames))):
        w = cv2.VideoWriter(paths[v], cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for Rw, tw in poses[first:first + n_poses]:
            R, t = R0 @ Rw, R0 @ tw + t0
            H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
            w.write(cv2.cvtColor(cv2.warpPerspective(tex, H, size, borderValue=150), cv2.COLOR_GRAY2BGR))
        w.release()
    return paths, Rb, tb


class BoardCalibrationWithOffset(unittest.TestCase):
    def test_offset_applied(self):
        import cv2
        import calibrate_cameras as cc
        d = tempfile.mkdtemp()
        fps, k = 30, 15
        (a, b), R_true, t_true = make_board_videos(d, k, fps=fps)
        res = cc.calibrate_board(a, b, offset_s=k / fps, seconds=0, per_second=fps)
        rot_err = np.degrees(np.linalg.norm(cv2.Rodrigues(res['R'] @ R_true.T)[0]))
        self.assertLess(rot_err, 2.5)
        self.assertLess(np.linalg.norm(res['t_cm'] * 10 - t_true) / np.linalg.norm(t_true), 0.05)
        self.assertAlmostEqual(res['square_mean_mm'], 38.0, delta=0.6)
        self.assertLess(res['stereo_rms_px'], 1.0)
        # Lockstep (the old behaviour = offset 0) pairs different moments: it must fail or look clearly wrong
        try:
            bad = cc.calibrate_board(a, b, offset_s=0.0, seconds=0, per_second=fps)
            self.assertTrue(bad['stereo_rms_px'] > 5 or abs(bad['square_mean_mm'] - 38) > 3, bad['stereo_rms_px'])
            self.assertFalse(bad['ok'])
        except ValueError:
            pass


if __name__ == '__main__':
    unittest.main()
