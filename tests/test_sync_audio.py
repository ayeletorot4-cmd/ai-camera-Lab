"""Tests for the sound offset (sync_videos.py): afconvert on macOS, ffmpeg elsewhere, none -> a clear message.

    venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sync_videos as sv  # noqa: E402


def fake_tool(sounds, commands):
    """subprocess.run stand-in: writes the 'sound track' of the video named in the command to the .wav."""
    def run(cmd, check, capture_output):
        commands.append(cmd)
        video = next(a for a in cmd if os.path.basename(a) in sounds)
        wavfile.write(cmd[-1], sv.AUDIO_RATE, sounds[os.path.basename(video)])
    return run


class AudioOffset(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        noise = (rng.normal(0, 3000, 8 * sv.AUDIO_RATE)).astype(np.int16)
        self.lag = 1.25                                                 # B started 1.25 s into A
        k = int(self.lag * sv.AUDIO_RATE)
        self.sounds = {'A.MP4': noise, 'B.MP4': noise[k:]}
        self.dir = tempfile.mkdtemp()
        for name in self.sounds:
            open(os.path.join(self.dir, name), 'w').close()

    def offset(self, tools):
        commands = []
        which = lambda name: tools.get(name)
        with mock.patch('shutil.which', side_effect=which), \
                mock.patch('subprocess.run', side_effect=fake_tool(self.sounds, commands)):
            off, conf = sv.audio_offset(os.path.join(self.dir, 'A.MP4'), os.path.join(self.dir, 'B.MP4'))
        return off, conf, commands

    def test_ffmpeg(self):
        off, conf, commands = self.offset({'ffmpeg': r'C:\ffmpeg\bin\ffmpeg.exe'})
        self.assertAlmostEqual(off, self.lag, places=3)
        self.assertGreater(conf, 1.5)
        self.assertEqual(commands[0][0], r'C:\ffmpeg\bin\ffmpeg.exe')
        self.assertIn('-ar', commands[0])
        self.assertEqual(commands[0][commands[0].index('-ar') + 1], str(sv.AUDIO_RATE))

    def test_afconvert_first(self):
        off, _, commands = self.offset({'afconvert': '/usr/bin/afconvert', 'ffmpeg': '/opt/homebrew/bin/ffmpeg'})
        self.assertAlmostEqual(off, self.lag, places=3)
        self.assertEqual(commands[0][0], 'afconvert')

    def test_no_tool(self):
        with mock.patch('shutil.which', return_value=None):
            self.assertIsNone(sv.audio_tool())
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg'):
                sv.audio_offset(os.path.join(self.dir, 'A.MP4'), os.path.join(self.dir, 'B.MP4'))


if __name__ == '__main__':
    unittest.main()
