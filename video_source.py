"""A video that may be split into GoPro chapters, read as one continuous video.

GoPro splits long recordings into ~4 GB chapters with the same 4-digit clip number and an increasing
chapter number: GX010123.MP4, GX020123.MP4, GX030123.MP4 (GX = HEVC, GH = H.264; also GL/GP). The user
picks the first chapter; the following ones are found automatically. VideoSource behaves like
cv2.VideoCapture (read / grab / retrieve / get / set / release) over all chapters: frame numbers and
timestamps (CAP_PROP_POS_MSEC) run on across the chapters, so the rest of the app doesn't need to know.

On macOS the frames are decoded by the hardware decoder (AVFoundation - also for GoPro HEVC/H.265);
jumping to a time there is approximate (see session_engine.py: workers jump early and skip frames).
On Windows / Linux the graphics card's decoder (FFmpeg + D3D11 / VAAPI ...) is used when LAB_HW_DECODE=1 is
set; if it can't open a video or fails in the middle, the normal decoder takes over. It is off by default: the
graphics card converts the colours a little differently (pixel values differ by up to ~20 levels), which can
change the measured switch light levels.
"""
import os
import re
import sys

import cv2

GOPRO_NAME = re.compile(r'^(G[HXLP])(\d{2})(\d{4})(\.mp4)$', re.IGNORECASE)


def hardware_decoding():
    """Use the graphics card's decoder where hardware=True is asked for? macOS: yes (AVFoundation); elsewhere
    only with LAB_HW_DECODE=1 (LAB_HW_DECODE=0 turns it off on macOS too)."""
    env = os.environ.get('LAB_HW_DECODE', '').strip()
    if env in ('0', '1'):
        return env == '1'
    return sys.platform == 'darwin'


def chapters(path):
    """All chapters of the clip `path` belongs to, in order (first chapter first). A video that is not a
    GoPro chapter -> [path]."""
    folder, name = os.path.split(os.path.abspath(path))
    m = GOPRO_NAME.match(name)
    if not m:
        return [path]
    prefix, _, clip, _ = m.groups()
    found = []
    for f in os.listdir(folder or '.'):
        mm = GOPRO_NAME.match(f)
        if mm and mm.group(1).upper() == prefix.upper() and mm.group(3) == clip:
            found.append((int(mm.group(2)), os.path.join(folder, f)))
    return [p for _, p in sorted(found)] or [path]


def first_chapter(path):
    return chapters(path)[0]


def total_size(path):
    """Bytes of all chapters (used to tell whether a saved result still belongs to this video)."""
    return sum(os.path.getsize(p) for p in chapters(path) if os.path.exists(p))


def _open(path, hardware=True, threads=0):
    """(capture, decoder name). threads: FFmpeg decoding threads (0 = OpenCV's default, one per CPU)."""
    params = [cv2.CAP_PROP_N_THREADS, int(threads)] if threads else []
    if hardware and hardware_decoding():
        if sys.platform == 'darwin':
            cap = cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap, 'avfoundation'
        else:
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                                   params + [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            if cap.isOpened() and cap.get(cv2.CAP_PROP_HW_ACCELERATION) not in (0, cv2.VIDEO_ACCELERATION_NONE):
                return cap, 'ffmpeg-hw'
            cap.release()
    if params:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
        if cap.isOpened():
            return cap, 'ffmpeg'
        cap.release()
    return cv2.VideoCapture(path), 'ffmpeg'


class VideoSource:
    """cv2.VideoCapture-like reader over all chapters of a clip (or a single video)."""

    def __init__(self, path, hardware=True, chapter_paths=None, threads=0):
        self.paths = chapter_paths or chapters(path)
        self.hardware = hardware
        self.threads = threads
        self.decoder = None      # of the open chapter: 'ffmpeg', 'ffmpeg-hw' or 'avfoundation'
        self.info = []          # per chapter: frames, fps, width, height, duration_ms
        for p in self.paths:
            cap = cv2.VideoCapture(p)   # metadata (frame count etc.) is most reliable from FFmpeg
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.info.append(dict(path=p, frames=n, fps=fps, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), duration_ms=n / fps * 1000))
            cap.release()
        self.start_frame, self.start_ms = [], []
        f = t = 0
        for c in self.info:
            self.start_frame.append(f)
            self.start_ms.append(t)
            f += c['frames']
            t += c['duration_ms']
        self.total_frames, self.total_ms = f, t
        self.k = 0
        self.cap = None
        if self.paths:
            self._open_chapter(0)
        self.pos_frame = 0       # frame number of the next frame to read (global)
        self.last_ms = 0.0
        self._chapter_ms = None  # chapter time of the last frame read from this chapter (None: none yet)
        self._seek = None        # the last jump in this chapter: (property, chapter value)
        self._after_ms = None    # after a decoder change: skip frames up to this chapter time (already read)

    # --- cv2.VideoCapture interface ---
    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def get(self, prop):
        c = self.info[0] if self.info else {}
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        if prop == cv2.CAP_PROP_FPS:
            return c.get('fps', 0.0)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(c.get('width', 0))
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(c.get('height', 0))
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.last_ms
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos_frame)
        return self.cap.get(prop) if self.cap is not None else 0.0

    def _open_chapter(self, k, hardware=None):
        if self.cap is not None:
            self.cap.release()
        self.k = k
        self.cap, self.decoder = _open(self.paths[k], self.hardware if hardware is None else hardware, self.threads)

    def _next_chapter(self):
        if self.k + 1 >= len(self.paths):
            return False
        self._open_chapter(self.k + 1)
        self._chapter_ms = self._seek = None
        self.pos_frame = self.start_frame[self.k]
        return self.cap.isOpened()

    def _software_fallback(self):
        """The graphics card's decoder stopped before the end of the chapter: go on with the normal decoder after
        the last frame read (by its timestamp - frame numbers are only estimates after a jump by time). False at
        the chapter's real end or if the normal decoder is already in use."""
        c = self.info[self.k]
        frame_ms = 1000 / c['fps']
        last = self._chapter_ms
        if self.decoder != 'ffmpeg-hw' or (last is not None and last >= c['duration_ms'] - 1.5 * frame_ms):
            return False
        self._open_chapter(self.k, hardware=False)
        if not self.cap.isOpened():
            return False
        if last is not None:            # a little before the last frame read, then skip up to it
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(last / frame_ms)) - 2))
            self._after_ms = last + 0.25 * frame_ms
        elif self._seek is not None:    # nothing read since the jump: the same jump again
            self.cap.set(*self._seek)
        return True

    def grab(self):
        while self.cap is not None:
            if self.cap.grab():
                local_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if self._after_ms is not None:
                    if local_ms <= self._after_ms:
                        continue
                    self._after_ms = None
                self._chapter_ms = local_ms
                self.last_ms = self.start_ms[self.k] + local_ms
                self.pos_frame += 1
                return True
            if self._software_fallback():
                continue
            if not self._next_chapter():
                return False
        return False

    def retrieve(self):
        return self.cap.retrieve()

    def read(self):
        if not self.grab():
            return False, None
        return self.retrieve()

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            k = max(i for i, s in enumerate(self.start_frame) if s <= max(0, value)) if self.info else 0
            local = value - self.start_frame[k]
            self._goto(k)
            self.pos_frame = int(value)
            self._chapter_ms, self._seek = None, (cv2.CAP_PROP_POS_FRAMES, local)
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, local)
        if prop == cv2.CAP_PROP_POS_MSEC:
            k = max(i for i, s in enumerate(self.start_ms) if s <= max(0.0, value)) if self.info else 0
            local = value - self.start_ms[k]
            self._goto(k)
            self.pos_frame = self.start_frame[k] + int(local / 1000 * self.info[k]['fps'])
            self._chapter_ms, self._seek = None, (cv2.CAP_PROP_POS_MSEC, local)
            return self.cap.set(cv2.CAP_PROP_POS_MSEC, local)
        return self.cap.set(prop, value)

    def _goto(self, k):
        if k != self.k:
            self._open_chapter(k)

    def release(self):
        if self.cap is not None:
            self.cap.release()

    # --- checks ---
    def check(self):
        """Warnings: chapters with another frame rate / size than the first one."""
        out = []
        if len(self.info) > 1:
            c0 = self.info[0]
            for c in self.info[1:]:
                if (c['width'], c['height']) != (c0['width'], c0['height']) or abs(c['fps'] - c0['fps']) > 0.01:
                    out.append(f"{os.path.basename(c['path'])}: {c['width']}x{c['height']} {c['fps']:.3f} fps, "
                               f"unlike the first chapter ({c0['width']}x{c0['height']} {c0['fps']:.3f} fps)")
        return out


def read_frame(path, index):
    """One frame by its (global) frame number - exact (FFmpeg decoder, not the hardware one)."""
    src = VideoSource(path, hardware=False)
    if index:
        src.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = src.read()
    src.release()
    return frame if ok else None
