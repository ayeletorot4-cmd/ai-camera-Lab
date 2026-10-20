"""A video that may be split into GoPro chapters, read as one continuous video.

GoPro splits long recordings into ~4 GB chapters with the same 4-digit clip number and an increasing
chapter number: GX010123.MP4, GX020123.MP4, GX030123.MP4 (GX = HEVC, GH = H.264; also GL/GP). The user
picks the first chapter; the following ones are found automatically. VideoSource behaves like
cv2.VideoCapture (read / grab / retrieve / get / set / release) over all chapters: frame numbers and
timestamps (CAP_PROP_POS_MSEC) run on across the chapters, so the rest of the app doesn't need to know.

On macOS the frames are decoded by the hardware decoder (AVFoundation - also for GoPro HEVC/H.265);
jumping to a time there is approximate (see session_engine.py: workers jump early and skip frames).
"""
import os
import re
import sys

import cv2

GOPRO_NAME = re.compile(r'^(G[HXLP])(\d{2})(\d{4})(\.mp4)$', re.IGNORECASE)


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


def _open(path, hardware=True):
    if hardware and sys.platform == 'darwin':
        cap = cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(path)


class VideoSource:
    """cv2.VideoCapture-like reader over all chapters of a clip (or a single video)."""

    def __init__(self, path, hardware=True, chapter_paths=None):
        self.paths = chapter_paths or chapters(path)
        self.hardware = hardware
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
        self.cap = _open(self.paths[0], hardware) if self.paths else None
        self.pos_frame = 0       # frame number of the next frame to read (global)
        self.last_ms = 0.0

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

    def _next_chapter(self):
        if self.k + 1 >= len(self.paths):
            return False
        self.cap.release()
        self.k += 1
        self.cap = _open(self.paths[self.k], self.hardware)
        self.pos_frame = self.start_frame[self.k]
        return self.cap.isOpened()

    def grab(self):
        while self.cap is not None:
            if self.cap.grab():
                self.last_ms = self.start_ms[self.k] + self.cap.get(cv2.CAP_PROP_POS_MSEC)
                self.pos_frame += 1
                return True
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
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, local)
        if prop == cv2.CAP_PROP_POS_MSEC:
            k = max(i for i, s in enumerate(self.start_ms) if s <= max(0.0, value)) if self.info else 0
            local = value - self.start_ms[k]
            self._goto(k)
            self.pos_frame = self.start_frame[k] + int(local / 1000 * self.info[k]['fps'])
            return self.cap.set(cv2.CAP_PROP_POS_MSEC, local)
        return self.cap.set(prop, value)

    def _goto(self, k):
        if k != self.k:
            self.cap.release()
            self.k = k
            self.cap = _open(self.paths[k], self.hardware)

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
