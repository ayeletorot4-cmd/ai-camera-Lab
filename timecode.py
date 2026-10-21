"""Start timecode of a (GoPro) video and the offset between two cameras synced by timecode.

GoPro HERO cameras synced in the Quik app ("Sync Timecode", QR code) write the time of day as a timecode
track ('tmcd') into every file. Each camera starts recording on its own, so two files don't start at the same
moment; the start timecodes tell how they line up:

    offset_s = start(B) - start(A)      = "seconds into video A at which video B started"
                                          (the convention of offset_s everywhere in this project)

A timecode HH:MM:SS:FF (non-drop) or HH:MM:SS;FF (drop-frame) counts frames at the nominal rate (60 for
59.94 fps); it is turned into a frame number and then into seconds with the file's real frame rate, so a
59.94 fps camera's clock is right.

Reading: ffprobe (part of FFmpeg) if installed - the 'timecode' tag of the tmcd stream or of the file; else
the timecode track is read straight from the MP4/MOV file (built-in, no FFmpeg needed).
"""
import json
import os
import shutil
import struct
import subprocess

import video_source as vs

FFPROBE_HELP = ('ffprobe (FFmpeg) is not installed. On a Mac: install Homebrew (https://brew.sh), then run '
                '"brew install ffmpeg"; on Windows: download FFmpeg from https://ffmpeg.org/download.html and add '
                'its bin folder to PATH. (The app can also read the timecode itself; ffprobe is a second opinion.)')


class TimecodeError(Exception):
    pass


# -----------------------------
# Timecode <-> frames
# -----------------------------
def nominal_rate(fps):
    """Frames per timecode second: 59.94 -> 60, 29.97 -> 30, 25 -> 25."""
    return int(round(fps))


def timecode_to_frames(tc, fps, drop_frame=None):
    """'HH:MM:SS:FF' / 'HH:MM:SS;FF' (';' or '.' = drop-frame) -> frame number since 00:00:00:00."""
    tc = tc.strip()
    if drop_frame is None:
        drop_frame = ';' in tc or '.' in tc       # HH:MM:SS;FF (or HH:MM:SS.FF) = drop-frame
    parts = tc.replace(';', ':').replace('.', ':').split(':')
    if len(parts) != 4:
        raise TimecodeError(f'Not a timecode: {tc!r}')
    h, m, s, f = (int(p) for p in parts)
    rate = nominal_rate(fps)
    if not 0 <= f < rate:
        raise TimecodeError(f'Frame {f} out of range for {rate} frames per second in {tc!r}')
    frames = ((h * 60 + m) * 60 + s) * rate + f
    if drop_frame:
        # Drop-frame: frame numbers 0..k-1 are skipped at the start of every minute except every 10th minute
        # (k = 2 at 29.97, 4 at 59.94), so that the timecode keeps up with the real clock
        k = round(rate / 15)
        minutes = h * 60 + m
        frames -= k * (minutes - minutes // 10)
    return frames


def frames_to_timecode(n, fps, drop_frame=False):
    rate = nominal_rate(fps)
    if drop_frame:
        k = round(rate / 15)
        per10 = rate * 600 - 9 * k          # real frames in 10 minutes
        per1 = rate * 60 - k                # real frames in a minute that drops
        d, r = divmod(n, per10)
        n += 9 * k * d + (k * ((r - k) // per1) if r > k else 0)
    f = n % rate
    s = n // rate
    sep = ';' if drop_frame else ':'
    return f'{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}{sep}{f:02d}'


def timecode_seconds(tc, fps, drop_frame=None):
    """Seconds since 00:00:00:00, using the real frame rate (e.g. 59.94)."""
    return timecode_to_frames(tc, fps, drop_frame) / fps


# -----------------------------
# Reading a file's start timecode
# -----------------------------
def _ffprobe(path):
    exe = shutil.which('ffprobe')
    if exe is None:
        return None
    out = subprocess.run([exe, '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', path],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise TimecodeError(f'ffprobe failed on {path}: {out.stderr.strip()[:300]}')
    return json.loads(out.stdout)


def _rate(text):
    if not text or text in ('0/0', '0'):
        return None
    if '/' in text:
        a, b = text.split('/')
        return float(a) / float(b) if float(b) else None
    return float(text)


def parse_ffprobe(info):
    """ffprobe JSON -> (timecode string, video frame rate). Prefers the tmcd stream's tag."""
    streams = info.get('streams', [])
    video = next((s for s in streams if s.get('codec_type') == 'video'), {})
    fps = _rate(video.get('r_frame_rate')) or _rate(video.get('avg_frame_rate'))
    tc = None
    for s in streams:
        if s.get('codec_tag_string') == 'tmcd' or s.get('codec_type') == 'data':
            tc = (s.get('tags') or {}).get('timecode') or tc
    tc = tc or (video.get('tags') or {}).get('timecode') or (info.get('format', {}).get('tags') or {}).get('timecode')
    return tc, fps


def _boxes(f, start, end):
    """(type, data start, box end) of the MP4 boxes in [start, end)."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, typ = struct.unpack('>I4s', f.read(8))
        hdr = 8
        if size == 1:
            size = struct.unpack('>Q', f.read(8))[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            return
        yield typ.decode('latin1'), pos + hdr, pos + size
        pos += size


def _child(f, start, end, typ):
    return next(((s, e) for t, s, e in _boxes(f, start, end) if t == typ), None)


def read_tmcd(path):
    """The timecode track straight from an MP4/MOV file -> (frame counter, frames per second of the timecode,
    drop_frame) or None if the file has no timecode track."""
    with open(path, 'rb') as f:
        size = os.path.getsize(path)
        moov = _child(f, 0, size, 'moov')
        if moov is None:
            return None
        for typ, s, e in _boxes(f, *moov):
            if typ != 'trak':
                continue
            mdia = _child(f, s, e, 'mdia')
            hdlr = mdia and _child(f, *mdia, 'hdlr')
            if not hdlr:
                continue
            f.seek(hdlr[0] + 8)
            if f.read(4) != b'tmcd':
                continue
            minf = _child(f, *mdia, 'minf')
            stbl = minf and _child(f, *minf, 'stbl')
            stsd = stbl and _child(f, *stbl, 'stsd')
            if not stsd:
                continue
            # stsd: version/flags(4) count(4), entry: size(4) 'tmcd'(4) reserved(6) dref(2) reserved(4)
            #       flags(4) timescale(4) frameDuration(4) numberOfFrames(1)
            f.seek(stsd[0] + 8 + 8 + 8 + 4)
            flags, timescale, frame_dur, n_frames = struct.unpack('>IIIB', f.read(13))
            co = _child(f, *stbl, 'stco')
            if co:
                f.seek(co[0] + 8)
                offset = struct.unpack('>I', f.read(4))[0]
            else:
                co = _child(f, *stbl, 'co64')
                if not co:
                    continue
                f.seek(co[0] + 8)
                offset = struct.unpack('>Q', f.read(8))[0]
            f.seek(offset)
            counter = struct.unpack('>I', f.read(4))[0]
            fps = timescale / frame_dur if frame_dur else n_frames
            return counter, fps, bool(flags & 1)
    return None


def start_timecode(path, chapter_only=False):
    """dict(timecode, fps, seconds, source) for the file (its first chapter, for GoPro chapters - or, with
    chapter_only, the start of that chapter itself)."""
    if not chapter_only:
        path = vs.first_chapter(path)
    info = None
    try:
        info = _ffprobe(path)
    except (TimecodeError, OSError, ValueError, subprocess.TimeoutExpired):
        info = None
    if info is not None:
        tc, fps = parse_ffprobe(info)
        if tc and fps:
            return dict(timecode=tc, fps=fps, seconds=timecode_seconds(tc, fps), source='ffprobe')
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    t = read_tmcd(path)
    if t is None:
        raise TimecodeError(f'{path} has no timecode track (is it a GoPro file recorded after "Sync Timecode"?)'
                            + ('' if shutil.which('ffprobe') else '\n' + FFPROBE_HELP))
    counter, tc_rate, drop = t
    tc = frames_to_timecode(counter, tc_rate, drop)
    # The counter counts frames at the timecode's rate; seconds with the video's real frame rate
    return dict(timecode=tc, fps=fps or tc_rate, seconds=counter / (fps or tc_rate), source='file')


def timecode_offset(path_a, path_b):
    """Seconds into video A at which video B started, from the start timecodes (+ both readings)."""
    a, b = start_timecode(path_a), start_timecode(path_b)
    if abs(a['fps'] - b['fps']) > 0.01:
        raise TimecodeError(f"The videos have different frame rates ({a['fps']:.3f} / {b['fps']:.3f} fps) - "
                            'record both cameras with the same settings.')
    off = b['seconds'] - a['seconds']
    if off > 12 * 3600:          # recorded across midnight
        off -= 24 * 3600
    elif off < -12 * 3600:
        off += 24 * 3600
    return off, a, b


def chapter_gaps(path):
    """Check that a clip's chapters follow each other without gaps: each chapter's start timecode should be the
    previous one's start + its duration. Returns warning strings (empty = fine or not checkable)."""
    paths = vs.chapters(path)
    if len(paths) < 2:
        return []
    src = vs.VideoSource(path, hardware=False)
    out = []
    try:
        starts = [start_timecode(p, chapter_only=True)['seconds'] for p in paths]
    except TimecodeError:
        return []   # no timecode: only the frame counts can be summed (done in VideoSource)
    for k in range(1, len(paths)):
        expected = starts[k - 1] + src.info[k - 1]['duration_ms'] / 1000
        gap = starts[k] - expected
        if abs(gap) > 1.5 / src.info[k - 1]['fps']:
            out.append(f'{os.path.basename(paths[k])} starts {gap * 1000:+.0f} ms after the end of the previous chapter'
                       ' - the chapters are not continuous.')
    return out
