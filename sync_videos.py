"""Cut two videos of the same session (two cameras recording together) so they are frame-by-frame in sync.

The time offset between the videos is found by cross-correlating their sound tracks (macOS: afconvert), or given
with --offset. Both outputs cover only the time both cameras recorded and have the same number of frames:
frame i of one output was taken at the same moment as frame i of the other (to within half a frame, ~8 ms at 60 fps).
Each frame of the second video is the one whose timestamp is closest to the first video's frame, so a small
frame-rate difference between the cameras does not build up.

Outputs (in media/synced/): <video A>_synced.mp4, <video B>_synced.mp4 (no sound) and a .json with the offset.
For cameras that record with a shared timecode (e.g. two synced GoPros) this step is not needed.

  venv/bin/python sync_videos.py media/IMG_0122.mov media/IMG_7557.mov
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

AUDIO_RATE = 16000


def read_audio(video_path, tmp_dir, max_s=600):
    """Mono sound track of the video (its GoPro chapters one after the other) at AUDIO_RATE, at most max_s
    seconds (needs macOS afconvert)."""
    import video_source as vs
    parts, total = [], 0
    for k, p in enumerate(vs.chapters(video_path)):
        wav = os.path.join(tmp_dir, f'{k}_{os.path.basename(p)}.wav')
        subprocess.run(['afconvert', '-f', 'WAVE', '-d', f'LEI16@{AUDIO_RATE}', '-c', '1', p, wav],
                       check=True, capture_output=True)
        _, x = wavfile.read(wav)
        parts.append(x.astype(np.float64))
        total += len(x)
        if max_s and total >= max_s * AUDIO_RATE:
            break
    x = np.concatenate(parts)[:int(max_s * AUDIO_RATE) if max_s else None]
    return (x - x.mean()) / (x.std() + 1e-9)


def audio_offset(path_a, path_b, max_s=600):
    """Seconds into video A at which video B started (negative: B started first), and a confidence ratio.
    Only the first max_s seconds of each video are used (enough unless the cameras started > max_s apart)."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = read_audio(path_a, tmp, max_s), read_audio(path_b, tmp, max_s)
    c = np.abs(correlate(a, b, mode='full', method='fft'))
    peak = int(np.argmax(c))
    # Confidence: the peak compared with the best peak more than 0.5 s away from it
    far = c.copy()
    far[max(0, peak - AUDIO_RATE // 2):peak + AUDIO_RATE // 2] = 0
    return (peak - (len(b) - 1)) / AUDIO_RATE, float(c[peak] / max(far.max(), 1e-9))


def open_video_writer(path, fps, size):
    """MP4 writer: H.264 (plays everywhere, also in browsers) if this OpenCV build has it, else MPEG-4 Part 2."""
    for fourcc in ('avc1', 'mp4v'):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
    return writer


def frame_times(path):
    """Timestamp (s) of every frame, read from the container (handles variable frame rate)."""
    cap = cv2.VideoCapture(path)
    times = []
    while cap.grab():
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    cap.release()
    return np.array(times)


def main():
    parser = argparse.ArgumentParser(description='Cut two videos of the same session so they are in sync.')
    parser.add_argument('video_a')
    parser.add_argument('video_b')
    parser.add_argument('--offset', type=float,
                        help='Seconds into video A at which video B started (default: found from the sound)')
    args = parser.parse_args()

    if args.offset is None:
        if sys.platform != 'darwin':
            sys.exit('Finding the offset from the sound needs macOS (afconvert) - give it with --offset.')
        offset, confidence = audio_offset(args.video_a, args.video_b)
        print(f'Offset from the sound: video B starts {offset:+.4f} s into video A '
              f'(peak {confidence:.1f}x the next best match)')
        if confidence < 1.5:
            print('WARNING: the sound match is not clear - check the result, or give --offset.')
    else:
        offset = args.offset

    print('Reading frame timestamps...')
    ta, tb = frame_times(args.video_a), frame_times(args.video_b)
    tb_in_a = tb + offset  # video B's frame times on video A's clock
    half_frame = 0.5 * np.median(np.diff(tb))
    keep_a = np.nonzero((ta >= tb_in_a[0] - half_frame) & (ta <= tb_in_a[-1] + half_frame))[0]
    match_b = np.clip(np.searchsorted(tb_in_a, ta[keep_a]), 1, len(tb) - 1)
    match_b -= (ta[keep_a] - tb_in_a[match_b - 1]) < (tb_in_a[match_b] - ta[keep_a])  # nearest
    err = np.abs(tb_in_a[match_b] - ta[keep_a])
    print(f'Common part: {ta[keep_a[0]]:.2f}-{ta[keep_a[-1]]:.2f} s of video A, {len(keep_a)} frames '
          f'(timing error mean {err.mean() * 1000:.1f} ms, max {err.max() * 1000:.1f} ms)')

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media', 'synced')
    os.makedirs(out_dir, exist_ok=True)
    names = [os.path.splitext(os.path.basename(p))[0] for p in (args.video_a, args.video_b)]
    out_paths = [os.path.join(out_dir, f'{n}_synced.mp4') for n in names]
    fps = 1 / np.median(np.diff(ta))

    # Write both outputs in one pass over each input: frame i of output A <- frame keep_a[i] of A,
    # frame i of output B <- frame match_b[i] of B (a B frame may be used twice or skipped)
    for path, wanted, out_path in ((args.video_a, keep_a, out_paths[0]), (args.video_b, match_b, out_paths[1])):
        cap = cv2.VideoCapture(path)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = open_video_writer(out_path, fps, (w, h))
        idx, frame, written = -1, None, 0
        for target in wanted:
            while idx < target:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
            writer.write(frame)
            written += 1
            if written % 120 == 0:
                print(f'\r  {os.path.basename(out_path)}: frame {written}/{len(wanted)}', end='', flush=True)
        writer.release()
        cap.release()
        print(f'\r  {os.path.basename(out_path)}: {written} frames written      ')

    info = dict(video_a=os.path.basename(args.video_a), video_b=os.path.basename(args.video_b),
                offset_s=round(offset, 4), fps=round(fps, 4), frames=len(keep_a),
                start_in_a_s=round(float(ta[keep_a[0]]), 4), start_in_b_s=round(float(tb[match_b[0]]), 4),
                outputs=[os.path.basename(p) for p in out_paths])
    with open(os.path.join(out_dir, f'{names[0]}+{names[1]}.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print('Saved:', *out_paths, sep='\n  ')


if __name__ == '__main__':
    main()
