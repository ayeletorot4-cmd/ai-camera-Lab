"""One pass over a session video: everything the analysis needs, per frame, saved to a cache.

For every frame: MediaPipe pose (33 points: x, y px, visibility), the camera movement (first frame -> this
frame), each switch's light level and, per switch, the distance of the closest hand / foot point to the
switch box (in box sizes, <= 0 = inside). The analysis (experiment_design.py, trial_measures.py) then runs in
seconds and can be re-run with other settings.

Speed: on a Mac the video is decoded by the hardware decoder (AVFoundation: ~5 ms per 4K frame instead of
~55 ms); MediaPipe Pose (~25 ms per frame) then runs in several worker processes at the same time, each on its
own part of the video. Parts are cut by time (AVFoundation can't jump to an exact frame number): a worker
jumps 1 s before its part and skips frames until its first timestamp, so the parts join without gaps.

Cache: <output folder>/<video name>_session.npz (re-used if the video and the switch boxes didn't change).
"""
import json
import multiprocessing as mproc
import os
import sys
import time

import cv2
import numpy as np

import video_source as vs

MP_MAX_SIDE = 1280        # frames are shrunk to this before MediaPipe (its models work on ~256 px crops)
MOTION_EVERY = 10         # camera movement is measured every 10th frame (it changes slowly)
TOUCH_POINTS = {          # pose points that can touch a switch (both sides)
    'upper': [15, 16, 17, 18, 19, 20, 21, 22],   # wrists, pinky, index, thumb
    'lower': [27, 28, 29, 30, 31, 32],           # ankles, heels, toes
}


def open_video(path):
    """The video (all its GoPro chapters) with the hardware decoder on macOS - sequential reading; jumping is
    approximate there (see video_source.py)."""
    return vs.VideoSource(path, hardware=True)


def default_workers():
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def cache_path(video_path, out_dir):
    return os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0] + '_session.npz')


def load_cache(video_path, out_dir, boxes=None):
    """The cached per-frame data, or None if missing/outdated (other video file or other switch boxes)."""
    path = cache_path(video_path, out_dir)
    if not os.path.exists(path) or not os.path.exists(video_path):
        return None
    d = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(d['meta']))
    if meta.get('video_size') != vs.total_size(video_path):
        return None
    if boxes is not None and (np.shape(meta['boxes']) != np.shape(boxes) or not np.allclose(meta['boxes'], boxes)):
        return None
    if not meta.get('complete'):   # older caches used a too strict test: check against the video's length
        src = vs.VideoSource(video_path, hardware=False)
        meta['complete'] = meta['frames'] >= src.total_frames - max(2, meta['fps']) * len(src.paths)
    d['meta'] = meta
    return d


def _process_range(job):
    """Worker: frames [start, end) of the video -> per-frame arrays (saved to a temporary .npz)."""
    import extract_kinematics as ek
    import mediapipe as mp
    video_path, boxes, t_start, t_end, out_file, counter = job   # this part: frames with t_start <= ms < t_end
    cap = open_video(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = cap.read()
    camera = ek.CameraMotion(first)
    cap.release()
    cap = open_video(video_path)
    if t_start > 1000:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_start - 1000)
    skip = 3                                   # timestamps right after a jump are not reliable
    scale = min(1.0, MP_MAX_SIDE / max(w, h))
    n_sw = len(boxes)
    light_corners = [ek.box_corners(b, w, h, ek.LIGHT_BOX_MARGIN) for b in boxes]
    touch_corners = [ek.box_corners(b, w, h, 0.0) for b in boxes]
    sizes = np.array([np.sqrt((b[2] - b[0]) * w * (b[3] - b[1]) * h) for b in boxes])
    times, pose, motion, light, tu, tl = [], [], [], [], [], []
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                model_complexity=1) as model:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t_start > 1000 and skip > 0:
                skip -= 1
                continue
            if t < t_start:
                continue
            if t >= t_end:
                break
            times.append(t)
            if i % MOTION_EVERY == 0:
                camera.update(frame)
            motion.append(camera.M.copy())
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            r = model.process(rgb)
            p = (np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in r.pose_landmarks.landmark])
                 if r.pose_landmarks else np.full((33, 3), np.nan))
            pose.append(p)
            lv, du, dl = np.zeros(n_sw), np.full(n_sw, np.inf), np.full(n_sw, np.inf)
            pts = {k: [q for q in p[v, :2] if not np.isnan(q[0])] for k, v in TOUCH_POINTS.items()}
            for s in range(n_sw):
                lv[s] = ek.light_energy(frame, camera.apply(light_corners[s]))
                poly = camera.apply(touch_corners[s]).astype(np.float32)
                for key, out in (('upper', du), ('lower', dl)):
                    if pts[key]:
                        out[s] = min(-cv2.pointPolygonTest(poly, (float(x), float(y)), True)
                                     for x, y in pts[key]) / sizes[s]
            light.append(lv)
            tu.append(du)
            tl.append(dl)
            i += 1
            if counter is not None and i % 15 == 0:
                with counter.get_lock():
                    counter.value += 15
    cap.release()
    np.savez(out_file, times=np.array(times), pose=np.array(pose, np.float32).reshape(-1, 33, 3),
             motion=np.array(motion, np.float32).reshape(-1, 2, 3), light=np.array(light, np.float32).reshape(-1, n_sw),
             touch_upper=np.array(tu, np.float32).reshape(-1, n_sw), touch_lower=np.array(tl, np.float32).reshape(-1, n_sw))
    return out_file


def process_video(video_path, boxes, out_dir, progress=None, should_stop=None, workers=None):
    """Run the pass (in parallel parts) and save the cache. `progress(done_frames, total, seconds_left)` is
    called now and then; `should_stop()` -> True stops it (nothing is saved). Returns the cache dict."""
    src = vs.VideoSource(video_path, hardware=False)    # all chapters of a GoPro clip
    total = src.total_frames
    fps = src.get(cv2.CAP_PROP_FPS) or 30.0
    size = [int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    workers = workers or default_workers()
    workers = max(1, min(workers, total // 300 or 1))   # parts of at least ~5 s
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    # Part boundaries in ms, half a frame before a frame's time (so each frame falls clearly in one part)
    edges = [-1e9] + [(k * total / workers - 0.5) / fps * 1000 for k in range(1, workers)] + [1e12]
    ctx = mproc.get_context('spawn')
    counter = ctx.Value('i', 0)
    parts = [os.path.join(out_dir, f'.{stem}_part{k}.npz') for k in range(workers)]
    jobs = [(video_path, boxes, edges[k], edges[k + 1], parts[k], counter) for k in range(workers)]
    t0 = time.time()
    if workers == 1:
        import threading
        th = threading.Thread(target=_process_range, args=(jobs[0],))
        th.start()
        alive = th.is_alive
        pool = None
    else:
        pool = ctx.Pool(workers, initializer=_init_worker, initargs=(counter,))
        res = pool.map_async(_range_with_counter, [j[:-1] + (None,) for j in jobs])
        alive = lambda: not res.ready()
    stopped = False
    while alive():
        time.sleep(0.5)
        n = counter.value
        if progress:
            rate = n / max(time.time() - t0, 1e-6)
            progress(n, total, (total - n) / rate if rate > 0 else None)
        if should_stop and should_stop():
            stopped = True
            if pool is not None:
                pool.terminate()
            break
    if pool is not None:
        if not stopped:
            res.get()   # raises a worker's error, if any
        pool.close()
        pool.join()
    else:
        th.join()
    if stopped:
        for p in parts:
            if os.path.exists(p):
                os.remove(p)
        return None
    chunks = [dict(np.load(p)) for p in parts]
    for p in parts:
        os.remove(p)
    d = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    d['light'], d['touch_upper'], d['touch_lower'] = d['light'].T, d['touch_upper'].T, d['touch_lower'].T
    meta = dict(video=os.path.abspath(video_path), video_size=vs.total_size(video_path), fps=fps, size=size,
                chapters=[os.path.basename(p) for p in src.paths],
                boxes=boxes, frames=len(d['times']), complete=len(d['times']) >= total - max(2, fps) * len(src.paths),
                seconds=round(time.time() - t0, 1), workers=workers)
    np.savez_compressed(cache_path(video_path, out_dir), meta=json.dumps(meta), **d)
    d['meta'] = meta
    return d


_counter = None


def _init_worker(counter):
    """Pool initializer: the shared frame counter (can't be passed through map's arguments)."""
    global _counter
    _counter = counter


def _range_with_counter(job):
    return _process_range(job[:-1] + (_counter,))
