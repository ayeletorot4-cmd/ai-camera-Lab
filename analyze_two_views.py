"""Track one limb in 3D from two synchronised videos of the same session (two camera angles).

Both videos must be in sync frame by frame (frame i of A taken at the same moment as frame i of B): use
sync_videos.py first, or cameras with a shared timecode. Steps:
  1. MediaPipe Holistic on every frame of both videos (cached in outputs/, so re-runs skip this step).
  2. The camera geometry (lens settings and where camera B is relative to camera A) is read from the file made
     by calibrate_cameras.py (printed board, or clicked matching points).
  3. Every joint of the chosen limb (and the 21 hand points for the upper limb) is triangulated to 3D on every
     frame both cameras see it. A point that doesn't fit both views (e.g. MediaPipe mixed up left and right
     in one view) is dropped.
Outputs (in outputs/):
  <A>+<B>_<limb>_<side>_3d.csv - per frame: each joint's 3D position (camera A coordinates, see units below),
      its reprojection error (px) and its fused 2D position in each view; joint angles from 3D and, for
      comparison, the same angles measured in 2D in each view.
  <A>+<B>_<limb>_<side>_3d.mp4 - both views side by side: thin grey = MediaPipe in that view alone,
      coloured = the 3D result drawn back into the view.
  <A>+<B>_<limb>_<side>_switch_events.csv - like extract_kinematics.py's: Switch, Start_ms, Touching_ms,
      Turn_off_ms (ms of the synced videos). The lights are measured in both views (an event both cameras
      saw: earliest start, latest turn-off); touching = first frame the hand (foot) is on the switch in BOTH views, or in one view
      if never both. Switch boxes: the original videos' saved boxes, moved to the synced videos.
Units: cm with a board calibration (or clicks with --known); otherwise relative (distance between the
cameras = 1). Angles don't depend on units.

  venv/bin/python calibrate_cameras.py clicks media/synced/IMG_0122_synced.mp4 media/synced/IMG_7557_synced.mp4
  venv/bin/python analyze_two_views.py media/synced/IMG_0122_synced.mp4 media/synced/IMG_7557_synced.mp4 --limb upper --side right
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

import mediapipe as mp
from extract_kinematics import (LIMBS, HAND_POINT_NAMES, HAND_CONNECTIONS, SIDE_COLOR, COLOR_LINE,
                                CameraMotion, draw_limb, default_output_path, open_video_writer)
from calibrate_cameras import cameras_path, load_cameras, project, triangulate_px
import extract_kinematics as ek

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
JOINT_MIN_VIS = 0.2          # a joint is triangulated when both views see it at least this well (low: the
                             # check that both views agree, MAX_REPROJ_PX, removes the bad ones)
MAX_REPROJ_PX = 40           # (4K) a 3D point that misses a view by more than this is dropped


# -----------------------------
# 1. MediaPipe on each video (cached)
# -----------------------------
def detect(video_path):
    """Per frame: pose (33 x [x_px, y_px, visibility]), left/right hand (21 x [x_px, y_px]) or NaN, and the
    camera motion matrix (first frame -> this frame). Cached next to the outputs."""
    cache = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(video_path))[0] + '_mediapipe_cache.npz')
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(video_path):
        print(f'  using cached MediaPipe results: {os.path.basename(cache)}')
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose, hands, motion, times = [], {'left': [], 'right': []}, [], []
    camera = None
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                        model_complexity=1) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            if camera is None:
                camera = CameraMotion(frame)
            camera.update(frame)
            motion.append(camera.M.copy())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            r = holistic.process(rgb)
            if r.pose_landmarks:
                pose.append([[lm.x * w, lm.y * h, lm.visibility] for lm in r.pose_landmarks.landmark])
            else:
                pose.append(np.full((33, 3), np.nan))
            for side, lms in (('left', r.left_hand_landmarks), ('right', r.right_hand_landmarks)):
                hands[side].append([[lm.x * w, lm.y * h] for lm in lms.landmark] if lms else np.full((21, 2), np.nan))
            if len(pose) % 60 == 0:
                print(f'\r  {os.path.basename(video_path)}: frame {len(pose)}/{total}', end='', flush=True)
    print(f'\r  {os.path.basename(video_path)}: {len(pose)} frames      ')
    d = dict(pose=np.array(pose, float), hand_left=np.array(hands['left'], float),
             hand_right=np.array(hands['right'], float), motion=np.array(motion, np.float32),
             times=np.array(times), size=np.array([w, h]))
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(cache, **d)
    return d


def to_first_frame(pts, M):
    """Undo the camera movement: points of a frame (..., 2) -> first-frame coordinates."""
    inv = cv2.invertAffineTransform(M)
    return pts @ inv[:, :2].T + inv[:, 2]


def to_frame(pts, M):
    return pts @ M[:, :2].T + M[:, 2]


# -----------------------------
# 2. Camera geometry
# -----------------------------
def view_px(X, cam):
    """3D points (camera A coordinates) -> pixels in a view."""
    return project(X, cam['K'], cam['dist'], cam['R'], cam['t'])


# -----------------------------
# 3. Angles
# -----------------------------
def angle(a, b, c):
    """Angle at b (degrees) between b->a and b->c; works for 2D or 3D, NaN if a point is missing."""
    u, v = a - b, c - b
    cosang = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


ANGLES = {  # name: (point, vertex, point) - pose landmark names of the chosen side
    'upper': {'elbow_angle': ('shoulder', 'elbow', 'wrist'),
              'shoulder_angle': ('hip', 'shoulder', 'elbow')},       # arm vs trunk
    'lower': {'knee_angle': ('hip', 'knee', 'ankle'),
              'hip_angle': ('shoulder', 'hip', 'knee'),               # thigh vs trunk
              'ankle_angle': ('knee', 'ankle', 'toe')},
}
POSE_INDEX = {'left': {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23, 'knee': 25, 'ankle': 27,
                       'heel': 29, 'toe': 31},
              'right': {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24, 'knee': 26, 'ankle': 28,
                        'heel': 30, 'toe': 32}}


# -----------------------------
# Switch events from both views
# -----------------------------
def switch_boxes_for(video_path):
    """Switch boxes (first-frame fractions) for a synced video: saved for it, or the original video's boxes
    moved to the synced video's first frame, or found now (the user confirms them)."""
    boxes = ek.load_saved_switches(video_path)
    if boxes:
        return boxes
    synced_first = cv2.VideoCapture(video_path).read()[1]
    h, w = synced_first.shape[:2]
    original = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(video_path))),
                            os.path.basename(video_path).replace('_synced.mp4', '.mov'))
    orig_boxes = ek.load_saved_switches(original) if os.path.exists(original) else None
    if orig_boxes:
        # The synced video starts later than the original: follow the camera movement in between
        orig_first = cv2.VideoCapture(original).read()[1]
        camera = ek.CameraMotion(orig_first)
        camera.update(synced_first)
        boxes = []
        for b in orig_boxes:
            c = camera.apply(ek.box_corners(b, w, h, 0.0))
            boxes.append([round(float(c[:, 0].min()) / w, 4), round(float(c[:, 1].min()) / h, 4),
                          round(float(c[:, 0].max()) / w, 4), round(float(c[:, 1].max()) / h, 4)])
        print(f'  {os.path.basename(video_path)}: switch boxes from {os.path.basename(original)}')
    else:
        cap = cv2.VideoCapture(video_path)
        screen_w, screen_h = ek.get_screen_work_area()
        display_size = ek.fit_to_screen(w, h, screen_w, screen_h)
        cv2.namedWindow(ek.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(ek.WINDOW_NAME, *display_size)
        boxes = ek.auto_select_switches(cap, synced_first, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                        cap.get(cv2.CAP_PROP_FPS) or 30, display_size)
        cv2.destroyAllWindows()
    if boxes:
        ek.save_switches(video_path, boxes)
    return boxes


def view_switch_signals(video_path, D, boxes, limb, n):
    """Per switch: light level per frame, and per frame the distance (in box sizes, <= 0 inside) of the
    closest hand/foot point to the box."""
    w, h = D['size']
    light_corners = [ek.box_corners(b, w, h, ek.LIGHT_BOX_MARGIN) for b in boxes]
    touch_corners = [ek.box_corners(b, w, h, 0.0) for b in boxes]
    sizes = [float(np.sqrt((b[2] - b[0]) * w * (b[3] - b[1]) * h)) for b in boxes]
    pose_ids = [i for side in ('left', 'right') for i in limb['touch_points'][side]]
    light = np.zeros((len(boxes), n))
    dist = np.full((len(boxes), n), np.inf)
    cap = cv2.VideoCapture(video_path)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        M = D['motion'][i]
        pts = [D['pose'][i, j, :2] for j in pose_ids if D['pose'][i, j, 2] > 0.5]
        if limb['hand']:
            for hand in (D['hand_left'][i], D['hand_right'][i]):
                pts += [p for p in hand if not np.isnan(p[0])]
        for s in range(len(boxes)):
            light[s, i] = ek.light_energy(frame, to_frame(light_corners[s], M))
            poly = to_frame(touch_corners[s], M).astype(np.float32)
            if pts:
                dist[s, i] = min(-cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in pts) / sizes[s]
        if i % 120 == 0:
            print(f'\r  {os.path.basename(video_path)}: frame {i}/{n}', end='', flush=True)
    print(f'\r  {os.path.basename(video_path)}: {n} frames      ')
    return light, dist


def merge_events(ev_a, ev_b, fps, max_gap_s=0.2):
    """Light events of one switch seen by the two cameras -> one list of (on, off) frames (None = unknown).
    For an event both cameras saw: the earlier start and the later turn-off (a hand over the LED can only
    make a light look off in one view, never on); an event only one camera saw is kept."""
    gap = max_gap_s * fps
    merged, used = [], set()
    for on_a, off_a in ev_a:
        match = next((k for k, (on_b, _) in enumerate(ev_b) if k not in used and abs(on_b - on_a) <= gap), None)
        if match is None:
            merged.append((on_a, off_a))
            continue
        used.add(match)
        on_b, off_b = ev_b[match]
        on = min(on_a, on_b) if on_a > 0 and on_b > 0 else max(on_a, on_b)  # 0 = already on at the start
        off = None if off_a is None or off_b is None else max(off_a, off_b)  # None = still on at the end
        merged.append((on, off))
    merged += [e for k, e in enumerate(ev_b) if k not in used]
    return sorted(merged)


def two_view_switch_events(path, args, A, B, limb, n, times, fps):
    """Switch events (start, touching, turn off) from both cameras -> CSV like extract_kinematics.py's.
    Touching: each camera's first frame with the hand (foot) on the switch; the later of the two = the moment
    both cameras have seen it (one view alone can see a hand in front of a switch it doesn't touch yet). If
    that is after the light went off (pressing turns it off, so the touch can't be later), or only one camera
    saw a touch, the earliest touch is used."""
    print('Switch events')
    boxes = [switch_boxes_for(args.video_a), switch_boxes_for(args.video_b)]
    if not all(boxes):
        print('  no switch boxes - skipping switch events')
        return
    la, da = view_switch_signals(args.video_a, A, boxes[0], limb, n)
    lb, db = view_switch_signals(args.video_b, B, boxes[1], limb, n)
    touch_window = int(round(ek.TOUCH_AFTER_OFF_S * fps))
    rows, both_count, one_count = [], 0, 0
    for s in range(ek.N_SWITCHES):
        ev_a, why_a = ek.find_light_events(la[s], fps)
        ev_b, why_b = ek.find_light_events(lb[s], fps)
        events = merge_events(ev_a, ev_b, fps)
        if not events:
            print(f'  switch {s + 1}: no light events (A: {why_a}; B: {why_b})')
            rows.append([s + 1, '', '', ''])
            continue
        print(f'  switch {s + 1}: {len(events)} light event(s) (camera A saw {len(ev_a)}, B saw {len(ev_b)})')
        for k, (on, off) in enumerate(events):
            end = n if off is None else min(n, off + touch_window)
            if k + 1 < len(events):
                end = min(end, events[k + 1][0])
            firsts = [on + int(np.argmax(d[s, on:end] <= ek.TOUCH_DIST))
                      for d in (da, db) if (d[s, on:end] <= ek.TOUCH_DIST).any()]
            touch = None
            if len(firsts) == 2 and (off is None or max(firsts) <= off + 1):
                touch, both_count = max(firsts), both_count + 1
            elif firsts:
                touch, one_count = min(firsts), one_count + 1
            rows.append([s + 1, round(float(times[on]), 1) if on > 0 else '',
                         round(float(times[touch]), 1) if touch is not None else '',
                         round(float(times[off]), 1) if off is not None else ''])
    rows.sort(key=lambda r: (r[0], r[1] if r[1] != '' else -1))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Switch', 'Start_ms', 'Touching_ms', 'Turn_off_ms'])
        w.writerows(rows)
    print(f'  touches confirmed by both cameras: {both_count}, from one camera: {one_count}')
    print(f'Switch events saved to {path}')


def main():
    parser = argparse.ArgumentParser(description='Track one limb in 3D from two synchronised videos.')
    parser.add_argument('video_a')
    parser.add_argument('video_b')
    parser.add_argument('--limb', choices=['upper', 'lower'], required=True)
    parser.add_argument('--side', choices=['left', 'right'], required=True, help="The subject's own side")
    parser.add_argument('--cameras', help='Camera file from calibrate_cameras.py (default: the one for these videos)')
    parser.add_argument('--no-switches', action='store_true', help='Skip the switch events')
    args = parser.parse_args()

    limb = LIMBS[args.limb]
    joints = limb['joints'][args.side]
    color = SIDE_COLOR[args.side]
    names = [os.path.splitext(os.path.basename(p))[0].replace('_synced', '') for p in (args.video_a, args.video_b)]
    tag = f'{names[0]}+{names[1]}_{args.limb}_{args.side}'

    print('1. MediaPipe on both videos')
    A, B = detect(args.video_a), detect(args.video_b)
    n = min(len(A['pose']), len(B['pose']))
    if abs(len(A['pose']) - len(B['pose'])) > 1:
        print(f"WARNING: the videos have {len(A['pose'])} and {len(B['pose'])} frames - are they synchronised?")

    # Landmarks in first-frame coordinates (camera movement removed)
    def steady_pose(D):
        xy = np.stack([to_first_frame(D['pose'][i, :, :2], D['motion'][i]) for i in range(n)])
        return xy, D['pose'][:n, :, 2]
    xa, va = steady_pose(A)
    xb, vb = steady_pose(B)

    cam_file = args.cameras or cameras_path(args.video_a, args.video_b)
    if not os.path.exists(cam_file):
        sys.exit(f'No camera file {cam_file} - run calibrate_cameras.py first (see the top of this file).')
    (cam_a, cam_b), units = load_cameras(cam_file)
    print(f'2. Cameras: {os.path.basename(cam_file)} (units: {units})')

    def fuse(pa, pb, ok):
        """3D points (NaN where not ok) and their reprojection error (max of the two views)."""
        X = np.full((len(pa), 3), np.nan)
        err = np.full(len(pa), np.nan)
        if ok.any():
            X[ok] = triangulate_px(pa[ok], pb[ok], cam_a['K'], cam_b['K'], cam_b['R'], cam_b['t'],
                                   cam_a['dist'], cam_b['dist'])
            ea = np.linalg.norm(view_px(X[ok], cam_a) - pa[ok], axis=1)
            eb = np.linalg.norm(view_px(X[ok], cam_b) - pb[ok], axis=1)
            err[ok] = np.maximum(ea, eb)
        return X, err

    print('3. 3D joints')
    point_names = list(POSE_INDEX[args.side])  # all of the side's joints (angles need hip/shoulder too)
    P3, E3, dropped = {}, {}, {}
    for name in point_names:
        j = POSE_INDEX[args.side][name]
        ok = (va[:, j] >= JOINT_MIN_VIS) & (vb[:, j] >= JOINT_MIN_VIS)
        X, err = fuse(xa[:, j], xb[:, j], ok)
        bad = err > MAX_REPROJ_PX
        dropped[name] = int(bad.sum())
        X[bad], err[bad] = np.nan, np.nan
        P3[name], E3[name] = X, err
    hand3 = None
    if limb['hand']:
        ha = np.stack([to_first_frame(A[f'hand_{args.side}'][i], A['motion'][i]) for i in range(n)])
        hb = np.stack([to_first_frame(B[f'hand_{args.side}'][i], B['motion'][i]) for i in range(n)])
        hand3 = np.full((n, 21, 3), np.nan)
        for k in range(21):
            ok = ~np.isnan(ha[:, k, 0]) & ~np.isnan(hb[:, k, 0])
            X, err = fuse(ha[:, k], hb[:, k], ok)
            X[err > MAX_REPROJ_PX] = np.nan
            hand3[:, k] = X
    for name in joints:
        got = ~np.isnan(P3[name][:, 0])
        print(f'  {name:9s} 3D in {got.mean():5.1%} of frames, median reprojection error '
              f'{np.nanmedian(E3[name]) if got.any() else float("nan"):.1f} px'
              + (f' ({dropped[name]} frames dropped: views disagree)' if dropped[name] else ''))
    if hand3 is not None:
        print(f'  hand      3D in {(~np.isnan(hand3[:, 0, 0])).mean():5.1%} of frames')

    # Angles: 3D, and the same angle in 2D in each view (to see what the second camera changes)
    angles = {}
    for aname, (p, v, q) in ANGLES[args.limb].items():
        angles[f'{aname}_3d'] = angle(P3[p], P3[v], P3[q])
        for view, xv, vv in (('A', xa, va), ('B', xb, vb)):
            ip, iv, iq = (POSE_INDEX[args.side][s] for s in (p, v, q))
            a2 = angle(xv[:, ip], xv[:, iv], xv[:, iq])
            a2[(vv[:, [ip, iv, iq]] < JOINT_MIN_VIS).any(axis=1)] = np.nan
            angles[f'{aname}_2d_{view}'] = a2
    for k, a in angles.items():
        if k.endswith('_3d') and not np.isnan(a).all():
            print(f'  {k}: median {np.nanmedian(a):.0f} deg, range {np.nanpercentile(a, 5):.0f}-'
                  f'{np.nanpercentile(a, 95):.0f} deg')

    # -----------------------------
    # CSV
    # -----------------------------
    csv_path = default_output_path(tag, '3d', '.csv')
    times = A['times'][:n]
    header = ['Frame', 'Timestamp_ms']
    for name in joints:
        header += [f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_reproj_px',
                   f'{name}_A_x_px', f'{name}_A_y_px', f'{name}_B_x_px', f'{name}_B_y_px']
    if hand3 is not None:
        for hn in HAND_POINT_NAMES:
            header += [f'hand_{hn}_x', f'hand_{hn}_y', f'hand_{hn}_z']
    header += list(angles)
    reproj = {}
    for name in joints:  # fused 3D point drawn back into each view, in that frame's own coordinates
        X = P3[name]
        ok = ~np.isnan(X[:, 0])
        ra, rb = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
        if ok.any():
            ra[ok], rb[ok] = view_px(X[ok], cam_a), view_px(X[ok], cam_b)
        ra = np.stack([to_frame(ra[i], A['motion'][i]) for i in range(n)])
        rb = np.stack([to_frame(rb[i], B['motion'][i]) for i in range(n)])
        reproj[name] = (ra, rb)
    fmt = lambda v, d=1: '' if np.isnan(v) else round(float(v), d)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            row = [i, round(float(times[i]), 2)]
            for name in joints:
                row += [fmt(c, 4) for c in P3[name][i]] + [fmt(E3[name][i])]
                row += [fmt(c) for c in reproj[name][0][i]] + [fmt(c) for c in reproj[name][1][i]]
            if hand3 is not None:
                for k in range(21):
                    row += [fmt(c, 4) for c in hand3[i, k]]
            row += [fmt(angles[k][i]) for k in angles]
            w.writerow(row)
    # -----------------------------
    # Video: both views side by side
    # -----------------------------
    video_path = default_output_path(tag, '3d', '.mp4')
    caps = [cv2.VideoCapture(args.video_a), cv2.VideoCapture(args.video_b)]
    W, H = 1920, 1080  # each view
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    writer = open_video_writer(video_path, fps, (2 * W, H))
    first_angle = next(k for k in angles if k.endswith('_3d'))
    print('4. Video')
    for i in range(n):
        frames = [c.read()[1] for c in caps]
        if any(fr is None for fr in frames):
            break
        tiles = []
        for v, (fr, D) in enumerate(zip(frames, (A, B))):
            sx, sy = W / fr.shape[1], H / fr.shape[0]
            tile = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            # MediaPipe in this view alone (thin grey)
            raw = {name: D['pose'][i, j, :2] for name, j in joints.items()
                   if D['pose'][i, j, 2] >= JOINT_MIN_VIS}
            for a, b in limb['segments']:
                if a in raw and b in raw:
                    cv2.line(tile, (int(raw[a][0] * sx), int(raw[a][1] * sy)),
                             (int(raw[b][0] * sx), int(raw[b][1] * sy)), (160, 160, 160), 2, cv2.LINE_AA)
            # 3D result drawn back into the view
            fused = {name: (int(reproj[name][v][i][0] * sx), int(reproj[name][v][i][1] * sy))
                     for name in joints if not np.isnan(reproj[name][v][i][0])}
            hand_px = None
            if hand3 is not None and not np.isnan(hand3[i, 0, 0]):
                hp = to_frame(view_px(hand3[i], (cam_a, cam_b)[v]), D['motion'][i]) * [sx, sy]
                if not np.isnan(hp).any():
                    hand_px = [(int(x), int(y)) for x, y in hp]
            draw_limb(tile, fused, limb['segments'], hand_px, color, 1.5)
            label = f"{'AB'[v]}: {names[v]}"
            cv2.putText(tile, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
            cv2.putText(tile, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 2)
            tiles.append(tile)
        out = np.hstack(tiles)
        a3 = angles[first_angle][i]
        text = f"{first_angle.replace('_3d', '').replace('_', ' ')} (3D): " + ('-' if np.isnan(a3) else f'{a3:.0f} deg')
        cv2.putText(out, text, (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
        cv2.putText(out, text, (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        writer.write(out)
        if i % 120 == 0:
            print(f'\r  frame {i}/{n}', end='', flush=True)
    writer.release()
    print(f'\r  {n} frames      ')
    saved = [csv_path, video_path]
    if not args.no_switches:
        events_path = default_output_path(tag, 'switch_events', '.csv')
        two_view_switch_events(events_path, args, A, B, limb, n, times, fps)
        saved.append(events_path)
    print('Saved:', *saved, sep='\n  ')


if __name__ == '__main__':
    main()
