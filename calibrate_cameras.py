"""Find where two cameras are relative to each other (and their lens settings), for analyze_two_views.py.

Two ways:

  board  (best; for a fixed setup such as two GoPros on mounts)
      1. Print the board once:   venv/bin/python calibrate_cameras.py make-board
         (outputs/charuco_board.png; print at 100% scale on A4, glue it to something flat, and measure one
         black square - it should be 38 mm).
      2. At the start of a session, with both cameras recording, hold the board where both cameras see it and
         slowly turn/tilt it and move it around the area where the subject will be, for ~20 s.
      3. venv/bin/python calibrate_cameras.py board VIDEO_A VIDEO_B [--square-mm 38]
      Gives each camera's focal length and lens distortion, the cameras' positions, and real units (cm).
      If the cameras are not moved, one calibration serves every recording of that session.
      (GoPro: use the "Linear" lens setting.)

  clicks (for existing videos without a board, e.g. two phones)
      venv/bin/python calibrate_cameras.py clicks VIDEO_A VIDEO_B [--focal-35mm 28]
      The first frames of both videos open side by side. Click the same fixed point in A, then in B
      (switch buttons, table corners, monitor corners...). The videos are in sync, so points on the subject in
      that first frame count too (knuckle, ring, elbow tip, ear) - use them: points that are not all on the
      table top give a much better result. At least 8 points, spread over the scene. Units are relative (distance between the cameras = 1)
      unless you give a known distance: --known 1 5 52.0 = points 1 and 5 are 52.0 cm apart.

The videos must be in sync (sync_videos.py). Output: outputs/<A>+<B>_cameras.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
WINDOW = 'Camera calibration'

# ChArUco board: 5 x 7 squares of 38 mm on A4 (190 x 266 mm), 4x4 ArUco markers
BOARD_SQUARES = (5, 7)
BOARD_SQUARE_MM = 38.0
BOARD_MARKER_RATIO = 0.75
BOARD_DICT = cv2.aruco.DICT_4X4_50


def pair_name(video_a, video_b):
    names = [os.path.splitext(os.path.basename(p))[0].replace('_synced', '') for p in (video_a, video_b)]
    return f'{names[0]}+{names[1]}'


def cameras_path(video_a, video_b):
    return os.path.join(OUT_DIR, f'{pair_name(video_a, video_b)}_cameras.json')


def save_cameras(path, cam_a, cam_b, R, t, units, method, extra=None):
    """Camera A at the origin; a point X (camera A coordinates) is at R @ X + t in camera B coordinates."""
    data = dict(method=method, units=units,
                camera_a=dict(K=np.asarray(cam_a['K']).tolist(), dist=np.asarray(cam_a['dist']).ravel().tolist(),
                              size=list(map(int, cam_a['size']))),
                camera_b=dict(K=np.asarray(cam_b['K']).tolist(), dist=np.asarray(cam_b['dist']).ravel().tolist(),
                              size=list(map(int, cam_b['size'])),
                              R=np.asarray(R).tolist(), t=np.asarray(t).ravel().tolist()),
                **(extra or {}))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved: {path}')


def load_cameras(path):
    with open(path) as f:
        d = json.load(f)
    a, b = d['camera_a'], d['camera_b']
    cams = [dict(K=np.array(a['K']), dist=np.array(a['dist']), R=np.eye(3), t=np.zeros(3), size=a['size']),
            dict(K=np.array(b['K']), dist=np.array(b['dist']), R=np.array(b['R']), t=np.array(b['t']), size=b['size'])]
    return cams, d['units']


def first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f'Could not read {path}')
    return frame


def K_matrix(f, size):
    return np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]], float)


# -----------------------------
# Clicks
# -----------------------------
def click_points(img_a, img_b, screen=(1500, 900), start=None):
    """Show both frames side by side; the user clicks matching points (A then B). Returns two (n, 2) arrays.
    `start`: point pairs already clicked (shown, can be undone, new ones are added after them)."""
    h = max(img_a.shape[0], img_b.shape[0])
    scale = min(screen[0] / (img_a.shape[1] + img_b.shape[1]), screen[1] / h)
    da = cv2.resize(img_a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    db = cv2.resize(img_b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    off_b = da.shape[1]
    canvas0 = np.zeros((max(da.shape[0], db.shape[0]), da.shape[1] + db.shape[1], 3), np.uint8)
    canvas0[:da.shape[0], :off_b] = da
    canvas0[:db.shape[0], off_b:] = db
    pts = {'a': [], 'b': []}
    if start is not None:
        pts = {'a': [tuple(p) for p in start[0]], 'b': [tuple(p) for p in start[1]]}
    mouse = [0, 0]

    def full_res(x, y):
        """Display position -> (view, full-resolution point) or None."""
        if x < off_b and y < da.shape[0]:
            return 'a', (x / scale, y / scale)
        if x >= off_b and y < db.shape[0]:
            return 'b', ((x - off_b) / scale, y / scale)
        return None

    def on_mouse(event, x, y, flags, param):
        mouse[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = full_res(x, y)
            want = 'a' if len(pts['a']) == len(pts['b']) else 'b'
            if hit and hit[0] == want:
                pts[want].append(hit[1])

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        canvas = canvas0.copy()
        for view, off in (('a', 0), ('b', off_b)):
            for i, (x, y) in enumerate(pts[view]):
                p = (int(x * scale) + off, int(y * scale))
                cv2.drawMarker(canvas, p, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.putText(canvas, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(canvas, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Magnifier: 4x of the full-resolution image around the mouse, in the bottom corner of the canvas
        hit = full_res(*mouse)
        if hit:
            img = img_a if hit[0] == 'a' else img_b
            cx, cy = int(hit[1][0]), int(hit[1][1])
            r = 40
            patch = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT)[cy:cy + 2 * r, cx:cx + 2 * r]
            zoom = cv2.resize(patch, (8 * r, 8 * r), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(zoom, (4 * r, 4 * r), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            zh, zw = zoom.shape[:2]
            x0 = canvas.shape[1] - zw if hit[0] == 'a' else 0  # opposite side, so it doesn't hide the target
            canvas[canvas.shape[0] - zh:, x0:x0 + zw] = zoom
        n = len(pts['b'])
        want = 'A (left)' if len(pts['a']) == len(pts['b']) else 'B (right)'
        lines = [f'Point {len(pts["a"]) + (want[0] == "A")}: click it in {want}.   {n} pairs so far (need 8+).',
                 'Switch buttons, table corners... and the subject at this moment (knuckle, ring, elbow) - not all on the table.',
                 'U = undo   ENTER = done   C = cancel']
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(canvas, line, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('u'), ord('U')):
            if len(pts['a']) > len(pts['b']):
                pts['a'].pop()
            elif pts['b']:
                pts['b'].pop()
        elif key in (13, 10) and n >= 8 and len(pts['a']) == n:
            break
        elif key in (ord('c'), ord('C'), 27):
            cv2.destroyAllWindows()
            return None, None
    cv2.destroyAllWindows()
    return np.array(pts['a'], float), np.array(pts['b'], float)


def calibrate_from_points(pa, pb, size_a, size_b, f_a, f_b, refine_focal=None):
    """Relative pose (R, t with |t| = 1) and focal lengths from matched points, plus the 3D points and
    the reprojection errors (px)."""
    n = len(pa)
    if refine_focal is None:
        refine_focal = n >= 20
    na = (pa - np.array(size_a) / 2) / f_a
    nb = (pb - np.array(size_b) / 2) / f_b
    E, mask = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.LMEDS)
    best = None
    for Ei in E.reshape(-1, 3, 3):  # the 5-point method can return several solutions
        good, R, t, _ = cv2.recoverPose(Ei, na, nb, np.eye(3))
        if best is None or good > best[0]:
            best = (good, R, t.ravel())
    _, R, t = best

    def unpack(p):  # one focal scale shared by both cameras (similar lenses): steadier with few points
        fa, fb = (f_a * p[0], f_b * p[0]) if refine_focal else (f_a, f_b)
        q = p[1:] if refine_focal else p
        return fa, fb, cv2.Rodrigues(q[:3])[0], q[3:6] / np.linalg.norm(q[3:6]), q[6:].reshape(n, 3)

    def residuals(p):
        fa, fb, Rb, tb, X = unpack(p)
        ra = project(X, K_matrix(fa, size_a), np.zeros(5), np.eye(3), np.zeros(3)) - pa
        rb = project(X, K_matrix(fb, size_b), np.zeros(5), Rb, tb) - pb
        return np.concatenate([ra.ravel(), rb.ravel()])

    X0 = triangulate_px(pa, pb, K_matrix(f_a, size_a), K_matrix(f_b, size_b), R, t)
    p0 = np.concatenate([[1.0] if refine_focal else [], cv2.Rodrigues(R)[0].ravel(), t, X0.ravel()])
    res = least_squares(residuals, p0, x_scale='jac', loss='soft_l1', f_scale=5.0)
    fa, fb, R, t, X = unpack(res.x)
    err = np.hypot(*res.fun.reshape(2, n, 2).transpose(2, 0, 1)).max(axis=0)
    return fa, fb, R, t, X, err


def project(X, K, dist, R, t):
    p, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), cv2.Rodrigues(R)[0], np.asarray(t, float),
                             K, np.asarray(dist, float))
    return p.reshape(-1, 2)


def triangulate_px(pa, pb, Ka, Kb, R, t, dist_a=np.zeros(5), dist_b=np.zeros(5)):
    """3D points (camera A coordinates) from pixel positions in both views (lens distortion removed first)."""
    na = cv2.undistortPoints(np.asarray(pa, np.float64).reshape(-1, 1, 2), Ka, dist_a).reshape(-1, 2)
    nb = cv2.undistortPoints(np.asarray(pb, np.float64).reshape(-1, 1, 2), Kb, dist_b).reshape(-1, 2)
    X = cv2.triangulatePoints(np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, np.reshape(t, (3, 1))]),
                              na.T, nb.T)
    return (X[:3] / X[3]).T


def run_clicks(args):
    img_a, img_b = first_frame(args.video_a), first_frame(args.video_b)
    size_a, size_b = (img_a.shape[1], img_a.shape[0]), (img_b.shape[1], img_b.shape[0])
    clicks_file = os.path.join(OUT_DIR, f'{pair_name(args.video_a, args.video_b)}_clicks.json')
    saved = None
    if os.path.exists(clicks_file) and not args.again:
        with open(clicks_file) as f:
            c = json.load(f)
        saved = np.array(c['a']), np.array(c['b'])
    if args.drop and saved is not None:
        keep = [i for i in range(len(saved[0])) if i + 1 not in args.drop]
        saved = saved[0][keep], saved[1][keep]
        print(f'Dropped point(s) {", ".join(map(str, args.drop))}; the rest are renumbered 1-{len(keep)}.')
    if saved is not None and not args.edit:
        pa, pb = saved
        print(f'Using the {len(pa)} saved point pairs in {clicks_file} '
              f'(--edit to add/remove points, --again to start over).')
    else:
        pa, pb = click_points(img_a, img_b, start=saved)
        if pa is None:
            sys.exit('Cancelled.')
    if args.drop or args.edit or saved is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(clicks_file, 'w') as f:
            json.dump(dict(a=pa.round(1).tolist(), b=pb.round(1).tolist()), f, indent=1)
    f_a = max(size_a) * args.focal_35mm / 36
    f_b = max(size_b) * args.focal_35mm / 36
    fa, fb, R, t, X, err = calibrate_from_points(pa, pb, size_a, size_b, f_a, f_b)
    print(f'{len(pa)} point pairs: focal A {fa:.0f} px, focal B {fb:.0f} px, '
          f'reprojection error median {np.median(err):.1f} px, max {err.max():.1f} px (point {np.argmax(err) + 1})')
    print('  error per point (px): ' + '  '.join(f'{i + 1}: {e:.0f}' for i, e in enumerate(err)))
    if err.max() > 15:
        print(f'WARNING: some points fit badly - check that each pair was clicked on the same spot '
              f'(e.g. --drop {np.argmax(err) + 1}, or --edit).')
    # Flatness check: all points close to one plane makes the result unreliable
    sv = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    if sv[2] / sv[0] < 0.05:
        print('WARNING: the points are almost on one plane - add points off the table (monitor, socket, wall).')
    units, scale = 'relative (distance between the cameras = 1)', 1.0
    if args.known:
        i, j, cm = int(args.known[0]) - 1, int(args.known[1]) - 1, float(args.known[2])
        scale = cm / np.linalg.norm(X[i] - X[j])
        units = 'cm'
        print(f'Scale: points {i + 1}-{j + 1} = {cm} cm -> cameras are {scale:.1f} cm apart')
    save_cameras(cameras_path(args.video_a, args.video_b),
                 dict(K=K_matrix(fa, size_a), dist=np.zeros(5), size=size_a),
                 dict(K=K_matrix(fb, size_b), dist=np.zeros(5), size=size_b),
                 R, t * scale, units, 'clicks',
                 dict(points=len(pa), reprojection_px_median=round(float(np.median(err)), 2)))


# -----------------------------
# Board
# -----------------------------
def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(BOARD_DICT)
    board = cv2.aruco.CharucoBoard(BOARD_SQUARES, BOARD_SQUARE_MM, BOARD_SQUARE_MM * BOARD_MARKER_RATIO, dictionary)
    return board, cv2.aruco.CharucoDetector(board)


def run_make_board(args):
    board, _ = make_board()
    dpi = 300
    px_per_mm = dpi / 25.4
    a4 = (int(210 * px_per_mm), int(297 * px_per_mm))
    bw, bh = int(BOARD_SQUARES[0] * BOARD_SQUARE_MM * px_per_mm), int(BOARD_SQUARES[1] * BOARD_SQUARE_MM * px_per_mm)
    img = board.generateImage((bw, bh), marginSize=0)
    page = np.full((a4[1], a4[0]), 255, np.uint8)
    y0, x0 = (a4[1] - bh) // 2, (a4[0] - bw) // 2
    page[y0:y0 + bh, x0:x0 + bw] = img
    cv2.putText(page, f'Print at 100% (no "fit to page"). Each square must measure {BOARD_SQUARE_MM:.0f} mm.',
                (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    path = os.path.join(OUT_DIR, 'charuco_board.png')
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(path, page, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f'Saved: {path} (A4, {dpi} dpi)')


def detect_board(detector, frame, min_corners=6):
    corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < min_corners:
        return None
    return corners.reshape(-1, 2), ids.ravel()


def run_board(args):
    board, detector = make_board()
    square = args.square_mm / BOARD_SQUARE_MM  # scale the board's object points to the measured size
    caps = [cv2.VideoCapture(args.video_a), cv2.VideoCapture(args.video_b)]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps / args.per_second))
    sizes = [(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for c in caps]
    seen = [[], []]  # per camera: (frame, corners, ids)
    i = 0
    print('Looking for the board...')
    while True:
        oks = [c.grab() for c in caps]
        if not all(oks):
            break
        if i % step == 0:
            for v, c in enumerate(caps):
                ok, frame = c.retrieve()
                if ok:
                    small = cv2.resize(frame, None, fx=0.5, fy=0.5) if max(sizes[v]) > 2000 else frame
                    if detect_board(detector, small) is not None:  # quick check at half size first
                        d = detect_board(detector, frame)
                        if d is not None:
                            seen[v].append((i,) + d)
            if i % (step * 20) == 0:
                print(f'\r  frame {i}: board seen in {len(seen[0])} (A) / {len(seen[1])} (B) frames', end='', flush=True)
        i += 1
        if args.seconds and i > args.seconds * fps:
            break
    print()
    for v in range(2):
        if len(seen[v]) < 10:
            sys.exit(f'The board was found in only {len(seen[v])} frames of video {"AB"[v]} - need 10+.')

    board_xyz = board.getChessboardCorners().reshape(-1, 3) * square  # 3D position of each corner id (mm)

    def object_image(corners, ids):
        return board_xyz[ids].astype(np.float32), corners.astype(np.float32)

    # Each camera's lens: focal length and distortion from all its board views (up to 60, spread over time)
    cams = []
    for v in range(2):
        views = seen[v][::max(1, len(seen[v]) // 60)]
        obj, img = zip(*(object_image(c, ids) for _, c, ids in views))
        rms, K, dist, _, _ = cv2.calibrateCamera(list(obj), list(img), sizes[v], None, None,
                                                 flags=cv2.CALIB_FIX_K3)
        print(f'Camera {"AB"[v]}: focal {K[0, 0]:.0f} px, lens fit error {rms:.2f} px ({len(views)} views)')
        cams.append(dict(K=K, dist=dist, size=sizes[v]))

    # Relative position from the frames where both cameras see the board (same corners in both)
    by_frame = {f: dict(zip(ids, c)) for f, c, ids in seen[1]}
    obj_s, img_a, img_b = [], [], []
    for f, ca, ia in seen[0]:
        if f not in by_frame:
            continue
        in_a = dict(zip(ia, ca))
        common = sorted(set(in_a) & set(by_frame[f]))
        if len(common) < 6:
            continue
        obj_s.append(board_xyz[common].astype(np.float32))
        img_a.append(np.float32([in_a[k] for k in common]))
        img_b.append(np.float32([by_frame[f][k] for k in common]))
    if len(obj_s) < 5:
        sys.exit(f'Both cameras saw the board together in only {len(obj_s)} frames - need 5+.')
    rms, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(obj_s, img_a, img_b, cams[0]['K'], cams[0]['dist'],
                                                     cams[1]['K'], cams[1]['dist'], sizes[0],
                                                     flags=cv2.CALIB_FIX_INTRINSIC)
    t_cm = t.ravel() / 10
    print(f'Both cameras: {len(obj_s)} frames, fit error {rms:.2f} px; cameras are {np.linalg.norm(t_cm):.0f} cm apart')
    save_cameras(cameras_path(args.video_a, args.video_b), cams[0], cams[1], R, t_cm, 'cm', 'board',
                 dict(frames_both=len(obj_s), stereo_rms_px=round(float(rms), 3)))


def main():
    parser = argparse.ArgumentParser(description='Find where two cameras are relative to each other.')
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('make-board', help='Save the printable calibration board')
    pc = sub.add_parser('clicks', help='Click matching points in the first frames of both videos')
    pc.add_argument('video_a')
    pc.add_argument('video_b')
    pc.add_argument('--focal-35mm', type=float, default=28,
                    help='Lens focal length, 35 mm equivalent (phone main camera in 4K video: ~28)')
    pc.add_argument('--known', nargs=3, metavar=('POINT', 'POINT', 'CM'),
                    help='Two clicked points and their real distance in cm (gives real units)')
    pc.add_argument('--again', action='store_true', help='Start the clicking over (forget the saved points)')
    pc.add_argument('--edit', action='store_true', help='Open the saved points to add or undo points')
    pc.add_argument('--drop', type=int, nargs='+', metavar='POINT', help='Remove these saved points')
    pb = sub.add_parser('board', help='Find the printed board in both videos')
    pb.add_argument('video_a')
    pb.add_argument('video_b')
    pb.add_argument('--square-mm', type=float, default=BOARD_SQUARE_MM, help='Measured size of one board square')
    pb.add_argument('--per-second', type=float, default=4, help='Frames checked per second of video')
    pb.add_argument('--seconds', type=float, help='Only look at the first N seconds')
    args = parser.parse_args()
    {'make-board': run_make_board, 'clicks': run_clicks, 'board': run_board}[args.mode](args)


if __name__ == '__main__':
    main()
