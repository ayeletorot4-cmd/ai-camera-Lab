"""Track one limb (upper or lower, left or right) in a lab video with MediaPipe Holistic.

The whole frame is analyzed, but only the chosen limb is saved and drawn:
  upper limb - shoulder, elbow, wrist + the 21 hand/finger points (red cross = hand-model wrist)
  lower limb - hip, knee, ankle, heel, toe
Outputs (in outputs/): a CSV of the limb's points and a video of the full original frame with the limb skeleton.
With --switches: also a CSV of light/touch events for the 5 switches (boxes with lights) on the table.
"""
import cv2
import mediapipe as mp
import numpy as np
import csv
import sys
import os
import ctypes
import argparse
import json
from datetime import datetime

WINDOW_NAME = 'MediaPipe Kinematics Visualization'

# -----------------------------
# Limb definitions
# -----------------------------
# MediaPipe Pose landmark indices per side. Left/Right are the SUBJECT's own sides.
LIMBS = {
    'upper': {
        'joints': {'left':  {'shoulder': 11, 'elbow': 13, 'wrist': 15},
                   'right': {'shoulder': 12, 'elbow': 14, 'wrist': 16}},
        'segments': [('shoulder', 'elbow'), ('elbow', 'wrist')],
        'hand': True,
        # Pose points that count for touching a switch (wrist, pinky, index, thumb) - plus the 21 hand points
        'touch_points': {'left': [15, 17, 19, 21], 'right': [16, 18, 20, 22]},
    },
    'lower': {
        'joints': {'left':  {'hip': 23, 'knee': 25, 'ankle': 27, 'heel': 29, 'toe': 31},
                   'right': {'hip': 24, 'knee': 26, 'ankle': 28, 'heel': 30, 'toe': 32}},
        'segments': [('hip', 'knee'), ('knee', 'ankle'), ('ankle', 'heel'), ('heel', 'toe'), ('ankle', 'toe')],
        'hand': False,
        'touch_points': {'left': [27, 29, 31], 'right': [28, 30, 32]},  # ankle, heel, toe
    },
}
HAND_POINT_NAMES = [lm.name.lower() for lm in mp.solutions.hands.HandLandmark]  # wrist, thumb_cmc, ...
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# Drawing colours (BGR): subject's left = orange, right = cyan; red cross on the hand-model wrist
SIDE_COLOR = {'left': (0, 138, 255), 'right': (231, 217, 0)}
COLOR_LINE, COLOR_CROSS = (255, 255, 255), (0, 0, 255)


def get_screen_work_area():
    """Usable screen size in real pixels (excludes the taskbar), DPI-scaling aware."""
    try:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()

        class RECT(ctypes.Structure):
            _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long),
                        ('right', ctypes.c_long), ('bottom', ctypes.c_long)]
        r = RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0)  # SPI_GETWORKAREA
        return r.right - r.left, r.bottom - r.top
    except Exception:
        return 1280, 720  # safe fallback (non-Windows)


def fit_to_screen(frame_w, frame_h, screen_w, screen_h, margin=0.9):
    """Largest size that keeps the aspect ratio and fits inside the screen (leaves room for the title bar)."""
    scale = min(screen_w * margin / frame_w, (screen_h - 40) * margin / frame_h, 1.0)
    return int(frame_w * scale), int(frame_h * scale)


def default_output_path(video_path, suffix, ext):
    """outputs/<video name>_<suffix><ext> next to this script; never overwrites an existing file."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    path = os.path.join(out_dir, f'{stem}_{suffix}{ext}')
    if os.path.exists(path):
        path = os.path.join(out_dir, f'{stem}_{suffix}_{datetime.now():%Y%m%d_%H%M%S}{ext}')
    return path


# -----------------------------
# Switch events
# -----------------------------
# The switch boxes are marked once on the first frame (saved per video and offered again next time).
# Each box follows small camera movements, its red light is measured every frame, and touching =
# a point of the tracked hand (or foot) inside the box.
SWITCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'switch_settings.json')
N_SWITCHES = 5
LIGHT_BOX_MARGIN = 0.25   # light is measured in the switch box enlarged by 25% (LED + its glow)
# Touching = a hand/foot point inside the switch box or within this fraction of the box size outside it.
# The button sticks out on the subject's side, so fingertips pressing it can sit just outside the box.
TOUCH_DIST = 0.25
TOUCH_AFTER_OFF_S = 1.0   # a touch up to 1 s after the light turned off still belongs to that event
MIN_EVENT_S = 0.07        # light changes shorter than this are ignored (noise, arm passing through the beam)
MERGE_GAP_S = 0.15        # a light hidden for less than this (hand/foot passing over the LED) is one event
# A switch whose light level between events is above this fraction of its level during events is lit all
# the time (blinking switches: <= 0.15, always-lit: >= 0.58); its "events" are only the hand passing by.
MAX_OFF_ON_RATIO = 0.35
LIGHT_ON_FRACTION = 0.2   # light counts as on above 20% of the way from its off level to its on level
# Slow changes (daylight, camera moving, glow of a neighbouring switch) are removed by subtracting the
# light level's running median over this window. Light events must be shorter than half of it (~0.5 s here).
BASELINE_WINDOW_S = 4.0
# On must stand out from off by this many times the off-level noise. Blinking switches score ~30-1500,
# steady (always-lit or never-lit) ones ~2-13. (A ratio test failed for switches on a reddish background.)
MIN_LIGHT_CONTRAST = 20.0


def load_saved_switches(video_path):
    try:
        with open(SWITCH_FILE) as f:
            return json.load(f).get(os.path.basename(video_path))
    except (OSError, ValueError):
        return None


def save_switches(video_path, boxes):
    try:
        with open(SWITCH_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data[os.path.basename(video_path)] = boxes
    os.makedirs(os.path.dirname(SWITCH_FILE), exist_ok=True)
    with open(SWITCH_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def select_switches(first_frame, display_size):
    """Drag a box around each switch in order 1..5. Returns [x0, y0, x1, y1] fractions per switch, or None."""
    dw, dh = display_size
    boxes = []
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    for n in range(1, N_SWITCHES + 1):
        shown = cv2.resize(first_frame, display_size, interpolation=cv2.INTER_AREA)
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            p0, p1 = (int(x0 * dw), int(y0 * dh)), (int(x1 * dw), int(y1 * dh))
            cv2.rectangle(shown, p0, p1, (0, 255, 255), 2)
            cv2.putText(shown, str(k + 1), (p0[0], p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        for i, line in enumerate([f"Drag a box around SWITCH {n} of {N_SWITCHES}, then press ENTER.",
                                  "Number the switches 1-5 from RIGHT to LEFT.",
                                  "Press C to cancel switch events."]):
            cv2.putText(shown, line, (15, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(shown, line, (15, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        x, y, w, h = cv2.selectROI(WINDOW_NAME, shown, showCrosshair=False)
        if w < 5 or h < 5:
            boxes = None
            break
        boxes.append([round(x / dw, 4), round(y / dh, 4), round((x + w) / dw, 4), round((y + h) / dh, 4)])
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
    return boxes


def box_corners(box, frame_w, frame_h, margin):
    """Corners of a fractional [x0, y0, x1, y1] box in pixels, enlarged by `margin` of its size."""
    x0, y0, x1, y1 = box[0] * frame_w, box[1] * frame_h, box[2] * frame_w, box[3] * frame_h
    mx, my = (x1 - x0) * margin / 2, (y1 - y0) * margin / 2
    return np.float32([[x0 - mx, y0 - my], [x1 + mx, y0 - my], [x1 + mx, y1 + my], [x0 - mx, y1 + my]])


class CameraMotion:
    """Follows small camera movements: maps first-frame coordinates to the current frame.

    Matches ORB features of each frame to the first frame (no drift) and fits a rotation +
    shift + scale; the moving subject is rejected as outliers by RANSAC."""
    SCALE = 0.25  # features are found on a quarter-size copy (fast, enough for whole-frame motion)

    def __init__(self, first_frame):
        self.orb = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self._small_gray(first_frame), None)
        self.M = np.float32([[1, 0, 0], [0, 1, 0]])

    def _small_gray(self, frame):
        small = cv2.resize(frame, None, fx=self.SCALE, fy=self.SCALE, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def update(self, frame):
        kp, des = self.orb.detectAndCompute(self._small_gray(frame), None)
        if des is None or self.ref_des is None or len(kp) < 10:
            return  # keep the last known camera position
        matches = self.matcher.match(self.ref_des, des)
        if len(matches) < 15:
            return
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in matches]) / self.SCALE
        dst = np.float32([kp[m.trainIdx].pt for m in matches]) / self.SCALE
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=12)
        if M is not None and inliers is not None and inliers.sum() >= 15:
            self.M = M.astype(np.float32)

    def apply(self, pts):
        return cv2.transform(pts.reshape(-1, 1, 2), self.M).reshape(-1, 2)


def light_energy(frame, corners):
    """Amount of pink LED light (the LED and its glow) inside the polygon's bounding box, per pixel.

    Lit LED/glow pixels are bright pink: red near saturation, green and blue clearly lower but still
    high, blue not below green. This leaves out the deep-red mushroom button (low green/blue), the
    white table (R = G = B) and skin (orange: blue well below green), which all sit in or near the box."""
    h, w = frame.shape[:2]
    x0, y0 = np.clip(np.floor(corners.min(axis=0)).astype(int), 0, [w, h])
    x1, y1 = np.clip(np.ceil(corners.max(axis=0)).astype(int), 0, [w, h])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    patch = frame[y0:y1, x0:x1].astype(np.int32)
    B, G, R = patch[..., 0], patch[..., 1], patch[..., 2]
    excess = R - np.maximum(G, B)
    pink = (R > 200) & (G > 90) & (B >= G - 15) & (excess > 20)
    return float(excess[pink].sum()) * 100 / ((x1 - x0) * (y1 - y0))


def otsu_threshold(values):
    """Value that best splits the samples into two groups (Otsu's method on a 256-bin histogram)."""
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    return centers[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)]


def find_light_events(signal, fps):
    """(first ON frame, first OFF frame or None) for each time the light was on, or ([], reason)."""
    raw = np.asarray(signal, dtype=float)
    if len(raw) < 3 or raw.max() - raw.min() < 1e-6:
        return [], 'light never changed'
    # Mirrored padding, so a light that is on at the very start or end doesn't lift the baseline there
    half = max(1, int(BASELINE_WINDOW_S * fps) // 2)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(raw, half, mode='reflect'), 2 * half + 1)
    s = raw - np.median(windows, axis=1)
    thr = otsu_threshold(s)
    low, high = s[s <= thr], s[s > thr]
    # On/off must be clearly different relative to the off-level noise (robust spread: scaled MAD), so an
    # unlit or always-lit switch with only small changes (drift, arm passing) gives no events
    if len(high) == 0:
        return [], 'no clear on/off changes of the light'
    noise = max(1.4826 * np.median(np.abs(low - np.median(low))), 1e-6)
    if (np.median(high) - np.median(low)) / noise < MIN_LIGHT_CONTRAST:
        return [], 'no clear on/off changes of the light'
    # Final on/off threshold close to the off level, so a light partly hidden by the hand/foot still counts as on
    on = s > np.median(low) + LIGHT_ON_FRACTION * (np.median(high) - np.median(low))
    min_len = max(2, int(round(MIN_EVENT_S * fps)))
    runs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(on)])
    merged = []
    max_gap = max(min_len, int(round(MERGE_GAP_S * fps)))
    for r in runs:  # join an event the hand/foot briefly hid
        if merged and r[0] - merged[-1][1] < max_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    # Real events reach the light's full level at some point; weaker bumps are the hand/foot passing by
    full = np.median(low) + 0.5 * (np.median(high) - np.median(low))
    merged = [r for r in merged if r[1] - r[0] >= min_len and s[r[0]:r[1]].max() >= full]
    if not merged:
        return [], 'no clear on/off changes of the light'
    lit = np.zeros(len(raw), bool)
    for a, b in merged:
        lit[a:b] = True
    if lit.all() or np.median(raw[~lit]) > MAX_OFF_ON_RATIO * np.median(raw[lit]):
        return [], 'light is on the whole time (only brief changes, e.g. the hand passing)'
    return [(a, b if b < len(raw) else None) for a, b in merged], ''


def draw_limb(image, joints_px, segments, hand_px, color, scale):
    """Draw the limb's segments and joints, the fingers, and a red cross on the hand-model wrist."""
    line_t = max(2, int(3 * scale))
    radius = max(4, int(8 * scale))
    for a, b in segments:
        if a in joints_px and b in joints_px:
            cv2.line(image, joints_px[a], joints_px[b], COLOR_LINE, line_t, cv2.LINE_AA)
    for p in joints_px.values():
        cv2.circle(image, p, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, p, radius, COLOR_LINE, max(1, line_t // 2), cv2.LINE_AA)
    if hand_px:
        finger_t = max(1, int(2 * scale))
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, hand_px[a], hand_px[b], COLOR_LINE, finger_t, cv2.LINE_AA)
        for p in hand_px[1:]:
            cv2.circle(image, p, max(2, int(4 * scale)), color, -1, cv2.LINE_AA)
        cv2.drawMarker(image, hand_px[0], COLOR_CROSS, cv2.MARKER_CROSS,
                       max(12, int(18 * scale)), max(2, int(3 * scale)), cv2.LINE_AA)


def main():
    # -----------------------------
    # 1. Configuration
    # -----------------------------
    parser = argparse.ArgumentParser(description='Track one limb in a video and save a CSV + skeleton video.')
    parser.add_argument('video', nargs='?', default='input_video.mov', help='Path to the video file')
    parser.add_argument('--limb', choices=['upper', 'lower'], required=True,
                        help='upper = shoulder/elbow/wrist + fingers; lower = hip/knee/ankle/heel/toe')
    parser.add_argument('--side', choices=['left', 'right'], required=True,
                        help="The subject's own left or right side")
    parser.add_argument('--switches', action='store_true',
                        help='Also detect light start / touching / light off events for the 5 switches')
    parser.add_argument('--select-switches', action='store_true',
                        help='Mark the switches again even if they are saved for this video')
    args = parser.parse_args()

    limb = LIMBS[args.limb]
    joints = limb['joints'][args.side]
    color = SIDE_COLOR[args.side]
    tag = f'{args.limb}_{args.side}'
    input_video_path = args.video
    output_csv_path = default_output_path(input_video_path, f'{tag}_landmarks', '.csv')
    output_video_path = default_output_path(input_video_path, f'{tag}_skeleton', '.mp4')

    # Threshold for data quality (0.0 to 1.0).
    # If average visibility of the limb's joints falls below this, a warning triggers.
    QUALITY_THRESHOLD = 0.65

    # Holistic finds each hand from the pose wrist, so the hand is always matched to the right side
    mp_holistic = mp.solutions.holistic

    # -----------------------------
    # 2. Setup CSV Header (chosen limb only)
    # -----------------------------
    header = ['Frame', 'Timestamp_ms']
    for name in joints:
        header.extend([f'{name}_x_px', f'{name}_y_px', f'{name}_z_rel', f'{name}_visibility'])
    if limb['hand']:
        for name in HAND_POINT_NAMES:
            header.extend([f'hand_{name}_x_px', f'hand_{name}_y_px', f'hand_{name}_z_rel'])

    # -----------------------------
    # 3. Process Video
    # -----------------------------
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        sys.exit(1)

    # Preview window: resizable, sized to fit the screen with the whole frame visible
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    screen_w, screen_h = get_screen_work_area()
    display_size = fit_to_screen(frame_w, frame_h, screen_w, screen_h)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, *display_size)
    cv2.moveWindow(WINDOW_NAME, (screen_w - display_size[0]) // 2, 0)

    # Output video: the whole original frame (full resolution) with the limb skeleton
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (frame_w, frame_h))
    if not video_writer.isOpened():
        print(f"Error: Could not create output video {output_video_path}")
        sys.exit(1)

    # Line/point sizes scale with the resolution so the skeleton is visible on 4K video
    draw_scale = max(frame_w, frame_h) / 1280
    text_scale = max(1.0, draw_scale * 0.8)
    text_t = max(2, int(2 * draw_scale))

    # Switches: saved boxes for this video, or mark them on the first frame
    switch_boxes = None
    if args.switches or args.select_switches:
        ret, first_frame = cap.read()
        if not ret:
            print(f"Error: Could not read frames from {input_video_path}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        switch_boxes = None if args.select_switches else load_saved_switches(input_video_path)
        if switch_boxes is None:
            switch_boxes = select_switches(first_frame, display_size)
            if switch_boxes:
                save_switches(input_video_path, switch_boxes)
            else:
                print("Switch marking cancelled - continuing without switch events.")
        if switch_boxes:
            camera = CameraMotion(first_frame)
            light_corners = [box_corners(b, frame_w, frame_h, LIGHT_BOX_MARGIN) for b in switch_boxes]
            touch_corners = [box_corners(b, frame_w, frame_h, 0.0) for b in switch_boxes]
            touch_sizes = [float(np.sqrt((b[2] - b[0]) * frame_w * (b[3] - b[1]) * frame_h)) for b in switch_boxes]
            events_csv_path = default_output_path(input_video_path, f'{tag}_switch_events', '.csv')
    # Touch detection also uses the hand-only model: Holistic finds hands only through the body pose, so it
    # misses them when the body is mostly outside the frame (e.g. only the arm reaches in)
    hands_model = (mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.5)
                   if switch_boxes and limb['hand'] else None)
    light_signal = [[] for _ in range(N_SWITCHES)]
    touching = [[] for _ in range(N_SWITCHES)]
    frame_times = []

    print(f"Video:  {input_video_path}  ({frame_w}x{frame_h}, {total_frames} frames)")
    print(f"Limb:   {args.side} {args.limb} limb (subject's own side)")
    print(f"Output: {output_csv_path}")
    print(f"        {output_video_path}")
    if switch_boxes:
        print(f"        {events_csv_path}")
    print("Press 'q' in the video window to stop early.")

    frame_idx = 0
    hand_found = 0
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1) as holistic:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # The whole frame is analyzed by MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)

                row_data = [frame_idx, round(timestamp_ms, 2)]
                warnings = []

                # -----------------------------
                # 4. Limb joints from the pose model (pixels, full frame)
                # -----------------------------
                joints_px = {}
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    for name, i in joints.items():
                        x, y = lms[i].x * frame_w, lms[i].y * frame_h
                        joints_px[name] = (int(x), int(y))
                        row_data.extend([round(x, 1), round(y, 1), round(lms[i].z * frame_w, 1),
                                         round(lms[i].visibility, 3)])
                    avg_visibility = sum(lms[i].visibility for i in joints.values()) / len(joints)
                    if avg_visibility < QUALITY_THRESHOLD:
                        warnings.append(f"WARNING: Low Pose Quality (Vis: {avg_visibility:.2f})")
                else:
                    warnings.append("WARNING: No Pose Detected")
                    row_data.extend([''] * (4 * len(joints)))

                # -----------------------------
                # 5. Hand / fingers (upper limb)
                # -----------------------------
                hand_px = None
                if limb['hand']:
                    hand = results.left_hand_landmarks if args.side == 'left' else results.right_hand_landmarks
                    if hand:
                        hand_found += 1
                        hand_px = []
                        for lm in hand.landmark:
                            x, y = lm.x * frame_w, lm.y * frame_h
                            hand_px.append((int(x), int(y)))
                            row_data.extend([round(x, 1), round(y, 1), round(lm.z * frame_w, 1)])
                    else:
                        warnings.append("WARNING: No Hand Detected")
                        row_data.extend([''] * (3 * len(HAND_POINT_NAMES)))

                csv_writer.writerow(row_data)

                # -----------------------------
                # 6. Switches: light level and touching (measured on the clean frame, before drawing)
                # -----------------------------
                if switch_boxes:
                    frame_times.append(timestamp_ms)
                    camera.update(frame)
                    # Points that can touch a switch: both hands/fingers (upper limb) or both feet (lower limb),
                    # so touching doesn't depend on MediaPipe telling the subject's left and right apart
                    touch_pts = []
                    if limb['hand']:
                        extra = hands_model.process(image_rgb).multi_hand_landmarks or []
                        for h in [results.left_hand_landmarks, results.right_hand_landmarks] + list(extra):
                            if h:
                                touch_pts += [(lm.x * frame_w, lm.y * frame_h) for lm in h.landmark]
                    if results.pose_landmarks:
                        lms = results.pose_landmarks.landmark
                        touch_pts += [(lms[i].x * frame_w, lms[i].y * frame_h)
                                      for side in ('left', 'right') for i in limb['touch_points'][side]
                                      if lms[i].visibility > 0.5]
                    cur_touch = []
                    for s in range(len(switch_boxes)):
                        light_signal[s].append(light_energy(frame, camera.apply(light_corners[s])))
                        # Distance of the closest point outside the box (negative = inside), in box sizes
                        poly = camera.apply(touch_corners[s])
                        dist = min((-cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in touch_pts),
                                   default=np.inf) / touch_sizes[s]
                        touching[s].append(dist)
                        cur_touch.append(dist <= TOUCH_DIST)

                # -----------------------------
                # 7. Draw the limb on the full frame -> output video; preview adds warnings and switches
                # -----------------------------
                draw_limb(frame, joints_px, limb['segments'], hand_px, color, draw_scale)
                video_writer.write(frame)

                for k, text in enumerate(warnings):
                    cv2.putText(frame, text, (30, int((50 + 45 * k) * text_scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_t)
                if switch_boxes:
                    # Switch outline: green = being touched, otherwise yellow; number above the box
                    for s in range(len(switch_boxes)):
                        poly = camera.apply(touch_corners[s]).astype(np.int32)
                        sw_color = (0, 255, 0) if cur_touch[s] else (0, 255, 255)
                        cv2.polylines(frame, [poly], True, sw_color, max(2, int(2 * draw_scale)), cv2.LINE_AA)
                        cv2.putText(frame, str(s + 1), (int(poly[:, 0].min()), int(poly[:, 1].min()) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, sw_color, text_t)

                # -----------------------------
                # 8. Display Video
                # -----------------------------
                display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
                frame_idx += 1
                cv2.imshow(WINDOW_NAME, display_frame)
                if frame_idx == 1:
                    # Bring the preview in front of other windows once, then let it behave normally
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
                    cv2.waitKey(1)
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    print(f"\r  frame {frame_idx}/{total_frames}", end='', flush=True)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n  Stopped early by user.")
                    break
                # Closing the window with the X button also stops processing
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n  Window closed - stopped early.")
                    break

    cap.release()
    video_writer.release()
    if hands_model:
        hands_model.close()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Data saved to {output_csv_path}")
    print(f"Skeleton video saved to {output_video_path}")
    if limb['hand'] and frame_idx:
        print(f"Hand/fingers found in {hand_found}/{frame_idx} frames ({100 * hand_found / frame_idx:.0f}%)")
    if switch_boxes and frame_times:
        write_switch_events(events_csv_path, light_signal, touching, frame_times, fps)


def write_switch_events(path, light_signal, touching, frame_times, fps):
    """One row per light event: switch number, light start, first touch, light off (ms, video time).

    Empty cell = unknown: start of a light already on when the video begins, off of a light still on
    when it ends, or no touch found. A switch with no light events gets one row with only its number."""
    touch_window = int(round(TOUCH_AFTER_OFF_S * fps))
    rows = []
    print("\nSwitch events:")
    for s in range(len(light_signal)):
        events, reason = find_light_events(light_signal[s], fps)
        if not events:
            print(f"  switch {s + 1}: no light events ({reason})")
            rows.append([s + 1, '', '', ''])  # keep every switch in the file
            continue
        print(f"  switch {s + 1}: {len(events)} light event(s)")
        n = len(frame_times)
        for k, (on, off) in enumerate(events):
            # Touch search: from light start until 1 s after it turns off (but not into the next event)
            end = n if off is None else min(n, off + touch_window)
            if k + 1 < len(events):
                end = min(end, events[k + 1][0])
            touch = next((f for f in range(on, end) if touching[s][f] <= TOUCH_DIST), None)
            rows.append([s + 1, round(frame_times[on], 1) if on > 0 else '',
                         round(frame_times[touch], 1) if touch is not None else '',
                         round(frame_times[off], 1) if off is not None else ''])
    rows.sort(key=lambda r: (r[0], r[1] if r[1] != '' else -1))
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Switch', 'Start_ms', 'Touching_ms', 'Turn_off_ms'])
        writer.writerows(rows)
    print(f"Switch events saved to {path}")


if __name__ == "__main__":
    main()
