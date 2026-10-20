"""A whole session: per-frame caches (1 or 2 cameras) -> trials synced with the design -> measures -> files.

The session's settings live in <session folder>/session.json (made by the app, lab_app.py):
  subject, session, cam_a, cam_b (optional), offset_s (seconds into video A at which video B started),
  calibration (camera file for 2 cameras), boxes_a/boxes_b (switch boxes), design (tested_side, contralateral,
  practice_limb, practice_side, retest), settings (experiment_design.Settings), parts_override, touse_override.
Outputs in the session folder:
  <subject>_<session>_trials.csv   - every trial: part/block/sequence/trial/target, light on, touch, hit,
                                     light off, trial end, outcome, anticipation, corrected
  <subject>_<session>_analysis.xlsx - measures per trial + mean/median/sd/iqr by part/block/sequence
  trials.json                       - the same rows, for the all-subjects export
"""
import json
import os
from dataclasses import asdict

import cv2
import numpy as np

import experiment_design as ed
import session_engine as se
import trial_measures as tm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')


def default_state(subject='S01', session='d1'):
    return dict(subject=subject, session=session, cam_a='', cam_b='', offset_s=0.0, calibration='',
                boxes_a=None, boxes_b=None,
                design=dict(tested_side='right', contralateral=False, practice_limb='upper',
                            practice_side='right', retest=session != 'd1'),
                settings=asdict(ed.Settings()), parts_override=None, touse_override={},
                scan_start_s=0.0, scan_minutes=5.0)


def session_dir(state):
    return os.path.join(RESULTS_DIR, state['subject'], state['session'])


def load_state(folder):
    with open(os.path.join(folder, 'session.json')) as f:
        s = default_state()
        s.update(json.load(f))
        return s


def save_state(state):
    folder = session_dir(state)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'session.json'), 'w') as f:
        json.dump(state, f, indent=1)


def template_of(state):
    d = state['design']
    return ed.session_template(d['tested_side'], d['contralateral'], d['practice_limb'], d['practice_side'],
                               d.get('retest', False))


def settings_of(state):
    return ed.Settings(**state['settings'])


def trial_key(r):
    return f"{r['Part']}|{r['Block']}|{r['Sequence']}|{r['Trial']}"


def _first_frame_xy(cache, pts, frames):
    """Points (frames, 2) of a camera -> that camera's first-frame coordinates (camera movement removed)."""
    out = np.full_like(pts, np.nan, dtype=float)
    for k, i in enumerate(frames):
        if i is None or np.isnan(pts[k, 0]):
            continue
        inv = cv2.invertAffineTransform(cache['motion'][i])
        out[k] = inv[:, :2] @ pts[k] + inv[:, 2]
    return out


class Session:
    """Loaded caches + analysis results of one session."""

    def __init__(self, state):
        self.state = state
        self.folder = session_dir(state)
        self.A = se.load_cache(state['cam_a'], self.folder, state['boxes_a']) if state['cam_a'] else None
        self.B = (se.load_cache(state['cam_b'], self.folder, state['boxes_b'])
                  if state.get('cam_b') and state.get('boxes_b') else None)
        self.result = None
        self.kin = {}

    @property
    def ready(self):
        return self.A is not None

    @property
    def two_cameras(self):
        return self.B is not None and bool(self.state.get('calibration'))

    def b_frame_for(self):
        """For each frame of camera A, the camera-B frame taken closest in time (or -1)."""
        ta = self.A['times'] / 1000
        tb = self.B['times'] / 1000 + self.state.get('offset_s', 0.0)
        j = np.clip(np.searchsorted(tb, ta), 1, len(tb) - 1)
        j = j - ((ta - tb[j - 1]) < (tb[j] - ta))
        j[np.abs(tb[j] - ta) > 0.5 / max(1, self.B['meta']['fps'])] = -1
        return j

    def a_frame_for_b(self, fb):
        ta = self.A['times'] / 1000
        t = self.B['times'][fb] / 1000 + self.state.get('offset_s', 0.0)
        return int(np.clip(np.searchsorted(ta, t), 0, len(ta) - 1))

    def analyze(self):
        st, A = self.state, self.A
        fps = A['meta']['fps']
        times = A['times']
        events = ed.light_events(A['light'], fps)
        touch = {'upper': A['touch_upper'].copy(), 'lower': A['touch_lower'].copy()}
        units = 'px (camera A image)'
        if self.B is not None:
            bf = self.b_frame_for()
            ev_b = []
            for s, on, off in ed.light_events(self.B['light'], self.B['meta']['fps']):
                ev_b.append((s, self.a_frame_for_b(on) if on > 0 else 0,
                             None if off is None else self.a_frame_for_b(off)))
            events = ed.merge_camera_events(events, ev_b, fps)
            ok = bf >= 0
            for limb in touch:
                tb = self.B[f'touch_{limb}']
                touch[limb][:, ok] = np.minimum(touch[limb][:, ok], tb[:, bf[ok]])
        template = template_of(st)
        res = ed.analyze_session(events, times, fps, touch, template, settings_of(st), st.get('parts_override'))

        # Marker tracks and switch positions (2D in camera A, or 3D with two calibrated cameras)
        w, h = A['meta']['size']
        boxes = np.array(st['boxes_a'])
        centers_a = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2 * w, (boxes[:, 1] + boxes[:, 3]) / 2 * h])
        sizes = np.sqrt((boxes[:, 2] - boxes[:, 0]) * w * (boxes[:, 3] - boxes[:, 1]) * h)
        tracks, n = {}, len(times)
        tracks_a = {}
        for p in template:
            key = (p.limb, p.side)
            if key not in tracks_a:
                tracks_a[key] = _first_frame_xy(A, tm.marker_track(A['pose'], *key, A['meta']['size']), range(n))
        fallback = None
        if self.two_cameras:
            from calibrate_cameras import load_cameras, triangulate_px
            (ca, cb), units = load_cameras(st['calibration'])
            bf = self.b_frame_for()
            bb = np.array(st['boxes_b'])
            wb, hb = self.B['meta']['size']
            centers_b = np.column_stack([(bb[:, 0] + bb[:, 2]) / 2 * wb, (bb[:, 1] + bb[:, 3]) / 2 * hb])
            tri = lambda pa, pb: triangulate_px(pa, pb, ca['K'], cb['K'], cb['R'], cb['t'], ca['dist'], cb['dist'])
            switch_pos = tri(centers_a, centers_b)
            switch_size = np.ones(len(switch_pos))
            for p in template:
                key = (p.limb, p.side)
                if key in tracks:
                    continue
                pa = _first_frame_xy(A, tm.marker_track(A['pose'], *key, A['meta']['size']), range(n))
                mb = tm.marker_track(self.B['pose'], *key, self.B['meta']['size'])
                pb = _first_frame_xy(self.B, np.where((bf >= 0)[:, None], mb[np.maximum(bf, 0)], np.nan),
                                     [int(x) if x >= 0 else None for x in bf])
                ok = ~np.isnan(pa[:, 0]) & ~np.isnan(pb[:, 0])
                X = np.full((n, 3), np.nan)
                if ok.any():
                    X[ok] = tri(pa[ok], pb[ok])
                tracks[key] = X
            units = f'3D, {units}'
            fallback = (tracks_a, centers_a, sizes, 'camera A')
            source = '3D'
        else:
            switch_pos, switch_size, tracks, source = centers_a, sizes, tracks_a, 'camera A'
        tm.measure_trials(res['trials'], times, tracks, switch_pos, switch_size, source, fallback)
        self.fallback = fallback
        detected = [r for r in res['trials'] if r['on_frame'] is not None and r.get('off_frame') is not None]
        with_data = [r for r in detected if not np.isnan(r['responsetime'])]
        if detected and len(with_data) < 0.8 * len(detected):
            limbs = ', '.join(sorted({f"{r['Limb']} {r['Side']}" for r in detected}))
            res['warnings'].append(
                f'The marker ({limbs}: wrist / ankle) was usable in only {len(with_data)} of {len(detected)} trials - '
                'MediaPipe does not see it well from this camera (e.g. the far arm, hidden by the body).')
        over = st.get('touse_override') or {}
        for r in res['trials']:
            if trial_key(r) in over:
                r['touse'] = bool(over[trial_key(r)])
        self.tracks, self.switch_pos, self.switch_size, self.units = tracks, switch_pos, switch_size, units
        self.result = res
        return res

    def trial_curves(self, r):
        """Kinematics of one trial for the graphs: dict(time, xyz, tangvel, stim) + measures (or None)."""
        if r.get('on_frame') is None or r.get('off_frame') is None:
            return None, None
        t_s = self.A['times'] / 1000
        a, b = r['window_start'], r['off_frame']
        key = (r['Limb'], r['Side'])
        tracks, switch_pos = self.tracks, self.switch_pos
        if r.get('Data') == 'camera A' and self.fallback:
            tracks, switch_pos = self.fallback[0], self.fallback[1]
        k = tm.trial_kinematics(t_s[a:b + 1], tracks[key][a:b + 1], t_s[r['on_frame']])
        m = tm.calculate_measures(k, switch_pos[r['Target'] - 1]) if k else None
        return k, m

    def set_touse(self, r, value):
        r['touse'] = bool(value)
        self.state.setdefault('touse_override', {})[trial_key(r)] = bool(value)
        save_state(self.state)

    def write_outputs(self):
        st = self.state
        base = os.path.join(self.folder, f"{st['subject']}_{st['session']}")
        trials = self.result['trials']
        tm.write_trials_csv(base + '_trials.csv', trials)
        rows = [tm.measure_row(st['subject'], st['session'], r) for r in trials]
        info = {'Subject': st['subject'], 'Session': st['session'], 'Camera A': st['cam_a'],
                'Camera B': st.get('cam_b') or '-', 'Units': self.units,
                'Switch order': ', '.join(map(str, self.result['pattern'])),
                'Warnings': ' | '.join(self.result['warnings']) or '-'}
        tm.write_analysis_xlsx(base + '_analysis.xlsx', {st['subject']: rows}, info, self.units)
        with open(os.path.join(self.folder, 'trials.json'), 'w') as f:
            json.dump([{k: tm._cell(v) for k, v in r.items()} for r in rows], f)
        return base + '_trials.csv', base + '_analysis.xlsx'


def export_all(results_dir=RESULTS_DIR, path=None):
    """makeCSVall: every analysed session under results/ (subject/session/trials.json) -> one Excel file.
    The sessions of a subject (day 1, re-test day) are combined."""
    by_subject = {}
    for subject in sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []:
        sdir = os.path.join(results_dir, subject)
        if not os.path.isdir(sdir):
            continue
        for session in sorted(os.listdir(sdir)):
            f = os.path.join(sdir, session, 'trials.json')
            if os.path.exists(f):
                with open(f) as fh:
                    rows = json.load(fh)
                for r in rows:
                    for m in tm.MEASURES:
                        r[m] = np.nan if r.get(m) in ('', None) else float(r[m])
                    r['touse'] = bool(r.get('touse'))
                by_subject.setdefault(subject, []).extend(rows)
    path = path or os.path.join(results_dir, 'analysis_all_subjects.xlsx')
    tm.write_analysis_xlsx(path, by_subject, {'Subjects': ', '.join(by_subject) or '-'})
    return path, len(by_subject)
