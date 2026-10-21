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

import numpy as np

import experiment_design as ed
import extract_kinematics as ek
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


def _nearest(ref_times, query, tol):
    """Index of the nearest ref time for each query time, -1 if further than tol."""
    if len(ref_times) == 0:
        return np.full(len(query), -1)
    j = np.clip(np.searchsorted(ref_times, query), 1, max(1, len(ref_times) - 1))
    j = np.minimum(j, len(ref_times) - 1)
    j = j - ((j > 0) & (np.abs(query - ref_times[j - 1]) < np.abs(ref_times[j] - query)))
    j[np.abs(ref_times[j] - query) > tol] = -1
    return j


class Session:
    """Loaded caches + analysis results of one session.

    Timeline: the session clock is camera A's clock (camera B's time + the sync offset). It holds every frame of
    camera A, plus camera B's frames from before camera A started and after it stopped - so if one camera
    stops early (or starts late), the analysis continues with the other camera alone."""

    def __init__(self, state):
        self.state = state
        self.folder = session_dir(state)
        self.cams = {}
        for c in ('a', 'b'):
            if state.get(f'cam_{c}') and state.get(f'boxes_{c}'):
                cache = se.load_cache(state[f'cam_{c}'], self.folder, state[f'boxes_{c}'])
                if cache is not None:
                    self.cams[c] = cache
        self.A, self.B = self.cams.get('a'), self.cams.get('b')
        self.result = None
        self._timeline()

    @property
    def ready(self):
        return bool(self.cams)

    @property
    def two_cameras(self):
        return self.A is not None and self.B is not None and bool(self.state.get('calibration'))

    def _timeline(self):
        if not self.cams:
            self.times, self.idx = np.zeros(0), {}
            return
        off = self.state.get('offset_s', 0.0) * 1000 if self.A is not None else 0.0
        t = {c: self.cams[c]['times'] + (off if c == 'b' else 0.0) for c in self.cams}
        fps = {c: self.cams[c]['meta']['fps'] for c in self.cams}
        main = 'a' if 'a' in t else 'b'
        parts = [t[main]]
        if main == 'a' and 'b' in t:
            half = 500 / fps['b']
            parts = [t['b'][t['b'] < t['a'][0] - half], t['a'], t['b'][t['b'] > t['a'][-1] + half]]
        self.times = np.concatenate(parts)
        self.fps = fps[main]
        self.idx = {c: _nearest(t[c], self.times, 500 / fps[c]) for c in t}
        self.cam_times = t
        self.coverage = {c: (t[c][0] / 1000, t[c][-1] / 1000) for c in t}

    def to_master(self, c, frame):
        """A camera's frame -> the session timeline frame closest in time."""
        tt = self.cam_times[c][frame]
        return int(np.clip(np.searchsorted(self.times, tt - 1e-6), 0, len(self.times) - 1))

    def per_frame(self, c, arr):
        """Per-frame array of camera c (frames first) on the session timeline (NaN where c has no frame)."""
        i = self.idx[c]
        out = np.full((len(self.times),) + arr.shape[1:], np.nan, dtype=float)
        ok = i >= 0
        out[ok] = arr[i[ok]]
        return out

    def track_2d(self, c, key):
        """Marker of camera c on the session timeline, in that camera's first-frame image coordinates."""
        cache = self.cams[c]
        xy = tm.marker_track(cache['pose'], *key, cache['meta']['size'])
        xy = ek.to_first_frame(xy, cache['motion'][:len(xy)])
        return self.per_frame(c, xy)

    def analyze(self):
        st = self.state
        n = len(self.times)
        events = []
        for c, cache in self.cams.items():
            ev = [(s, self.to_master(c, on) if on > 0 else 0, None if off is None else self.to_master(c, off))
                  for s, on, off in ed.light_events(cache['light'], cache['meta']['fps'])]
            # A light 'still on' at the end of a camera that stopped early isn't a light still on in the session
            events = ev if not events else ed.merge_camera_events(events, ev, self.fps)
        touch = {}
        for limb in ('upper', 'lower'):
            arrs = [self.per_frame(c, cache[f'touch_{limb}'].T) for c, cache in self.cams.items()]
            touch[limb] = np.nan_to_num(np.fmin.reduce(arrs) if len(arrs) > 1 else arrs[0], nan=np.inf).T
        template = template_of(st)
        res = ed.analyze_session(events, self.times, self.fps, touch, template, settings_of(st),
                                 st.get('parts_override'))
        if len(self.cams) == 2:
            (a0, a1), (b0, b1) = self.coverage['a'], self.coverage['b']
            for name, (x0, x1), other in (('A', (a0, a1), (b0, b1)), ('B', (b0, b1), (a0, a1))):
                if x1 < other[1] - 5:
                    res['warnings'].append(f'Camera {name} stopped at {x1 / 60:.1f} min - from there on only the '
                                           f'other camera is used.')
                if x0 > other[0] + 5:
                    res['warnings'].append(f'Camera {name} started at {x0 / 60:.1f} min - before that only the '
                                           f'other camera is used.')

        # Marker sources: 3D (two calibrated cameras), then each camera's own image
        keys = {(p.limb, p.side) for p in template}
        sources, units = [], 'px (camera image)'
        for c, cache in self.cams.items():
            w, h = cache['meta']['size']
            boxes = np.array(st[f'boxes_{c}'])
            centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2 * w, (boxes[:, 1] + boxes[:, 3]) / 2 * h])
            sizes = np.array([ek.box_size(b, w, h) for b in boxes])
            sources.append(dict(name=f'camera {c.upper()}', cam=c, switch_pos=centers, switch_size=sizes,
                                track={k: self.track_2d(c, k) for k in keys}))
        if self.two_cameras:
            from calibrate_cameras import load_cameras, triangulate_px
            (ca, cb), cal_units = load_cameras(st['calibration'])
            tri = lambda pa, pb: triangulate_px(pa, pb, ca['K'], cb['K'], cb['R'], cb['t'], ca['dist'], cb['dist'])
            sa_, sb_ = sources
            track3 = {}
            for k in keys:
                pa, pb = sa_['track'][k], sb_['track'][k]
                ok = ~np.isnan(pa[:, 0]) & ~np.isnan(pb[:, 0])
                X = np.full((n, 3), np.nan)
                if ok.any():
                    X[ok] = tri(pa[ok], pb[ok])
                track3[k] = X
            sources.insert(0, dict(name='3D', cam=None, switch_pos=tri(sa_['switch_pos'], sb_['switch_pos']),
                                   switch_size=np.ones(len(sa_['switch_pos'])), track=track3))
            units = f'3D, {cal_units} (trials marked camera A/B: px of that camera)'
        tm.measure_trials(res['trials'], self.times, sources)
        self.sources = {src['name']: src for src in sources}
        detected = [r for r in res['trials'] if r['on_frame'] is not None and r.get('off_frame') is not None]
        with_data = [r for r in detected if r.get('Data')]
        if detected and len(with_data) < 0.8 * len(detected):
            limbs = ', '.join(sorted({f"{r['Limb']} {r['Side']}" for r in detected}))
            res['warnings'].append(
                f'The marker ({limbs}: wrist / ankle) was usable in only {len(with_data)} of {len(detected)} trials - '
                'MediaPipe does not see it well from the camera(s) (e.g. the far arm, hidden by the body).')
        over = st.get('touse_override') or {}
        for r in res['trials']:
            if trial_key(r) in over:
                r['touse'] = bool(over[trial_key(r)])
        self.units = units
        self.switch_pos = sources[0]['switch_pos']
        self.result = res
        return res

    def trial_curves(self, r):
        """Kinematics of one trial for the graphs: dict(time, xyz, tangvel, stim) + measures (or None)."""
        if r.get('on_frame') is None or r.get('off_frame') is None or not r.get('Data'):
            return None, None
        src = self.sources[r['Data']]
        t_s = self.times / 1000
        a, b = r['window_start'], r['off_frame']
        k = tm.trial_kinematics(t_s[a:b + 1], src['track'][(r['Limb'], r['Side'])][a:b + 1], t_s[r['on_frame']])
        m = tm.calculate_measures(k, src['switch_pos'][r['Target'] - 1]) if k else None
        return k, m

    def trial_switch_pos(self, r):
        return self.sources[r['Data']]['switch_pos'] if r.get('Data') in self.sources else self.switch_pos

    def frame_ref(self, f, prefer=None):
        """Session-timeline frame -> (camera, that camera's frame) to show; prefer a camera if it has the frame."""
        order = ([prefer] if prefer in self.idx else []) + [c for c in ('a', 'b') if c in self.idx]
        for c in order:
            if self.idx[c][f] >= 0:
                return c, int(self.idx[c][f])
        return None, None

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
                'Light gap filled from timing (s)': self.result.get('delay'),
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
