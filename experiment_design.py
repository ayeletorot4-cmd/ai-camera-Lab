"""Turn a session's light events into the experiment's trials: part / block / sequence / trial.

Design: a block = 5 sequences x 6 reaching trials toward the lit switch, always the same 6-switch order;
30 s pause between blocks. A session = parts in a fixed order, e.g.
  Familiarization (2 blocks), pre_test (3) [+ pre_test_contralateral (3)], practice (10),
  post_test (3) [+ post_test_contralateral (3)]        - re-test day: re_test (3) [+ contralateral (3)]
Trial = light on (start) -> button hit (end; the light goes off when the button is pressed) or, if the
button is not hit within the 1 s window, the light turning off by itself (end; a miss).

Steps: light events from the per-frame light levels (one or two cameras) -> lights no camera saw (e.g. an LED
facing away) filled in from the timing: the next light always comes on 1 s after a trial ends, so a gap with
room for a trial between two lights is a hidden light (start = previous end + 1 s, end = next start - 1 s; the
switch = where the hand/foot goes) -> blocks (gaps > block_gap_s between lights) -> parts (blocks taken in
order from the template; the user can change it) -> inside a block, each light is matched to its place in the
6-switch order (a missed or extra light doesn't shift the rest) -> trial rows.
"""
from dataclasses import dataclass, field

import numpy as np

import extract_kinematics as ek

PART_ORDER = ['Familiarization', 'pre_test', 'pre_test_contralateral', 'practice',
              'post_test', 'post_test_contralateral', 're_test', 're_test_contralateral']
IGNORE = '(ignore)'


@dataclass
class Settings:
    trials_per_sequence: int = 6
    sequences_per_block: int = 5
    block_gap_s: float = 10.0        # a pause longer than this between two lights starts a new block
    light_window_s: float = 1.0      # the light stays on at most this long when the button is not hit
    miss_margin_s: float = 0.06      # light on >= window - margin -> nobody pressed it (a miss)
    min_block_trials: int = 10       # shorter groups of lights (e.g. testing the lights) are ignored
    fill_hidden_lights: bool = True  # infer lights no camera saw from the 1 s timing between trials
    pattern: list = field(default_factory=list)   # the 6-switch order; empty = found from the data


@dataclass
class Part:
    name: str
    blocks: int
    limb: str     # 'upper' / 'lower'
    side: str     # 'right' / 'left'


def session_template(tested_side='right', contralateral=False, practice_limb='upper', practice_side=None,
                     retest=False):
    """The parts of a session in order (the user can edit them)."""
    other = 'left' if tested_side == 'right' else 'right'
    practice_side = practice_side or tested_side
    test = lambda name: [Part(name, 3, 'upper', tested_side)] + (
        [Part(f'{name}_contralateral', 3, 'upper', other)] if contralateral else [])
    if retest:
        return test('re_test')
    return ([Part('Familiarization', 2, practice_limb, practice_side)] + test('pre_test')
            + [Part('practice', 10, practice_limb, practice_side)] + test('post_test'))


# -----------------------------
# Light events
# -----------------------------
def light_events(light, fps):
    """light: (switches, frames) -> list of (switch number 1..n, on frame, off frame or None), by time.
    An on frame of 0 means the light was already on when the video started."""
    events = []
    for s in range(light.shape[0]):
        ev, _ = ek.find_light_events(light[s], fps)
        events += [(s + 1, on, off) for on, off in ev]
    return sorted(events, key=lambda e: e[1])


def merge_camera_events(ev_a, ev_b, fps, max_gap_s=0.2):
    """Events of the same switches seen by two cameras (frames on camera A's clock) -> one list. Seen by
    both: earliest on, latest off (a hand over the LED can only make a light look off in one view)."""
    out, used = [], set()
    for s, on, off in ev_a:
        k = next((i for i, (s2, on2, _) in enumerate(ev_b)
                  if i not in used and s2 == s and abs(on2 - on) <= max_gap_s * fps), None)
        if k is None:
            out.append((s, on, off))
            continue
        used.add(k)
        _, on2, off2 = ev_b[k]
        on_m = min(on, on2) if on > 0 and on2 > 0 else max(on, on2)
        # off None = still on when that camera's video ended: the other camera knows (unless it ended too)
        offs = [o for o in (off, off2) if o is not None]
        off_m = max(offs) if offs else None
        out.append((s, on_m, off_m))
    out += [e for i, e in enumerate(ev_b) if i not in used]
    return sorted(out, key=lambda e: e[1])


def inter_trial_delay(events, times_ms):
    """Typical time from a light going off to the next light coming on (s), or None."""
    d = [(times_ms[b[1]] - times_ms[a[2]]) / 1000 for a, b in zip(events[:-1], events[1:])
         if a[2] is not None and b[1] > 0]
    d = [x for x in d if 0.3 < x < 1.6]
    return float(np.median(d)) if len(d) >= 5 else None


def _touched_switch(touch_any, on, off):
    """The switch the hand/foot is on at the end of a hidden light (the press is what turns a light off - on the
    way the hand can pass over a neighbouring switch), or 0 if it isn't near any switch then."""
    n = touch_any.shape[1]
    end = min(n, max(on + 1, off + 3))
    seg = touch_any[:, max(on, end - 9):end]          # the last ~0.15 s
    if seg.size == 0 or not np.isfinite(seg).any():
        return 0
    best = np.nanmin(np.where(np.isfinite(seg), seg, np.nan), axis=1)
    s = int(np.nanargmin(best))
    return s + 1 if best[s] <= 3 * ek.TOUCH_DIST else 0


def _hidden_end(touch_any, s, on, fps, settings, limit):
    """End of a hidden light: the first frame the hand/foot is on the switch (= the press turns it off; s = 0:
    on any switch, after first leaving the one it started at), else the end of the light window (a miss)."""
    window = int(round(settings.light_window_s * fps))
    stop = min(on + window, limit)
    rows = touch_any[[s - 1]] if s > 0 else touch_any
    seg = rows[:, on:stop] <= ek.TOUCH_DIST
    if s == 0:
        seg &= ~(rows[:, on:on + 1] <= ek.TOUCH_DIST)     # not the switch it was already touching
    hit = np.nonzero(seg.any(axis=0))[0]
    return on + int(hit[0]) if len(hit) else stop


def fill_hidden_lights(events, times_ms, fps, touch_any, settings):
    """Insert the lights no camera saw. Events become (switch, on, off, source): source 'video' or 'timing'.
    Switch 0 = hidden light whose switch couldn't be told (no hand/foot near any switch)."""
    events = [tuple(e[:3]) + ('video',) for e in events]
    delay = inter_trial_delay(events, times_ms) if settings.fill_hidden_lights else None
    if delay is None:
        return events, None
    n = len(times_ms)
    to_frame = lambda ms: int(np.clip(np.searchsorted(times_ms, ms), 0, n - 1))
    min_trial = 0.15
    out = []
    for k, e in enumerate(events):
        out.append(e)
        nxt = events[k + 1] if k + 1 < len(events) else None
        if e[2] is None or nxt is None or nxt[1] <= 0:
            continue
        end_ms, next_on_ms = times_ms[e[2]], times_ms[nxt[1]]
        # Room for one more trial in this gap? (it's not the pause between blocks)
        while (next_on_ms - end_ms) / 1000 >= 2 * delay + min_trial and \
                (next_on_ms - end_ms) / 1000 < settings.block_gap_s:
            on = to_frame(end_ms + delay * 1000)
            off_by_timing = to_frame(next_on_ms - delay * 1000)
            if (times_ms[off_by_timing] - times_ms[on]) / 1000 <= settings.light_window_s + 0.1:
                off = off_by_timing                       # the next light came 1 s after this one ended
            else:                                         # two or more hidden lights: end from the touch
                off = _hidden_end(touch_any, 0, on, fps, settings, off_by_timing)
            s = _touched_switch(touch_any, on, off + 2)
            out.append((s, on, off, 'timing'))
            end_ms = times_ms[off]
    return out, delay


# -----------------------------
# Blocks, parts, pattern
# -----------------------------
def split_blocks(events, times_ms, settings):
    """Groups of event indices separated by pauses longer than block_gap_s."""
    blocks, cur = [], []
    for i, e in enumerate(events):
        on = e[1]
        if cur and (times_ms[on] - times_ms[events[cur[-1]][1]]) / 1000 > settings.block_gap_s:
            blocks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    return blocks


def assign_parts(blocks, template, settings):
    """Default part for each block: blocks too short to be real are ignored, the rest take the template's
    parts in order. Returns [(part name, block number in the part)] and warnings."""
    out, warnings = [], []
    slots = [(p.name, b + 1) for p in template for b in range(p.blocks)]
    real = [i for i, b in enumerate(blocks) if len(b) >= settings.min_block_trials]
    k = 0
    for i, b in enumerate(blocks):
        if i not in real:
            out.append((IGNORE, 0))
        elif k < len(slots):
            out.append(slots[k])
            k += 1
        else:
            out.append((IGNORE, 0))
    if len(real) != len(slots):
        warnings.append(f'Found {len(real)} blocks but the session plan has {len(slots)} - check the parts.')
    short = len(blocks) - len(real)
    if short:
        warnings.append(f'{short} short group(s) of lights (< {settings.min_block_trials}) ignored.')
    return out, warnings


def parse_pattern(text):
    """'1,4,3,5,4,2', '1 4 3 5 4 2' or '143542' (switch numbers are single digits) -> [1, 4, 3, 5, 4, 2]."""
    text = str(text).strip()
    if not text:
        return []
    if text.isdigit():
        return [int(c) for c in text]
    return [int(x) for x in text.replace(';', ',').replace(' ', ',').split(',') if x.strip()]


def pattern_problem(pattern, settings):
    """Why a typed switch order can't be right, or None."""
    if not pattern:
        return None
    if any(not 1 <= p <= ek.N_SWITCHES for p in pattern):
        return f'switch numbers must be 1-{ek.N_SWITCHES}'
    if len(pattern) != settings.trials_per_sequence:
        return f'{len(pattern)} switches, but a sequence has {settings.trials_per_sequence} trials'
    return None


def find_pattern(blocks, events, settings):
    """The 6-switch order: the first 6 lights of the blocks (most common), or the settings' pattern."""
    if settings.pattern and not pattern_problem(settings.pattern, settings):
        return list(settings.pattern)
    n = settings.trials_per_sequence
    starts = [tuple(events[i][0] for i in b[:n]) for b in blocks
              if len(b) >= settings.min_block_trials]
    starts = [st for st in starts if 0 not in st and len(st) == n] or starts
    if not starts:
        return []
    values, counts = np.unique(np.array(starts), axis=0, return_counts=True)
    return [int(v) for v in values[np.argmax(counts)]]


def align_block(targets, pattern, n_sequences):
    """Match a block's lights (switch numbers, in time order) to the expected order (pattern repeated
    n_sequences times). Returns {slot index: light index}. A light that is missing (not seen) leaves its
    slot empty; an extra light is left out; the lights may start later than slot 0 (a partial block)."""
    expected = list(pattern) * n_sequences
    E, O = len(expected), len(targets)
    MATCH, MISMATCH, SKIP_E, SKIP_O = 2.0, -3.0, -1.0, -2.0
    # score[i][j]: best for expected[:i], observed[:j]; skipping expected slots before the first light is free
    score = np.full((E + 1, O + 1), -np.inf)
    back = np.zeros((E + 1, O + 1), int)  # 1 = diagonal, 2 = skip expected, 3 = skip observed
    score[:, 0] = 0.0
    for j in range(1, O + 1):
        score[0, j] = score[0, j - 1] + SKIP_O
        back[0, j] = 3
    for i in range(1, E + 1):
        for j in range(1, O + 1):
            t = targets[j - 1]   # 0 = a hidden light whose switch is unknown: fits anywhere, but weaker
            d = score[i - 1, j - 1] + (MATCH if expected[i - 1] == t else 1.0 if t == 0 else MISMATCH)
            u = score[i - 1, j] + SKIP_E
            l = score[i, j - 1] + SKIP_O
            best = max(d, u, l)
            score[i, j] = best
            back[i, j] = 1 if best == d else (2 if best == u else 3)
    # The block may also end early (video cut): free skipping of the remaining expected slots
    i, j = int(np.argmax(score[:, O])), O
    mapping = {}
    while i > 0 and j > 0:
        if back[i, j] == 1:
            if expected[i - 1] == targets[j - 1] or targets[j - 1] == 0:
                mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif back[i, j] == 2:
            i -= 1
        else:
            j -= 1
    return mapping


# -----------------------------
# Trials
# -----------------------------
def build_trials(events, times_ms, fps, blocks, parts, template, settings, touch, pattern, delay=None):
    """Trial rows for all blocks that belong to a part. `touch`: {'upper'/'lower': (switches, frames)}
    hand/foot distance to each switch box (box sizes)."""
    limb_of = {p.name: (p.limb, p.side) for p in template}
    window_frames = int(round((settings.light_window_s - settings.miss_margin_s) * fps))
    n = len(times_ms)
    rows = []
    ms = lambda f: round(float(times_ms[f]), 1) if f is not None and 0 <= f < n else None
    for bi, (block, (part, block_no)) in enumerate(zip(blocks, parts)):
        if part == IGNORE:
            continue
        limb, side = limb_of.get(part, ('upper', 'right'))
        # A light found from the timing fits its place whatever switch the hand was nearest to (less certain)
        mapping = align_block([0 if len(events[i]) > 3 and events[i][3] == 'timing' else events[i][0]
                               for i in block], pattern, settings.sequences_per_block)
        first_slot = min(mapping) if mapping else 0
        for slot in range(first_slot, len(pattern) * settings.sequences_per_block):
            seq, trial = slot // len(pattern) + 1, slot % len(pattern) + 1
            row = dict(Part=part, Block=block_no, Sequence=seq, Trial=trial, Target=pattern[slot % len(pattern)],
                       Limb=limb, Side=side, Light_on_ms=None, Hit_ms=None, Light_off_ms=None, Trial_end_ms=None,
                       Touch_ms=None, Outcome='not detected', on_frame=None, off_frame=None, block_index=bi,
                       Light_source='')
            ev = events[block[mapping[slot]]] if slot in mapping else None
            prev = rows[-1] if rows and rows[-1]['block_index'] == bi else None
            if ev is None and delay is not None and prev is not None and prev['off_frame'] is not None \
                    and not any(k > slot for k in mapping):
                # A hidden light at the end of the block: 1 s after the previous trial, ended by the touch
                on = int(np.clip(np.searchsorted(times_ms, times_ms[prev['off_frame']] + delay * 1000), 0, n - 1))
                touch_any = np.minimum(touch['upper'], touch['lower'])
                target = pattern[slot % len(pattern)]
                window = int(round(settings.light_window_s * fps))
                near = touch_any[target - 1, on:min(n, on + window + 2)]
                # Only if the hand/foot really went to that switch (else the block simply ended)
                if on < n - 1 and np.isfinite(near).any() and np.nanmin(near) <= 3 * ek.TOUCH_DIST:
                    ev = (target, on, _hidden_end(touch_any, target, on, fps, settings, n - 1), 'timing')
            if ev is not None:
                s, on, off = ev[:3]
                row['Light_source'] = ev[3] if len(ev) > 3 else 'video'
                row.update(on_frame=on, off_frame=off, Light_on_ms=ms(on) if on > 0 else None,
                           Light_off_ms=ms(off))
                if off is None or on == 0:
                    row['Outcome'] = 'unknown'          # light still on at the end / already on at the start
                elif off - on >= window_frames:
                    row['Outcome'] = 'miss'             # the light went off by itself: no hit in time
                else:
                    row['Outcome'] = 'hit'
                    row['Hit_ms'] = ms(off)             # pressing the button turns the light off
                row['Trial_end_ms'] = ms(off)
                # First frame the hand/foot is on the lit switch (from the video)
                end = n if off is None else min(n, off + 2)
                d = touch[limb][row['Target'] - 1, on:end]
                hit = np.nonzero(d <= ek.TOUCH_DIST)[0]
                if len(hit):
                    row['Touch_ms'] = ms(on + int(hit[0]))
            rows.append(row)
        # Data window of each trial: from the previous trial's end (the home phase) to this trial's end
        trials = [r for r in rows if r['block_index'] == bi and r['on_frame'] is not None]
        for k, r in enumerate(trials):
            prev_end = trials[k - 1]['off_frame'] if k > 0 else None
            r['window_start'] = prev_end if prev_end is not None else max(0, r['on_frame'] - int(2 * fps))
    return rows


def analyze_session(light, times_ms, fps, touch, template, settings, parts_override=None):
    """Everything from per-frame light levels (or a list of light events) to trial rows. Returns dict(events,
    blocks, parts, pattern, trials, warnings, delay). `parts_override`: the user's part for each block."""
    events = light_events(light, fps) if not isinstance(light, list) else light
    seen = {e[0] for e in events}
    warnings = [f'Switch {s}: its light was not seen by any camera - its trials are found from the timing '
                '(1 s after the previous trial) and from where the hand/foot goes.'
                for s in range(1, ek.N_SWITCHES + 1) if s not in seen] if events else []
    problem = pattern_problem(settings.pattern, settings)
    if problem:
        warnings.append(f'The typed switch order {settings.pattern} is ignored ({problem}) - it is found from the '
                        'lights instead. Type it as e.g. 1,4,3,5,4,2.')
    touch_any = np.minimum(touch['upper'], touch['lower'])
    events, delay = fill_hidden_lights(events, times_ms, fps, touch_any, settings)
    blocks = split_blocks(events, times_ms, settings)
    parts, w = assign_parts(blocks, template, settings)
    warnings += w
    if parts_override is not None and len(parts_override) == len(blocks):
        parts = [tuple(p) for p in parts_override]
        warnings = [x for x in warnings if 'session plan has' not in x]
    pattern = find_pattern(blocks, events, settings)
    if not pattern or 0 in pattern:
        return dict(events=events, blocks=blocks, parts=parts, pattern=pattern, trials=[], delay=delay,
                    warnings=warnings + ['Could not find the switch order - type it in (Session > Advanced).'])
    trials = build_trials(events, times_ms, fps, blocks, parts, template, settings, touch, pattern, delay)
    expected = settings.trials_per_sequence * settings.sequences_per_block
    for bi, ((part, no), b) in enumerate(zip(parts, blocks)):
        if part == IGNORE:
            continue
        got = sum(1 for r in trials if r['block_index'] == bi and r['on_frame'] is not None)
        if got != expected:
            warnings.append(f'{part} block {no}: {got} trials found instead of {expected}.')
    return dict(events=events, blocks=blocks, parts=parts, pattern=pattern, trials=trials, warnings=warnings,
                delay=delay)
