"""Turn a session's light events into the experiment's trials: part / block / sequence / trial.

Design: a block = 5 sequences x 6 reaching trials toward the lit switch, always the same 6-switch order;
30 s pause between blocks. A session = parts in a fixed order, e.g.
  Familiarization (2 blocks), pre_test (3) [+ pre_test_contralateral (3)], practice (10),
  post_test (3) [+ post_test_contralateral (3)]        - re-test day: re_test (3) [+ contralateral (3)]
Trial = light on (start) -> button hit (end; the light goes off when the button is pressed) or, if the
button is not hit within the 1 s window, the light turning off by itself (end; a miss).

Steps: light events from the per-frame light levels (one or two cameras) -> blocks (gaps > block_gap_s
between lights) -> parts (blocks taken in order from the template; the user can change it) -> inside a
block, each light is matched to its place in the 6-switch order (a missed or extra light doesn't shift
the rest) -> trial rows.
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
        off_m = None if off is None or off2 is None else max(off, off2)
        out.append((s, on_m, off_m))
    out += [e for i, e in enumerate(ev_b) if i not in used]
    return sorted(out, key=lambda e: e[1])


# -----------------------------
# Blocks, parts, pattern
# -----------------------------
def split_blocks(events, times_ms, settings):
    """Groups of event indices separated by pauses longer than block_gap_s."""
    blocks, cur = [], []
    for i, (_, on, _) in enumerate(events):
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


def find_pattern(blocks, events, settings):
    """The 6-switch order: the first 6 lights of the blocks (most common), or the settings' pattern."""
    if settings.pattern:
        return list(settings.pattern)
    n = settings.trials_per_sequence
    starts = [tuple(events[i][0] for i in b[:n]) for b in blocks
              if len(b) >= settings.min_block_trials]
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
            d = score[i - 1, j - 1] + (MATCH if expected[i - 1] == targets[j - 1] else MISMATCH)
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
            if expected[i - 1] == targets[j - 1]:
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
def build_trials(events, times_ms, fps, blocks, parts, template, settings, touch, pattern):
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
        mapping = align_block([events[i][0] for i in block], pattern, settings.sequences_per_block)
        first_slot = min(mapping) if mapping else 0
        for slot in range(first_slot, len(pattern) * settings.sequences_per_block):
            seq, trial = slot // len(pattern) + 1, slot % len(pattern) + 1
            row = dict(Part=part, Block=block_no, Sequence=seq, Trial=trial, Target=pattern[slot % len(pattern)],
                       Limb=limb, Side=side, Light_on_ms=None, Hit_ms=None, Light_off_ms=None, Trial_end_ms=None,
                       Touch_ms=None, Outcome='not detected', on_frame=None, off_frame=None, block_index=bi)
            if slot in mapping:
                s, on, off = events[block[mapping[slot]]]
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
                d = touch[limb][s - 1, on:end]
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
    """Everything from per-frame light levels to trial rows. Returns dict(events, blocks, parts, pattern,
    trials, warnings). `parts_override`: the user's part for each block (same order as blocks)."""
    events = light_events(light, fps) if not isinstance(light, list) else light
    blocks = split_blocks(events, times_ms, settings)
    parts, warnings = assign_parts(blocks, template, settings)
    if parts_override is not None and len(parts_override) == len(blocks):
        parts = [tuple(p) for p in parts_override]
        warnings = []
    pattern = find_pattern(blocks, events, settings)
    if not pattern:
        return dict(events=events, blocks=blocks, parts=parts, pattern=[], trials=[],
                    warnings=warnings + ['No blocks found - no trials.'])
    trials = build_trials(events, times_ms, fps, blocks, parts, template, settings, touch, pattern)
    for (part, no), b in zip(parts, blocks):
        if part != IGNORE and len(b) != settings.trials_per_sequence * settings.sequences_per_block:
            warnings.append(f'{part} block {no}: {len(b)} lights instead of '
                            f'{settings.trials_per_sequence * settings.sequences_per_block}.')
    return dict(events=events, blocks=blocks, parts=parts, pattern=pattern, trials=trials, warnings=warnings)
