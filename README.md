# MediaPipe Limb Tracker ("only visuals" copy)

Tracks **one limb** in a lab video with MediaPipe Holistic. The whole frame is analyzed, but only the chosen limb is saved and drawn. Optionally also detects **switch events** (light on / touch / light off) for the 5 switches on the table.

## Quick start
1. Double-click **`Run Kinematics.bat`** (Windows) or **`Run Kinematics.command`** (Mac) and pick the video (on Mac the dialog opens in `media/`; on Windows, in the folder above this one).
2. Type **U** (upper limb) or **L** (lower limb).
3. Type **L** or **R** for the subject's own **left** or **right** side.
4. Type **Y** to detect switch events, or press ENTER to skip (see below).
5. Watch the preview. Press `Esc` or close the window to stop early. (Typing elsewhere while the preview is in front goes to the preview, so letters don't stop it.)

| Limb | Points tracked |
|---|---|
| upper | shoulder, elbow, wrist + 21 hand/finger points (red cross = wrist from the hand model) |
| lower | hip, knee, ankle, heel, toe (red cross = ankle) |

Colours: subject's left side = orange, right side = cyan.

## Videos
The videos are not in git (too big); the 10 raw lab videos (IMG_0119-0123, IMG_7554-7558, ~300 MB each) are files of the GitHub release **lab-videos** of this (private) repo. To get them into `media/`:
```
gh release download lab-videos -R ofir-code/ai-camera-lab -D media
```
(or download them from the repo's Releases page). The synced videos (`media/synced/`) and everything in `outputs/` can be made again from them with the scripts; the sync offsets are kept in `media/synced/*.json`.

## Switch events
The first time for each video, the switches are **found automatically**: one pass over the video (about a minute for a 30 s 4K video) looks for the black boxes with a red button whose pink light turns on during the video. The first frame then shows them numbered **1 to 5 from the subject's RIGHT to LEFT** (the numbering follows the arc of switches; the subject faces its middle). Check them and press:

| Key | |
|---|---|
| ENTER | OK (when all 5 are found) |
| F | flip the numbering (1 to 5 the other way round) |
| 1-5 | redraw that switch's box (drag, then ENTER) |
| A | add a switch that was not found (e.g. hidden by the subject in IMG_7554) |
| M | mark all 5 by hand: drag a box around each switch in order 1 to 5, ENTER after each |
| C | continue without switch events |

A box should include the switch's light. The boxes are saved in `outputs/switch_settings.json` and offered again next time for the same video name (type R in the launcher to find them again). Boxes for IMG_0120 and IMG_0123 are already saved.

For each switch:
- **Start** - the light turns on.
- **Touching** - the first frame a hand/finger point (upper limb: either hand) or foot point (lower limb: either foot) is inside the box or within 25% of the box size from it, from light start until 1 s after the light turns off. (The button sticks out on the subject's side, so fingertips pressing it sit at the edge of the box.)
- **Turn off** - the light turns off.

How the light is measured: the pink LED light and its glow inside the box enlarged by 25% (the deep-red button, white table and skin are left out), minus its slowly changing background. The boxes follow camera movement automatically. A light partly hidden by the hand or foot still counts as on. A switch whose light stays on the whole video (e.g. IMG_0119, IMG_7554) has no light events.

Pressing the button turns the light off, so Touching is usually 0-3 frames (0-50 ms) before Turn off. An empty cell means unknown: the light was already on when the video started (no Start), was still on when it ended (no Turn off), no touch was found, or (a row with only the switch number) that switch had no light events. Times are video time in ms; at 60 fps they are accurate to one frame (~17 ms).

## Reaching Lab app (experiment sessions)
A window app for whole experiment sessions: double-click **`Lab App.command`** (Mac) or **`Lab App.bat`** (Windows), or run `venv/bin/python lab_app.py`. Six steps on the left:

1. **Session** - subject ID, day (d1 = familiarization, tests and practice; d2 = re-test day), the session video (one recording of the whole session; optional second camera + sync offset + calibration for 3D), and the design: tested arm, contralateral test yes/no, practice limb and side. The session plan is shown, e.g. Familiarization (2 blocks) → pre_test (3) → practice (10) → post_test (3) = 18 blocks. *Advanced*: trials per sequence (6), sequences per block (5), light window (1 s), pause that starts a new block (10 s), switch order (found automatically).
2. **Switches** - finds the 5 switches automatically in the first minutes with lights (or from a later time), numbered from the subject's right to left; flip, redraw, add or remove a box by dragging.
3. **Process** - one pass over each video (MediaPipe pose, the switch lights, hand/foot positions), saved so it runs once. On a Mac about 1.5 x the video's length (hardware video decoding + 3-4 MediaPipe workers). The video is cut into parts that run at the same time, one worker per 2 logical CPUs (`LAB_WORKERS=6` sets another number); in each worker the video is decoded beside MediaPipe. On Windows the graphics card can decode the video with `LAB_HW_DECODE=1` (off by default: it converts the colours slightly differently, which can change the measured light levels - compare it once on a real session first; `LAB_HW_DECODE=0` turns it off on a Mac). The decoder used is saved in the cache (`meta['decoder']`).
4. **Blocks & parts** - the groups of lights found (blocks = lights separated by the 30 s pauses) with the part each one belongs to, taken in order from the session plan; change a part or block number if needed (e.g. a block that was repeated). Warnings if the count doesn't match the plan, a block doesn't have 30 lights, or the wrist/ankle was poorly visible.
5. **Trials** - like the old dataViewer: every trial with its outcome and marks, RT and MT, a *touse* tick box, the speed (or x / y / z / path) graph with light on, movement onset/end and hit, and the video frame at light on / movement onset / movement end / trial end with the marker and the target switch.
6. **Export** - this session's two files, and one Excel file for all subjects.

**Trials.** Each light (in the 6-switch order, matched so a missed or extra light doesn't shift the rest) is a trial: start = light on; end = the button hit (the light goes off when the button is pressed) or, if not hit within the 1 s window, the light turning off by itself (**miss**). Trials are marked when the participant **started moving before the light** (the reach toward the target began at home, 30-500 ms before the light) and when the participant **first went to another switch and then corrected** (the first movement ended at another switch and the right one was hit afterwards; not a failure - marked so you can decide).

**Lights no camera sees** (e.g. an LED facing away): the next light always comes on ~1 s after a trial ends, so a gap with room for a trial is a hidden light - start = previous trial's end + 1 s, end = next light - 1 s (or the moment the hand/foot reaches the switch, at the end of a block); the switch = the one the hand/foot is on when it ends. These trials have *Light_source* = timing (video = the light was seen).

**Two cameras, one stops early** (or starts late): the session timeline covers both videos; where only one camera recorded, only that one is used. Each trial uses 3D when both cameras see the marker, otherwise the camera that sees it best (column *Data*).

**Measures** - the same method as the previous MATLAB system (`calculateMeasures.m`): the marker (wrist for the upper limb, ankle for the lower limb; 3D with two calibrated cameras, otherwise its position in the camera image) is resampled to 100 Hz, filtered (Butterworth 2nd order, 10 Hz, zero-phase) and its speed computed; movement onset = last sample before the first speed peak after the light with speed < 5% of the peak (reaction time = onset - light on; 0 if already moving), movement end = first sample after the peak below 5% (movement time = end - onset); also AT, DT, AT/DT, number of speed peaks, straight-line deviation, end-point distance. One-frame MediaPipe glitches are removed first.

**Outputs** (in `results/<subject>/<session>/`):
- `<subject>_<session>_trials.csv` - every trial synced with the design: Part, Block, Sequence, Trial, Target, Limb, Side, Light_on_ms, Touch_ms (hand/foot reaches the switch in the video), Hit_ms, Light_off_ms, Trial_end_ms, Outcome (hit / miss / not detected), Anticipation (+ ms before the light), Corrected (+ the switch it went to first).
- `<subject>_<session>_analysis.xlsx` - sheet *measures* (one row per trial: responsetime, movementtime, AT, DT, ..., touse) and the summary sheets of the old `makeCSVall.m`: mean / median / sd / iqr by part, by block and by sequence (familiarization and touse = 0 left out), plus *info*.
- `results/analysis_all_subjects.xlsx` (Export → all subjects) - every analysed session together, a row per subject.

`results/` holds participant data and is not uploaded to git.

## Two GoPro cameras (recommended setup)
**Camera settings** - the same on both cameras: GoPro HERO13 Black, **4K, 60 fps, lens Linear, HyperSmooth (stabilization) OFF**, on fixed mounts (don't move them during the day - then one calibration serves every recording of that day).

**The board** - `venv/bin/python calibrate_cameras.py make-board` writes `outputs/charuco_board.pdf` (exact A4 scale) and `charuco_board.png`: a ChArUco board of 5 x 7 squares, 38 mm, `DICT_4X4_50` markers. Print the PDF **at 100% / Actual size** (not "fit to page"), check the 100 mm bar and one square with a ruler (38 mm; note the measured size), and glue it to something flat and stiff.

**Each session:**
1. **Sync the timecode**: in the GoPro Quik app, *Sync Timecode* (both cameras scan the QR code). Then each camera can start / stop on its own - the timecode tells how the files line up.
2. **Record**: start both cameras; hold the board where both see it and slowly tilt / turn / move it around the subject's area for ~20-30 s; then run the experiment.
3. **App → Session**: choose each camera's video - the **first chapter** (`GX01xxxx.MP4`). GoPro splits long recordings into ~4 GB chapters (`GX010123.MP4`, `GX020123.MP4`, ...); the following chapters of the same clip are found automatically and used as one continuous video (the app shows them and warns if they don't follow each other).
4. **Offset**: filled in automatically from the start timecodes (*From timecode*), and checked against the sound (warning if they differ by more than one frame). Offset = seconds into video A at which video B started.
5. **Board calibration…**: source = the session videos (time window; default the first 60 s) or separate calibration clips; enter the measured square size. It shows the frames with the board in A / B / both, each lens' fit error, the stereo fit error, the distance between the cameras and a check (the board's squares measured from the 3D result vs the entered size, and the reprojection error) - **PASS** or warnings (e.g. fewer than 20 frames seen by both, stereo error > 1 px). The result is saved with the session and used as its calibration. *Use an earlier calibration…* re-uses the file of an earlier recording of the same day (cameras not moved).
6. **Switches → Process → Blocks & parts → Trials → Export** as before.

The timecode is read with `ffprobe` (FFmpeg) if installed - on a Mac `brew install ffmpeg` (Homebrew: https://brew.sh); on Windows from https://ffmpeg.org/download.html, added to PATH - otherwise straight from the MP4's timecode track (no FFmpeg needed). Timecodes HH:MM:SS:FF and drop-frame HH:MM:SS;FF are converted with the file's real frame rate (e.g. 59.94).

Command line: `venv/bin/python calibrate_cameras.py board GX010123.MP4 GX010456.MP4 [--offset S] [--start S] [--seconds 60] [--square-mm 38]` (offset from the timecodes if not given).

**Tests**: `venv/bin/python -m unittest discover -s tests -v` (timecodes incl. drop-frame and 59.94, the offset's sign, chapters, pairing of board frames with an offset, the printed board).

## Two cameras (3D)
Two cameras filming the same session from different angles give each joint's position in 3D, which fixes what one camera can't see (depth, a joint hidden behind the body, angles distorted by perspective). Three steps, on a Mac, from this folder:

**1. Sync the videos** (not needed for timecode-synced GoPros: the app takes the offset from the timecodes - see above):
```
venv/bin/python sync_videos.py media/IMG_0122.mov media/IMG_7557.mov
```
Finds the time offset from the sound and writes both videos cut to the common part, frame-by-frame in sync (to within half a frame), to `media/synced/` (`<video>_synced.mp4` + a .json with the offset). If the sound match is unclear, give the offset yourself: `--offset 1.838` (seconds into video A at which video B started). Already done for IMG_0122+IMG_7557 (offset 1.838 s) and IMG_0123+IMG_7558 (0.648 s); both offsets were checked against the switch lights seen by both cameras (agree within 1 ms).

**2. Calibrate the cameras** (where camera B is relative to camera A, and their lenses) - one of:
- **Printed board** (best; for a fixed setup - in the app: *Board calibration…*): print `outputs/charuco_board.pdf` on A4 at 100% (each square 38 mm - measure it), glue it to something flat. At the start of the session, with both cameras recording, hold it where both cameras see it and slowly tilt/turn/move it around the subject's area for ~20 s. Then
  `venv/bin/python calibrate_cameras.py board VIDEO_A VIDEO_B [--square-mm 38] [--offset S]` (frames of the two videos are paired by time with the offset) - gives real units (cm) and each lens' distortion. Cameras on fixed mounts: one calibration serves the whole session. GoPro: use the "Linear" lens setting. (`calibrate_cameras.py make-board` makes the board image again.)
- **Clicks** (existing videos without a board):
  `venv/bin/python calibrate_cameras.py clicks media/synced/IMG_0122_synced.mp4 media/synced/IMG_7557_synced.mp4`
  Both first frames open side by side; click the same point in A (left), then in B (right), at least 8 times (more is better; 20+ also corrects the lens focal length). Use the magnifier. Good points: the centres of the red switch buttons, corners of the white labels, table corners, and the subject in that first frame (knuckle, ring, elbow tip) - points that are not all on the table top. U = undo, ENTER = done. The clicks are saved (`--again` to redo). Add `--known 1 3 45.5` if you know the real distance between two clicked points (here points 1 and 3, 45.5 cm) to get cm.

**3. Analyze:**
```
venv/bin/python analyze_two_views.py media/synced/IMG_0122_synced.mp4 media/synced/IMG_7557_synced.mp4 --limb upper --side right
```
Outputs in `outputs/`: `<A>+<B>_<limb>_<side>_3d.csv` (per frame: each joint's 3D position, how well it fits both views in px, its position in each view; joint angles in 3D - elbow and shoulder, or knee, hip and ankle - and the same angles measured in 2D in each view, for comparison) `..._switch_events.csv` (the 3 switch events - Start, Touching, Turn off - per light event, like the one-camera file, but the lights are measured by both cameras and Touching needs the hand on the switch in both views, so a hand passing in front of a switch in one view doesn't count; times are ms of the synced videos) and `..._3d.mp4` (both views side by side: grey = MediaPipe in that view alone, coloured = the 3D result drawn back into each view). A joint is in 3D only on frames where both cameras see it; frames where the two views disagree (e.g. MediaPipe mixed up left and right in one view) are left empty. MediaPipe's results are cached per video (`*_mediapipe_cache.npz`), so re-running after a new calibration takes a few minutes.

Tip for the camera placement: MediaPipe works best when it sees the person roughly from the front or side, upright. A camera straight above (like IMG_012x) sees the arm well but guesses hips/legs under the table.

## Outputs (`outputs\`)
- `<video>_<limb>_<side>_landmarks.csv` - per frame: `<joint>_x_px`, `_y_px`, `_z_rel`, `_visibility` for the limb joints, plus `hand_<point>_x_px`, `_y_px`, `_z_rel` for the 21 hand points (upper limb). Coordinates are pixels in the full frame. Empty cells = not detected in that frame.
- `<video>_<limb>_<side>_skeleton.mp4` - the whole original video (full resolution) with the limb skeleton drawn on it.
- `<video>_<limb>_<side>_switch_events.csv` (with switch events) - one row per light event: `Switch`, `Start_ms`, `Touching_ms`, `Turn_off_ms` (video time in ms), sorted by switch.

A timestamp is added to a name if that file already exists, so nothing is overwritten. If you stop early, all files contain the frames processed so far.

From a terminal (uses the venv in the main folder):
```
..\venv\Scripts\python.exe extract_kinematics.py "path\to\video.mov" --limb upper|lower --side left|right [--switches] [--select-switches]
```
On a Mac (uses `venv` in this folder):
```
venv/bin/python extract_kinematics.py "path/to/video.mov" --limb upper|lower --side left|right [--switches] [--select-switches]
```

## Setup on macOS
The script uses MediaPipe's older `solutions` API, which only exists up to MediaPipe 0.10.21, and that version needs **Python 3.12 or older** (newer MediaPipe for Python 3.13+ no longer has it). To (re)create the environment in this folder:
```
python3 -m pip install --user uv
~/Library/Python/3.13/bin/uv python install 3.12
~/Library/Python/3.13/bin/uv venv --seed -p 3.12 venv
venv/bin/pip install -r requirements.txt
```
(Adjust `3.13` in the uv path to your `python3 --version`.) The first time you double-click `Run Kinematics.command`, macOS may ask to allow Terminal to run it; if it says the file can't be opened, right-click it → Open.
