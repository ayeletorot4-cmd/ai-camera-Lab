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

## Two cameras (3D)
Two cameras filming the same session from different angles give each joint's position in 3D, which fixes what one camera can't see (depth, a joint hidden behind the body, angles distorted by perspective). Three steps, on a Mac, from this folder:

**1. Sync the videos** (not needed for cameras that record a shared timecode, e.g. synced GoPros):
```
venv/bin/python sync_videos.py media/IMG_0122.mov media/IMG_7557.mov
```
Finds the time offset from the sound and writes both videos cut to the common part, frame-by-frame in sync (to within half a frame), to `media/synced/` (`<video>_synced.mp4` + a .json with the offset). If the sound match is unclear, give the offset yourself: `--offset 1.838` (seconds into video A at which video B started). Already done for IMG_0122+IMG_7557 (offset 1.838 s) and IMG_0123+IMG_7558 (0.648 s); both offsets were checked against the switch lights seen by both cameras (agree within 1 ms).

**2. Calibrate the cameras** (where camera B is relative to camera A, and their lenses) - one of:
- **Printed board** (best; for a fixed setup): print `outputs/charuco_board.png` on A4 at 100% (each square 38 mm - measure it), glue it to something flat. At the start of the session, with both cameras recording, hold it where both cameras see it and slowly tilt/turn/move it around the subject's area for ~20 s. Then
  `venv/bin/python calibrate_cameras.py board VIDEO_A VIDEO_B [--square-mm 38]` - gives real units (cm) and each lens' distortion. Cameras on fixed mounts: one calibration serves the whole session. GoPro: use the "Linear" lens setting. (`calibrate_cameras.py make-board` makes the board image again.)
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
