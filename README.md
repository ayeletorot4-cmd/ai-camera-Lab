# MediaPipe Limb Tracker ("only visuals" copy)

Tracks **one limb** in a lab video with MediaPipe Holistic. The whole frame is analyzed, but only the chosen limb is saved and drawn. Optionally also detects **switch events** (light on / touch / light off) for the 5 switches on the table.

## Quick start
1. Double-click **`Run Kinematics.bat`** and pick the video (the dialog opens in the main "AI camera for the lab" folder).
2. Type **U** (upper limb) or **L** (lower limb).
3. Type **L** or **R** for the subject's own **left** or **right** side.
4. Type **Y** to detect switch events, or press ENTER to skip (see below).
5. Watch the preview. Press `q` or close the window to stop early.

| Limb | Points tracked |
|---|---|
| upper | shoulder, elbow, wrist + 21 hand/finger points (red cross = wrist from the hand model) |
| lower | hip, knee, ankle, heel, toe |

Colours: subject's left side = orange, right side = cyan.

## Switch events
The first time for each video, the first frame opens and you drag a box around each switch **in order 1 to 5, from the subject's RIGHT to LEFT**, pressing ENTER after each (C cancels). Include the switch's light in the box. The boxes are saved in `outputs\switch_settings.json` and offered again next time for the same video name (type R in the launcher to mark them again). Boxes for IMG_0119-0123 and IMG_7554-7558 are already saved.

For each switch:
- **Start** - the light turns on.
- **Touching** - the first frame a hand/finger point (upper limb: either hand) or foot point (lower limb: either foot) is inside the box or within 25% of the box size from it, from light start until 1 s after the light turns off. (The button sticks out on the subject's side, so fingertips pressing it sit at the edge of the box.)
- **Turn off** - the light turns off.

How the light is measured: the pink LED light and its glow inside the box enlarged by 25% (the deep-red button, white table and skin are left out), minus its slowly changing background. The boxes follow camera movement automatically. A light partly hidden by the hand or foot still counts as on. A switch whose light stays on the whole video (e.g. IMG_0119, IMG_7554) has no light events.

Pressing the button turns the light off, so Touching is usually 0-3 frames (0-50 ms) before Turn off. An empty cell means unknown: the light was already on when the video started (no Start), was still on when it ended (no Turn off), no touch was found, or (a row with only the switch number) that switch had no light events. Times are video time in ms; at 60 fps they are accurate to one frame (~17 ms).

## Outputs (`outputs\`)
- `<video>_<limb>_<side>_landmarks.csv` - per frame: `<joint>_x_px`, `_y_px`, `_z_rel`, `_visibility` for the limb joints, plus `hand_<point>_x_px`, `_y_px`, `_z_rel` for the 21 hand points (upper limb). Coordinates are pixels in the full frame. Empty cells = not detected in that frame.
- `<video>_<limb>_<side>_skeleton.mp4` - the whole original video (full resolution) with the limb skeleton drawn on it.
- `<video>_<limb>_<side>_switch_events.csv` (with switch events) - one row per light event: `Switch`, `Start_ms`, `Touching_ms`, `Turn_off_ms` (video time in ms), sorted by switch.

A timestamp is added to a name if that file already exists, so nothing is overwritten. If you stop early, all files contain the frames processed so far.

From a terminal (uses the venv in the main folder):
```
..\venv\Scripts\python.exe extract_kinematics.py "path\to\video.mov" --limb upper|lower --side left|right [--switches] [--select-switches]
```
