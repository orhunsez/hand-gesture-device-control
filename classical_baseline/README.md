# Classical baseline: skin colour + contour (the old Gesture Mouse)

This folder holds the first version of the project: a classical image-processing
pipeline (skin colour in YCrCb, contour, finger counting) that controls the mouse.
It is **not** the main system any more. The main system uses MediaPipe landmarks
(see the README in the repository root). This code is kept as a possible
**classical baseline** to compare against the landmark-based approach under
different backgrounds and lighting conditions.

Limits to keep in mind: it only counts fingers, so it can separate gestures such as
fist and open hand but not thumb up/down or swipes, and it depends on skin colour,
so lighting and background change its result a lot.

## Run it (from the repository root)

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe classical_baseline\main.py
```

`pyautogui` (in `requirements.txt`) is needed because this code moves the real mouse.

Works on Windows, macOS and Linux with Python 3.8+ and OpenCV 4 or 5.

Keys: `q` quit, `p` pause. Push your real mouse into a screen corner for an emergency stop.

| Gesture | Action |
|---|---|
| 1 finger | move cursor |
| fist | click |
| 2 fingers | scroll (move hand up/down) |
| open hand | rest pose, does nothing |

A gesture must stay the same for `STABLE_FRAMES` frames before the mouse reacts,
so one wrong frame never clicks. The window shows both the steady gesture and
what the current frame looks like.

## One-time setup per operating system

**Windows:** nothing extra.

**macOS:** the first run asks for permissions. Allow both for the app you run Python from
(Terminal, iTerm or VS Code), then restart that app:
- System Settings → Privacy & Security → **Camera**
- System Settings → Privacy & Security → **Accessibility** (without this the mouse does not move)

**Linux:** the mouse can only be controlled in an **X11 / Xorg** session, not Wayland.
On the login screen pick "Ubuntu on Xorg" (or similar). You may also need
`sudo apt install python3-tk python3-dev`.

## If something goes wrong

| Message | Fix |
|---|---|
| Could not open any camera | Close other apps using the camera (Teams, Zoom, browser). On macOS allow camera access (above). |
| Wrong camera opens | Change `CAMERA_ID` in `config.py` to 1 or 2. |
| OpenCV cannot open windows | `pip uninstall -y opencv-python-headless opencv-python` then `pip install opencv-python` |
| Hand not found / face detected | Tune `SKIN_LOWER` / `SKIN_UPPER` in `config.py` while watching the "Skin mask" window. |

## Files (follow the pipeline top to bottom)
`config.py` settings → `preprocess.py` → `segmentation.py` → `features.py` → `gestures.py` → `controller.py` → `display.py` → `main.py`

## Old TODO list (from the first version)
- [ ] Tune skin range in `config.py` for your own lighting
- [ ] Experiment: HSV vs YCrCb, with/without histogram equalization (report section 5)
- [ ] Measure accuracy: finger-count correct % per gesture, in several lighting conditions
- [ ] Remove the face/arm from the mask (e.g. skip the top region or use background subtraction)
