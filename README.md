# Hand Gesture Device Control (CENG391)

Real-time hand gesture interface: a webcam finds the hand, MediaPipe gives 21 hand
landmarks, and the gestures (static poses and short swipes) are mapped to device
commands such as media and slide control. Term project for CENG391, 2026-27 Fall.

**Current state:** data collection works (landmark recording and viewing). The
recognition part (smoothing, segmentation, classifiers, calibration, false-trigger
prevention, command mapping) is not implemented yet. The gesture set and target
device are still a proposal. For the full methodology, decisions and open items, read
[`TEAM_BRIEF_methodology.md`](TEAM_BRIEF_methodology.md). The report outline is in
[`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md).

## Setup

Python 3.10 was used and tested; other versions were not tried. From the project folder:

```
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Always run the scripts with `.venv\Scripts\python.exe` (plain `python` may not work on
Windows). The hand model file `models/hand_landmarker.task` must exist. If it is
missing, download it into `models/` from
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`.

Note: MediaPipe 1.2.x has no `mp.solutions.hands`; only the Tasks API (`HandLandmarker`)
is available, so older MediaPipe tutorials will not run as written.

## Record and look at data

```
.venv\Scripts\python.exe main.py
```

A menu lets you record clips, list recorded clips, replay a clip, or print one frame as a
21x3 table. Recording keys: `SPACE` starts a clip (3 second countdown, then 2 seconds of
recording), `v` replays the last clip, `q` quits. Add `--show` to replay every clip
automatically after saving. After pulling updated scripts, close and reopen any running
recorder window (a running script keeps the old code).

Rules for recording:
- Use your own user id in the form `u01`, `u02`, ... and always the same one.
- Use the agreed gesture names; do not invent new spellings.
- Never commit raw video (`data/videos/`); faces are visible. It is in `.gitignore`.

## Data layout

```
data/landmarks/<user>/<gesture>.json      e.g. data/landmarks/u01/open_palm.json
```

One folder per user, one file per gesture, numbered clips inside. Each clip stores its
own background, lighting, distance and duration, and a list of frames
`{"t": seconds, "landmarks": [[x, y, z] x 21]}` (`null` when no hand was found). Values
are raw MediaPipe output: not smoothed, normalised or mirrored. Training and test sets
are split by person, never by frame.

## Files

| Path | What it is |
|---|---|
| `main.py` | Menu for recording and viewing |
| `record_landmarks.py` | Webcam to labelled landmark clips |
| `view_clip.py` | List, replay and print recorded clips |
| `models/` | MediaPipe hand model (pretrained by Google, not trained by us) |
| `data/` | Recorded landmark data |
| `classical_baseline/` | The first version (skin colour + contour mouse control), kept as a possible classical baseline. Has its own README. |
| `docs/` | Report outline |
| `TEAM_BRIEF_methodology.md` | Methodology, data format, decisions and open items |

## Team split (from Preferred Work)
- Murathan: report sections 1, 2, 7
- Abidin and Orhun: code, report sections 4, 5, 6
- Berzan: data collection and test videos, report sections 2, 3, 4
