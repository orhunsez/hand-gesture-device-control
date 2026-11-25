# CENG391 Gesture Interface: Methodology Brief and Team Requests

**Purpose.** This document (1) explains the proposed methodology of our real-time gesture interface, and (2) lists exactly what the Problem Definition, Dataset and Related Work teams need to deliver so that all sections of the Progress Report (due 20 November 2026) stay consistent and can be written in parallel.

**Status of assumptions.** Items marked **[ASSUMPTION]** have not yet been confirmed by the whole team (target device, gesture set). Everything else is the current proposal. Literature figures quoted below are from memory and **must be verified against the original papers** before they appear in the report.

---

## Part A. Methodology

### A.1 Problem in one paragraph

The system receives a continuous stream of RGB frames from a single ordinary webcam and must decide, in real time, whether the user is performing an intentional hand gesture and, if so, which device command it corresponds to. The difficulty is not only recognising a hand shape. The system must also (i) work for different users, backgrounds, lighting conditions and camera distances, (ii) find where a gesture starts and ends inside a never-ending stream (temporal segmentation), and (iii) avoid firing commands for incidental movements such as scratching the face or reaching for the keyboard (false-trigger prevention, the "Midas touch" problem).

### A.2 Gesture set (decided) and target device (not decided)

- Commands are **not defined for now**: the gestures are only recognised, nothing is sent to a device yet. The command mapping (4.10) and the target device are decided later. An earlier idea was a PC controlled for media playback and slide presentation with `pyautogui`; it is not confirmed.
- Static gestures (identity is the hand shape in a single frame, hand held still): `open_palm`, `fist`, `thumb_up`, `thumb_down`, `point` (index finger extended).
- Dynamic gestures (identity is the motion over time): `swipe_right` and `swipe_left`, both performed with the **open palm** as an **out-and-back movement**: the hand goes to the user's right (or left) and comes back, and the direction of the first stroke gives the label. No swipe up.
- One **`no_gesture`** class (negative examples: still poses that are not a gesture and movements that are not a swipe). There is **no activation gesture** (decided, D.4 item 10).
- The exact label names live in `data_tools/dataset_config.py`.

### A.3 Design principle

We do **not** classify raw pixels. A pretrained hand model (MediaPipe Hands) converts each frame into 21 hand landmarks, and all further processing works on these coordinates. Colour-based segmentation (skin colour + contour, used in our older code) depends on pixel intensity and therefore breaks when lighting, background or skin tone changes. Landmarks describe hand *geometry*, which is much less sensitive to these factors. This is the central reason for our design.

### A.4 Pipeline overview

Calibration runs once at start-up and feeds parameters into later stages. At run time every frame passes through the following chain:

```
camera frame
   -> hand detection + 21 landmarks        (4.2, MediaPipe, pretrained)
   -> landmark smoothing (1-euro filter)   (4.4)
   -> normalisation (translation + scale)  (4.3)
        |-> STATIC branch:  per-frame classifier -> hold (stability) check
        |-> DYNAMIC branch: speed-based segmentation -> window classifier
   -> false-trigger prevention             (4.9)
   -> command mapping + device control     (4.10)

User calibration (4.8, once at start-up) -> sets scale, segmentation thresholds, confidence threshold
```

Note on order: in the report the normalisation section (4.3) precedes smoothing (4.4), but in the actual pipeline smoothing is applied first, on raw landmarks, and normalisation second. The system-overview figure (4.1) and one sentence in the text must state this explicitly.

Each stage can be switched off, which allows ablation experiments (Section A.12). In addition, latency should be measured after every stage (experiment E9).

### A.5 Stage 4.2: Hand detection and landmark extraction

MediaPipe Hands works in two steps: a palm detector locates the hand in the full frame, and a landmark model then estimates 21 keypoints (wrist, finger joints, fingertips) on the cropped hand. For each landmark it outputs `(x, y, z)`: `x` and `y` are normalised image coordinates, `z` is a relative depth. Settings to document: maximum number of hands (1), minimum detection confidence, minimum tracking confidence. If no hand is detected in a frame, that frame is treated as "no gesture". The model is pretrained and is **not** our contribution. The report must say so openly and justify the choice (A.3, and the course asks for 21-landmark output).

### A.6 Stage 4.3: Landmark normalisation

Raw coordinates depend on where the hand is in the image and how far it is from the camera. Two steps remove these effects:

- *Translation invariance:* subtract the wrist (landmark 0): `q_i = p_i - p_0`.
- *Scale invariance:* divide by a hand-size measure, the wrist-to-middle-finger-base distance `s = || p_9 - p_0 ||`, giving `q^_i = (p_i - p_0) / s`.

The resulting vector (21 x 3 numbers, optionally plus joint angles and fingertip distances) is the feature vector for classification. A fist then looks numerically similar anywhere in the frame and at any distance.

Two details are part of the implementation (`processing/landmark_features.py`). First, MediaPipe divides x by the image width and y by the image height, so on a 4:3 camera equal steps in x and y are different lengths; x, y and z are therefore multiplied by the image width, height and width before any distance is computed, so that the hand size `s` and all distances are measured equally in both directions. Second, rotation of the hand in the image plane is deliberately not removed, because thumb up and thumb down differ by orientation. Frames without a hand stay NaN, and a degenerate hand (size almost zero) also gives NaN instead of a division by zero.

### A.7 Stage 4.4: Temporal smoothing

Even a perfectly still hand makes landmarks jitter from frame to frame. Smoothing has a cost: the stronger the filter, the larger the lag. We use two levels:

1. **Landmark level: 1-euro filter** (Casiez et al., 2012), an adaptive low-pass filter. For each coordinate: estimate the speed `dx^`, set the cut-off `f_c = f_cmin + beta * |dx^|`, then `alpha = 1 / (1 + tau / T_e)` with `tau = 1 / (2 * pi * f_c)` and `T_e` the sampling period. When the hand is slow the cut-off is low (strong smoothing, little jitter); when it moves fast the cut-off rises (little lag). Parameters: `f_cmin`, `beta`, `d_cutoff`. Tuning procedure: first set `beta = 0` and tune `f_cmin` for minimal jitter, then raise `beta` until lag becomes acceptable.
2. **Class level:** majority voting or exponential moving average over the last few predicted labels.

Measurement (experiment E2): jitter (standard deviation of a landmark on a still hand) and added latency, with and without filtering.

Implementation (`processing/landmark_smoothing.py`, evaluated by `evaluation/evaluate_smoothing.py`): the filter is causal, works in pixel units on the raw landmarks, and a frame without a hand resets it. The metrics are the landmark jitter and the palm-centre speed noise on still-hand clips, and the added delay and the flattening of the movement on swipe clips. **Provisional result on the pilot clips of one person (user `u99`, near distance, plain background, bright light; 18 still clips and 12 swipe clips; to be repeated on the training users):** with `f_cmin` = 0.5 Hz and `beta` = 0.02 the jitter fell by 84 % and the speed noise (95th percentile) by 68 %, at the price of 27 ms of added delay (about half a frame at 20 frames per second) and a swipe flattened to 97 % of its amplitude. Larger `beta` lowers the delay but also the noise reduction; a larger `f_cmin` does the same. Be careful when quoting these numbers: the unfiltered jitter is already very small (0.0029 hand sizes, well under one pixel at this distance) and the unfiltered speed noise (0.19 hand sizes per second) is small next to a swipe (about 4 hand sizes per second), so at this distance the gain is real but modest. The effect is expected to be larger with a smaller hand (far distance, E8), which has not been recorded yet.

### A.8 Stage 4.5: Gesture segmentation

A classifier always outputs *some* label, so the system must decide which part of the stream is a gesture candidate. Static and dynamic gestures need different logic:

- **Static gestures: stability-based.** The same class must be predicted for several consecutive frames while the hand is nearly still (a *hold* condition). Transition frames between two poses produce unstable output and are rejected naturally.
- **Dynamic gestures: motion-energy-based.** The speed of the **palm centre** `c_t` (mean of landmarks 0, 5, 9, 13, 17), `v_t = || c_t - c_(t-1) || / dt`, normalised by hand size, `v^_t = v_t / s`, is compared with an onset threshold `tau_on` (gesture starts) and an offset threshold `tau_off` (gesture ends). The palm centre is used instead of the wrist because, in the pilot recording, the wrist moved only about half as far as the palm centre (the hand pivots at the wrist). A swipe is an **out-and-back cycle** (see A.12), so the candidate window runs from the onset of the outward stroke until the hand has been still again for a short time (speed below `tau_off` for about 0.2 s). The window therefore contains both strokes, about 10 to 20 frames (0.5 to 1 s) in the pilot, and is resampled to a fixed length before it goes to the dynamic classifier.

Three mechanisms make this robust: **hysteresis** (`tau_on > tau_off`, so speed fluctuating around one threshold does not start and stop the gesture repeatedly), a **minimum duration** (very short movements are discarded as tremor), and a **ring buffer** of the last N frames.

**Boundary with 4.9:** Segmentation answers "does this interval form a gesture candidate?". Stage 4.9 answers "should this candidate become a command?". Do not mix the two in the text.

### A.9 Stages 4.6 and 4.7: Classification (the only part where we train ML)

- **Static classifier (4.6):** input is the normalised landmark vector of one frame; output is one of {open palm, fist, thumb up, thumb down, point, no gesture}. Two models are compared (experiment E1): a **rule-based baseline** (for example, finger extended or not, computed from landmark distances and angles) and a **learned model** (SVM or a small MLP).
- **Dynamic classifier (4.7):** input is the landmark sequence of the segmented window, resampled to a fixed length; output is {swipe right, swipe left, (swipe up), no gesture}. Again a rule-based baseline (direction and distance of wrist displacement) is compared with a small learned model (SVM on resampled features; 1D-CNN or LSTM only as an optional extension if enough data exist).

Rationale for small models: the input is a low-dimensional landmark vector, not an image, and our dataset will be small. Large deep models are unlikely to pay off and add risk.

**Pilot finding for 4.6: the first classifier used the orientation of the hand (`recognition/train_static.py`, `docs/results/static_classifier_pilot_2026-11-21.txt`).** An SVM trained on one person's four gestures (fist, open palm, thumb up, thumb down) reached 100 % in clip-level cross-validation, which only shows that the chain works (same person, and the four poses are far apart). Applied to the open-palm frames of an earlier recording session (user id `u01`; both ids `u01` and `u99` are the hand of the same person, so this is a test across sessions, not across people) it labelled 78 % correctly; the wrong frames came exactly from the clips in which the hand was tilted by 61 to 68 degrees, and all of them were called thumb up. In the training data only the thumb-up clips were tilted (mean tilt 57 degrees, the other gestures 7 to 8), so the tilt was tied to the class. When those frames were rotated in the picture plane to the tilt of the training open palm, 100 % were correct. Limits: six clips of one hand, and only rotation in the picture plane was examined. Nothing here says how the system treats other people. Three consequences for the design: (1) every gesture must be recorded with different hand orientations; (2) the features should describe the finger configuration with quantities that do not depend on rotation (distances between landmarks, joint angles) and keep the orientation only where it carries meaning (thumb up versus thumb down); (3) as an alternative or addition, the training frames of the gestures for which orientation does not matter can be rotated randomly. Rotation was deliberately not removed in 4.3, which is right for thumb up and thumb down, but the feature set has to take it into account.

### A.10 Stage 4.9: False-trigger prevention

Two behaviours are decided (D.4 items 10 and 11). (1) **No activation gesture:** a command starts as soon as its gesture is recognised, like a phone's volume key, which acts the moment it is pressed; no clutch gesture has to open a window first. (2) **Holding repeats:** while a static gesture is held, its command keeps being applied step by step, like holding the volume key, and it stops when the gesture is released. A swipe cannot be held, so it gives exactly one command per out-and-back cycle.

A recognised gesture becomes a command only if all of the following hold: the classifier confidence exceeds a threshold; the stability check (static, 4.5) or the minimum duration (swipe) is satisfied; and the sample is not classified as **no gesture**. After a swipe a **cooldown** (refractory period) blocks further commands. For a held static gesture the repeat timing (a repeat every 0.2 s, the same for every gesture, D.4 item 13) takes the place of the cooldown, and the gesture has to be released before it can start a new command. Because there is no clutch, these safeguards are the only protection against accidental triggers, so experiment E4 (false-trigger rate with and without them) carries more weight.

### A.11 Stage 4.8 and 4.10: Calibration and command mapping

- **Calibration (4.8):** at start-up the user holds each static gesture for about 2 seconds and makes one swipe in each direction (about 20 seconds of recording; the program says which gesture to show, so the label never comes from the user's choice). From these recordings the system sets, always within fixed limits around the population values: the stability thresholds of the static hold check (from the palm speed and pose change of the user's still hand), the finger thresholds of the rule-based classifier, the confidence threshold of 4.9 (only lowered, never below a floor), and suggested segmentation thresholds `tau_on` / `tau_off` (from the user's swipe speed). The learned classifier can also be retrained with the user's calibration frames added (a variant in E3). Each calibration gesture is first checked with the uncalibrated rules; if it looks like a different gesture, the step is repeated, because a wrong pose would spoil the calibration. Calibration does **not** set the scale `s`: normalisation already divides every frame by its own hand size, and a fixed calibrated `s` would be wrong as soon as the hand moves closer or further away; the measured hand size only produces a warning when it is outside the range the validator accepts. There is no separate neutral-pose step; the noise of a still hand is measured on the held gestures. Effect measured in E3. (Decided 23 November 2026, D.4 item 14; code in `recognition/calibration.py`.)
- **Command mapping (4.10):** a table maps each confirmed gesture to a keyboard or media command sent with `pyautogui`.

### A.12 Open palm used for two gestures (decided: separate by motion, to be validated)

Open palm is both a static gesture (hand still) and the hand shape used for swiping (hand moving). The hand shape is identical; only motion differs. The decision is to keep one shape and separate the two by motion with three rules:

1. **Speed gate:** the static branch is suppressed while the normalised wrist speed is above the segmentation threshold (`tau_on`, 4.5); the dynamic branch only starts above it.
2. **Minimum hold time (open palm only):** the static open palm counts only after the hand has been stable for about half a second (about 10 frames at 20 frames per second). A swipe is shorter than this. The other static gestures are not the swipe hand shape and start as soon as they are recognised (D.4 items 10 and 12).
3. **Cooldown after a swipe:** static results are blocked for about one second after a swipe cycle, because the palm stops at the end of a swipe and would otherwise be recognised as a static open palm.

**Swipe definition (decided after the pilot recording).** In all 6 pilot clips (`swipe_left`, one person) the hand came back by itself after the outward stroke, taking 8 to 21 frames. A one-way swipe would therefore make the return stroke look like a swipe in the opposite direction. Instead of suppressing the return, the gesture is defined as the whole **out-and-back cycle**, and the direction of the first stroke gives the label:

- `swipe_left`: the hand goes to the user's left, then comes back. The saved data is not mirrored, so the palm-centre x first increases and then decreases (6 of 6 pilot clips).
- `swipe_right`: the hand goes to the user's right, then comes back (x first decreases, then increases; not yet recorded).
- The loader must express the movement **relative to its starting position** (subtract the position at onset), so that the same swipe started from different places on the screen looks the same.
- Cost: the command can only be issued after the return stroke, about one second after the movement starts. Measure it in E9.
- Between two swipes in a row the user must pause briefly so the cycles can be separated.

**Recording a swipe:** wait about half a second after the countdown ends (3 of 6 pilot clips already moved in the first frame, leaving no still lead-in for the onset threshold), make the out-and-back movement once at a steady speed, then keep the hand still. Swipe and `no_gesture` clips are recorded for 3 seconds by default (`DURATIONS` in `data_tools/dataset_config.py`), to leave room for slower users and a longer lead-in (in the pilot one cycle took 10 to 20 frames, see D.3).

Data consequence: record static `open_palm` clips with the hand still and swipe clips with the hand moving, and add hard negatives to `no_gesture` (very slow palm movement, stopping the palm after a swipe). Risk: a slow swipe below the speed threshold can look like a static palm; the thresholds are tuned per user by calibration (4.8) and measured in E4. Fallback if this proves too unreliable: move the static open palm to another hand shape; the data of the other gestures is not affected.

### A.13 Experiments linked to the methodology (from the report outline)

| ID | Question | Section |
|---|---|---|
| E1 | Rule-based baseline vs learned classifier | 5.3 |
| E2 | Effect of temporal smoothing (jitter, latency) | 5.4 |
| E3 | Effect of user calibration | 5.5 |
| E4 | Effect of false-trigger prevention | 5.6 |
| E5 | Cross-user evaluation (unseen users) | 5.7 |
| E6 | Effect of background | 5.8 |
| E7 | Effect of lighting | 5.9 |
| E8 | Effect of camera distance | 5.10 |
| E9 | Runtime: FPS and per-stage / end-to-end latency | 5.11 |

Optional extra: run our older skin-colour + contour code as a *classical baseline* in E6 and E7 to show empirically why landmark-based processing is more robust.

---

## Part B. Requests to the other teams

### B.1 Rules for everyone

- Use the **same terminology** as Part A (landmark, normalisation, segmentation, hold, cooldown, "no gesture" class). Do not rename concepts.
- Progress Report length: Section 3 at most about 1.5 pages; Related Work about 1 to 1.5 pages. Final Report will expand both.
- Verify every citation against the original paper and add it to `literature.bib`.
- Keep a note of every generative-AI tool used and for what; the **GenAI declaration is mandatory** in the Progress Report.
- Do not promise any feature that is not in Part A.
- Drafts are due to the methodology team by **[DATE TO BE FILLED IN]** so that integration is possible before 20 November 2026.

### B.2 Problem Definition team (Sections 3.1 and 3.2)

**Why we need this:** every methodology decision (number of classes, which gesture is dynamic, what segmentation must separate) depends on the exact task and gesture list.

**3.1 Task definition** must contain:
- Input: continuous RGB stream from a single webcam. Output: a device command or "no command".
- The three sub-tasks: static classification, dynamic classification, temporal segmentation.
- The difficulties we will evaluate: lighting, background, camera distance, user differences, intentional vs incidental movement. (These are also the course's evaluation criteria.)
- System constraints: real-time operation, ordinary webcam, no extra hardware.

**3.2 Gesture set and command mapping** must contain:
- One table: gesture name, type (static or dynamic), triggered command.
- The "no gesture" class and, if adopted, the activation gesture, listed separately.
- A one-sentence justification per gesture (visually distinct, easy to perform, not confusable).
- The resolution of the open-palm issue (A.12) once decided.
- The target device in one sentence.

### B.3 Dataset team (Sections 3.3 to 3.6; 3.7 and 3.8 can wait for the Final Report)

**Why we need this:** the course asks for evaluation across users, backgrounds, lighting and camera distance. This is only possible if every sample records the condition under which it was captured. Public datasets (HaGRID, Jester, IPN Hand) are not recorded under controlled, labelled conditions, so our **main data will be self-collected**; public data may supplement static gestures at most.

Requirements:
- **Stored format:** MediaPipe landmark sequences (21 x 3 per frame) rather than raw video; frame order must be kept for dynamic gestures.
- **Metadata for every sample:** user ID, gesture label, background, lighting, camera distance.
- **Conditions:** several users, at least two backgrounds, two lighting conditions and two distances. The team states the real numbers it can achieve; the report must match them exactly.
- **Negative data:** record realistic non-gesture behaviour (talking with the hands, reaching for the keyboard, adjusting hair, lowering the hand). Without it the false-trigger experiment (E4) is meaningless.
- **Split:** training, validation and test sets are separated **by person**; test users are never seen during training.
- **Ethics:** obtain participant consent and do not share raw images.
- **For the Progress Report:** if collection is not finished, write the section as a *planned protocol*; real statistics are added in the Final Report.
- **Please report back** which conditions (number of users, backgrounds) you can really cover, so the experiment plan can be fixed.

### B.4 Related Work team (Section 2)

**Why we need this:** each subsection must justify one methodological choice. Write it as an argument, not a list of papers.

| Subsection | Must justify | Topics to cover |
|---|---|---|
| 2.1 Hand detection and landmarks | Why MediaPipe | Classical methods (skin colour, contour); MediaPipe Hands two-stage design |
| 2.2 Datasets and benchmarks | Why our own data | HaGRID, Jester, IPN Hand: size, classes, static/dynamic, fit for device control |
| 2.3 Static and dynamic recognition | Why landmark-based, small models | Landmark-based vs video-based methods; classical vs learned classifiers (e.g. Molchanov et al., 2016) |
| 2.4 Continuous recognition and false triggers | Why segmentation, hysteresis, cooldown | Gesture spotting (e.g. Alon et al., 2009); the Midas touch problem; how false positives and latency are measured |
| 2.5 Temporal smoothing | Why the 1-euro filter | Jitter vs lag trade-off (Casiez et al., 2012) |
| 2.6 Positioning | What is different in our work | Short comparison table plus one sentence |

Suggested positioning sentence: existing work mostly reports accuracy under a single condition, while we evaluate robustness across users, backgrounds, lighting and distance, and measure false-trigger rate and latency explicitly.

### B.5 Introduction (written last by the methodology team)

Only 1.1 Motivation, 1.2 Problem statement and 1.4 Proposed approach are needed for the Progress Report, about half to one page. It is written after the other sections exist.

---

## Part C. Open items

The full, current list of open items is in **D.5**. The short version: the gesture set and target device are still unconfirmed (A.12 open-palm conflict), the label lists, validator and data loader are not written yet, and the classical baseline is undecided.

---

## Part D. Project status (as of 21 November 2026)

This part is the "catch-up" section: anyone joining or returning to the project should be able to read it and know what exists, how to run it, what was decided and what is still open.

### D.1 What exists in the repository

| Path | What it is |
|---|---|
| `main.py` | Menu: record clips, list clips, replay a clip, print a frame as a 21x3 table |
| `data_tools/dataset_config.py` | Shared label lists (gestures with their type, backgrounds, lightings, distances); the recorder refuses any value that is not in it |
| `data_tools/record_landmarks.py` | Webcam to MediaPipe to labelled landmark clips (see D.3) |
| `data_tools/view_clip.py` | Lists, replays and prints recorded clips (also reachable from `main.py`) |
| `processing/landmark_features.py` | Landmark normalisation (4.3): wrist as origin, hand size as unit, aspect-ratio correction; `normalize`, `hand_size`, `flatten`. Used for training data and the live system alike |
| `processing/landmark_smoothing.py` | 1-euro filter (4.4): `OneEuroFilter`, `LandmarkSmoother` (one frame at a time, for the live system) and `smooth_clip` (a recorded clip). Applied to raw landmarks before normalisation |
| `evaluation/evaluate_smoothing.py` | Measures jitter, speed noise, delay and amplitude loss with and without the filter and tunes `f_cmin` and `beta` (experiment E2) |
| `data_tools/dataset_loader.py` | Turns the clips of static gestures into the training table: skips clips with a validator ERROR, smooths, normalises, keeps only stable frames (hand present, palm speed and pose change low in a 5-frame window), one row of 63 numbers per frame with label, user, clip, frame and condition; `split` divides by user, `leave_one_user_out` gives cross-validation folds. Swipes are not handled yet (they need the segmentation) |
| `tests/test_dataset_loader.py` | 25 checks of the loader on made-up data |
| `recognition/train_static.py` | Static classifier (4.6): SVM on the 63 normalised numbers; `evaluate_clip_cv` (clips never split), `evaluate_leave_one_user_out` (experiment E5, needs several users). The pilot run is only a check of the chain |
| `data_tools/validate_dataset.py` | Checks all recorded clips before training: labels and names, numbering, timestamps, frames with a hand, hand size, stillness of static gestures, swipe direction and size, duplicate clips; prints dataset statistics. ERROR clips must not be used for training (`bad_clips()` lists them for the loader) |
| `tests/test_validate_dataset.py` | 17 checks of the validator on made-up good and bad data |
| `tests/test_landmark_smoothing.py` | 17 checks of the filter. Run with `.venv\Scripts\python.exe tests/test_landmark_smoothing.py` |
| `tests/test_landmark_features.py` | 15 checks of the normalisation (translation and scale invariance, rotation kept, NaN handling). Run with `.venv\Scripts\python.exe tests/test_landmark_features.py` |
| `models/hand_landmarker.task` | MediaPipe hand model bundle (palm detector + 21-landmark model), pretrained by Google, 7.8 MB |
| `data/landmarks/<user>/<gesture>.json` | Recorded data, one folder per user, one file per gesture |
| `classical_baseline/` | The old skin-colour + contour code (mouse control). Kept as a possible classical baseline, see D.5 |
| `docs/REPORT_OUTLINE.md` | Report section outline |
| `docs/results/` | Measured pilot results with the data used, the way to reproduce them and the caveats (smoothing, validation, loader, static classifier) |
| `report/` | LaTeX template for the Progress and Final Report, one file per section so that several people can write at once. Compiled on Overleaf without errors (reported by the group); LaTeX is not installed on the development machine; see `report/README.md` |
| `requirements.txt` | `opencv-python`, `numpy`, `pyautogui`, `mediapipe` |

Not in the repository: `.venv/` (each person creates their own) and `data/videos/` (raw video shows faces and must never be committed; both are in `.gitignore`).

### D.2 How to run

Python 3.10 was used and tested. Other versions were not tried. From the project folder:

1. `py -3.10 -m venv .venv`
2. `.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. Make sure `models\hand_landmarker.task` exists. If it is missing, download it from `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` into `models\`.
4. `.venv\Scripts\python.exe main.py`

Notes: do not use plain `python` (the Windows Store stub is broken on the main development machine); always use `.venv\Scripts\python.exe`. The installed MediaPipe version (1.2.3) has no `mp.solutions.hands`; it only has the Tasks API (`HandLandmarker`), so older MediaPipe tutorials will not run as written. After pulling updated scripts, close and restart any open recorder window, because a running script keeps the old code in memory.

Recording keys: `SPACE` starts a clip (3 second countdown, then 2 seconds of recording), `v` replays the last clip, `q` quits. `--show` replays every clip automatically after saving. `--save-video` also saves the raw video (privacy: never commit it).

### D.3 Data format (for the Dataset team)

- Path: `data/landmarks/<user>/<gesture>.json`. The user id must look like `u01`, `u02`, ... (the recorder rejects anything else). One person always records under one id, in one folder.
- The file holds numbered clips. Each clip stores its own `background`, `lighting`, `distance` and `duration`, and a list of frames.
- Each frame is `{"t": seconds, "landmarks": [[x, y, z] x 21]}`; `"landmarks"` is `null` when MediaPipe found no hand in that frame.
- Values are raw MediaPipe output: not smoothed, not normalised, not mirrored. Normalisation and frame selection happen later, in the loader (D.5), so those choices can change without re-recording.
- There is no separate index file. Overview tables are produced from the JSON files themselves (the list option in `main.py`). An `index.csv` was tried and removed because it would cause merge conflicts and duplicates information.
- Measured so far: a 2 second clip has about 40 frames, so the recording runs at about 20 frames per second on the development machine, and a clip is about 26 KB. Current data: `u01/open_palm.json` with 6 clips (5 near, 1 far; plain background, bright light).
- Each clip recorded from now on also stores `image_width` and `image_height` (the camera picture size, 640 x 480 on the development camera). Older clips do not have them; `processing/landmark_features.py` then assumes 640 x 480.
- Correction to the pilot numbers below: they were computed on normalised coordinates without the aspect-ratio correction. With the correction the swipe amplitude at the palm centre is 0.88 to 1.67 hand sizes (mean 1.20) for `swipe_left` and 0.34 to 1.19 (mean 0.75) for `swipe_right`, instead of 0.66 to 1.26 and 0.26 to 0.91; `swipe_right` clips 4, 5 and 6 (0.46, 0.45, 0.34) are still weak, and clip 5 still goes the wrong way. Static jitter hardly changed (medians 0.0029 fist, 0.0019 open palm, 0.0039 thumb up). The directions and the comparison between left and right do not change.
- Clip length is set per gesture in `data_tools/dataset_config.py` (`DURATIONS`): 2 seconds for static gestures, 3 seconds for swipes and `no_gesture`. `--duration` overrides it.
- Pilot recording (one person, user id `u99`, 6 clips of `swipe_left`, plain background, bright light, near; to be deleted before the real collection): the hand was found in every frame; the outward stroke took 9 to 22 frames (0.4 to 1.05 s) and the return 8 to 21 frames; the travel of the outward stroke was on average 0.45 hand sizes at the wrist, 0.93 at the palm centre and 1.44 at the index fingertip. Caveats: one person, six clips, and the start of the stroke is estimated as the lowest x before the peak, which overstates the duration of slow drifts. The static `u01/open_palm.json` clips were not perfectly still (the wrist x range reached 0.44 hand sizes and the y range 1.5), so recording instructions for static gestures must say "hold the hand completely still" and the loader must select stable frames.
- Second pilot analysis (same person, 6 clips each of `swipe_left` and `swipe_right`, recorded for 2 seconds, motion measured at the palm centre in hand sizes, with onset and settle defined as leaving and reaching the start and end positions by more than 0.15 hand sizes): the hand was found in every frame. The first stroke went in the expected direction in 6 of 6 `swipe_left` clips (x increases) and in 5 of 6 `swipe_right` clips (x decreases); the exception (`swipe_right` clip 5) has a first stroke in the opposite direction and no clear cycle, so it is probably a mis-performed clip and is the kind of label error the validator and the later cross-check must catch. The active movement lasted 10 to 15 frames (0.46 to 0.70 s) for `swipe_left` and 16 to 20 frames (0.75 to 0.96 s) for `swipe_right`, after a still lead-in of 8 to 17 frames, and all complete cycles ended by frame 35 of 39, so 2 seconds were enough for this person. The stroke amplitude was 0.66 to 1.26 hand sizes (mean 0.91) for `swipe_left` but only 0.26 to 0.91 (mean 0.58) for `swipe_right`; clips 4 and 6 of `swipe_right` (0.35 and 0.26) are about as large as the movement seen in "still" clips, which would make them hard to tell from `no_gesture`. Recording instruction: make the movement large, at least about one hand size. The hand returned close to its start (net displacement between -0.2 and +0.3 hand sizes), which supports the out-and-back definition. The estimates in the first pilot analysis (20 to 40 frames per cycle) were too high; the numbers in this bullet replace them.

### D.4 Decisions taken so far

1. MediaPipe Hands is the hand model (a classical-only pipeline would not give 21 landmarks and a custom landmark model is out of scope).
2. Only landmarks are stored, not images. Recording videos is optional and local only.
3. JSON, human-readable, one line per frame. One folder per user, one file per gesture, clips numbered inside the file.
4. Training and test sets are split by person, never by frame or by clip, because frames of one clip are nearly identical and would leak into the test set.
5. Static data is stored in the same format as dynamic data. Which frames count as "the gesture" is decided in the loader, not at recording time.
6. The old OpenCV code is kept under `classical_baseline/` instead of being deleted.
7. Thresholds and model choices are tuned on training and validation users only. Test users are used once, at the end.
8. A swipe is an out-and-back cycle whose first stroke gives the direction (A.12); the motion cue for segmentation is the palm centre, not the wrist (A.8).
9. The static classifier is an SVM (RBF kernel) on standardised inputs; every result is measured per clip or per user, never per frame (`recognition/train_static.py`). The pilot run is a check of the chain, not a result (A.9).
10. **No activation gesture** (decided 23 November 2026). A command starts as soon as its gesture is recognised, the way a phone's volume key acts the moment it is pressed. There is no clutch gesture that has to be shown first. Protection against accidental commands therefore rests on the other safeguards of 4.9 (A.10).
11. **Holding a gesture repeats its command** (decided 23 November 2026). While a static gesture is held, its command keeps being applied in small steps, the way holding a phone's volume key keeps raising the volume; releasing the gesture stops it. A swipe cannot be held and gives one command per out-and-back cycle (A.12).
12. **Only the static open palm waits** (decided 23 November 2026). The half-second stability wait of A.12 rule 2 applies to the open palm only, because it is also the swipe hand shape. Fist, thumb up, thumb down and point start as soon as they are recognised (a few consecutive frames, 4.9).
13. **Repeat timing** (decided 23 November 2026). A held static gesture repeats its command every 0.2 s, with the same setting for every gesture. The first repeat also comes 0.2 s after the first command.
14. **Calibration design** (decided 23 November 2026, details in A.11). Calibration records each static gesture once and one swipe per direction, checks every calibration gesture with the uncalibrated rules, and adjusts only thresholds, each within fixed limits around the population value. It does not change the scale `s` (normalisation already uses the hand size of every frame) and has no separate neutral-pose step.

### D.5 Open items

**Decisions needed from the team**

1. The gesture set is decided (A.2) and the open-palm conflict is handled by the three rules in A.12, which still have to be validated with real recordings. Commands and the target device are deliberately postponed (4.10). The activation gesture is decided: there is none (D.4 item 10).
1a. Resolved 23 November 2026: the half-second wait of A.12 rule 2 applies to the open palm only (D.4 item 12). Still to check on real data: in the pilot swipe clips the palm was held still for 8 to 17 frames (0.4 to 0.85 s) before the movement, so the static open palm can still fire before a swipe whose still lead-in is longer than half a second (`evaluation/evaluate_triggers.py` counts this).
2. Decide whether the **classical baseline** (skin colour + contour) is part of the project. If yes: it can only compare a subset of static gestures (open palm, fist, perhaps one and two fingers), because it only counts fingers, and it needs either its per-frame output saved at recording time or raw videos for the test conditions (E6 to E8). It also needs `pyautogui` installed to run. It is a scope increase and could be dropped if time is short.
3. Ask the instructor or teaching assistant whether a MediaPipe-based pipeline with a classical baseline satisfies the course expectation of classical image processing content.
4. Data-collection capacity: how many users, which backgrounds, lighting conditions and distances the Dataset team can really cover.
5. Static data handling: use every stable frame as a sample (recommended) or one sample per clip; keep or drop the z coordinate; which feature set (63 raw values, 42 without z, derived distances and angles) is chosen by experiment.
6. Team member names and dates for the Gantt chart; draft deadlines for each team.

**Work not done yet**

7. **Fixed label lists** are now in `data_tools/dataset_config.py` and enforced by the recorder (a typo such as `openpalm` is rejected). The validator and the loader still have to use the same file.
8. **Validation script**: written (`data_tools/validate_dataset.py`), it also gives the numbers for Section 3.5. On the pilot data it found 1 error (the mis-performed `swipe_right` clip 5) and 19 warnings; the output is in `docs/results/validation_pilot_2026-11-21.txt`. Still open: it does not yet cross-check clips with a model (that needs a trained classifier, see 10), and its thresholds come from one person.
9. **Data loader**: the static part is written (`data_tools/dataset_loader.py`: validator, smoothing, normalisation, stable frames, split by user; run on the pilot data it kept 1035 of 1172 frames, details and the choice of the two thresholds in `docs/results/loader_pilot_2026-11-21.txt`). The stability thresholds are provisional (one person). Still open: the dynamic part (resampling the out-and-back window to a fixed length, wrist-relative trajectory features), which needs the segmentation (4.5); the split itself cannot be chosen yet because there are only two users.
10. **Guided recording mode** (the program announces which gesture to perform, so the label never comes from a person's choice) and a **live sanity check** that warns when a clip does not look like its label. After training, a cross-validated model can flag suspicious clips.
11. Everything from Section 4.4 onward in code: smoothing, segmentation, classifiers, calibration, false-trigger prevention, command mapping. None of it is implemented yet; only data collection exists.
12. Runtime: recording runs at about 20 frames per second; measure per-stage latency (E9) before choosing window lengths for dynamic gestures.

**Housekeeping**

13. `README.md` still describes the old mouse-control project and must be rewritten for the current tools.
14. Nothing from the current work is committed. Use `git add -A` so the moved files are tracked as renames. Decide whether `models/hand_landmarker.task` is committed (7.8 MB) or downloaded by each person, and whether `data/landmarks/` is committed (landmarks only, no faces).
15. `docs/REPORT_OUTLINE.md` exists on GitHub; a second copy lives in the private notes folder and may differ. The repository copy is the reference.
16. The Progress Report deadline in the handoff note was 20 November 2026. Today is 21 November, so its submission status or any extension needs to be confirmed.
17. **Orientation of the hand in the static classifier** (A.9): record each gesture with different tilts, add rotation-independent features, and/or rotate training frames; then check again with several users. Until then the classifier should not be trusted on rotated hands.
18. The static classifier has only been tried on one complete user. It needs `thumb_down`, `point` and `no_gesture` data from several people before experiments E1 and E5 can be run, and a rule-based baseline for E1 is not written yet.
