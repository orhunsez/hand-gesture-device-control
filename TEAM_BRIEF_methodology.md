# CENG391 Gesture Interface: Methodology Brief and Team Requests

**Purpose.** This document (1) explains the proposed methodology of our real-time gesture interface, and (2) lists exactly what the Problem Definition, Dataset and Related Work teams need to deliver so that all sections of the Progress Report (due 20 November 2026) stay consistent and can be written in parallel.

**Status of assumptions.** Items marked **[ASSUMPTION]** have not yet been confirmed by the whole team (target device, gesture set). Everything else is the current proposal. Literature figures quoted below are from memory and **must be verified against the original papers** before they appear in the report.

---

## Part A. Methodology

### A.1 Problem in one paragraph

The system receives a continuous stream of RGB frames from a single ordinary webcam and must decide, in real time, whether the user is performing an intentional hand gesture and, if so, which device command it corresponds to. The difficulty is not only recognising a hand shape. The system must also (i) work for different users, backgrounds, lighting conditions and camera distances, (ii) find where a gesture starts and ends inside a never-ending stream (temporal segmentation), and (iii) avoid firing commands for incidental movements such as scratching the face or reaching for the keyboard (false-trigger prevention, the "Midas touch" problem).

### A.2 Target device and gesture set [ASSUMPTION]

- Target device: a PC controlled for media playback and slide presentation (commands sent with `pyautogui`).
- Static gestures (identity is the hand shape in a single frame): open palm (play/pause), fist (mute), thumb up / thumb down (volume up / down).
- Dynamic gestures (identity is the motion over time): swipe right / swipe left (next / previous slide), optionally swipe up.
- A **"no gesture"** class, and optionally an **activation gesture** (see A.10).

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

### A.7 Stage 4.4: Temporal smoothing

Even a perfectly still hand makes landmarks jitter from frame to frame. Smoothing has a cost: the stronger the filter, the larger the lag. We use two levels:

1. **Landmark level: 1-euro filter** (Casiez et al., 2012), an adaptive low-pass filter. For each coordinate: estimate the speed `dx^`, set the cut-off `f_c = f_cmin + beta * |dx^|`, then `alpha = 1 / (1 + tau / T_e)` with `tau = 1 / (2 * pi * f_c)` and `T_e` the sampling period. When the hand is slow the cut-off is low (strong smoothing, little jitter); when it moves fast the cut-off rises (little lag). Parameters: `f_cmin`, `beta`, `d_cutoff`. Tuning procedure: first set `beta = 0` and tune `f_cmin` for minimal jitter, then raise `beta` until lag becomes acceptable.
2. **Class level:** majority voting or exponential moving average over the last few predicted labels.

Measurement (experiment E2): jitter (standard deviation of a landmark on a still hand) and added latency, with and without filtering.

### A.8 Stage 4.5: Gesture segmentation

A classifier always outputs *some* label, so the system must decide which part of the stream is a gesture candidate. Static and dynamic gestures need different logic:

- **Static gestures: stability-based.** The same class must be predicted for several consecutive frames while the hand is nearly still (a *hold* condition). Transition frames between two poses produce unstable output and are rejected naturally.
- **Dynamic gestures: motion-energy-based.** The wrist speed `v_t = || p_0(t) - p_0(t-1) || / dt`, normalised by hand size, `v^_t = v_t / s`, is compared with an onset threshold `tau_on` (gesture starts) and an offset threshold `tau_off` (gesture ends). The frames between onset and offset (about 15 to 30 frames) are passed to the dynamic classifier.

Three mechanisms make this robust: **hysteresis** (`tau_on > tau_off`, so speed fluctuating around one threshold does not start and stop the gesture repeatedly), a **minimum duration** (very short movements are discarded as tremor), and a **ring buffer** of the last N frames.

**Boundary with 4.9:** Segmentation answers "does this interval form a gesture candidate?". Stage 4.9 answers "should this candidate become a command?". Do not mix the two in the text.

### A.9 Stages 4.6 and 4.7: Classification (the only part where we train ML)

- **Static classifier (4.6):** input is the normalised landmark vector of one frame; output is one of {open palm, fist, thumb up, thumb down, no gesture}. Two models are compared (experiment E1): a **rule-based baseline** (for example, finger extended or not, computed from landmark distances and angles) and a **learned model** (SVM or a small MLP).
- **Dynamic classifier (4.7):** input is the landmark sequence of the segmented window, resampled to a fixed length; output is {swipe right, swipe left, (swipe up), no gesture}. Again a rule-based baseline (direction and distance of wrist displacement) is compared with a small learned model (SVM on resampled features; 1D-CNN or LSTM only as an optional extension if enough data exist).

Rationale for small models: the input is a low-dimensional landmark vector, not an image, and our dataset will be small. Large deep models are unlikely to pay off and add risk.

### A.10 Stage 4.9: False-trigger prevention

A prediction becomes a command only if all of the following hold: the classifier confidence exceeds a threshold; the hold time or minimum duration is satisfied; after a command, a **cooldown** (refractory period) blocks further commands; and the sample is not classified as **no gesture**. Optionally, an **activation gesture** (a clutch) opens a short window in which commands are accepted, which is one of the most effective protections against accidental triggers. Experiment E4 compares the system with and without these safeguards using false-trigger rate.

### A.11 Stage 4.8 and 4.10: Calibration and command mapping

- **Calibration (4.8):** at start-up the user holds a neutral pose and performs each gesture once or twice. The system measures the user's hand size and typical speed and adjusts the scale `s`, the thresholds `tau_on` / `tau_off` and the confidence threshold. Effect measured in E3.
- **Command mapping (4.10):** a table maps each confirmed gesture to a keyboard or media command sent with `pyautogui`.

### A.12 Known design issue: open palm used for two gestures [DECISION NEEDED]

Open palm is both the static "play/pause" gesture (hand still) and the natural hand shape for swiping (hand moving). The hand shape is identical; only motion differs. Two possible fixes:

1. Keep one shape and let the static branch be suppressed whenever hand speed exceeds `tau_on` (needs a clear rule in 4.5 / 4.9).
2. Use a different hand shape for swipes (for example, index finger extended).

The team must choose one before the gesture table (Section 3.2) is finalised.

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
| `record_landmarks.py` | Webcam to MediaPipe to labelled landmark clips (see D.3) |
| `view_clip.py` | Lists, replays and prints recorded clips (also reachable from `main.py`) |
| `models/hand_landmarker.task` | MediaPipe hand model bundle (palm detector + 21-landmark model), pretrained by Google, 7.8 MB |
| `data/landmarks/<user>/<gesture>.json` | Recorded data, one folder per user, one file per gesture |
| `classical_baseline/` | The old skin-colour + contour code (mouse control). Kept as a possible classical baseline, see D.5 |
| `docs/REPORT_OUTLINE.md` | Report section outline |
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

### D.4 Decisions taken so far

1. MediaPipe Hands is the hand model (a classical-only pipeline would not give 21 landmarks and a custom landmark model is out of scope).
2. Only landmarks are stored, not images. Recording videos is optional and local only.
3. JSON, human-readable, one line per frame. One folder per user, one file per gesture, clips numbered inside the file.
4. Training and test sets are split by person, never by frame or by clip, because frames of one clip are nearly identical and would leak into the test set.
5. Static data is stored in the same format as dynamic data. Which frames count as "the gesture" is decided in the loader, not at recording time.
6. The old OpenCV code is kept under `classical_baseline/` instead of being deleted.
7. Thresholds and model choices are tuned on training and validation users only. Test users are used once, at the end.

### D.5 Open items

**Decisions needed from the team**

1. Confirm the **target device** and the **gesture set**, including how to resolve the open-palm conflict (A.12). Until then the gesture names in the code are provisional.
2. Decide whether the **classical baseline** (skin colour + contour) is part of the project. If yes: it can only compare a subset of static gestures (open palm, fist, perhaps one and two fingers), because it only counts fingers, and it needs either its per-frame output saved at recording time or raw videos for the test conditions (E6 to E8). It also needs `pyautogui` installed to run. It is a scope increase and could be dropped if time is short.
3. Ask the instructor or teaching assistant whether a MediaPipe-based pipeline with a classical baseline satisfies the course expectation of classical image processing content.
4. Data-collection capacity: how many users, which backgrounds, lighting conditions and distances the Dataset team can really cover.
5. Static data handling: use every stable frame as a sample (recommended) or one sample per clip; keep or drop the z coordinate; which feature set (63 raw values, 42 without z, derived distances and angles) is chosen by experiment.
6. Team member names and dates for the Gantt chart; draft deadlines for each team.

**Work not done yet**

7. **Fixed label lists** for gesture, background, lighting and distance, shared by the recorder, validator and loader (a typo such as `openpalm` would currently create a new class).
8. **Validation script**: unknown labels, clips with few hand frames, unexpected frame counts, class balance, clips per condition and user. It also gives the numbers for Section 3.5.
9. **Data loader**: JSON to training matrices, with stable-frame selection, normalisation (the same function must be used at training time and at run time) and the person-based split.
10. **Guided recording mode** (the program announces which gesture to perform, so the label never comes from a person's choice) and a **live sanity check** that warns when a clip does not look like its label. After training, a cross-validated model can flag suspicious clips.
11. Everything from Section 4.4 onward in code: smoothing, segmentation, classifiers, calibration, false-trigger prevention, command mapping. None of it is implemented yet; only data collection exists.
12. Runtime: recording runs at about 20 frames per second; measure per-stage latency (E9) before choosing window lengths for dynamic gestures.

**Housekeeping**

13. `README.md` still describes the old mouse-control project and must be rewritten for the current tools.
14. Nothing from the current work is committed. Use `git add -A` so the moved files are tracked as renames. Decide whether `models/hand_landmarker.task` is committed (7.8 MB) or downloaded by each person, and whether `data/landmarks/` is committed (landmarks only, no faces).
15. `docs/REPORT_OUTLINE.md` exists on GitHub; a second copy lives in the private notes folder and may differ. The repository copy is the reference.
16. The Progress Report deadline in the handoff note was 20 November 2026. Today is 21 November, so its submission status or any extension needs to be confirmed.
