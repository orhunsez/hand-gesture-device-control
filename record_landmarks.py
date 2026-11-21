"""Record labelled hand-landmark clips for the gesture dataset.

Each clip is a short recording of MediaPipe landmarks, saved together with the
condition metadata needed for the cross-user / background / lighting / distance
experiments.

Example:
    .venv\\Scripts\\python.exe record_landmarks.py ^
        --user u01 --gesture open_palm --background plain ^
        --lighting bright --distance near

Keys in the preview window:
    SPACE  start a clip (3-2-1 countdown, then DURATION seconds are recorded)
    v      replay the last saved clip as a landmark skeleton (v or q stops it)
    q      quit
Add --show to replay every clip automatically right after it is saved.

Saved data (raw MediaPipe output, NOT smoothed and NOT normalised, so the
processing choices can still be changed later):
    data/landmarks/<user>/<gesture>.json      one folder per user, one file per gesture
        (the user id must look like u01, u02, ...)
        clips: numbered 1, 2, 3 ... each new recording is appended to the file
        each clip has its own background, lighting, distance, duration
        frames: one line per frame, {"t": seconds, "landmarks": [[x, y, z] x 21]}
        "landmarks" is null when no hand was found in that frame.
        Open it in any text editor.
    (No index file: view_clip.py lists all clips straight from the JSON files.)
    data/videos/     only with --save-video (keep OUT of git: faces are visible)

Images are NOT mirrored for processing. Only the preview is mirrored.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import dataset_config as cfg

USER_ID = re.compile(r"^u\d{2,}$")   # u01, u02, ... keeps one person = one folder
MODEL_PATH = "models/hand_landmarker.task"
DATA_DIR = Path("data")
COUNTDOWN_S = 3

# Pairs of landmark indices to draw the hand skeleton in the preview.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", help="participant id, e.g. u01")
    p.add_argument("--gesture", help="one of the gestures in dataset_config.py")
    p.add_argument("--background", help="one of the backgrounds in dataset_config.py")
    p.add_argument("--lighting", help="one of the lightings in dataset_config.py")
    p.add_argument("--distance", help="one of the distances in dataset_config.py")
    p.add_argument("--duration", type=float, default=None,
                   help="clip length in seconds (default: per gesture, see "
                        "DURATIONS in dataset_config.py)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--show", action="store_true",
                   help="replay each clip automatically right after recording")
    p.add_argument("--save-video", action="store_true",
                   help="also save the raw video (privacy: do not commit)")
    args = p.parse_args()

    # Any label not given on the command line is asked interactively.
    # Press Enter to accept the value in brackets.
    # Values must come from dataset_config.py, so a typo can never become a new class.
    defaults = {"user": "u01", "gesture": "open_palm", "background": "plain",
                "lighting": "bright", "distance": "near"}
    allowed = {"gesture": cfg.GESTURES, "background": cfg.BACKGROUNDS,
               "lighting": cfg.LIGHTINGS, "distance": cfg.DISTANCES}

    def valid(name, value):
        if name == "user":
            return bool(USER_ID.match(value))
        return value in allowed[name]

    def choices(name):
        return "u01, u02, ..." if name == "user" else ", ".join(allowed[name])

    for name, default in defaults.items():
        value = getattr(args, name)
        if value is None:
            while True:
                value = input(f"{name} ({choices(name)}) [{default}]: ").strip() or default
                if valid(name, value):
                    break
                print(f"Invalid {name}. Choose from: {choices(name)}")
            setattr(args, name, value)
        elif not valid(name, value):
            p.error(f"--{name} must be one of: {choices(name)}")
    if args.duration is None:
        args.duration = cfg.DURATIONS.get(args.gesture, cfg.DEFAULT_DURATION)
    print(f"Clip length for '{args.gesture}': {args.duration:g} s")
    return args


def clip_file(landmark_dir, user, gesture):
    """data/landmarks/<user>/<gesture>.json; the clips are numbered inside it."""
    return landmark_dir / user / f"{gesture}.json"


def load_clips(path):
    """Return the list of clips already stored in the file (or [])."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["clips"]


def render_file(user, gesture, clips):
    """Readable JSON text: one line per frame, one block per clip.

    Each frame is {"t": seconds, "landmarks": [[x, y, z] x 21]} and
    "landmarks" is null when MediaPipe found no hand in that frame.
    """
    lines = ["{", f'  "user": {json.dumps(user)},',
             f'  "gesture": {json.dumps(gesture)},', '  "clips": [']
    for ci, clip in enumerate(clips):
        lines.append("    {")
        keys = ["clip", "background", "lighting", "distance", "duration"]
        keys += [k for k in ("image_width", "image_height") if k in clip]
        for key in keys:
            lines.append(f"      {json.dumps(key)}: {json.dumps(clip[key])},")
        lines.append('      "frames": [')
        frames = clip["frames"]
        for fi, fr in enumerate(frames):
            comma = "," if fi < len(frames) - 1 else ""
            lines.append(f'        {{"t": {fr["t"]:.4f}, '
                         f'"landmarks": {json.dumps(fr["landmarks"])}}}{comma}')
        lines.append("      ]")
        lines.append("    }" + ("," if ci < len(clips) - 1 else ""))
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


def append_clip(path, args, times, frames):
    """Add one clip to the (user, gesture) file and return its number.

    The file is rewritten through a temporary file so a crash while saving
    cannot corrupt the clips already recorded.
    """
    clips = load_clips(path)
    clip_id = len(clips) + 1
    clips.append({
        "clip": clip_id, "background": args.background,
        "lighting": args.lighting, "distance": args.distance,
        "duration": args.duration,
        # camera picture size, needed to measure x and y distances equally (4.3)
        **({"image_width": args.image_width, "image_height": args.image_height}
           if hasattr(args, "image_width") else {}),
        "frames": [
            {"t": float(t),
             "landmarks": None if np.isnan(arr[0, 0])
             else np.round(arr.astype(np.float64), 5).tolist()}
            for t, arr in zip(times, frames)
        ],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(render_file(args.user, args.gesture, clips), encoding="utf-8")
    os.replace(tmp, path)
    return clip_id


def draw_hand(frame, hand):
    """Draw landmarks on the mirrored preview frame."""
    h, w = frame.shape[:2]
    pts = [(int((1 - lm.x) * w), int(lm.y * h)) for lm in hand]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)


def replay_clip(times, frames, label):
    """Play a recorded clip once as a landmark skeleton in the same window."""
    size = 480
    for i, (t, arr) in enumerate(zip(times, frames)):
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        if not np.isnan(arr[0, 0]):
            pts = [(int((1 - x) * size), int(y * size)) for x, y, _ in arr]
            for a, b in CONNECTIONS:
                cv2.line(canvas, pts[a], pts[b], (0, 200, 0), 2)
            for pt in pts:
                cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
        else:
            cv2.putText(canvas, "no hand", (170, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        cv2.putText(canvas, f"REPLAY {label}  {i + 1}/{len(frames)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("record_landmarks", canvas)
        wait = 33
        if i + 1 < len(times):
            wait = max(1, int((times[i + 1] - t) * 1000))
        if cv2.waitKey(wait) & 0xFF in (ord("q"), ord("v")):
            break


def main():
    args = parse_args()
    landmark_dir = DATA_DIR / "landmarks"
    video_dir = DATA_DIR / "videos"
    landmark_dir.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Camera could not be opened.")

    program_start = time.time()
    last_ts_ms = -1

    def detect(frame_bgr):
        """Run MediaPipe on one frame. Timestamps must increase strictly."""
        nonlocal last_ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((time.time() - program_start) * 1000), last_ts_ms + 1)
        last_ts_ms = ts
        result = landmarker.detect_for_video(image, ts)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    state = "idle"          # idle | countdown | recording
    state_start = 0.0
    frames_lm, frames_t = [], []
    writer = None
    clip_name = ""
    saved = 0
    last_t, last_lm = [], []   # the most recently saved clip, for replay

    print(f"Recording '{args.gesture}' for {args.user}. SPACE = record, q = quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        if not hasattr(args, "image_width"):
            args.image_height, args.image_width = frame.shape[:2]
        now = time.time()
        hand = detect(frame)

        if state == "recording":
            arr = np.full((21, 3), np.nan, dtype=np.float32)
            if hand is not None:
                arr = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            frames_lm.append(arr)
            frames_t.append(now - state_start)
            if writer is not None:
                writer.write(frame)
            if now - state_start >= args.duration:
                n_found = int(np.sum(~np.isnan([f[0, 0] for f in frames_lm])))
                path = clip_file(landmark_dir, args.user, args.gesture)
                clip_id = append_clip(path, args, frames_t, frames_lm)
                clip_name = f"{args.user}/{path.name} clip {clip_id}"
                if writer is not None:
                    writer.release()
                    writer = None
                saved += 1
                print(f"saved {clip_name}: {len(frames_lm)} frames, "
                      f"{n_found} with a hand")
                state = "idle"
                last_t, last_lm = list(frames_t), list(frames_lm)
                if args.show:
                    replay_clip(last_t, last_lm, clip_name)

        elif state == "countdown" and now - state_start >= COUNTDOWN_S:
            state = "recording"
            state_start = now
            frames_lm, frames_t = [], []
            if args.save_video:
                h, w = frame.shape[:2]
                idx = len(load_clips(
                    clip_file(landmark_dir, args.user, args.gesture))) + 1
                vpath = video_dir / f"{args.user}_{args.gesture}_{idx:03d}.avi"
                writer = cv2.VideoWriter(
                    str(vpath), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))

        # Preview (mirrored, with overlay)
        view = cv2.flip(frame, 1)
        if hand is not None:
            draw_hand(view, hand)
        if state == "idle":
            msg = f"SPACE: record  v: replay last  q: quit ({saved} saved)"
        elif state == "countdown":
            msg = f"Get ready: {COUNTDOWN_S - (now - state_start):.1f}"
        else:
            msg = f"RECORDING {now - state_start:.1f}/{args.duration:.1f}s"
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.putText(view, f"{args.user} | {args.gesture}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("record_landmarks", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and state == "idle":
            state = "countdown"
            state_start = now
        if key == ord("v") and state == "idle" and last_lm:
            replay_clip(last_t, last_lm, clip_name)

    if writer is not None:
        writer.release()
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
