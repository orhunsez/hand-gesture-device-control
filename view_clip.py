"""Look at recorded landmark clips (JSON files in data/landmarks).

Layout: data/landmarks/<user>/<gesture>.json, with numbered clips inside.

    .venv\\Scripts\\python.exe view_clip.py                          # list all clips
    .venv\\Scripts\\python.exe view_clip.py u01/open_palm.json 2     # replay clip 2
    .venv\\Scripts\\python.exe view_clip.py u01/open_palm.json 2 --print --frame 5
                                                                   # 21x3 array of frame 5

In the replay window: SPACE pauses, q closes. The clip loops.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LANDMARK_DIR = Path("data/landmarks")
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def list_files():
    """Names like 'u01/open_palm.json' for every recording, sorted."""
    return sorted(p.relative_to(LANDMARK_DIR).as_posix()
                  for p in LANDMARK_DIR.glob("*/*.json"))


def read_file(name):
    path = Path(name)
    if not path.exists():
        path = LANDMARK_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def clip_arrays(clip):
    """Return (landmarks (T,21,3) with NaN where no hand, t (T,))."""
    frames = clip["frames"]
    t = np.array([f["t"] for f in frames], dtype=np.float32)
    lm = np.full((len(frames), 21, 3), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        if f["landmarks"] is not None:
            lm[i] = f["landmarks"]
    return lm, t


def load_clip(name, clip_number=1):
    """Return (landmarks, t, metadata) for one numbered clip of a file."""
    data = read_file(name)
    for clip in data["clips"]:
        if clip["clip"] == clip_number:
            lm, t = clip_arrays(clip)
            meta = {k: v for k, v in clip.items() if k != "frames"}
            meta.update(user=data["user"], gesture=data["gesture"])
            return lm, t, meta
    raise SystemExit(f"{name} has no clip {clip_number}.")


def list_clips():
    files = list_files()
    if not files:
        print("No files found in data/landmarks.")
        return
    for name in files:
        data = read_file(name)
        print(f"{name}: {len(data['clips'])} clips")
        for clip in data["clips"]:
            lm, _ = clip_arrays(clip)
            found = int(np.sum(~np.isnan(lm[:, 0, 0])))
            print(f"  clip {clip['clip']}: {len(lm)} frames, {found} with hand | "
                  f"{clip['background']} | {clip['lighting']} | {clip['distance']}")


def show_array(name, clip_number, frame):
    """Print the 21 x 3 landmark array of one frame in the terminal."""
    lm, t, _ = load_clip(name, clip_number)
    frame = min(max(frame, 0), len(lm) - 1)
    print(f"{name} clip {clip_number}  frame {frame} of {len(lm)}  "
          f"(t = {float(t[frame]):.3f} s)")
    print("id      x        y        z")
    for i, (x, y, z) in enumerate(lm[frame]):
        print(f"{i:2d}  {x:7.4f}  {y:7.4f}  {z:8.4f}")


def replay(name, clip_number):
    lm, t, meta = load_clip(name, clip_number)
    size = 640
    paused, i = False, 0
    print(f"{name} clip {clip_number}: {len(lm)} frames, {float(t[-1]):.2f} s. "
          "SPACE pause, q close.")
    while True:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        hand = lm[i]
        if not np.isnan(hand[0, 0]):
            # Mirror x so it looks like the recording preview.
            pts = [(int((1 - x) * size), int(y * size)) for x, y, _ in hand]
            for a, b in CONNECTIONS:
                cv2.line(canvas, pts[a], pts[b], (0, 200, 0), 2)
            for pt in pts:
                cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
        else:
            cv2.putText(canvas, "no hand", (230, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        cv2.putText(canvas, f"{meta['gesture']} #{clip_number}  "
                    f"frame {i + 1}/{len(lm)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("view_clip", canvas)

        wait = 50
        if i + 1 < len(t):
            wait = max(1, int((t[i + 1] - t[i]) * 1000))
        key = cv2.waitKey(wait) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if not paused:
            i = (i + 1) % len(lm)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="e.g. u01/open_palm.json")
    p.add_argument("clip", nargs="?", type=int, default=1, help="clip number")
    p.add_argument("--print", dest="print_array", action="store_true",
                   help="print the 21x3 array instead of replaying")
    p.add_argument("--frame", type=int, default=0, help="frame for --print")
    a = p.parse_args()
    if a.file is None:
        list_clips()
    elif a.print_array:
        show_array(a.file, a.clip, a.frame)
    else:
        replay(a.file, a.clip)
