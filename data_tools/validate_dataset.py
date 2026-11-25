"""Check the recorded landmark data before it is used for training.

    .venv\\Scripts\\python.exe validate_dataset.py            # whole data/landmarks folder
    .venv\\Scripts\\python.exe validate_dataset.py u99        # one user only

Prints one line per problem and a summary. ERROR = the clip must not be used for training
(wrong label, broken file, almost no hand found). WARNING = look at the clip (replay it with
main.py) and decide. The exit code is 1 when there is at least one ERROR.

What is checked, per file and per clip:
  - the folder and file names match the labels stored inside, and the labels are in
    dataset_config.py; clips are numbered 1, 2, 3, ...
  - timestamps increase, the frame rate is plausible, the clip length matches the setting
  - how many frames contain a hand, and how long the longest gap without a hand is
  - the hand is not absurdly small or large in the picture
  - static gestures: the hand really stayed still
  - swipes: the first stroke goes in the direction of the label and is large enough
  - no two clips are identical (a copied clip would leak into the test set)
Then it counts clips per user, gesture and condition, and lists gestures a user is missing.

The thresholds below come from the pilot clips of one person (user u99) and are provisional.
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import dataset_config as cfg
import landmark_features as lf

DATA_DIR = Path("data/landmarks")
PALM = [0, 5, 9, 13, 17]
USER_ID = r"^u\d{2,}$"

# Provisional thresholds (pilot, one person).
MIN_FRAMES = 15
FPS_RANGE = (10.0, 40.0)            # the development machine gives about 20
DURATION_TOLERANCE_S = 0.3          # last timestamp vs the stored clip length
HAND_RATIO_WARN, HAND_RATIO_ERROR = 0.90, 0.50
MAX_GAP_FRAMES = 3                  # longest run of frames without a hand
HAND_SIZE_PX = (40.0, 300.0)        # wrist to middle finger base on a 640 x 480 picture
STATIC_MAX_DRIFT = 0.30             # palm centre range in hand sizes (pilot: median 0.02, max 0.09)
SWIPE_MIN_AMPLITUDE = 0.50          # first stroke, hand sizes (pilot: 0.34 to 1.67)
SWIPE_FIRST_STROKE = {"swipe_left": +1, "swipe_right": -1}   # sign of the change of palm x


def clip_array(clip):
    frames = clip["frames"]
    lm = np.full((len(frames), 21, 3), np.nan)
    for i, f in enumerate(frames):
        if f["landmarks"] is not None:
            lm[i] = f["landmarks"]
    return lm, np.array([f["t"] for f in frames], dtype=float)


def palm_x_and_y(lm, width, height):
    px = lf.to_pixel_units(lm, width, height)
    s = np.nanmedian(lf.hand_size(lm, width, height))
    c = np.nanmean(px[:, PALM, :2], axis=1) / s
    return c[:, 0], c[:, 1], s


def first_stroke(x):
    """Signed size of the biggest departure from the start position, and its sign."""
    ok = ~np.isnan(x)
    x = x[ok]
    if len(x) < 5:
        return 0.0, 0
    dev = x - np.mean(x[:3])
    pk = int(np.argmax(np.abs(dev)))
    return float(dev[pk]), int(np.sign(dev[pk]))


def longest_gap(missing):
    best = run = 0
    for m in missing:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def check_file(path, data, issues):
    """Append (level, where, message) tuples to issues for one file."""
    import re
    user_dir, gesture_file = path.parent.name, path.stem
    where_file = f"{user_dir}/{path.name}"

    def add(level, msg, clip=None):
        issues.append((level, where_file + (f" clip {clip}" if clip else ""), msg))

    if not re.match(USER_ID, user_dir):
        add("ERROR", f"folder name '{user_dir}' is not a user id like u01")
    if data.get("user") != user_dir:
        add("ERROR", f"user inside the file is '{data.get('user')}', folder is '{user_dir}'")
    if gesture_file not in cfg.GESTURES:
        add("ERROR", f"file name '{gesture_file}' is not a gesture in dataset_config.py")
    if data.get("gesture") != gesture_file:
        add("ERROR", f"gesture inside the file is '{data.get('gesture')}', file name says '{gesture_file}'")
    gesture = gesture_file
    clips = data.get("clips", [])
    if not clips:
        add("ERROR", "the file has no clips")
        return []

    seen, arrays = [], []
    for pos, clip in enumerate(clips, 1):
        k = clip.get("clip")
        if k != pos:
            add("ERROR", f"clips are not numbered 1, 2, 3, ...: found {k} at position {pos}", pos)
        for name, allowed in (("background", cfg.BACKGROUNDS), ("lighting", cfg.LIGHTINGS),
                              ("distance", cfg.DISTANCES)):
            if clip.get(name) not in allowed:
                add("ERROR", f"{name} '{clip.get(name)}' is not one of {allowed}", pos)

        lm, t = clip_array(clip)
        n = len(t)
        width, height = clip.get("image_width"), clip.get("image_height")
        arrays.append((pos, lm))
        if n < MIN_FRAMES:
            add("ERROR", f"only {n} frames", pos)
            continue
        if not np.all(np.diff(t) > 0):
            add("ERROR", "timestamps do not increase", pos)
            continue
        fps = (n - 1) / (t[-1] - t[0])
        if not FPS_RANGE[0] <= fps <= FPS_RANGE[1]:
            add("WARNING", f"frame rate {fps:.1f} per second is unusual", pos)
        expected = cfg.DURATIONS.get(gesture, cfg.DEFAULT_DURATION)
        if abs(clip.get("duration", 0) - expected) > 1e-6:
            add("WARNING", f"recorded for {clip.get('duration')} s, the current setting for '{gesture}' is {expected} s", pos)
        if abs(t[-1] - clip.get("duration", t[-1])) > DURATION_TOLERANCE_S:
            add("WARNING", f"last timestamp {t[-1]:.2f} s does not match the clip length {clip.get('duration')} s", pos)

        missing = np.isnan(lm[:, 0, 0])
        ratio = 1.0 - missing.mean()
        if ratio < HAND_RATIO_ERROR:
            add("ERROR", f"hand found in only {ratio * 100:.0f}% of the frames", pos)
            continue
        if ratio < HAND_RATIO_WARN:
            add("WARNING", f"hand found in {ratio * 100:.0f}% of the frames", pos)
        gap = longest_gap(missing)
        if gap > MAX_GAP_FRAMES:
            add("WARNING", f"{gap} frames in a row without a hand", pos)

        px, py, s = palm_x_and_y(lm, width, height)
        size_px = float(np.nanmedian(lf.hand_size(lm, width, height)))
        if not HAND_SIZE_PX[0] <= size_px <= HAND_SIZE_PX[1]:
            add("WARNING", f"hand size {size_px:.0f} px is unusual (expected {HAND_SIZE_PX[0]:.0f} to {HAND_SIZE_PX[1]:.0f})", pos)

        kind = cfg.GESTURES.get(gesture)
        if kind == "static":
            drift = max(float(np.nanmax(px) - np.nanmin(px)), float(np.nanmax(py) - np.nanmin(py)))
            if drift > STATIC_MAX_DRIFT:
                add("WARNING", f"hand not still: palm centre moved {drift:.2f} hand sizes", pos)
        if gesture in SWIPE_FIRST_STROKE:
            amp, sign = first_stroke(px)
            if abs(amp) < SWIPE_MIN_AMPLITUDE:
                add("WARNING", f"swipe too small: first stroke {abs(amp):.2f} hand sizes (minimum {SWIPE_MIN_AMPLITUDE})", pos)
            if sign != SWIPE_FIRST_STROKE[gesture]:
                add("ERROR", f"first stroke goes the wrong way for '{gesture}' (amplitude {amp:+.2f} hand sizes); mislabelled or badly performed", pos)

    # identical clips
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            a, b = arrays[i][1], arrays[j][1]
            if a.shape == b.shape and np.array_equal(a, b, equal_nan=True) and not np.isnan(a).all():
                add("ERROR", f"clip {arrays[j][0]} is identical to clip {arrays[i][0]}", arrays[j][0])
    return clips


def validate(root=DATA_DIR, only_user=None):
    """Return (issues, stats). issues: list of (level, where, message)."""
    issues, stats = [], {"files": 0, "clips": 0, "per_user": defaultdict(Counter),
                         "conditions": Counter(), "users": set()}
    for path in sorted(Path(root).glob("*/*.json")):
        if only_user and path.parent.name != only_user:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            issues.append(("ERROR", f"{path.parent.name}/{path.name}", f"file cannot be read: {e}"))
            continue
        clips = check_file(path, data, issues)
        stats["files"] += 1
        stats["clips"] += len(clips)
        stats["users"].add(path.parent.name)
        stats["per_user"][path.parent.name][path.stem] += len(clips)
        for c in clips:
            stats["conditions"][(c.get("background"), c.get("lighting"), c.get("distance"))] += 1
    return issues, stats


def bad_clips(issues):
    """Set of (file, clip number) with an ERROR, so a loader can leave them out."""
    out = set()
    for level, where, _ in issues:
        if level == "ERROR" and " clip " in where:
            f, k = where.rsplit(" clip ", 1)
            out.add((f, int(k)))
        elif level == "ERROR":
            out.add((where, None))     # the whole file
    return out


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    issues, stats = validate(DATA_DIR, only)
    for level, where, msg in issues:
        print(f"{level:<7} {where}: {msg}")
    errors = sum(1 for i in issues if i[0] == "ERROR")
    warnings = len(issues) - errors

    print(f"\n{stats['files']} files, {stats['clips']} clips, {len(stats['users'])} users: "
          f"{errors} errors, {warnings} warnings")
    if stats["files"] == 0:
        return 1

    gestures = list(cfg.GESTURES)
    print("\nClips per user and gesture ('-' = none):")
    print(f"{'user':<6} " + " ".join(f"{g[:11]:>11}" for g in gestures))
    for user in sorted(stats["per_user"]):
        row = stats["per_user"][user]
        print(f"{user:<6} " + " ".join(f"{row[g] if row[g] else '-':>11}" for g in gestures))
    totals = Counter()
    for row in stats["per_user"].values():
        totals.update(row)
    print(f"{'total':<6} " + " ".join(f"{totals[g]:>11}" for g in gestures))

    for user in sorted(stats["per_user"]):
        missing = [g for g in gestures if not stats["per_user"][user][g]]
        if missing:
            print(f"Missing for {user}: {', '.join(missing)}")
    counts = [totals[g] for g in gestures if totals[g]]
    if counts and max(counts) > 2 * min(counts):
        print(f"Class balance: the biggest gesture has {max(counts)} clips and the smallest {min(counts)}; "
              "the classifier needs similar numbers.")
    print("\nClips per condition (background, lighting, distance):")
    for cond, n in sorted(stats["conditions"].items(), key=lambda kv: str(kv[0])):
        print(f"  {cond}: {n}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
