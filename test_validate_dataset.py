"""Checks for validate_dataset.py with made-up data. Run:  .venv\\Scripts\\python.exe test_validate_dataset.py"""
import json
import tempfile
from pathlib import Path

import numpy as np

import validate_dataset as vd

rng = np.random.default_rng(2)
W, H = 640, 480
SIZE = 100.0                      # hand size in pixels
DT = 0.05                         # 20 frames per second
BASE = rng.uniform(-SIZE, SIZE, size=(21, 2))
BASE[0] = [0.0, 0.0]
BASE[9] = [0.0, -SIZE]            # middle finger base straight above the wrist


def clip(number, gesture, x_px=None, y_px=None, n=40, hand=None, duration=2.0,
         background="plain", lighting="bright", distance="near"):
    """A made-up clip. x_px and y_px: palm movement per frame in pixels."""
    x_px = np.zeros(n) if x_px is None else x_px
    y_px = np.zeros(n) if y_px is None else y_px
    hand = np.ones(n, dtype=bool) if hand is None else hand
    frames = []
    for i in range(n):
        if not hand[i]:
            frames.append({"t": round(i * DT, 4), "landmarks": None})
            continue
        pts = np.zeros((21, 3))
        pts[:, :2] = BASE + [320 + x_px[i], 300 + y_px[i]] + rng.normal(0, 0.2, size=(21, 2))
        pts[:, 0] /= W
        pts[:, 1] /= H
        frames.append({"t": round(i * DT, 4), "landmarks": np.round(pts, 5).tolist()})
    return {"clip": number, "background": background, "lighting": lighting, "distance": distance,
            "duration": duration, "image_width": W, "image_height": H, "frames": frames}


def out_and_back(amp_hand_sizes, sign, n=60):
    """Palm x: still, a stroke in direction `sign`, back, still. n = 60 frames is 3 seconds."""
    x = np.zeros(n)
    up = np.linspace(0, 1, 8)
    x[10:18] = sign * amp_hand_sizes * SIZE * up
    x[18:26] = sign * amp_hand_sizes * SIZE * up[::-1]
    return x


def write(root, user, gesture, clips, user_inside=None, gesture_inside=None):
    d = Path(root) / user
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{gesture}.json").write_text(json.dumps({
        "user": user_inside or user, "gesture": gesture_inside or gesture, "clips": clips}), encoding="utf-8")


def has(issues, level, text):
    return any(l == level and text in w + " " + m for l, w, m in issues)


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


with tempfile.TemporaryDirectory() as tmp:
    # Good data: a still static gesture and a correct swipe each way.
    write(tmp, "u01", "fist", [clip(1, "fist"), clip(2, "fist")])
    write(tmp, "u01", "swipe_left", [clip(1, "swipe_left", out_and_back(1.2, +1), n=60, duration=3.0)])
    write(tmp, "u01", "swipe_right", [clip(1, "swipe_right", out_and_back(1.2, -1), n=60, duration=3.0)])
    issues, stats = vd.validate(tmp)
    check("good data has no errors", not any(i[0] == "ERROR" for i in issues))
    check("good data has no warnings", not issues)
    check("counts clips", stats["clips"] == 4 and stats["files"] == 3)

with tempfile.TemporaryDirectory() as tmp:
    # Bad data, one problem at a time.
    write(tmp, "u02", "openpalm", [clip(1, "open_palm")])                                  # wrong file name
    write(tmp, "x1", "fist", [clip(1, "fist")])                                            # wrong folder name
    write(tmp, "u03", "fist", [clip(1, "fist")], gesture_inside="open_palm")              # label mismatch
    write(tmp, "u04", "open_palm", [clip(1, "open_palm", x_px=np.linspace(0, 60, 40))])    # hand moves
    write(tmp, "u05", "swipe_left", [clip(1, "swipe_left", out_and_back(1.2, -1), n=60, duration=3.0)])   # wrong way
    write(tmp, "u06", "swipe_right", [clip(1, "swipe_right", out_and_back(0.2, -1), n=60, duration=3.0)])  # too small
    hand = np.zeros(40, dtype=bool)
    hand[:12] = True
    write(tmp, "u07", "fist", [clip(1, "fist", hand=hand)])                                # hand mostly missing
    same = clip(1, "fist")
    write(tmp, "u08", "fist", [same, dict(same, clip=2)])                                  # identical clips
    write(tmp, "u09", "fist", [clip(2, "fist")])                                           # numbering
    write(tmp, "u10", "fist", [clip(1, "fist", lighting="dark")])                          # bad condition
    write(tmp, "u11", "swipe_left", [clip(1, "swipe_left", out_and_back(1.2, +1, n=40), n=40, duration=2.0)])   # old duration
    issues, stats = vd.validate(tmp)

    check("file name that is not a gesture", has(issues, "ERROR", "u02/openpalm.json"))
    check("folder name that is not a user id", has(issues, "ERROR", "x1/fist.json"))
    check("label inside differs from the file name", has(issues, "ERROR", "gesture inside the file"))
    check("static hand that moves", has(issues, "WARNING", "u04/open_palm.json clip 1 hand not still"))
    check("swipe in the wrong direction", has(issues, "ERROR", "u05/swipe_left.json clip 1 first stroke goes the wrong way"))
    check("swipe too small", has(issues, "WARNING", "u06/swipe_right.json clip 1 swipe too small"))
    check("hand mostly missing", has(issues, "ERROR", "u07/fist.json clip 1 hand found in only"))
    check("identical clips", has(issues, "ERROR", "clip 2 is identical to clip 1"))
    check("clip numbering", has(issues, "ERROR", "not numbered"))
    check("condition outside the list", has(issues, "ERROR", "lighting 'dark'"))
    check("recording length differs from the setting", has(issues, "WARNING", "u11/swipe_left.json clip 1 recorded for 2.0 s"))
    check("the correct data produced no issue in a clean folder (u11 only warns)",
          not any(l == "ERROR" and "u11" in w for l, w, m in issues))

    bad = vd.bad_clips(issues)
    check("bad_clips lists clips with an error", ("u05/swipe_left.json", 1) in bad and ("u07/fist.json", 1) in bad)
    check("bad_clips leaves out clips with warnings only", ("u04/open_palm.json", 1) not in bad)

print("\nAll checks passed.")
