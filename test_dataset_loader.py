"""Checks for dataset_loader.py with made-up data. Run:  .venv\\Scripts\\python.exe test_dataset_loader.py"""
import json
import tempfile
from pathlib import Path

import numpy as np

import dataset_loader as dl

rng = np.random.default_rng(3)
W, H = 640, 480
SIZE = 100.0
DT = 0.05
BASE = rng.uniform(-SIZE, SIZE, size=(21, 2))
BASE[0] = [0.0, 0.0]
BASE[9] = [0.0, -SIZE]


def clip(number, x_px=None, n=40, hand=None, lighting="bright"):
    """A made-up clip; x_px is the palm movement per frame in pixels."""
    x_px = np.zeros(n) if x_px is None else x_px
    hand = np.ones(n, dtype=bool) if hand is None else hand
    frames = []
    for i in range(n):
        if not hand[i]:
            frames.append({"t": round(i * DT, 4), "landmarks": None})
            continue
        pts = np.zeros((21, 3))
        pts[:, :2] = BASE + [320 + x_px[i], 300] + rng.normal(0, 0.15, size=(21, 2))
        pts[:, 0] /= W
        pts[:, 1] /= H
        frames.append({"t": round(i * DT, 4), "landmarks": np.round(pts, 5).tolist()})
    return {"clip": number, "background": "plain", "lighting": lighting, "distance": "near",
            "duration": 2.0, "image_width": W, "image_height": H, "frames": frames}


def write(root, user, gesture, clips):
    d = Path(root) / user
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{gesture}.json").write_text(json.dumps({"user": user, "gesture": gesture, "clips": clips}),
                                       encoding="utf-8")


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


with tempfile.TemporaryDirectory() as tmp:
    # u01: two still fist clips, one open palm clip, one swipe clip (must be ignored).
    write(tmp, "u01", "fist", [clip(1), clip(2, lighting="dim")])
    write(tmp, "u01", "open_palm", [clip(1)])
    write(tmp, "u01", "swipe_left", [clip(1, n=60)])
    # u02: a still fist clip, and an open palm clip whose hand moves in the second half.
    moving = np.zeros(40)
    moving[20:] = np.cumsum(np.full(20, 8.0))               # 8 px per frame is fast
    write(tmp, "u02", "fist", [clip(1)])
    write(tmp, "u02", "open_palm", [clip(1, x_px=moving)])
    # u03: a clip with no hand in many frames (ERROR in the validator) and a good clip.
    hand = np.zeros(40, dtype=bool)
    hand[:8] = True
    write(tmp, "u03", "fist", [clip(1, hand=hand), clip(2)])

    ds, stats = dl.load_static(tmp)

    check("table has 63 numbers per row and no NaN", ds.X.shape[1] == 63 and not np.isnan(ds.X).any())
    check("all columns have the same length", all(len(getattr(ds, f)) == len(ds) for f in dl._FIELDS))
    check("the wrist columns are zero after normalising", np.allclose(ds.X[:, :3], 0))
    check("hand size is one after normalising", np.allclose(np.linalg.norm(ds.X[:, 27:29], axis=1), 1.0, atol=0.05))
    check("swipe clips are left out", "swipe_left" not in set(ds.y.tolist()))
    check("the labels come from the file names", set(ds.y.tolist()) == {"fist", "open_palm"})
    check("a still clip keeps all its frames", int(np.sum((ds.user == "u01") & (ds.y == "fist") & (ds.clip == 1))) == 40)
    check("a clip with an ERROR is skipped", ("u03/fist.json", 1) in stats["clips_skipped"]
          and int(np.sum((ds.user == "u03") & (ds.clip == 1))) == 0)
    check("the good clip of the same user is still used", int(np.sum((ds.user == "u03") & (ds.clip == 2))) == 40)

    moving_rows = (ds.user == "u02") & (ds.y == "open_palm")
    frames = ds.frame[moving_rows]
    check("moving frames are dropped", moving_rows.sum() < 40 and frames.max() < 20)
    check("the calm frames before the movement are kept", frames.min() == 0 and moving_rows.sum() >= 15)
    check("conditions are carried along", set(ds.lighting[(ds.user == "u01") & (ds.clip == 2)].tolist()) == {"dim"})
    check("stats count the frames", stats["frames_total"] > stats["frames_kept"] > 0)

    # Feature options.
    check("without z: 42 numbers", dl.load_static(tmp, keep_z=False)[0].X.shape[1] == 42)
    check("without the wrist: 60 numbers", dl.load_static(tmp, drop_wrist=True)[0].X.shape[1] == 60)
    check("without both: 40 numbers", dl.load_static(tmp, keep_z=False, drop_wrist=True)[0].X.shape[1] == 40)
    raw, _ = dl.load_static(tmp, smooth=False)
    check("smoothing changes the numbers a little but not the shape",
          raw.X.shape == ds.X.shape and not np.allclose(raw.X, ds.X) and np.abs(raw.X - ds.X).max() < 0.1)
    check("only the chosen users are loaded", set(dl.load_static(tmp, users=["u02"])[0].user.tolist()) == {"u02"})

    # Split by user.
    train, val, test = dl.split(ds, train=["u01"], val=["u02"], test=["u03"])
    check("split keeps each user in one set", set(train.user.tolist()) == {"u01"}
          and set(val.user.tolist()) == {"u02"} and set(test.user.tolist()) == {"u03"})
    check("split loses no rows", len(train) + len(val) + len(test) == len(ds))
    try:
        dl.split(ds, train=["u01", "u02"], val=["u02"])
        overlap_caught = False
    except ValueError:
        overlap_caught = True
    check("a user in two sets is refused", overlap_caught)
    try:
        dl.split(ds, train=["u01"], test=["u99"])
        unknown_caught = False
    except ValueError:
        unknown_caught = True
    check("a user without data is refused", unknown_caught)

    folds = list(dl.leave_one_user_out(ds))
    check("leave-one-user-out gives one fold per user", [u for u, _, _ in folds] == ["u01", "u02", "u03"])
    check("each test fold holds only its user and the training part never does",
          all(set(te.user.tolist()) == {u} and u not in set(tr.user.tolist()) for u, tr, te in folds))

    # Other feature sets (hand_features.py): same rows, other columns.
    import hand_features as hf
    inv, _ = dl.load_static(tmp, features="invariant+orient")
    check("feature set option keeps the same rows", len(inv) == len(ds) and np.array_equal(inv.frame, ds.frame))
    check("feature set option gives the hand_features columns",
          inv.X.shape[1] == len(hf.feature_names("invariant+orient"))
          and np.allclose(inv.X, hf.compute(hf.unflatten(ds.X), "invariant+orient"), equal_nan=True))
    try:
        dl.load_static(tmp, features="pixels")
        bad_name_caught = False
    except ValueError:
        bad_name_caught = True
    check("an unknown feature set is refused", bad_name_caught)

    # Whole clips (load_clips), used by calibration and the frame-by-frame experiments.
    unchecked = dl.load_clips(tmp, validate=False)       # the made-up swipe clip does not move: validator ERROR
    check("load_clips: static and dynamic clips, each with its landmarks and times",
          {"swipe_left", "fist"} <= {c["gesture"] for c in unchecked}
          and all(c["landmarks"].shape == (len(c["times"]), 21, 3) for c in unchecked))
    all_clips = dl.load_clips(tmp)
    skipped = {(f, k) for f, k in stats["clips_skipped"]}
    check("load_clips: skips the same validator ERROR clips as load_static",
          not any((c["file"], c["clip"]) in skipped for c in all_clips))
    check("load_clips: user and gesture filters",
          {(c["user"], c["gesture"]) for c in dl.load_clips(tmp, users=["u01"], gestures=["fist"])} == {("u01", "fist")})

with tempfile.TemporaryDirectory() as empty:
    ds0, st0 = dl.load_static(empty)
    check("an empty folder gives an empty table", len(ds0) == 0 and ds0.X.shape == (0, 63))

print("\nAll checks passed.")
