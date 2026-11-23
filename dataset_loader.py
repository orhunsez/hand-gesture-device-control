"""Turn the recorded clips into the tables a classifier is trained on (static gestures).

    .venv\\Scripts\\python.exe dataset_loader.py        # print what would be loaded

In code:
    import dataset_loader as dl
    ds, stats = dl.load_static()
    train, val, test = dl.split(ds, train=["u01", "u02"], val=["u03"], test=["u04"])
    model.fit(train.X, train.y)

What happens to every clip of a static gesture (and of no_gesture, whose stable frames are
the negative examples of the static classifier):
  1. The clip is skipped if validate_dataset.py finds an ERROR in it.
  2. The landmarks are smoothed (landmark_smoothing.py, raw landmarks first), then normalised
     (landmark_features.py). The same two steps are used by the live system.
  3. Only STABLE frames are kept: the hand is present and, in a window of 5 frames around the
     frame, the palm centre barely moves (speed) and the hand shape barely changes (pose change).
     Frames while the hand moves into or out of a pose are dropped.
  4. Every kept frame becomes one row of X (63 numbers: 21 points x, y, z), with its label and
     the user, clip, frame and recording condition it came from.
Dynamic gestures (swipes) are not handled here; they need the segmentation (4.5).

The stability window uses frames after the current one, which is fine for building training data
from recorded clips. The live system can only use past frames (a hold time, 4.5 and 4.9).
The two thresholds are PROVISIONAL, chosen on pilot clips of one person: with the hand held
still the palm speed stayed under 0.18 hand sizes per second and the pose change under 0.010
per frame; the thresholds are about twice that.

Rules for the split: the unit is the USER, never the frame or the clip, because frames of one clip
are nearly identical and would leak into the test set. Choose thresholds and models on the
training and validation users only; use the test users once, at the end.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import dataset_config as cfg
import hand_features as hf
import landmark_features as lf
import landmark_smoothing as ls
import validate_dataset as vd

DATA_DIR = Path("data/landmarks")
PALM = [0, 5, 9, 13, 17]
V_MAX = 0.3          # palm centre speed, hand sizes per second (provisional)
POSE_MAX = 0.02      # mean landmark change per frame, hand sizes (provisional)
WINDOW = 5           # frames that must all be calm around a stable frame


@dataclass
class Dataset:
    """Rows are samples. All arrays have the same length N."""
    X: np.ndarray            # (N, number of features)
    y: np.ndarray            # gesture name
    user: np.ndarray         # u01, u02, ...
    clip: np.ndarray         # clip number inside its file
    file: np.ndarray         # e.g. u99/fist.json
    frame: np.ndarray        # frame index inside the clip
    background: np.ndarray
    lighting: np.ndarray
    distance: np.ndarray

    def __len__(self):
        return len(self.y)

    def select(self, mask):
        mask = np.asarray(mask)
        return Dataset(**{name: getattr(self, name)[mask] for name in self.__dataclass_fields__})


_FIELDS = ["X", "y", "user", "clip", "file", "frame", "background", "lighting", "distance"]


def _empty(n_features):
    return Dataset(X=np.zeros((0, n_features)), y=np.array([], dtype=object),
                   user=np.array([], dtype=object), clip=np.array([], dtype=int),
                   file=np.array([], dtype=object), frame=np.array([], dtype=int),
                   background=np.array([], dtype=object), lighting=np.array([], dtype=object),
                   distance=np.array([], dtype=object))


def motion(smoothed, times, width=None, height=None):
    """Palm-centre speed (hand sizes per second) and pose change (mean landmark change in hand sizes)
    for every frame of a clip, both shape (T,).

    Frame i takes the change from frame i-1; the first frame copies the second. NaN where the frame or
    the one before it has no hand. The speed is divided by the median hand size of the clip.
    Needs T >= 2 and at least one frame with a hand.
    """
    lm = np.asarray(smoothed, dtype=np.float64)
    px = lf.to_pixel_units(lm, width, height)
    size = np.nanmedian(lf.hand_size(lm, width, height))
    centre = np.nanmean(px[:, PALM, :2], axis=1)
    q = lf.normalize(lm, width, height)
    dt = np.diff(times)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.linalg.norm(np.diff(centre, axis=0), axis=1) / dt / size
        pose = np.nanmean(np.linalg.norm(np.diff(q, axis=0), axis=2), axis=1)
    speed = np.concatenate([[speed[0]], speed])
    pose = np.concatenate([[pose[0]], pose])
    return speed, pose


def stable_frames(smoothed, times, width=None, height=None, v_max=V_MAX, pose_max=POSE_MAX,
                  window=WINDOW):
    """Boolean mask (T,) of frames where the hand is present and calm.

    smoothed: (T, 21, 3) smoothed landmarks in the stored units; times: (T,) seconds.
    """
    lm = np.asarray(smoothed, dtype=np.float64)
    T = len(lm)
    present = ~np.isnan(lm[:, 0, 0])
    if T < 2 or present.sum() < 2:
        return np.zeros(T, dtype=bool)
    speed, pose = motion(lm, times, width, height)
    calm = ~((speed > v_max) | (pose > pose_max) | np.isnan(speed) | np.isnan(pose)) & present
    moving = (~calm).astype(int)
    return np.convolve(moving, np.ones(window, dtype=int), mode="same") == 0


def load_static(root=DATA_DIR, users=None, smooth=True, keep_z=True, drop_wrist=False,
                v_max=V_MAX, pose_max=POSE_MAX, validate=True, features="raw63"):
    """Load all static clips. Returns (Dataset, stats).

    users      : only these user ids (default all).
    smooth     : apply the 1-euro filter before normalising.
    keep_z     : False leaves the depth column out (z is the least reliable).
    drop_wrist : True leaves the wrist out (it is always 0 after normalising).
    features   : "raw63" (default) = the normalised coordinates, shaped by keep_z and drop_wrist;
                 any other name from hand_features.FEATURE_SETS computes those features instead
                 (keep_z then chooses 3D or picture-plane angles and distances; drop_wrist is ignored).
    validate   : skip clips that validate_dataset.py marks as ERROR.
    stats      : clips used, clips skipped, frames seen and kept per gesture.
    """
    root = Path(root)
    bad = set()
    if validate:
        issues, _ = vd.validate(root)
        bad = vd.bad_clips(issues)

    stats = {"clips_used": 0, "clips_skipped": [], "frames_total": 0, "frames_kept": 0,
             "per_gesture": {}}
    parts = {name: [] for name in _FIELDS}
    if features != "raw63" and features not in hf.FEATURE_SETS:
        raise ValueError(f"unknown feature set {features!r}")

    def to_rows(q):
        if features == "raw63":
            return lf.flatten(q, drop_wrist, keep_z)
        return hf.compute(q, features, use_z=keep_z)

    n_features = to_rows(np.ones((1, 21, 3))).shape[1]

    for path in sorted(root.glob("*/*.json")):
        user, gesture = path.parent.name, path.stem
        if users is not None and user not in users:
            continue
        if cfg.GESTURES.get(gesture) in (None, "dynamic"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        file_name = f"{user}/{path.name}"
        for clip in data["clips"]:
            k = clip["clip"]
            if (file_name, k) in bad or (file_name, None) in bad:
                stats["clips_skipped"].append((file_name, k))
                continue
            lm, t = vd.clip_array(clip)
            w, h = clip.get("image_width"), clip.get("image_height")
            sm = ls.smooth_clip(lm, t, w, h) if smooth else lm
            keep = stable_frames(sm, t, w, h, v_max, pose_max)
            g = stats["per_gesture"].setdefault(gesture, [0, 0])
            g[0] += len(t)
            g[1] += int(keep.sum())
            stats["frames_total"] += len(t)
            stats["frames_kept"] += int(keep.sum())
            stats["clips_used"] += 1
            if not keep.any():
                continue
            X = to_rows(lf.normalize(sm[keep], w, h))
            n = len(X)
            parts["X"].append(X)
            parts["y"].append(np.full(n, gesture, dtype=object))
            parts["user"].append(np.full(n, user, dtype=object))
            parts["clip"].append(np.full(n, k, dtype=int))
            parts["file"].append(np.full(n, file_name, dtype=object))
            parts["frame"].append(np.flatnonzero(keep))
            for name in ("background", "lighting", "distance"):
                parts[name].append(np.full(n, clip[name], dtype=object))

    if not parts["X"]:
        return _empty(n_features), stats
    return Dataset(**{name: np.concatenate(parts[name]) for name in _FIELDS}), stats


def load_clips(root=DATA_DIR, users=None, gestures=None, validate=True):
    """Every recorded clip (static and dynamic) as a dict, for code that works on whole clips
    (calibration, the frame-by-frame pipeline, experiments E3 and E4).

    Keys: user, gesture, clip, file, landmarks (T, 21, 3; raw stored values, NaN = no hand), times (T,),
    width, height (None for old clips), background, lighting, distance.
    users / gestures: only these (default all). validate: skip clips with a validator ERROR.
    """
    root = Path(root)
    bad = set()
    if validate:
        issues, _ = vd.validate(root)
        bad = vd.bad_clips(issues)
    out = []
    for path in sorted(root.glob("*/*.json")):
        user, gesture = path.parent.name, path.stem
        if (users is not None and user not in users) or (gestures is not None and gesture not in gestures):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        file_name = f"{user}/{path.name}"
        for clip in data["clips"]:
            k = clip["clip"]
            if (file_name, k) in bad or (file_name, None) in bad:
                continue
            lm, t = vd.clip_array(clip)
            out.append({"user": user, "gesture": gesture, "clip": k, "file": file_name,
                        "landmarks": lm, "times": t,
                        "width": clip.get("image_width"), "height": clip.get("image_height"),
                        "background": clip["background"], "lighting": clip["lighting"],
                        "distance": clip["distance"]})
    return out


def split(ds, train, val=(), test=()):
    """Divide a Dataset by user. Returns (train, val, test) Datasets.

    Raises ValueError if a user is in two sets or a listed user has no data.
    """
    sets = {"train": list(train), "val": list(val), "test": list(test)}
    seen = {}
    for name, members in sets.items():
        for u in members:
            if u in seen:
                raise ValueError(f"user {u} is in both '{seen[u]}' and '{name}'")
            seen[u] = name
    present = set(ds.user.tolist())
    unknown = sorted(u for u in seen if u not in present)
    if unknown:
        raise ValueError(f"no data for user(s): {', '.join(unknown)}")
    return tuple(ds.select(np.isin(ds.user, members)) for members in sets.values())


def leave_one_user_out(ds):
    """Yield (user, train, test): each user in turn is the test set, all others train."""
    for u in sorted(set(ds.user.tolist())):
        yield u, ds.select(ds.user != u), ds.select(ds.user == u)


def main():
    ds, stats = load_static()
    print(f"clips used: {stats['clips_used']}, skipped because of errors: {stats['clips_skipped'] or 'none'}")
    print(f"frames seen: {stats['frames_total']}, kept as stable: {stats['frames_kept']} "
          f"({100 * stats['frames_kept'] / max(1, stats['frames_total']):.0f}%)")
    print(f"X: {ds.X.shape}, users: {sorted(set(ds.user.tolist()))}")
    print("\nsamples per gesture and user (kept / seen frames per gesture in brackets):")
    users = sorted(set(ds.user.tolist()))
    print(f"{'gesture':<12} " + " ".join(f"{u:>6}" for u in users) + "   kept/seen")
    for g in sorted(stats["per_gesture"]):
        row = [int(np.sum((ds.y == g) & (ds.user == u))) for u in users]
        kept, seen = stats["per_gesture"][g][1], stats["per_gesture"][g][0]
        print(f"{g:<12} " + " ".join(f"{n:>6}" for n in row) + f"   {kept}/{seen}")
    missing = [g for g, kind in cfg.GESTURES.items() if kind != "dynamic" and g not in stats["per_gesture"]]
    if missing:
        print(f"\nstatic gestures with no data yet: {', '.join(missing)}")


if __name__ == "__main__":
    main()
