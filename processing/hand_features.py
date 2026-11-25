"""Features of one hand pose for the static classifier (report sections 4.3 and 4.6).

Input is always NORMALISED landmarks (landmark_features.normalize): shape (21, 3) for one frame or
(N, 21, 3) for many, wrist at the origin, hand size 1, x to the right and y DOWN in the picture.

Why this file exists (pilot finding, docs/results/static_classifier_pilot_2026-11-21.txt): an SVM
on the 63 normalised coordinates learned the TILT of the hand, because only the thumb-up clips were
tilted in the training data, and it called tilted open palms "thumb up". Normalisation removes
position and size but deliberately keeps rotation (thumb up and thumb down differ only by it).
The fix is to describe the finger configuration with numbers that do not change when the hand is
turned in the picture plane, and to keep exactly one number for the orientation where it matters.

Feature sets (compute(Q, name)):
    raw63             the 63 normalised coordinates (what train_static.py used first). Depends on rotation.
    invariant         15 joint bend angles + 13 distances. Unchanged by rotation in the picture plane,
                      by position and by hand size.
    invariant+orient  invariant + the vertical direction of the thumb (thumb_up vs thumb_down).

Joint bend = 180 degrees minus the angle between the two bones at a joint, in radians: 0 for a
straight finger, larger the more it is bent. Distances are in hand sizes. By default both use x, y
and z; use_z=False uses the picture plane only (z is the least reliable coordinate).

augment_rotation() turns training frames in the picture plane for the gestures whose meaning does
not depend on orientation (fist, open palm, point, no gesture), so a learned model cannot tie a tilt
to a class. Thumb up and thumb down are never turned.
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import numpy as np

import landmark_features as lf

# MediaPipe landmark numbers, from the wrist to the fingertip.
CHAINS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}
FINGERS = list(CHAINS)
TIPS = [4, 8, 12, 16, 20]
THUMB_MCP, THUMB_TIP, INDEX_MCP = 2, 4, 5

# Pairs of landmarks whose distance is a feature: fingertip to wrist (5), neighbouring fingertips (4),
# thumb tip to the other fingertips not yet covered (3), thumb tip to the base of the index finger (1,
# small when the thumb is folded over the fingers as in a fist).
DIST_PAIRS = ([(0, t) for t in TIPS]
              + [(4, 8), (8, 12), (12, 16), (16, 20)]
              + [(4, 12), (4, 16), (4, 20)]
              + [(THUMB_TIP, INDEX_MCP)])

FEATURE_SETS = ("raw63", "invariant", "invariant+orient")

# Gestures whose meaning does not depend on how the hand is turned in the picture.
ORIENTATION_FREE = {"fist", "open_palm", "point", "no_gesture"}


def _cols(use_z):
    return slice(0, 3) if use_z else slice(0, 2)


def joint_bends(Q, use_z=True):
    """Bend angle (radians) at the 3 joints of each finger, shape (..., 15).

    Order: thumb (CMC, MCP, IP), index (MCP, PIP, DIP), middle, ring, pinky.
    NaN for a frame without a hand or a joint whose bones have zero length.
    """
    P = np.asarray(Q, dtype=np.float64)[..., _cols(use_z)]
    out = []
    for chain in CHAINS.values():
        for j in range(1, 4):
            a = P[..., chain[j - 1], :] - P[..., chain[j], :]
            b = P[..., chain[j + 1], :] - P[..., chain[j], :]
            with np.errstate(invalid="ignore", divide="ignore"):
                cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
            out.append(np.pi - np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.stack(out, axis=-1)


def distances(Q, use_z=True):
    """Distances (hand sizes) between the landmark pairs in DIST_PAIRS, shape (..., 13)."""
    P = np.asarray(Q, dtype=np.float64)[..., _cols(use_z)]
    return np.stack([np.linalg.norm(P[..., i, :] - P[..., j, :], axis=-1) for i, j in DIST_PAIRS], axis=-1)


def thumb_direction(Q):
    """Unit vector (x, y) in the picture from the thumb base (MCP) to the thumb tip, shape (..., 2).

    y points down, so a thumb pointing straight up gives (0, -1) and straight down (0, +1).
    """
    v = np.asarray(Q, dtype=np.float64)[..., THUMB_TIP, :2] - np.asarray(Q, dtype=np.float64)[..., THUMB_MCP, :2]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n < lf.EPS, np.nan, v / n)


def tilt_deg(Q):
    """Hand tilt in degrees: direction wrist -> middle-finger base, 0 = straight up, positive = towards
    picture-right. Same quantity as in the pilot diagnosis (docs/results). For reporting, not a feature."""
    v = np.asarray(Q, dtype=np.float64)[..., lf.MIDDLE_MCP, :2] - np.asarray(Q, dtype=np.float64)[..., lf.WRIST, :2]
    return np.degrees(np.arctan2(v[..., 0], -v[..., 1]))


def compute(Q, name="invariant+orient", use_z=True):
    """Feature vectors, one row per frame: (21, 3) -> (n,), (N, 21, 3) -> (N, n)."""
    Q = np.asarray(Q, dtype=np.float64)
    if name == "raw63":
        return lf.flatten(Q)
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {name!r}; choose one of {FEATURE_SETS}")
    parts = [joint_bends(Q, use_z), distances(Q, use_z)]
    if name == "invariant+orient":
        parts.append(thumb_direction(Q)[..., 1:2])
    return np.concatenate(parts, axis=-1)


def feature_names(name="invariant+orient"):
    if name == "raw63":
        return [f"{c}{i}" for i in range(21) for c in "xyz"]
    joints = {"thumb": ["cmc", "mcp", "ip"]}
    names = [f"bend_{f}_{j}" for f in FINGERS for j in joints.get(f, ["mcp", "pip", "dip"])]
    names += [f"dist_{i}_{j}" for i, j in DIST_PAIRS]
    if name == "invariant+orient":
        names.append("thumb_dir_y")
    return names


def unflatten(X):
    """Undo landmark_features.flatten for the default 63 columns: (N, 63) -> (N, 21, 3)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[-1] != 63:
        raise ValueError("unflatten needs the 63-column layout (wrist and z kept)")
    return X.reshape(*X.shape[:-1], 21, 3)


def rotate(Q, angle_rad):
    """Turn normalised landmarks in the picture plane around the wrist (the origin).

    angle_rad: a number, or one angle per frame (shape (N,)). Positive turns the hand clockwise on
    the screen (x right, y down). z is unchanged.
    """
    Q = np.asarray(Q, dtype=np.float64)
    a = np.asarray(angle_rad, dtype=np.float64)[..., None]
    c, s = np.cos(a), np.sin(a)
    out = Q.copy()
    out[..., 0] = c * Q[..., 0] - s * Q[..., 1]
    out[..., 1] = s * Q[..., 0] + c * Q[..., 1]
    return out


def augment_rotation(Q, y, copies=3, max_deg=90.0, rotatable=ORIENTATION_FREE, seed=0):
    """Add `copies` randomly turned versions (uniform in +-max_deg) of every frame whose label is in
    `rotatable`. Use on TRAINING data only.

    Returns (Q_new, src): the originals first, then the turned copies; src[i] is the index of the
    original frame that row i comes from, so labels and clip ids follow with y[src], groups[src].
    """
    Q, y = np.asarray(Q, dtype=np.float64), np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(np.isin(y, list(rotatable)))
    src = np.arange(len(Q))
    if copies <= 0 or len(idx) == 0:
        return Q, src
    rep = np.repeat(idx, copies)
    turned = rotate(Q[rep], np.radians(rng.uniform(-max_deg, max_deg, size=len(rep))))
    return np.concatenate([Q, turned]), np.concatenate([src, rep])
