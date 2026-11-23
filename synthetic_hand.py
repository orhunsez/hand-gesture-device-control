"""Made-up hands and clips for the checks (test_*.py). Not used by the system itself.

make_hand() builds a simple 3D hand skeleton in normalised units (wrist at the origin, hand size 1,
y down). A bent finger curls towards the camera (negative z) and back down, as in a real fist, so
its picture-plane projection is short. This lets the checks use all five gestures, including point,
which has not been recorded yet. (Moved here from test_static_classifiers.py unchanged, plus the
`curl` option, so that several test files can use the same hand.)

make_clip() turns normalised poses into a clip in the STORED units (x, y divided by the image size,
as MediaPipe gives them), with frame times, jitter and an optional movement of the whole hand.
"""
import numpy as np

import hand_features as hf

rng = np.random.default_rng(7)

MCP = {"index": (-0.30, -0.95), "middle": (0.0, -1.0), "ring": (0.25, -0.95), "pinky": (0.45, -0.85)}
BONES = {"index": (0.45, 0.28, 0.22), "middle": (0.50, 0.30, 0.24), "ring": (0.45, 0.28, 0.22),
         "pinky": (0.36, 0.22, 0.18)}
STRAIGHT, CURLED = (0.05, 0.05, 0.05), (1.4, 1.6, 1.0)        # joint bends in radians
THUMB_OUT = np.array([-0.8, -0.6]) / np.linalg.norm([-0.8, -0.6])

SHAPES = {
    "open_palm": dict(fingers="11111"),
    "fist": dict(fingers="00000"),
    "thumb_up": dict(fingers="10000", thumb_up_angle=0),
    "thumb_down": dict(fingers="10000", thumb_up_angle=180),
    "point": dict(fingers="01000"),
}


def make_hand(fingers="11111", thumb_up_angle=None, tilt_deg=0.0, noise=0.0, curl=None):
    """fingers: thumb, index, middle, ring, pinky as '1' extended / '0' bent.

    thumb_up_angle: turn the whole hand so the extended thumb points this many degrees from straight
    up (0 = up, 180 = down, 90 = sideways). Otherwise the hand is turned by tilt_deg.
    curl: joint bends (3 numbers, radians) of a bent finger; default CURLED (a tight fist).
    """
    bent = CURLED if curl is None else curl
    P = np.zeros((21, 3))
    for k, name in enumerate(["index", "middle", "ring", "pinky"], start=1):
        chain = hf.CHAINS[name]
        base = np.array([*MCP[name], 0.0])
        P[chain[1]] = base
        d0 = base / np.linalg.norm(base)
        theta, pos = 0.0, base
        for j, (bone, bend) in enumerate(zip(BONES[name], STRAIGHT if fingers[k] == "1" else bent)):
            theta += bend
            d = np.cos(theta) * d0 + np.sin(theta) * np.array([0.0, 0.0, -1.0])
            pos = pos + bone * d
            P[chain[j + 2]] = pos
    P[1] = [-0.35, -0.30, 0.0]
    P[2] = [-0.55, -0.55, 0.0]
    if fingers[0] == "1":
        P[3] = [*(P[2, :2] + 0.30 * THUMB_OUT), 0.0]
        P[4] = [*(P[3, :2] + 0.25 * THUMB_OUT), 0.0]
    else:                                           # folded over the fingers
        P[3] = [-0.35, -0.75, -0.10]
        P[4] = [-0.10, -0.85, -0.20]
    P = P + rng.normal(0, noise, size=P.shape) * (np.arange(21) > 0)[:, None]
    if thumb_up_angle is not None:
        current = np.arctan2(THUMB_OUT[1], THUMB_OUT[0])
        target = np.arctan2(-np.cos(np.radians(thumb_up_angle)), np.sin(np.radians(thumb_up_angle)))
        return hf.rotate(P, target - current)
    return hf.rotate(P, np.radians(tilt_deg))


def make_clip(Q, n=40, fps=20.0, size_px=100.0, wrist_px=(320.0, 330.0), jitter_px=0.3,
              shift_x=None, missing=(), width=640, height=480, seed=0):
    """A clip in stored units: returns (landmarks (n, 21, 3), times (n,)).

    Q        : one normalised pose (21, 3) held for the whole clip, or one pose per frame (n, 21, 3).
    size_px  : hand size in pixels; wrist_px: wrist position in pixels at the start.
    jitter_px: standard deviation of the noise added to every coordinate of every frame, in pixels.
    shift_x  : optional (n,) horizontal movement of the whole hand, in HAND SIZES (e.g. a swipe).
    missing  : frame numbers without a hand (NaN).
    """
    g = np.random.default_rng(seed)
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim == 2:
        Q = np.repeat(Q[None], n, axis=0)
    px = Q * size_px
    px[..., 0] += wrist_px[0]
    px[..., 1] += wrist_px[1]
    if shift_x is not None:
        px[..., 0] += np.asarray(shift_x, dtype=np.float64)[:, None] * size_px
    px = px + g.normal(0, jitter_px, size=px.shape)
    lm = px / np.array([width, height, width], dtype=np.float64)
    lm[list(missing)] = np.nan
    times = 0.05 + np.arange(n) / fps
    return lm, times


def swipe_path(n=60, fps=20.0, start_s=0.6, out_s=0.35, back_s=0.35, amplitude=1.2, sign=+1):
    """Horizontal palm path of an out-and-back swipe in hand sizes, shape (n,): still, out, back, still.

    sign +1: x increases first (swipe_left in the stored, not mirrored, picture), -1: decreases first.
    """
    t = np.arange(n) / fps
    x = np.zeros(n)
    out = (t >= start_s) & (t < start_s + out_s)
    back = (t >= start_s + out_s) & (t < start_s + out_s + back_s)
    x[out] = amplitude * 0.5 * (1 - np.cos(np.pi * (t[out] - start_s) / out_s))
    x[back] = amplitude * 0.5 * (1 + np.cos(np.pi * (t[back] - start_s - out_s) / back_s))
    return sign * x
