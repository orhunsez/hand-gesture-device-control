"""Checks for landmark_features.py. Run:  .venv\\Scripts\\python.exe test_landmark_features.py"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import numpy as np

import landmark_features as lf

rng = np.random.default_rng(0)
W, H = 640, 480


def random_hand():
    """A made-up hand in normalised image coordinates (x, y in 0..1, z small)."""
    base = rng.uniform(0.3, 0.6, size=(21, 3))
    base[:, 2] = rng.uniform(-0.05, 0.05, size=21)
    base[0, 2] = 0.0
    return base


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


hand = random_hand()
q = lf.normalize(hand, W, H)

# 1. The wrist is the origin and the hand size is 1.
check("wrist is the origin", np.allclose(q[0], 0))
size_after = np.linalg.norm(q[9, :2] - q[0, :2])
check("hand size is 1 after normalisation", np.isclose(size_after, 1.0))

# 2. Translation invariance: moving the hand in the picture changes nothing.
moved = hand.copy()
moved[:, 0] += 0.2
moved[:, 1] -= 0.1
check("translation invariant", np.allclose(lf.normalize(moved, W, H), q))

# 3. Scale invariance: a hand twice as far from the camera (half the size around its
#    own wrist) gives the same numbers.
far = hand.copy()
far[:, :3] = hand[0, :3] + 0.5 * (hand[:, :3] - hand[0, :3])
check("scale invariant", np.allclose(lf.normalize(far, W, H), q))

# 4. Rotation is NOT removed (thumb up and thumb down must stay different).
theta = np.pi / 2
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
turned = hand.copy()
turned[:, :2] = (hand[:, :2] - hand[0, :2]) @ rot.T + hand[0, :2]
check("rotation is kept (not invariant)", not np.allclose(lf.normalize(turned, W, H), q))

# 5. Aspect ratio: the same physical hand, rotated by 90 degrees in PIXEL space, keeps
#    its hand size once the image size is used.
px = np.zeros((21, 3))
px[:, :2] = rng.uniform(100, 300, size=(21, 2))           # a hand drawn in pixels
norm_in = px.copy()
norm_in[:, 0] /= W
norm_in[:, 1] /= H
size_px = np.linalg.norm(px[9, :2] - px[0, :2])
check("hand size equals the pixel distance when the image size is given",
      np.isclose(lf.hand_size(norm_in, W, H), size_px))
check("without the image size the default 640x480 is used",
      np.isclose(lf.hand_size(norm_in), size_px))

# 6. A clip of many frames, with a frame that has no hand.
clip = np.stack([random_hand() for _ in range(5)])
clip[2] = np.nan
qc = lf.normalize(clip, W, H)
check("clip shape is kept", qc.shape == clip.shape)
check("a frame without a hand stays NaN", np.isnan(qc[2]).all())
check("other frames are normal", not np.isnan(qc[[0, 1, 3, 4]]).any())
check("clip frames match single-frame results", np.allclose(qc[1], lf.normalize(clip[1], W, H)))

# 7. A degenerate hand (wrist on the middle finger base) gives NaN, not a crash or inf.
flat = random_hand()
flat[9] = flat[0]
with np.errstate(all="raise"):
    out = lf.normalize(flat, W, H)
check("degenerate hand gives NaN", np.isnan(out).all())

# 8. flatten
check("flatten one frame", lf.flatten(q).shape == (63,))
check("flatten a clip", lf.flatten(qc).shape == (5, 63))
check("flatten without wrist and z", lf.flatten(qc, drop_wrist=True, keep_z=False).shape == (5, 40))

print("\nAll checks passed.")
