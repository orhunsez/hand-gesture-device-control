"""Checks for hand_features.py. Run:  .venv\\Scripts\\python.exe test_hand_features.py"""
import numpy as np

import hand_features as hf
import landmark_features as lf

rng = np.random.default_rng(1)
W, H = 640, 480


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def random_hand():
    """A made-up hand in stored units (x, y in 0..1 of the picture, z small)."""
    base = rng.uniform(0.3, 0.6, size=(21, 3))
    base[:, 2] = rng.uniform(-0.05, 0.05, size=21)
    base[0, 2] = 0.0
    return base


Q = lf.normalize(random_hand(), W, H)
clipQ = np.stack([lf.normalize(random_hand(), W, H) for _ in range(6)])

# 1. Shapes and names.
for name, n in (("raw63", 63), ("invariant", 28), ("invariant+orient", 29)):
    check(f"{name}: {n} numbers for one frame and for a clip",
          hf.compute(Q, name).shape == (n,) and hf.compute(clipQ, name).shape == (6, n)
          and len(hf.feature_names(name)) == n)
check("invariant+orient = invariant plus one column",
      np.allclose(hf.compute(clipQ, "invariant+orient")[:, :-1], hf.compute(clipQ, "invariant")))
check("use_z=False gives the same number of columns",
      hf.compute(clipQ, "invariant", use_z=False).shape == (6, 28))
try:
    hf.compute(Q, "pixels")
    caught = False
except ValueError:
    caught = True
check("an unknown feature set is refused", caught)

# 2. Turning the hand in the picture plane: invariant features stay, the others change.
angles = rng.uniform(-np.pi, np.pi, size=6)
turned = hf.rotate(clipQ, angles)
for use_z in (True, False):
    check(f"invariant features unchanged by rotation (use_z={use_z})",
          np.allclose(hf.compute(turned, "invariant", use_z), hf.compute(clipQ, "invariant", use_z)))
check("raw63 changes with rotation (as expected)",
      not np.allclose(hf.compute(turned, "raw63"), hf.compute(clipQ, "raw63")))
upside_down = hf.rotate(clipQ, np.pi)
check("thumb direction flips sign when the hand is turned upside down",
      np.allclose(hf.compute(upside_down)[:, -1], -hf.compute(clipQ)[:, -1]))

# 3. Position and size in the picture do not matter (through the normalisation).
raw = random_hand()
moved_and_far = raw.copy()
moved_and_far[:, :3] = raw[0] + 0.5 * (raw - raw[0]) + [0.1, -0.05, 0.0]
check("invariant features unchanged by moving the hand and halving its size",
      np.allclose(hf.compute(lf.normalize(raw, W, H)), hf.compute(lf.normalize(moved_and_far, W, H))))

# 4. Joint bend angles on a known chain.
straight = np.zeros((21, 3))
for chain in hf.CHAINS.values():
    for j, lm in enumerate(chain[1:], start=1):
        straight[lm] = [0.0, -0.3 * j, 0.0]            # every finger a straight line upwards
check("a straight finger has bend 0", np.allclose(hf.joint_bends(straight), 0, atol=1e-6))
bent = straight.copy()
bent[[7, 8]] = straight[6] + np.array([[0.3, 0, 0], [0.6, 0, 0]])   # index turns 90 degrees at the PIP
check("a right angle at the index PIP gives pi/2 there", np.isclose(hf.joint_bends(bent)[4], np.pi / 2))
check("thumb pointing straight up gives direction (0, -1)", np.allclose(hf.thumb_direction(straight), [0, -1]))

# 5. Tilt: 0 for an upright hand, positive towards picture-right; rotate() turns clockwise on screen.
check("an upright hand has tilt 0", np.isclose(hf.tilt_deg(straight), 0.0))
check("rotate(+30 deg) gives tilt +30", np.isclose(hf.tilt_deg(hf.rotate(straight, np.radians(30))), 30.0))
check("rotate back and forth is the identity", np.allclose(hf.rotate(hf.rotate(clipQ, angles), -angles), clipQ))
check("the wrist stays at the origin", np.allclose(hf.rotate(clipQ, angles)[:, 0], 0))

# 6. Rotation augmentation.
y = np.array(["fist", "thumb_up", "open_palm", "thumb_down", "point", "no_gesture"])
Qa, src = hf.augment_rotation(clipQ, y, copies=3, max_deg=60, seed=0)
check("originals come first and unchanged", np.array_equal(Qa[:6], clipQ) and np.array_equal(src[:6], np.arange(6)))
check("3 copies of each orientation-free frame, none of thumb up / down",
      len(Qa) == 6 + 3 * 4 and set(y[src[6:]].tolist()) == {"fist", "open_palm", "point", "no_gesture"})
check("each copy is a turned version of its source frame",
      np.allclose(hf.compute(Qa[6:], "invariant"), hf.compute(clipQ[src[6:]], "invariant")))
copy_tilt = hf.tilt_deg(Qa[6:]) - hf.tilt_deg(clipQ[src[6:]])
copy_tilt = (copy_tilt + 180) % 360 - 180
check("copies are turned by at most max_deg", np.all(np.abs(copy_tilt) <= 60 + 1e-6))
Qn, srcn = hf.augment_rotation(clipQ, y, copies=0)
check("copies=0 adds nothing", len(Qn) == 6 and np.array_equal(srcn, np.arange(6)))

# 7. Frames without a hand stay NaN, without warnings or crashes.
clipQ[2] = np.nan
with np.errstate(all="raise"):
    F = hf.compute(clipQ)
check("a frame without a hand gives NaN features", np.isnan(F[2]).all())
check("other frames are normal", not np.isnan(F[[0, 1, 3, 4, 5]]).any())
check("unflatten undoes flatten", np.allclose(hf.unflatten(lf.flatten(clipQ)), clipQ, equal_nan=True))

print("\nAll checks passed.")
