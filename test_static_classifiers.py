"""Checks for rule_classifier.py and train_static.LandmarkSVM on made-up hands.

    .venv\\Scripts\\python.exe test_static_classifiers.py

make_hand() builds a simple 3D hand skeleton in normalised units (wrist at the origin, hand size 1,
y down). A bent finger curls towards the camera (negative z) and back down, as in a real fist, so
its picture-plane projection is short. This lets the rules be checked on all five gestures,
including point, which has not been recorded yet.
"""
import numpy as np

import hand_features as hf
import rule_classifier as rc
import train_static as ts

rng = np.random.default_rng(7)

MCP = {"index": (-0.30, -0.95), "middle": (0.0, -1.0), "ring": (0.25, -0.95), "pinky": (0.45, -0.85)}
BONES = {"index": (0.45, 0.28, 0.22), "middle": (0.50, 0.30, 0.24), "ring": (0.45, 0.28, 0.22),
         "pinky": (0.36, 0.22, 0.18)}
STRAIGHT, CURLED = (0.05, 0.05, 0.05), (1.4, 1.6, 1.0)        # joint bends in radians
THUMB_OUT = np.array([-0.8, -0.6]) / np.linalg.norm([-0.8, -0.6])


def make_hand(fingers="11111", thumb_up_angle=None, tilt_deg=0.0, noise=0.0):
    """fingers: thumb, index, middle, ring, pinky as '1' extended / '0' bent.

    thumb_up_angle: turn the whole hand so the extended thumb points this many degrees from straight
    up (0 = up, 180 = down, 90 = sideways). Otherwise the hand is turned by tilt_deg.
    """
    P = np.zeros((21, 3))
    for k, name in enumerate(["index", "middle", "ring", "pinky"], start=1):
        chain = hf.CHAINS[name]
        base = np.array([*MCP[name], 0.0])
        P[chain[1]] = base
        d0 = base / np.linalg.norm(base)
        theta, pos = 0.0, base
        for j, (bone, bend) in enumerate(zip(BONES[name], STRAIGHT if fingers[k] == "1" else CURLED)):
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


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


SHAPES = {
    "open_palm": dict(fingers="11111"),
    "fist": dict(fingers="00000"),
    "thumb_up": dict(fingers="10000", thumb_up_angle=0),
    "thumb_down": dict(fingers="10000", thumb_up_angle=180),
    "point": dict(fingers="01000"),
}

# ---------------------------------------------------------------- rules, default thresholds
rules = rc.RuleClassifier()
for g, kw in SHAPES.items():
    label, conf = rules.predict_with_confidence(make_hand(**kw))
    check(f"rules: {g} recognised (confidence {conf:.2f})", label == g and 0.5 <= conf <= 1.0)

for g in ("open_palm", "fist", "point"):
    labels = rules.predict(np.stack([make_hand(**SHAPES[g], tilt_deg=a) for a in (-75, -40, 40, 75)]))
    check(f"rules: {g} still recognised when the hand is tilted by up to 75 degrees", (labels == g).all())

check("rules: thumb tilted 30 degrees from vertical is still thumb up",
      rules.predict(make_hand(fingers="10000", thumb_up_angle=30)) == "thumb_up")
check("rules: thumb pointing sideways is no gesture",
      rules.predict(make_hand(fingers="10000", thumb_up_angle=90)) == "no_gesture")
check("rules: two fingers (not in the gesture set) is no gesture",
      rules.predict(make_hand(fingers="01100")) == "no_gesture")
check("rules: point with the thumb out is still point", rules.predict(make_hand(fingers="11000")) == "point")

batch = np.stack([make_hand(**SHAPES["fist"]), np.full((21, 3), np.nan), make_hand(**SHAPES["open_palm"])])
labels, conf = rules.predict_with_confidence(batch)
check("rules: a frame without a hand is no gesture with confidence 0",
      labels[1] == "no_gesture" and conf[1] == 0.0 and labels[0] == "fist" and labels[2] == "open_palm")
check("rules: confidences lie in 0..1", np.all((conf >= 0) & (conf <= 1)))
check("rules: classes_ lists the five gestures and no_gesture",
      sorted(rules.classes_.tolist()) == sorted(list(SHAPES) + ["no_gesture"]))

# ---------------------------------------------------------------- rules, fitted thresholds
train_Q, train_y = [], []
for g, kw in SHAPES.items():
    if g == "point":
        continue                                   # like the real data: no point recordings
    for _ in range(30):
        train_Q.append(make_hand(**kw, noise=0.02))
        train_y.append(g)
train_Q, train_y = np.stack(train_Q), np.array(train_y)
fitted = rc.RuleClassifier().fit(train_Q, train_y)
M = rc.finger_measures(train_Q)
ext = M[train_y == "open_palm"]
fist = M[train_y == "fist"]
check("fit: every threshold lies between the extended and the bent frames",
      np.all((fitted.thr > np.minimum(ext.max(0), fist.max(0))) & (fitted.thr < np.maximum(ext.min(0), fist.min(0)))))
check("fit: the fitted rules recognise the training gestures",
      np.mean(fitted.predict(train_Q) == train_y) == 1.0)
check("fit: point (absent from training) is still recognised", fitted.predict(make_hand(**SHAPES["point"])) == "point")
only_fists = rc.RuleClassifier().fit(train_Q[train_y == "fist"], train_y[train_y == "fist"])
check("fit: a finger seen in only one state keeps its default threshold", np.allclose(only_fists.thr, rc.DEFAULT_THR))
with_unknown = rc.RuleClassifier().fit(np.concatenate([train_Q, train_Q[:5]]),
                                       np.concatenate([train_y, ["no_gesture"] * 5]))
check("fit: frames with labels outside the templates are ignored", np.allclose(with_unknown.thr, fitted.thr))

# ---------------------------------------------------------------- LandmarkSVM interface
groups = np.repeat(np.arange(len(train_y) // 5), 5)          # 5 frames per made-up "clip"
for kw in (dict(features="raw63"), dict(features="invariant+orient", augment=True)):
    svm = ts.LandmarkSVM(**kw).fit(train_Q, train_y, groups=groups)
    labels, conf = svm.predict_with_confidence(train_Q)
    check(f"svm {kw}: labels from the training classes, confidence in 0..1",
          set(labels.tolist()) <= set(train_y.tolist()) and np.all((conf >= 0) & (conf <= 1)))
    check(f"svm {kw}: recognises its training frames", np.mean(labels == train_y) > 0.95)
one_label, one_conf = svm.predict_with_confidence(train_Q[0])
check("svm: one frame in, one label and one number out", isinstance(one_conf, float) and one_label == train_y[0])
P = svm.model.predict_proba(hf.compute(train_Q, "invariant+orient"))
check("svm: label is the class with the largest probability",
      np.array_equal(svm.classes_[P.argmax(1)], svm.predict(train_Q)) and np.allclose(P.max(1), conf))
ts.LandmarkSVM().fit(train_Q, train_y)
check("svm: also trains without clip groups", True)

print("\nAll checks passed.")
