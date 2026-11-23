"""Checks for calibration.py on made-up clips (synthetic_hand.py).

    .venv\\Scripts\\python.exe test_calibration.py
"""
import tempfile
from pathlib import Path

import numpy as np

import calibration as cal
import rule_classifier as rc
from synthetic_hand import SHAPES, make_clip, make_hand, swipe_path


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def static_segment(gesture=None, n=40, jitter_px=0.3, size_px=100.0, seed=0, **hand):
    kw = dict(SHAPES[gesture]) if gesture else {}
    kw.update(hand)
    lm, t = make_clip(make_hand(**kw), n=n, jitter_px=jitter_px, size_px=size_px, seed=seed)
    return lm, t, 640, 480


def swipe_segment(sign, amplitude=1.2, out_s=0.35, seed=0):
    x = swipe_path(n=60, amplitude=amplitude, out_s=out_s, back_s=out_s, sign=sign)
    lm, t = make_clip(make_hand(**SHAPES["open_palm"]), n=60, shift_x=x, seed=seed)
    return lm, t, 640, 480


def full_session(jitter_px=0.3, curl=None, **swipe_kw):
    seg = {}
    for k, g in enumerate(cal.STATIC_STEPS):
        extra = {"curl": curl} if curl is not None else {}
        seg[g] = static_segment(g, jitter_px=jitter_px, seed=k, **extra)
    seg["swipe_left"] = swipe_segment(+1, seed=10, **swipe_kw)
    seg["swipe_right"] = swipe_segment(-1, seed=11, **swipe_kw)
    return seg


def status(c, step):
    return next(s for s in c.steps if s.step == step).status


# ---------------------------------------------------------------- a clean session
c = cal.calibrate(full_session(), user="u98")
p = c.profile
check("clean session: every step ok", all(s.status == "ok" for s in c.steps) and not c.failed())
check("clean session: steady frames of all five held gestures are kept",
      sorted(set(c.y.tolist())) == sorted(cal.STATIC_STEPS) and c.Q.shape[1:] == (21, 3))
check("clean session: a calm hand keeps the default stability thresholds (they are only raised)",
      p.v_max == cal.DEFAULTS["v_max"] and p.pose_max == cal.DEFAULTS["pose_max"])
check("clean session: hand size measured (100 px)", abs(p.hand_size_px - 100) < 3)
lo, hi = cal.LIMITS
for name in ("tau_on", "tau_off", "swipe_min_amplitude"):
    d = cal.DEFAULTS[name]
    check(f"clean session: {name} = {getattr(p, name):.3f} lies within the limits around {d}",
          d * lo <= getattr(p, name) <= d * hi and name in p.calibrated)
check("clean session: tau_off = OFF_RATIO x tau_on (within its limits)",
      abs(p.tau_off - cal._limit(cal.OFF_RATIO * p.tau_on, cal.DEFAULTS["tau_off"])) < 1e-9)
check("clean session: tau_on >= 2 x v_max", p.tau_on >= 2 * p.v_max - 1e-9)
check("clean session: confidence threshold of the rules is never raised",
      p.confidence_threshold("rules") <= cal.DEFAULTS["conf_threshold"]["rules"])

# ---------------------------------------------------------------- a shaky user
shaky = cal.calibrate(full_session(jitter_px=2.5)).profile
very = cal.calibrate(full_session(jitter_px=6.0)).profile
check(f"shaky user: v_max raised ({shaky.v_max:.3f} > {cal.DEFAULTS['v_max']})", shaky.v_max > cal.DEFAULTS["v_max"])
check(f"very shaky user: v_max is capped at twice the default ({very.v_max:.3f})",
      very.v_max <= 2 * cal.DEFAULTS["v_max"] + 1e-9 and very.v_max >= shaky.v_max)

# ---------------------------------------------------------------- wrong pose and unknown pose
seg = full_session()
seg["open_palm"] = static_segment("fist", seed=20)               # the user shows a fist when asked for an open palm
c = cal.calibrate(seg)
step = next(s for s in c.steps if s.step == "open_palm")
check("wrong pose: the open_palm step fails and names the gesture it looks like",
      step.status == "fail" and "fist" in step.message and "open_palm" in c.failed())
check("wrong pose: its frames are not used", "open_palm" not in set(c.y.tolist()))

seg = full_session()
seg["point"] = static_segment(fingers="01100", seed=21)          # two fingers: no gesture for the rules
c = cal.calibrate(seg)
check("pose that looks like no gesture: the step is kept with a warning",
      status(c, "point") == "warn" and "point" in set(c.y.tolist()))

seg = full_session()
seg["fist"] = (np.full_like(seg["fist"][0], np.nan),) + seg["fist"][1:]
check("hand never found: the step fails", status(cal.calibrate(seg), "fist") == "fail")

# ---------------------------------------------------------------- a user whose fingers do not bend far
stiff = (0.7, 0.8, 0.5)                                          # sum 2.0 rad instead of 4.0
c = cal.calibrate(full_session(curl=stiff))
fist_Q = c.Q[c.y == "fist"]
before = rc.RuleClassifier().predict(fist_Q)
after = c.profile.rules().predict(fist_Q)
check(f"stiff fingers: uncalibrated rules miss the fist (fist in {np.mean(before == 'fist') * 100:.0f}% of frames)",
      np.mean(before == "fist") < 0.5 and status(c, "fist") == "warn")
check(f"stiff fingers: calibrated rules recognise it (fist in {np.mean(after == 'fist') * 100:.0f}% of frames)",
      np.mean(after == "fist") > 0.9 and "rule_thr" in c.profile.calibrated)
thr = np.array(c.profile.rule_thr)
check("stiff fingers: every finger threshold stays within RULE_LIMIT of the population threshold",
      np.all(np.abs(thr - rc.DEFAULT_THR) <= cal.RULE_LIMIT * rc.DEFAULT_THR + 1e-9))
check("stiff fingers: the open palm is still recognised after calibration",
      np.mean(c.profile.rules().predict(c.Q[c.y == "open_palm"]) == "open_palm") > 0.9)


# ---------------------------------------------------------------- confidence thresholds of other models
class FixedConfidence:
    """Labels from the default rules, a fixed confidence."""
    def __init__(self, value):
        self.value, self.rules = value, rc.RuleClassifier()

    def predict_with_confidence(self, Q):
        labels = self.rules.predict(Q)
        return labels, np.full(len(labels), self.value)


seg = full_session()
for value, expected in ((0.6, 0.6), (0.3, cal.CONF_FLOOR), (0.95, cal.DEFAULTS["conf_threshold"]["svm"])):
    got = cal.calibrate(seg, models={"svm": FixedConfidence(value)}).profile.confidence_threshold("svm")
    check(f"confidence {value} on the user's gestures -> threshold {got:.2f} (expected {expected})",
          abs(got - expected) < 1e-9)

# ---------------------------------------------------------------- swipes
seg = full_session()
seg["swipe_left"] = swipe_segment(-1, seed=30)                   # goes the wrong way first
c = cal.calibrate(seg)
check("swipe whose first stroke goes the wrong way fails", status(c, "swipe_left") == "fail")
seg = full_session()
seg["swipe_right"] = swipe_segment(-1, amplitude=0.25, out_s=1.2, seed=31)
check("swipe that is far too slow fails", status(cal.calibrate(seg), "swipe_right") == "fail")
slow = cal.calibrate(full_session(amplitude=0.8, out_s=0.6)).profile
fast = cal.calibrate(full_session(amplitude=1.5, out_s=0.25)).profile
check(f"slower swipes give a lower tau_on ({slow.tau_on:.2f} < {fast.tau_on:.2f})", slow.tau_on < fast.tau_on)

seg = {g: s for g, s in full_session().items() if g not in cal.SWIPE_STEPS}
c = cal.calibrate(seg)
check("no swipes recorded: swipe thresholds keep their defaults and the steps say 'missing'",
      c.profile.tau_on == cal.DEFAULTS["tau_on"] and "tau_on" not in c.profile.calibrated
      and status(c, "swipe_left") == "missing")
check("nothing recorded: the profile equals the defaults",
      not cal.calibrate({}).profile.calibrated and cal.calibrate({}).profile.v_max == cal.DEFAULTS["v_max"])

# ---------------------------------------------------------------- distance warning
seg = {g: static_segment(g, size_px=30.0, seed=k) for k, g in enumerate(cal.STATIC_STEPS)}
c = cal.calibrate(seg)
check("hand too small (30 px): distance warning", any(s.step == "hand size" and "closer" in s.message for s in c.steps))

# ---------------------------------------------------------------- steadiest stretch
lm, t = make_clip(make_hand(**SHAPES["fist"]), n=40, shift_x=np.concatenate([np.linspace(0, 1.5, 15), np.full(25, 1.5)]))
import dataset_loader as dl
import landmark_smoothing as ls
sm = ls.smooth_clip(lm, t)
idx = cal.steadiest(*dl.motion(sm, t), t)
check("steadiest(): the stretch lies in the still part after the movement", len(idx) >= 15 and idx[0] >= 15)

# ---------------------------------------------------------------- saving and helpers
c = cal.calibrate(full_session(curl=stiff), user="u98")
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "sub" / "u98.json"
    c.profile.save(path)
    back = cal.UserProfile.load(path)
check("profile survives saving and loading", back == c.profile)
check("loaded profile gives the same rules", np.allclose(back.rules().thr, c.profile.rules().thr))

clips = [{"user": "u98", "gesture": g, "clip": k, "landmarks": None, "times": None, "width": 640, "height": 480}
         for g in ("fist", "swipe_left", "no_gesture") for k in (1, 2)]
segs = cal.segments_from_clips(clips, "u98", clip=2)
check("segments_from_clips: one segment per gesture, only the asked clip number, no no_gesture",
      sorted(segs) == ["fist", "swipe_left"])

pop_Q = np.stack([make_hand(**SHAPES[g], noise=0.02) for g in ("fist", "open_palm") for _ in range(20)])
pop_y = np.repeat(["fist", "open_palm"], 20)
pop_groups = np.repeat(np.arange(8), 5)
c = cal.calibrate(full_session(curl=stiff))
svm = cal.adapted_svm(pop_Q, pop_y, pop_groups, c.Q, c.y, features="invariant+orient")
check("adapted_svm: learns the user's calibration gestures as well",
      set(svm.classes_.tolist()) == set(cal.STATIC_STEPS) and np.mean(svm.predict(c.Q) == c.y) > 0.95)

print("\nAll checks passed.")
