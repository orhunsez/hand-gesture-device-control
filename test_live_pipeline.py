"""Checks for live_pipeline.py on made-up clips, also end to end with the trigger gate.

    .venv\\Scripts\\python.exe test_live_pipeline.py
"""
import numpy as np

import calibration as cal
import live_pipeline as lp
import rule_classifier as rc
import trigger_gate as tg
from synthetic_hand import SHAPES, make_clip, make_hand, swipe_path


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == object or b.dtype == object:
        return bool(np.all(a == b))
    return bool(np.allclose(a, b, equal_nan=True))


def events_of(result, config=None, swipes=()):
    return tg.TriggerGate(config).run(result["t"], result["label"], result["conf"], result["calm"],
                                      result["moving"], swipes=swipes)


rules = rc.RuleClassifier()
analyzer = lp.FrameAnalyzer(rules)

# ---------------------------------------------------------------- frame by frame == whole clip
lm, t = make_clip(make_hand(**SHAPES["fist"]), n=40, missing=(10, 11, 12), jitter_px=0.5)
clip = analyzer.process_clip(lm, t, 640, 480)
analyzer.reset()
frames = [analyzer.process(lm[i] if i not in (10, 11, 12) else None, t[i], 640, 480) for i in range(len(t))]
for key in ("label", "conf", "present", "speed", "pose", "calm", "moving", "centre", "size", "smoothed"):
    check(f"process_clip() == process() frame by frame: {key}", same(clip[key], [f[key] for f in frames]))
check("centre, size and smoothed are NaN in the frames without a hand",
      np.isnan(clip["centre"][10:13]).all() and np.isnan(clip["size"][10:13]).all()
      and np.isnan(clip["smoothed"][10:13]).all() and not np.isnan(clip["centre"][13]).any())
c0 = lm[0][lp.PALM, :2].mean(axis=0) * [640, 480]
check(f"centre: the palm centre in pixels (frame 0: {clip['centre'][0].round(1)}, raw {c0.round(1)})",
      np.allclose(clip["centre"][0], c0) and 90 < np.nanmedian(clip["size"]) < 110)

check("frames without a hand: no_gesture, confidence 0, not calm, not moving",
      all(clip["label"][i] == "no_gesture" and clip["conf"][i] == 0 and not clip["calm"][i]
          and not clip["moving"][i] for i in (10, 11, 12)))
check("first frame and the first frame after the hand comes back have no speed",
      np.isnan(clip["speed"][0]) and np.isnan(clip["speed"][13]))
check("still fist: recognised in every frame with a hand",
      np.all(clip["label"][clip["present"]] == "fist"))
check("still fist: calm and not moving in every frame that has a speed",
      np.all(clip["calm"][~np.isnan(clip["speed"])]) and not clip["moving"].any())

# ---------------------------------------------------------------- motion flags follow the profile
strict = cal.UserProfile(v_max=1e-6)
check("a profile with a tiny v_max: the same still hand is never calm",
      not lp.FrameAnalyzer(rules, strict).process_clip(lm, t, 640, 480)["calm"].any())
x = swipe_path(n=60, amplitude=1.2, sign=+1)
lm_s, t_s = make_clip(make_hand(**SHAPES["open_palm"]), n=60, shift_x=x)
sw = analyzer.process_clip(lm_s, t_s, 640, 480)
move_idx = np.flatnonzero(sw["moving"])
check(f"swipe: fast frames flagged as moving (frames {move_idx.min()} to {move_idx.max()}), "
      f"none in the still lead-in", len(move_idx) >= 3 and move_idx.min() >= 12)
check("swipe: the hand is not calm while it moves", not np.any(sw["calm"] & sw["moving"]))

# ---------------------------------------------------------------- end to end with the gate
lm, t = make_clip(make_hand(**SHAPES["thumb_up"]), n=40)
ev = events_of(analyzer.process_clip(lm, t, 640, 480))
press = [e for e in ev if e.kind == "press"]
check(f"thumb up held 2 s: pressed at {press[0].t - t[0]:.2f} s after the first frame (about 0.15 s)",
      len(press) == 1 and press[0].gesture == "thumb_up" and press[0].t - t[0] < 0.2)
check(f"thumb up held 2 s: {sum(e.kind == 'repeat' for e in ev)} repeats", sum(e.kind == "repeat" for e in ev) >= 7)

lm, t = make_clip(make_hand(**SHAPES["open_palm"]), n=40)
press = [e for e in events_of(analyzer.process_clip(lm, t, 640, 480)) if e.kind == "press"]
check(f"open palm held 2 s: pressed after half a second ({press[0].t - t[0]:.2f} s)",
      len(press) == 1 and 0.5 <= press[0].t - t[0] < 0.65)

ev = events_of(sw)
check("swipe with a 0.6 s still lead-in and no swipe event: the open palm presses before the movement",
      any(e.kind == "press" and e.t < t_s[12] for e in ev))
x = swipe_path(n=60, start_s=0.3, amplitude=1.2, sign=+1)
lm_s, t_s = make_clip(make_hand(**SHAPES["open_palm"]), n=60, shift_x=x)
sw = analyzer.process_clip(lm_s, t_s, 640, 480)
end = t_s[np.flatnonzero(sw["moving"]).max()] + 0.2
ev = events_of(sw, swipes=[(end, "swipe_left", 1.0)])
quiet = end + tg.SWIPE_COOLDOWN_S + tg.HOLD_S["open_palm"]
check("swipe with a 0.3 s lead-in and its swipe event: only the swipe until the cooldown and the open-palm "
      "wait are over", [e.kind for e in ev if e.t < quiet] == ["swipe"])
check(f"a palm kept still and raised for longer than that (here until 3 s) does press open palm "
      f"(at {next(e.t for e in ev if e.kind == 'press'):.2f} s; intended: it is a held open palm)",
      any(e.kind == "press" and e.gesture == "open_palm" and e.t >= quiet - 1e-9 for e in ev))
ev = events_of(sw, config=tg.GateConfig().without("cooldown"), swipes=[(end, "swipe_left", 1.0)])
check("the same without the cooldown: the palm resting after the swipe presses open palm",
      any(e.kind == "press" and e.gesture == "open_palm" and e.t > end for e in ev))

print("\nAll checks passed.")
