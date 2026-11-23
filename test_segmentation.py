"""Checks for swipe_segmentation.py on made-up clips (no camera, no recorded data).

    .venv\\Scripts\\python.exe test_segmentation.py
"""
import numpy as np

import calibration as cal
import live_pipeline as lp
import rule_classifier as rc
import swipe_segmentation as ss
import trigger_gate as tg
from synthetic_hand import SHAPES, make_clip, make_hand, swipe_path

FPS = 20.0
PALM = make_hand(**SHAPES["open_palm"])
analyzer = lp.FrameAnalyzer(rc.RuleClassifier())


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def path(*parts):
    """Palm path in hand sizes from (seconds, x_end, y_end) parts: each part moves smoothly (half a cosine)
    from where the previous one ended to (x_end, y_end); equal start and end = the hand stays still."""
    xs, ys, x, y = [], [], 0.0, 0.0
    for seconds, x1, y1 in parts:
        n = int(round(seconds * FPS))
        s = 0.5 * (1 - np.cos(np.pi * np.arange(1, n + 1) / n))
        xs.append(x + (x1 - x) * s)
        ys.append(y + (y1 - y) * s)
        x, y = x1, y1
    return np.concatenate(xs), np.concatenate(ys)


def clip(x, y=None, missing=(), seed=0):
    """FrameAnalyzer result of an open palm moving along x (and y), in hand sizes."""
    n = len(x)
    Q = np.repeat(PALM[None], n, axis=0)
    if y is not None:
        Q[..., 1] += np.asarray(y)[:, None]          # make_clip only moves the hand in x
    lm, t = make_clip(Q, n=n, fps=FPS, shift_x=x, missing=missing, seed=seed)
    return analyzer.process_clip(lm, t, 640, 480), lm, t


def segment(x, y=None, missing=(), profile=None):
    r, _, _ = clip(x, y, missing)
    seg = ss.SwipeSegmenter(profile)
    return seg.segment_clip(r), seg, r


def reasons(seg):
    return [rej["reason"] for rej in seg.rejects]


# ---------------------------------------------------------------- thresholds come from the profile
seg = ss.SwipeSegmenter()
check("no profile: tau_on, tau_off and the minimum amplitude are calibration.DEFAULTS",
      (seg.tau_on, seg.tau_off, seg.min_amplitude) ==
      (cal.DEFAULTS["tau_on"], cal.DEFAULTS["tau_off"], cal.DEFAULTS["swipe_min_amplitude"]))
seg = ss.SwipeSegmenter(cal.UserProfile(tau_on=1.4, tau_off=0.6, swipe_min_amplitude=0.35))
check("a profile's own values are used", (seg.tau_on, seg.tau_off, seg.min_amplitude) == (1.4, 0.6, 0.35))

# ---------------------------------------------------------------- clean swipes
for sign, label in ((+1, "swipe_left"), (-1, "swipe_right")):
    sw, seg, r = segment(swipe_path(n=60, amplitude=1.2, sign=sign))
    check(f"clean {label} (x {'increases' if sign > 0 else 'decreases'} first): exactly one swipe, label {label}",
          len(sw) == 1 and sw[0].label == label and not seg.rejects)
    s = sw[0]
    check(f"{label}: amplitude {s.amplitude:.2f} close to the 1.2 hand sizes of the movement", 1.0 < s.amplitude <= 1.25)
    onset = r["t"][np.flatnonzero(r["speed"] > seg.tau_on)[0]]
    check(f"{label}: start = first frame above tau_on ({s.t_start:.2f} s), event after the return "
          f"({s.t_end:.2f} s; the movement ends at {r['t'][0] + 1.3:.2f} s)",
          s.t_start == onset and r["t"][0] + 1.25 <= s.t_end < r["t"][0] + 1.6)
    check(f"{label}: window from the start position to the event, {len(s.times)} frames, shapes agree",
          s.times[0] < s.t_start and s.times[-1] == s.t_end and s.landmarks.shape == (len(s.times), 21, 3)
          and s.path.shape == (len(s.times), 2))
    check(f"{label}: path relative to the start (first point {s.path[0, 0]:+.3f}, last {s.path[-1, 0]:+.3f} hand sizes)",
          abs(s.path[0, 0]) < 0.05 and abs(s.path[-1, 0]) <= ss.RETURN_TOL)
    check(f"{label}: confidence {s.conf} and event() gives (t_end, label, conf)",
          s.conf == ss.CONFIDENCE and s.event() == (s.t_end, label, s.conf))

# ---------------------------------------------------------------- the pilot finding: a pause at the turning point
x, _ = path((0.6, 0, 0), (0.35, -1.2, 0), (0.4, -1.2, 0), (0.35, 0, 0), (0.8, 0, 0))
sw, seg, r = segment(x)
slow = np.flatnonzero((r["speed"] < seg.tau_off) & (r["t"] > r["t"][0] + 0.9) & (r["t"] < r["t"][0] + 1.4))
check(f"pause at the turning point: the speed stays below tau_off there for {len(slow)} frames "
      f"(> 0.2 s, so 'still for 0.2 s' would end the cycle there)", len(slow) * 1 / FPS > 0.2)
check("pause of 0.4 s at the turning point: still ONE swipe, swipe_right",
      len(sw) == 1 and sw[0].label == "swipe_right" and not seg.rejects)
low = cal.UserProfile(swipe_min_amplitude=0.25)
x, _ = path((0.6, 0, 0), (0.3, 0.35, 0), (0.4, 0.35, 0), (0.3, 0, 0), (0.8, 0, 0))
sw, seg, _ = segment(x, profile=low)
check(f"calibrated minimum amplitude 0.25 and a 0.35 swipe with a pause at the turning point: one swipe "
      f"(the turning point is within RETURN_TOL but not within half the excursion)",
      len(sw) == 1 and sw[0].label == "swipe_left" and sw[0].t_end > sw[0].t_start + 0.8)

# ---------------------------------------------------------------- no swipe
sw, seg, r = segment(np.zeros(60))
check(f"still hand for 3 s (peak speed {np.nanmax(r['speed']):.2f}): no swipe, no cycle started",
      sw == [] and seg.rejects == [])
x, _ = path((0.6, 0, 0), (0.4, 1.2, 0), (1.5, 1.2, 0))
sw, seg, r = segment(x)
check(f"one-way movement (1.2 hand sizes out, then still): no swipe ({reasons(seg)})",
      sw == [] and reasons(seg) == ["no return (one-way movement)"])
after = seg.rejects[0]["t"] - (r["t"][0] + 1.0)
check(f"one-way movement: rejected {after:.2f} s after the movement stopped (STILL_REJECT_S {ss.STILL_REJECT_S} s "
      f"plus the smoothing lag)", ss.STILL_REJECT_S <= after < ss.STILL_REJECT_S + 0.3)
sw, seg, _ = segment(swipe_path(n=60, amplitude=0.3, sign=+1))
check(f"too small (0.3 hand sizes out and back): no swipe ({reasons(seg)})",
      sw == [] and reasons(seg) == ["below minimum amplitude"])
x, y = path((0.6, 0, 0), (0.35, 0.6, -1.5), (0.35, 0, 0), (0.8, 0, 0))
sw, seg, _ = segment(x, y)
check(f"mostly vertical (0.6 sideways, 1.5 up, and back): no swipe ({reasons(seg)})",
      sw == [] and len(seg.rejects) == 1 and seg.rejects[0]["reason"].startswith("mostly vertical"))
x, y = path((0.6, 0, 0), (0.35, 1.2, -0.6), (0.35, 0, 0), (0.8, 0, 0))
sw, _, _ = segment(x, y)
check("a diagonal swipe (1.2 sideways, 0.6 up) still counts", len(sw) == 1 and sw[0].label == "swipe_left")

t = np.arange(100) / FPS
wave = np.where((t >= 0.5) & (t < 3.5), 0.8 * np.sin(2 * np.pi * 1.5 * (t - 0.5)), 0.0)
sw, seg, _ = segment(wave)
check(f"waving back and forth for 3 s (1.5 per second, 0.8 hand sizes): no swipe ({reasons(seg)})",
      sw == [] and reasons(seg)[:1] == ["too long (over 2.0 s)"])
check("waving: no second cycle starts while the waving goes on after the rejection", len(seg.rejects) == 1)

# A jerk of 0.1 s (1.2 hand sizes out and back) lasts 0.25 to 0.30 s from onset to end after smoothing at
# 20 frames per second, i.e. about MIN_DURATION_S itself; so the minimum duration is checked with a larger value.
x, _ = path((0.6, 0, 0), (0.05, 1.2, 0), (0.05, 0, 0), (1.0, 0, 0))
sw, seg, _ = segment(x)
measured = sw[0].t_end - sw[0].t_start if sw else float("nan")
check(f"a 0.1 s jerk measures {measured:.2f} s from onset to end; with MIN_DURATION_S {ss.MIN_DURATION_S} it passes "
      f"(the rule only stops even faster ones; see the module constants)", len(sw) == 1 and 0.2 <= measured < 0.35)
default_min = ss.MIN_DURATION_S
ss.MIN_DURATION_S = 0.4
sw, seg, _ = segment(x)
ss.MIN_DURATION_S = default_min
check(f"minimum duration raised to 0.4 s: the same jerk gives no swipe ({reasons(seg)})",
      sw == [] and len(seg.rejects) == 1 and seg.rejects[0]["reason"].startswith("too short"))

# ---------------------------------------------------------------- two swipes in a row
x, _ = path((0.5, 0, 0), (0.35, 1.2, 0), (0.35, 0, 0), (0.5, 0, 0), (0.35, -1.2, 0), (0.35, 0, 0), (0.8, 0, 0))
sw, seg, _ = segment(x)
check(f"two swipes with a 0.5 s pause between them: two swipes "
      f"({', '.join(f'{s.label} {s.t_start:.2f}-{s.t_end:.2f} s' for s in sw)})",
      [s.label for s in sw] == ["swipe_left", "swipe_right"] and not seg.rejects and sw[0].t_end < sw[1].t_start)

# ---------------------------------------------------------------- lost hand
base = swipe_path(n=60, amplitude=1.2, sign=+1)
sw, seg, r = segment(base, missing=(16, 17))
check("hand lost for 2 frames during the out stroke: still one swipe_left, the lost frames are NaN in the window",
      len(sw) == 1 and sw[0].label == "swipe_left" and np.isnan(sw[0].path[:, 0]).sum() == 2)
sw, seg, r = segment(base, missing=(16, 17, 18))
check(f"hand lost for 3 frames: the cycle is dropped and the return stroke does not become a swipe ({reasons(seg)})",
      sw == [] and reasons(seg) == ["hand lost"])
x, _ = path((0.6, 0, 0), (0.6, 0, 0), (0.35, -1.2, 0), (0.35, 0, 0), (0.8, 0, 0))
sw, seg, _ = segment(x, missing=(3, 4, 5, 6))
check("hand lost in the still lead-in: the swipe afterwards is found",
      len(sw) == 1 and sw[0].label == "swipe_right" and not seg.rejects)

# ---------------------------------------------------------------- frame by frame == whole clip
x, _ = path((0.5, 0, 0), (0.35, 1.2, 0), (0.35, 0, 0), (0.5, 0, 0), (0.35, -1.2, 0), (0.35, 0, 0), (0.8, 0, 0))
missing = (5, 6, 7, 8, 30)
r, lm, t = clip(x, missing=missing)
whole = ss.SwipeSegmenter().segment_clip(r)
analyzer.reset()
seg = ss.SwipeSegmenter()
frames = [seg.update(analyzer.process(None if i in missing else lm[i], t[i], 640, 480)) for i in range(len(t))]
one = [s for s in frames if s is not None]


def same_swipe(a, b):
    return (a.t_start == b.t_start and a.t_end == b.t_end and a.label == b.label and a.conf == b.conf
            and a.amplitude == b.amplitude and np.array_equal(a.times, b.times)
            and np.array_equal(a.landmarks, b.landmarks, equal_nan=True) and np.array_equal(a.path, b.path, equal_nan=True))


check(f"segment_clip() == update() frame by frame ({len(whole)} swipes, every field identical)",
      len(whole) == len(one) == 2 and all(same_swipe(a, b) for a, b in zip(whole, one)))

# ---------------------------------------------------------------- end to end with the trigger gate
x, _ = path((0.5, 0, 0), (0.35, 1.2, 0), (0.35, 0, 0), (0.1, 0, 0), (0.35, -1.2, 0), (0.35, 0, 0), (1.6, 0, 0))
r, _, _ = clip(x)
sw = ss.SwipeSegmenter().segment_clip(r)
ev = tg.TriggerGate().run(r["t"], r["label"], r["conf"], r["calm"], r["moving"], swipes=[s.event() for s in sw])
got = [e for e in ev if e.kind == "swipe"]
check(f"through the gate: two swipe candidates {[s.label for s in sw]}, the second within the 1 s cooldown, "
      f"so one swipe command", len(sw) == 2 and len(got) == 1 and got[0].gesture == "swipe_left"
      and sw[1].t_end - sw[0].t_end < tg.SWIPE_COOLDOWN_S)

print("\nAll checks passed.")
