"""Checks for trigger_gate.py on made-up frame results (no camera, no classifier).

    .venv\\Scripts\\python.exe test_trigger_gate.py
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import numpy as np

import calibration as cal
import trigger_gate as tg

FPS = 20.0


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def stream(*parts):
    """parts: (label, seconds, dict of per-frame values) -> arrays for TriggerGate.run."""
    labels, conf, calm, moving = [], [], [], []
    for label, seconds, kw in parts:
        n = int(round(seconds * FPS))
        labels += [label] * n
        conf += [kw.get("conf", 0.9)] * n
        calm += [kw.get("calm", True)] * n
        moving += [kw.get("moving", False)] * n
    times = np.arange(len(labels)) / FPS
    return times, labels, conf, calm, moving


def run(*parts, config=None, swipes=()):
    s = stream(*parts)
    return tg.TriggerGate(config).run(*s, swipes=swipes), s[0]


def kinds(events, kind=None, gesture=None):
    return [e for e in events if (kind is None or e.kind == kind) and (gesture is None or e.gesture == gesture)]


# ---------------------------------------------------------------- start at once, repeat while held
ev, times = run(("fist", 1.0, {}))
press = kinds(ev, "press")
check("fist: one press, after N_ON = 3 frames (0.10 s)", len(press) == 1 and abs(press[0].t - times[2]) < 1e-9)
rep = [e.t for e in kinds(ev, "repeat")]
check(f"fist held 1 s: first repeat 0.2 s after the press ({rep[0] - press[0].t:.2f} s)",
      abs(rep[0] - press[0].t - 0.2) < 1e-9)
check(f"fist held 1 s: a repeat every 0.2 s ({len(rep)} repeats: {', '.join(f'{t:.2f}' for t in rep)})",
      len(rep) == 4 and np.allclose(np.diff(rep), 0.2))
ev, _ = run(("fist", 3.0, {}))
rep = [e.t for e in kinds(ev, "repeat")]
check(f"held 3 s: {len(rep)} repeats, mean interval {np.mean(np.diff(rep)):.3f} s",
      len(rep) == 14 and abs(np.mean(np.diff(rep)) - 0.2) < 0.01)
check("no release while the gesture is still held", not kinds(ev, "release"))

for g in ("thumb_up", "thumb_down", "point"):
    ev, times = run((g, 0.5, {}))
    check(f"{g} starts at once too (press at {kinds(ev, 'press')[0].t:.2f} s)", kinds(ev, "press")[0].t < 0.15)

ev, times = run(("open_palm", 1.0, {}))
p = kinds(ev, "press")
check(f"open palm waits half a second (press at {p[0].t:.2f} s)", len(p) == 1 and 0.5 - 1e-9 <= p[0].t < 0.55)
ev, _ = run(("open_palm", 0.45, {}), ("no_gesture", 0.5, {}))
check("open palm shown for less than half a second: no press", not kinds(ev, "press"))

# ---------------------------------------------------------------- release and re-press
ev, times = run(("fist", 1.0, {}), ("no_gesture", 0.5, {}))
rel = kinds(ev, "release")
check("release on the 3rd frame without the gesture (2 frames tolerated)",
      len(rel) == 1 and abs(rel[0].t - times[20 + 2]) < 1e-9)
check("nothing after the release", all(e.t <= rel[0].t for e in ev))

ev, _ = run(("fist", 0.5, {}), ("no_gesture", 0.1, {}), ("fist", 0.5, {}))
check("a 2-frame dropout inside a hold: still one press, no release", len(kinds(ev, "press")) == 1 and not kinds(ev, "release"))
ev, _ = run(("fist", 0.5, {}), (None, 0.1, {}), ("fist", 0.5, {}))
check("the same with the hand lost for 2 frames", len(kinds(ev, "press")) == 1 and not kinds(ev, "release"))
ev, _ = run(("fist", 0.5, {}), ("no_gesture", 0.05, {}), ("fist", 0.5, {}), config=tg.GateConfig().without("hysteresis"))
check("without hysteresis a 1-frame dropout gives a second press", len(kinds(ev, "press")) == 2)

ev, times = run(("fist", 0.5, {}), ("no_gesture", 0.2, {}), ("fist", 0.5, {}))
p = kinds(ev, "press")
check("gesture released and shown again: a new press after N_ON frames",
      len(p) == 2 and abs(p[1].t - times[14 + 2]) < 1e-9)

ev, times = run(("fist", 0.5, {}), ("thumb_up", 0.5, {}))
order = [(e.kind, e.gesture) for e in ev if e.kind != "repeat"]
check("direct change fist -> thumb up: release fist, then press thumb up in the same frame",
      order == [("press", "fist"), ("release", "fist"), ("press", "thumb_up")]
      and kinds(ev, "release")[0].t == kinds(ev, "press", "thumb_up")[0].t)

# ---------------------------------------------------------------- confidence, stillness, speed gate
ev, _ = run(("fist", 1.0, {"conf": 0.6}))
check("confidence below conf_on (0.6 < 0.7): no press", not ev)
ev, _ = run(("fist", 0.3, {}), ("fist", 1.0, {"conf": 0.6}))
check("hysteresis: an already held gesture continues at confidence 0.6 (>= conf_hold 0.5)",
      not kinds(ev, "release") and len(kinds(ev, "repeat")) >= 5)
ev, _ = run(("fist", 0.3, {}), ("fist", 0.5, {"conf": 0.4}))
check("confidence 0.4 (< conf_hold) ends the hold", len(kinds(ev, "release")) == 1)
ev, _ = run(("fist", 1.0, {"calm": False}))
check("hand not calm: no press", not ev)
ev, _ = run(("fist", 0.3, {}), ("fist", 1.0, {"calm": False}))
check("hysteresis: a held gesture continues while the hand moves a little", not kinds(ev, "release"))
ev, _ = run(("fist", 0.3, {}), ("fist", 0.5, {"calm": False, "moving": True}))
check("fast movement (speed gate) ends a hold", len(kinds(ev, "release")) == 1)
ev, _ = run(("no_gesture", 1.0, {}), (None, 1.0, {"conf": 0.0}))
check("no_gesture and frames without a hand never press", not ev)
ev, _ = run(("swipe_left", 1.0, {}))
check("a dynamic label in the static stream is ignored", not ev)

# ---------------------------------------------------------------- swipes and cooldown
ev, times = run(("open_palm", 2.0, {"calm": False}), swipes=[(0.5, "swipe_left", 0.9)])
check("swipe: one swipe event at its time", [(e.kind, e.gesture) for e in ev] == [("swipe", "swipe_left")])
ev, times = run(("no_gesture", 0.5, {}), ("fist", 1.5, {}), swipes=[(0.4, "swipe_right", 0.9)])
p = kinds(ev, "press")
check(f"cooldown: no press until 1 s after the swipe, then N_ON frames (cooldown ends 1.40 s, press at {p[0].t:.2f} s)",
      len(p) == 1 and 1.45 <= p[0].t <= 1.55 + 1e-9)
gate = tg.TriggerGate()
first = gate.on_swipe(0.0, "swipe_left")
second = gate.on_swipe(0.5, "swipe_right")
third = gate.on_swipe(1.2, "swipe_right")
check("cooldown: a second swipe within 1 s is blocked and counted, a later one passes",
      len(first) == 1 and second == [] and len(third) == 1 and gate.blocked_swipes == 1)
ev, _ = run(("fist", 1.0, {}), swipes=[(0.5, "swipe_left", 0.9)])
check("a swipe ends a held gesture: release, then swipe",
      [(e.kind, e.gesture) for e in ev if e.kind != "repeat"] == [("press", "fist"), ("release", "fist"), ("swipe", "swipe_left")])
check("low-confidence swipe is ignored", tg.TriggerGate().on_swipe(0.0, "swipe_left", conf=0.3) == [])
ev, _ = run(("fist", 1.5, {}), swipes=[(0.0, "swipe_left", 0.9)], config=tg.GateConfig().without("cooldown"))
check("without cooldown the fist presses right after the swipe", kinds(ev, "press")[0].t < 0.15)

# ---------------------------------------------------------------- ablation switches
off = tg.GateConfig().without(*tg.SAFEGUARDS)
ev, times = run(("fist", 0.5, {"conf": 0.1, "calm": False}), config=off)
check("all safeguards off: press on the very first frame, even at confidence 0.1",
      kinds(ev, "press")[0].t == times[0])
ev, _ = run(("open_palm", 0.5, {}), config=tg.GateConfig().without("confirmation"))
check("without confirmation the open palm presses at once", kinds(ev, "press")[0].t == 0.0)
check("without() leaves the original config unchanged", tg.GateConfig().use_cooldown and not off.use_cooldown)

# ---------------------------------------------------------------- timing details
gate = tg.TriggerGate()
for t in np.arange(0, 0.3, 0.05):
    gate.update(t, "fist", 0.9)
burst = gate.update(2.0, "fist", 0.9)
check("a long gap between frames gives one repeat, not a burst", len(burst) == 1)

profile = cal.UserProfile(conf_threshold={"rules": 0.55, "svm": 0.6})
c = tg.GateConfig.from_profile(profile, "rules")
check("from_profile: the calibrated confidence threshold is used", c.conf_on == 0.55 and c.conf_hold == 0.5)
check("from_profile: unknown model falls back to the default", tg.GateConfig.from_profile(profile, "mlp").conf_on == 0.7)

print("\nAll checks passed.")
