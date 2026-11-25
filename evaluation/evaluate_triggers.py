"""Experiment E4 (report section 5.6): do the false-trigger safeguards help?

    .venv\\Scripts\\python.exe evaluate_triggers.py [--model svm|rules] [--train-user u99] [--profile p.json]

Every recorded clip is played frame by frame through the static branch (live_pipeline.FrameAnalyzer)
and the trigger gate (trigger_gate.py), once per gate setting: all safeguards on, each one switched
off alone, all off. The analysis of a clip is done once; only the gate is run again per setting.

Classifier: trained on the static clips of the training user. A static clip of the training user is
analysed by a model trained WITHOUT it (clip-level folds); every other clip by a model trained on all
of them. svm = LandmarkSVM inv+orient with rotation augmentation (the E1 recommendation), rules =
RuleClassifier fitted on the same frames. Without --profile the default thresholds are used.

Measures
  static clips (the gesture is held for the whole clip, about 2 s)
    hit       share of clips with a press of the right gesture
    wrong     presses of another gesture, per clip
    extra     presses of the right gesture after the first one (a hold that broke up), per clip
    latency   median time from the first frame with a hand to the first right press, in ms
    interval  mean time between two consecutive commands of one hold (press, repeat, repeat, ...),
              in s; the setting is 0.2 s
  swipe clips (here no static gesture should press)
    static presses per clip before the movement starts and after it
    The dynamic branch (segmentation 4.5, classifier 4.7) is not written yet. So that the cooldown can
    act, a STAND-IN swipe event is placed where a simple segmentation would end the movement: after the
    palm speed exceeded tau_on, the first moment it has stayed below tau_off for 0.2 s (A.8), with the
    clip's own label (an oracle).
  no_gesture clips: false presses and false actions (press + repeat) per minute. None recorded yet.
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import argparse
import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import calibration as cal
import dataset_loader as dl
import hand_features as hf
import live_pipeline as lp
import rule_classifier as rc
import train_static as ts
import trigger_gate as tg

SETTLE_S = 0.2


def make_model(kind, Q, y, groups):
    if kind == "rules":
        return rc.RuleClassifier().fit(Q, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ts.LandmarkSVM(features="invariant+orient", augment=True).fit(Q, y, groups=groups)


def stand_in_swipe(r, profile):
    """(onset time, swipe event time) of the first movement of a clip, or (None, None)."""
    fast = np.flatnonzero(r["speed"] > profile.tau_on)
    if not len(fast):
        return None, None
    onset = fast[0]
    t = r["t"]
    slow = np.nan_to_num(r["speed"], nan=np.inf) < profile.tau_off
    start = None
    for j in range(onset + 1, len(t)):
        if slow[j]:
            start = j if start is None else start
            if t[j] - t[start] >= SETTLE_S - 1e-9:
                return t[onset], t[j]
        else:
            start = None
    return t[onset], t[-1]


def analyse_all(clips, kind, train_user, profile, folds=5, seed=0):
    """FrameAnalyzer output for every clip: {(file, clip): result}."""
    ds, _ = dl.load_static(users=[train_user])
    Q, y, groups = hf.unflatten(ds.X), ds.y, ts.clip_groups(ds)
    out = {}
    key = lambda c: f"{c['file']}#{c['clip']}"
    own_static = {key(c): c for c in clips if c["user"] == train_user and c["gesture"] in tg.STATIC}
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        splits = list(cv.split(Q, y, groups))
    for tr, te in splits:
        model = make_model(kind, Q[tr], y[tr], groups[tr])
        analyzer = lp.FrameAnalyzer(model, profile)
        for g in sorted(set(groups[te].tolist())):
            c = own_static[g]
            out[g] = analyzer.process_clip(c["landmarks"], c["times"], c["width"], c["height"])
    model = make_model(kind, Q, y, groups)
    analyzer = lp.FrameAnalyzer(model, profile)
    for c in clips:
        if key(c) not in out:
            out[key(c)] = analyzer.process_clip(c["landmarks"], c["times"], c["width"], c["height"])
    return out, sorted(set(y.tolist()))


def gate_settings(profile, kind):
    base = tg.GateConfig.from_profile(profile, kind)
    settings = {"all on": base}
    for s in tg.SAFEGUARDS:
        settings[f"without {s}"] = base.without(s)
    settings["all off"] = base.without(*tg.SAFEGUARDS)
    return settings


def action_intervals(events):
    """Times between consecutive commands (press, repeats) within each hold."""
    out, last = [], None
    for e in events:
        if e.kind == "press":
            last = e.t
        elif e.kind == "repeat" and last is not None:
            out.append(e.t - last)
            last = e.t
        elif e.kind in ("release", "swipe"):
            last = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["svm", "rules"], default="svm")
    ap.add_argument("--train-user", default="u99")
    ap.add_argument("--profile", help="calibration profile (JSON) to use instead of the defaults")
    args = ap.parse_args()
    profile = cal.UserProfile.load(args.profile) if args.profile else cal.UserProfile()

    clips = dl.load_clips()
    results, classes = analyse_all(clips, args.model, args.train_user, profile)
    key = lambda c: f"{c['file']}#{c['clip']}"
    static = [c for c in clips if c["gesture"] in tg.STATIC]
    swipes = [c for c in clips if c["gesture"] in tg.DYNAMIC]
    negatives = [c for c in clips if c["gesture"] == "no_gesture"]
    stand_in = {key(c): stand_in_swipe(results[key(c)], profile) for c in swipes}
    print(f"Model: {args.model}, trained on {args.train_user} ({classes}); thresholds: "
          f"{'profile ' + args.profile if args.profile else 'defaults (no calibration)'}")
    print(f"Clips: {len(static)} static, {len(swipes)} swipe, {len(negatives)} no_gesture")
    print(f"Gate: N_ON {tg.N_ON} frames, open-palm wait {tg.HOLD_S['open_palm']} s, repeat every {tg.REPEAT_S} s, "
          f"conf_on {tg.GateConfig.from_profile(profile, args.model).conf_on:.2f}, "
          f"conf_hold {tg.CONF_HOLD}, {tg.DROP_FRAMES} frames tolerated, swipe cooldown {tg.SWIPE_COOLDOWN_S} s")
    found = [v for v in stand_in.values() if v[0] is not None]
    print(f"Stand-in swipe events: {len(found)} of {len(swipes)} swipe clips reach tau_on {profile.tau_on:.2f}")

    header = (f"\n{'gate setting':<24} {'hit':>6} {'wrong':>6} {'extra':>6} {'latency':>8} {'interval':>9}"
              f"   swipe clips: presses before / after the movement")
    print(header)
    per_gesture = {}
    for name, config in gate_settings(profile, args.model).items():
        gate = tg.TriggerGate(config)
        hits, wrong, extra, lat, gaps = [], [], [], [], []
        for c in static:
            r = results[key(c)]
            ev = gate.run(r["t"], r["label"], r["conf"], r["calm"], r["moving"])
            right = [e for e in ev if e.kind == "press" and e.gesture == c["gesture"]]
            hits.append(bool(right))
            wrong.append(sum(e.kind == "press" and e.gesture != c["gesture"] for e in ev))
            extra.append(max(0, len(right) - 1))
            first_hand = r["t"][np.flatnonzero(r["present"])[0]] if r["present"].any() else None
            if right and first_hand is not None:
                lat.append((right[0].t - first_hand) * 1000)
            gaps += action_intervals(ev)
            if name == "all on":
                pg = per_gesture.setdefault((c["user"], c["gesture"]), [[], [], []])
                pg[0].append(bool(right))
                pg[1].append(lat[-1] if right and first_hand is not None else np.nan)
                pg[2].append(wrong[-1])
        before, after = [], []
        for c in swipes:
            r = results[key(c)]
            onset, t_event = stand_in[key(c)]
            sw = [(t_event, c["gesture"], 1.0)] if t_event is not None else []
            ev = gate.run(r["t"], r["label"], r["conf"], r["calm"], r["moving"], swipes=sw)
            presses = [e for e in ev if e.kind == "press"]
            cut = onset if onset is not None else np.inf
            before.append(sum(e.t < cut for e in presses))
            after.append(sum(e.t >= cut for e in presses))
        latency = f"{np.median(lat):.0f}" if lat else "-"
        rate = f"{np.mean(gaps):.3f}" if gaps else "-"
        print(f"{name:<24} {np.mean(hits) * 100:>5.0f}% {np.mean(wrong):>6.2f} {np.mean(extra):>6.2f} {latency:>8} "
              f"{rate:>9}   {np.mean(before):.2f} / {np.mean(after):.2f}")

    print("\nAll safeguards on, per user and gesture (static clips):")
    print(f"   {'user':<6} {'gesture':<12} {'clips':>5} {'hit':>6} {'latency ms':>11} {'wrong presses':>14}")
    for (u, g), (h, l, w) in sorted(per_gesture.items()):
        l = np.array(l)
        lat_txt = f"{np.nanmedian(l):.0f}" if np.isfinite(l).any() else "-"
        print(f"   {u:<6} {g:<12} {len(h):>5} {np.mean(h) * 100:>5.0f}% {lat_txt:>11} {int(np.sum(w)):>14}")

    print("\nAll safeguards on, swipe clips one by one (static presses that should not happen):")
    gate = tg.TriggerGate(gate_settings(profile, args.model)["all on"])
    for c in swipes:
        r = results[key(c)]
        onset, t_event = stand_in[key(c)]
        sw = [(t_event, c["gesture"], 1.0)] if t_event is not None else []
        ev = gate.run(r["t"], r["label"], r["conf"], r["calm"], r["moving"], swipes=sw)
        lead = f"{onset - r['t'][0]:.2f} s" if onset is not None else "no movement found"
        presses = ", ".join(f"{e.gesture} at {e.t:.2f} s" for e in ev if e.kind == "press") or "none"
        swipe_txt = f"swipe event at {t_event:.2f} s" if t_event is not None else "no swipe event"
        print(f"   {c['user']} {c['gesture']:<11} #{c['clip']}  still lead-in {lead:<18} {swipe_txt:<22} presses: {presses}")

    if not negatives:
        print("\nno_gesture: no clips recorded yet, so the false-trigger rate per minute cannot be measured.")
    else:
        minutes = sum(results[key(c)]["t"][-1] - results[key(c)]["t"][0] for c in negatives) / 60
        print(f"\nno_gesture: {len(negatives)} clips, {minutes:.2f} minutes")
        for name, config in gate_settings(profile, args.model).items():
            gate = tg.TriggerGate(config)
            presses = actions = 0
            for c in negatives:
                r = results[key(c)]
                ev = gate.run(r["t"], r["label"], r["conf"], r["calm"], r["moving"])
                presses += sum(e.kind == "press" for e in ev)
                actions += sum(e.kind in ("press", "repeat") for e in ev)
            print(f"   {name:<24} false presses per minute {presses / minutes:6.1f}, "
                  f"false actions per minute {actions / minutes:6.1f}")


if __name__ == "__main__":
    main()
