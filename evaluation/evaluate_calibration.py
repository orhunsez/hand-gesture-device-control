"""Experiment E3 (report section 5.5): does user calibration help?

    .venv\\Scripts\\python.exe evaluate_calibration.py [--calib-clip 1] [--copies 1]

For every user with at least two gestures that have two or more static clips:
  calibration = clip CALIB_CLIP of every gesture of that user (static gestures and swipes), run through
                calibration.calibrate() exactly as the live calibration would use it
  test        = the other static clips of that user (their stable frames, dataset_loader.load_static)
  population  = the static stable frames of all OTHER users; models are trained on them only
Compared on the test clips (frame accuracy, clip accuracy = majority label of the clip's frames):
  rules   thresholds fitted on the population          vs  calibrated thresholds
  svm     LandmarkSVM inv+orient +rot on the population vs  the same with the calibration frames added
          (calibration.adapted_svm), and the population SVM with a calibrated confidence threshold
  "passes" = share of test frames that are right AND reach the confidence threshold of the trigger gate
           (4.9), i.e. what is left for the gate; the threshold used is printed next to it
  still frames: share of the test-clip frames that the stability check calls calm (default vs calibrated
           v_max, pose_max)
  swipe thresholds: share of the user's other swipe clips whose peak palm speed exceeds tau_on, and share
           of the static test clips whose fastest frame exceeds it (a swipe onset where there is none)

Pilot mode: when no other user has data, the population is the SAME user (two halves of the
non-calibration clips by clip number, each half once the population and once the test). Calibration then
has nothing to fix: the numbers check the mechanics, they are not an E3 result.
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import argparse
import warnings

import numpy as np

import calibration as cal
import dataset_loader as dl
import hand_features as hf
import landmark_smoothing as ls
import rule_classifier as rc
import train_static as ts

SVM_ARGS = dict(features="invariant+orient", augment=True)


def clip_accuracy(truth, pred, groups):
    hits = []
    for g in np.unique(groups):
        m = groups == g
        labels, counts = np.unique(pred[m], return_counts=True)
        hits.append(labels[counts.argmax()] == truth[m][0])
    return float(np.mean(hits))


def clip_motion(clip):
    sm = ls.smooth_clip(clip["landmarks"], clip["times"], clip["width"], clip["height"])
    return dl.motion(sm, clip["times"], clip["width"], clip["height"])


def run_fold(pop, test, segments, copies, user=""):
    """Train on `pop`, calibrate with `segments`, evaluate on `test`. Returns (rows, calibration)."""
    pQ, py, pg = hf.unflatten(pop.X), pop.y, ts.clip_groups(pop)
    tQ, ty, tg = hf.unflatten(test.X), test.y, ts.clip_groups(test)
    rules_pop = rc.RuleClassifier().fit(pQ, py)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # few clips per class: uneven folds are expected
        svm_pop = ts.LandmarkSVM(**SVM_ARGS).fit(pQ, py, groups=pg)
        c = cal.calibrate(segments, user=user, rules=rules_pop, models={"svm": svm_pop})
        svm_cal = cal.adapted_svm(pQ, py, pg, c.Q, c.y, copies=copies, **SVM_ARGS) if len(c.y) else svm_pop
    p = c.profile
    d_rules, d_svm = cal.DEFAULTS["conf_threshold"]["rules"], cal.DEFAULTS["conf_threshold"]["svm"]
    variants = [
        ("rules, population", rules_pop, d_rules),
        ("rules, calibrated", p.rules(), p.confidence_threshold("rules")),
        ("svm, population", svm_pop, d_svm),
        ("svm, calibrated threshold", svm_pop, p.confidence_threshold("svm")),
        ("svm, adapted", svm_cal, d_svm),
    ]
    rows = []
    for name, model, thr in variants:
        pred, conf = model.predict_with_confidence(tQ)
        rows.append((name, ty, pred, conf, tg, thr))
    return rows, c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calib-clip", type=int, default=1, help="clip number used as the calibration (default 1)")
    ap.add_argument("--copies", type=int, default=1, help="copies of the calibration frames in the adapted SVM")
    args = ap.parse_args()

    ds, _ = dl.load_static()
    clips = dl.load_clips()
    users = sorted(set(ds.user.tolist()))
    print(f"Users with static data: {users}")
    for u in users:
        mine = ds.select(ds.user == u)
        n_clips = {g: len(set(mine.clip[mine.y == g].tolist())) for g in sorted(set(mine.y.tolist()))}
        gestures = [g for g, n in n_clips.items() if n >= 2]
        if len(gestures) < 2:
            print(f"\n{u}: skipped (clips per gesture {n_clips}; needs two or more gestures with at least 2 clips)")
            continue
        segments = cal.segments_from_clips(clips, u, args.calib_clip)
        rest = mine.select(np.isin(mine.y, gestures) & (mine.clip != args.calib_clip))
        others = ds.select((ds.user != u) & np.isin(ds.y, gestures))
        missing = sorted(set(gestures) - set(others.y.tolist()))
        if not missing:
            mode = f"population = users {sorted(set(others.user.tolist()))}"
            folds = [(others, rest)]
        else:
            mode = (f"PILOT MODE: the other users have no {missing}, so the population is the same user "
                    f"(MECHANICS CHECK ONLY)")
            even = rest.clip % 2 == 0
            folds = [(rest.select(even), rest.select(~even)), (rest.select(~even), rest.select(even))]
        print(f"\n=== {u}: calibration = clip {args.calib_clip} of {sorted(segments)}; "
              f"test = {len(set(ts.clip_groups(rest).tolist()))} other clips of {gestures}")
        print(f"    {mode}")

        pooled, calibs = {}, []
        for pop, test in folds:
            rows, c = run_fold(pop, test, segments, args.copies, user=u)
            calibs.append(c)
            for name, truth, pred, conf, groups, thr in rows:
                entry = pooled.setdefault(name, [[], [], [], [], []])
                for lst, val in zip(entry, (truth, pred, conf, groups, np.full(len(truth), thr))):
                    lst.append(val)
        print("    " + cal.describe(calibs[0]).replace("\n", "\n    "))
        if len(calibs) > 1:
            print("    (calibration of the first fold; the rule thresholds start from that fold's population)")

        print(f"\n    {'method':<28} {'frames':>8} {'clips':>8} {'passes':>8}  threshold")
        for name, entry in pooled.items():
            truth, pred, conf, groups, thr = (np.concatenate(v) for v in entry)
            right = pred == truth
            print(f"    {name:<28} {np.mean(right) * 100:>7.1f}% {clip_accuracy(truth, pred, groups) * 100:>7.1f}% "
                  f"{np.mean(right & (conf >= thr)) * 100:>7.1f}%  {', '.join(f'{t:.2f}' for t in np.unique(thr))}")
            wrong = ~right
            if wrong.any():
                pairs, counts = np.unique([f"{a}->{b}" for a, b in zip(truth[wrong], pred[wrong])], return_counts=True)
                print(f"    {'':<28} wrong: " + ", ".join(f"{pr} {n}" for pr, n in zip(pairs, counts)))

        p = calibs[0].profile
        test_clips = [c for c in clips if c["user"] == u and c["gesture"] in gestures and c["clip"] != args.calib_clip]
        motion = [clip_motion(c) for c in test_clips]
        for label, vmax, pmax in (("default", cal.DEFAULTS["v_max"], cal.DEFAULTS["pose_max"]),
                                  ("calibrated", p.v_max, p.pose_max)):
            calm = np.concatenate([(s <= vmax) & (q <= pmax) for s, q in motion])
            print(f"    still frames in the test clips, {label:<10} (v_max {vmax:.3f}, pose_max {pmax:.3f}): "
                  f"{np.mean(calm) * 100:.1f}%")
        swipes = [c for c in clips if c["user"] == u and c["gesture"] in cal.SWIPE_STEPS and c["clip"] != args.calib_clip]
        peaks = [np.nanmax(clip_motion(c)[0]) for c in swipes]
        still_max = [np.nanmax(s) for s, _ in motion]
        for label, tau in (("default", cal.DEFAULTS["tau_on"]), ("calibrated", p.tau_on)):
            found = f"{np.mean(np.array(peaks) > tau) * 100:.0f}% of {len(peaks)}" if peaks else "no other swipe clips"
            print(f"    tau_on {label:<10} {tau:.2f}: swipe clips reaching it {found}; "
                  f"static clips reaching it {np.mean(np.array(still_max) > tau) * 100:.0f}% of {len(still_max)}")


if __name__ == "__main__":
    main()
