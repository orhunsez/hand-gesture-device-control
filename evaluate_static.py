"""Experiment E1 (report section 5.3): rule-based baseline versus learned static classifiers.

    .venv\\Scripts\\python.exe evaluate_static.py

Methods (all take normalised landmarks; see rule_classifier.py, hand_features.py, train_static.py):
    rules                     finger extended / bent + thumb direction, thresholds fitted on training data
    svm raw63                 the first classifier: SVM on the 63 normalised coordinates
    svm raw63 +rot            the same, trained with randomly turned copies of fist / open palm / point
    svm invariant             SVM on joint angles and distances only (no orientation at all)
    svm inv+orient            invariant + vertical thumb direction
    svm inv+orient +rot       the same, with turned training copies

Tests:
  1. Clip-level cross-validation inside one user (a clip is never split between training and test).
  2. Rotation stress test, inside the same folds: the test frames of the gestures whose meaning does not
     depend on orientation are turned in the picture plane by a fixed angle; how much accuracy is left?
     Thumb up / down are left out of this test, because turning them changes their meaning.
  3. Other recording session: train on all clips of the training user, label every clip of the other
     user ids, one line per clip with its tilt and the finger pattern the rules see.
  4. Confidence: how many wrong answers a confidence threshold would hold back (input to 4.9).

What is measured is always per frame AND per clip (clip = majority label of its frames). With the pilot
data (one person) these numbers check the chain and show the rotation problem; they are not E1 results.
Choose methods and thresholds on training / validation users; use test users once, at the end.
"""
import argparse
import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import dataset_loader as dl
import hand_features as hf
import rule_classifier as rc
import train_static as ts

METHODS = {
    "rules": lambda: rc.RuleClassifier(),
    "svm raw63": lambda: ts.LandmarkSVM(features="raw63"),
    "svm raw63 +rot": lambda: ts.LandmarkSVM(features="raw63", augment=True),
    "svm invariant": lambda: ts.LandmarkSVM(features="invariant"),
    "svm inv+orient": lambda: ts.LandmarkSVM(features="invariant+orient"),
    "svm inv+orient +rot": lambda: ts.LandmarkSVM(features="invariant+orient", augment=True),
}
ANGLES = [-90, -60, -30, 0, 30, 60, 90]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def clip_majority_accuracy(truth, pred, groups):
    """Share of clips whose most frequent predicted label is the true label."""
    hits = []
    for g in np.unique(groups):
        m = groups == g
        labels, counts = np.unique(pred[m], return_counts=True)
        hits.append(labels[counts.argmax()] == truth[m][0])
    return float(np.mean(hits))


def finger_pattern(Q, rules):
    """Letters of the fingers the (fitted) rules see as extended in most frames, e.g. 'T----'."""
    E = np.nanmedian(rules.extended_prob(Q), axis=0) > 0.5
    return "".join(c if e else "-" for c, e in zip("TIMRP", E))


def cross_validation(ds, n_splits, seed):
    """Tests 1 and 2. Returns {method: {"angle": (truth, pred, conf, groups) per angle}}."""
    Q, y, groups = hf.unflatten(ds.X), ds.y, ts.clip_groups(ds)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {m: {a: ([], [], [], []) for a in ANGLES} for m in METHODS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        folds = list(cv.split(Q, y, groups))
    for tr, te in folds:
        for name, make in METHODS.items():
            model = make().fit(Q[tr], y[tr], groups=groups[tr])
            for a in ANGLES:
                idx = te if a == 0 else te[np.isin(y[te], list(hf.ORIENTATION_FREE))]
                Qt = Q[idx] if a == 0 else hf.rotate(Q[idx], np.radians(a))
                pred, conf = model.predict_with_confidence(Qt)
                for lst, val in zip(out[name][a], (y[idx], pred, conf, groups[idx])):
                    lst.append(val)
    return {m: {a: tuple(np.concatenate(v) for v in vals) for a, vals in per.items()} for m, per in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-user", default="u99", help="user whose clips are used for training (default u99)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds_all, _ = dl.load_static()
    train = ds_all.select(ds_all.user == args.train_user)
    others = sorted(set(ds_all.user.tolist()) - {args.train_user})
    classes = sorted(set(train.y.tolist()))
    print(f"Training user {args.train_user}: {len(train)} stable frames, "
          f"{len(set(ts.clip_groups(train).tolist()))} clips, classes {classes}")
    print(f"Other user ids: {others or 'none'}")
    print("Note: the rules can also answer point and no_gesture, which are not in this data; such answers count as wrong.")

    # ------------------------------------------------------------------ tests 1 and 2
    res = cross_validation(train, args.folds, args.seed)
    print(f"\n1) Clip-level cross-validation inside {args.train_user} ({args.folds} folds, seed {args.seed})")
    print(f"   {'method':<22} {'frames':>8} {'clips':>8}   (wrong = true->predicted, frames)")
    for name in METHODS:
        truth, pred, _, groups = res[name][0]
        w = pred != truth
        pairs, counts = np.unique([f"{a}->{b}" for a, b in zip(truth[w], pred[w])], return_counts=True)
        errors = ", ".join(f"{pr} {c}" for pr, c in zip(pairs, counts)) or "-"
        print(f"   {name:<22} {np.mean(pred == truth) * 100:>7.1f}% {clip_majority_accuracy(truth, pred, groups) * 100:>7.1f}%"
              f"   wrong: {errors}")

    free = sorted(set(classes) & hf.ORIENTATION_FREE)
    print(f"\n2) Rotation stress test: test frames of {free} turned by a fixed angle (frame accuracy)")
    print(f"   {'method':<22}" + "".join(f"{a:>+5d} deg" for a in ANGLES))
    for name in METHODS:
        print(f"   {name:<22}" + "".join(f"{np.mean(res[name][a][1] == res[name][a][0]) * 100:>8.1f}%" for a in ANGLES))
    print("   (the 0 deg column contains all test frames, the others only the turned gestures)")
    print("   Most frequent wrong answer per method over the turned frames:")
    for name in METHODS:
        wrong = np.concatenate([res[name][a][1][res[name][a][1] != res[name][a][0]] for a in ANGLES if a != 0])
        if len(wrong):
            labels, counts = np.unique(wrong, return_counts=True)
            top = ", ".join(f"{l} {c}" for l, c in sorted(zip(labels, counts), key=lambda t: -t[1])[:3])
            print(f"   {name:<22} {len(wrong)} wrong: {top}")
        else:
            print(f"   {name:<22} 0 wrong")

    # ------------------------------------------------------------------ test 3
    Qtr, ytr, gtr = hf.unflatten(train.X), train.y, ts.clip_groups(train)
    models = {name: make().fit(Qtr, ytr, groups=gtr) for name, make in METHODS.items()}
    print(f"\nFitted rule thresholds (all of {args.train_user}):\n   " + models["rules"].describe().replace("\n", "\n   "))
    session = {}
    if others:
        print(f"\n3) Trained on all of {args.train_user}; majority label (share of frames) for every clip of {others}")
        print("   fingers = what the rules see as extended (T I M R P); tilt = median, 0 = upright")
        short = {"rules": "rules", "svm raw63": "raw63", "svm raw63 +rot": "raw63+rot", "svm invariant": "inv",
                 "svm inv+orient": "inv+or", "svm inv+orient +rot": "inv+or+rot"}
        print(f"   {'clip':<22} {'tilt':>5} {'fingers':>8}  " + "".join(f"{short[n]:>19}" for n in METHODS))
        for u in others:
            sub = ds_all.select(ds_all.user == u)
            Qs = hf.unflatten(sub.X)
            preds = {name: m.predict_with_confidence(Qs) for name, m in models.items()}
            session[u] = (sub, preds)
            for g in sorted(set(sub.y.tolist())):
                for c in sorted(set(sub.clip[sub.y == g].tolist())):
                    m = (sub.y == g) & (sub.clip == c)
                    cells = []
                    for name in METHODS:
                        labels, counts = np.unique(preds[name][0][m], return_counts=True)
                        k = counts.argmax()
                        cells.append(f"{labels[k]} {counts[k] / m.sum() * 100:.0f}%")
                    head = f"{u} {g} #{c} ({m.sum()})"
                    print(f"   {head:<22} {np.median(hf.tilt_deg(Qs[m])):>5.0f} {finger_pattern(Qs[m], models['rules']):>8}  "
                          + "".join(f"{cell:>19}" for cell in cells))
            print(f"   frame accuracy against the RECORDED labels of {u}: " + ", ".join(
                f"{short[n]} {np.mean(preds[n][0] == sub.y) * 100:.0f}%" for n in METHODS))

    # ------------------------------------------------------------------ test 4
    print("\n4) Confidence threshold: of all answers in test 2 (all angles pooled), how many are kept, and how")
    print("   many of the kept ones are right. A good confidence keeps most right answers and few wrong ones.")
    print(f"   {'method':<22} {'threshold':>9} {'kept':>7} {'right among kept':>17} {'wrong held back':>16}")
    for name in METHODS:
        truth = np.concatenate([res[name][a][0] for a in ANGLES])
        pred = np.concatenate([res[name][a][1] for a in ANGLES])
        conf = np.concatenate([res[name][a][2] for a in ANGLES])
        wrong = pred != truth
        for t in THRESHOLDS:
            keep = conf >= t
            right = f"{np.mean(~wrong[keep]) * 100:.1f}%" if keep.any() else "-"
            held = f"{np.mean(~keep[wrong]) * 100:.1f}%" if wrong.any() else "- (none wrong)"
            print(f"   {name if t == THRESHOLDS[0] else '':<22} {t:>9.1f} {np.mean(keep) * 100:>6.1f}% {right:>17} {held:>16}")


if __name__ == "__main__":
    main()
