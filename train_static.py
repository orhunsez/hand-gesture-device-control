"""Train and evaluate a classifier for the static gestures (report section 4.6).

    .venv\\Scripts\\python.exe train_static.py

The model is an SVM with an RBF kernel on the 63 normalised landmark numbers, after
standardising every number (zero mean, unit spread, learned on the training rows only).

How a result is measured (this matters more than the model):
  - The unit of evaluation is the CLIP or the USER, never the frame. Frames of one clip are
    nearly identical; if some of them were in the training part and others in the test part,
    the test would only check memory and the accuracy would look far too good.
  - Within one person: StratifiedGroupKFold where all frames of a clip stay together.
  - Across people: train on some users and test on others (leave_one_user_out), which is the
    question of experiment E5. This needs several users and cannot be run on the pilot data.

With the pilot data (one person with four gestures) the numbers only show that the chain from
recording to prediction works. They say nothing about new users.
"""
import warnings

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import dataset_loader as dl


def make_model(C=1.0, gamma="scale"):
    """Standardise, then an RBF SVM."""
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma))


def clip_groups(ds):
    """One group id per clip (file and clip number), so a clip is never split."""
    return np.array([f"{f}#{c}" for f, c in zip(ds.file, ds.clip)])


def evaluate_clip_cv(ds, n_splits=5, seed=0, C=1.0, gamma="scale"):
    """Cross-validation over clips of the given data. Returns (accuracy, true labels, predicted)."""
    groups = clip_groups(ds)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    truth, pred = [], []
    for tr, te in cv.split(ds.X, ds.y, groups):
        model = make_model(C, gamma).fit(ds.X[tr], ds.y[tr])
        truth.extend(ds.y[te])
        pred.extend(model.predict(ds.X[te]))
    truth, pred = np.array(truth), np.array(pred)
    return float(np.mean(truth == pred)), truth, pred


def evaluate_leave_one_user_out(ds, C=1.0, gamma="scale"):
    """Train on all users but one, test on that one (experiment E5).

    Returns a list of (user, accuracy, number of test rows). A user whose gestures are not
    all present in the training users is reported with a warning, because the model cannot
    predict a class it never saw.
    """
    results = []
    for user, train, test in dl.leave_one_user_out(ds):
        unseen = set(test.y.tolist()) - set(train.y.tolist())
        if unseen:
            warnings.warn(f"user {user}: gesture(s) {sorted(unseen)} are in no other user's data")
        if len(set(train.y.tolist())) < 2:
            continue
        model = make_model(C, gamma).fit(train.X, train.y)
        results.append((user, float(np.mean(model.predict(test.X) == test.y)), len(test)))
    return results


def print_confusion(truth, pred):
    labels = sorted(set(truth.tolist()) | set(pred.tolist()))
    cm = confusion_matrix(truth, pred, labels=labels)
    header = "true / predicted"
    print(f"{header:<18}" + "".join(f"{l:>12}" for l in labels) + "   recall")
    for i, l in enumerate(labels):
        total = cm[i].sum()
        rec = cm[i, i] / total if total else float("nan")
        print(f"{l:<18}" + "".join(f"{n:>12}" for n in cm[i]) + f"   {rec * 100:5.1f}%")


def main():
    ds_all, _ = dl.load_static()
    users = sorted(set(ds_all.user.tolist()))
    print(f"Data: {len(ds_all)} stable frames from users {users}")

    # One person, four gestures: the only complete set at the moment.
    pilot = ds_all.select(ds_all.user == "u99")
    classes = sorted(set(pilot.y.tolist()))
    clips = len(set(clip_groups(pilot).tolist()))
    print(f"\n1) Clip-level cross-validation inside user u99: {len(pilot)} frames, {clips} clips, classes {classes}")
    acc, truth, pred = evaluate_clip_cv(pilot, n_splits=5)
    print(f"   accuracy {acc * 100:.1f}% (frames of a clip never in both training and test)")
    print_confusion(truth, pred)

    print("\n2) Same test with other feature choices (still one person, exploratory):")
    print(f"   {'features':<34} {'accuracy':>9}")
    for smooth in (True, False):
        for keep_z in (True, False):
            d, _ = dl.load_static(users=["u99"], smooth=smooth, keep_z=keep_z)
            a, _, _ = evaluate_clip_cv(d, n_splits=5)
            name = f"{'smoothed' if smooth else 'raw'}, {'x y z' if keep_z else 'x y only'}"
            print(f"   {name:<34} {a * 100:>8.1f}%")

    # A small look across user ids: train on u99, see how u01's open palm frames are labelled.
    # (Both ids are the same person's hand, so this is a test across sessions.)
    other = [u for u in users if u != "u99"]
    if other:
        print(f"\n3) Train on all u99 frames, then label the frames of {other} (only gestures they have):")
        model = make_model().fit(pilot.X, pilot.y)
        for u in other:
            sub = ds_all.select(ds_all.user == u)
            p = model.predict(sub.X)
            for g in sorted(set(sub.y.tolist())):
                m = sub.y == g
                labels, counts = np.unique(p[m], return_counts=True)
                share = ", ".join(f"{l} {c}" for l, c in zip(labels, counts))
                print(f"   {u} {g}: {m.sum()} frames -> {share}  (correct {np.mean(p[m] == g) * 100:.0f}%)")
        print("   This is recall for one gesture of another user id; check who recorded it before calling it a test\n"
              "   across people (u01 and u99 are the same person's hand). It says nothing about wrong detections.")
    else:
        print("\n3) No second user yet, so the cross-user test cannot be run.")


if __name__ == "__main__":
    main()
