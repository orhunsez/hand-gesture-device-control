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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import dataset_loader as dl
import hand_features as hf


def make_model(C=1.0, gamma="scale"):
    """Standardise, then an RBF SVM."""
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma))


class LandmarkSVM:
    """SVM on a chosen feature set of normalised landmarks, with a confidence for every prediction.

    Same interface as rule_classifier.RuleClassifier, so experiment E1 and the live system can use either:
        fit(Q, y, groups=None), predict(Q), predict_with_confidence(Q)    Q: normalised landmarks (N, 21, 3)

    features : a name from hand_features.FEATURE_SETS ("raw63", "invariant", "invariant+orient").
    augment  : add `copies` randomly turned versions (up to +-max_deg) of the training frames of the
               gestures whose meaning does not depend on orientation (hand_features.augment_rotation).
    Confidence = the largest class probability. The probabilities come from sigmoid (Platt) calibration
    of the SVM scores on held-out parts of the training data (CalibratedClassifierCV, one SVM trained on
    all data, ensemble=False). Pass groups (one clip id per frame) so that the held-out parts are whole
    clips: frames of one clip are nearly identical, and splitting by frame makes the probabilities far
    too confident. The label is the class with the largest probability, so label and confidence agree.
    """

    def __init__(self, features="invariant+orient", augment=False, copies=3, max_deg=90.0,
                 C=1.0, gamma="scale", use_z=True, calib_splits=3, seed=0):
        self.features, self.augment, self.copies, self.max_deg = features, augment, copies, max_deg
        self.C, self.gamma, self.use_z = C, gamma, use_z
        self.calib_splits, self.seed = calib_splits, seed

    def _X(self, Q):
        return hf.compute(Q, self.features, self.use_z)

    def fit(self, Q, y, groups=None):
        Q, y = np.asarray(Q, dtype=np.float64), np.asarray(y)
        if groups is not None:
            groups = np.asarray(groups)
        if self.augment:
            Q, src = hf.augment_rotation(Q, y, self.copies, self.max_deg, seed=self.seed)
            y = y[src]
            if groups is not None:
                groups = groups[src]      # a turned copy stays in the clip it came from
        if groups is not None:
            k = min(self.calib_splits, len(set(groups.tolist())))
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)   # few clips per class: uneven folds are expected
                cv = list(splitter.split(Q, y, groups))
        else:
            cv = StratifiedKFold(n_splits=self.calib_splits, shuffle=True, random_state=self.seed)
        self.model = CalibratedClassifierCV(make_model(self.C, self.gamma), method="sigmoid",
                                            cv=cv, ensemble=False).fit(self._X(Q), y)
        self.classes_ = self.model.classes_
        return self

    def predict_with_confidence(self, Q):
        Q = np.asarray(Q, dtype=np.float64)
        single = Q.ndim == 2
        P = self.model.predict_proba(self._X(Q[None] if single else Q))
        labels, conf = self.classes_[P.argmax(axis=1)], P.max(axis=1)
        return (labels[0], float(conf[0])) if single else (labels, conf)

    def predict(self, Q):
        return self.predict_with_confidence(Q)[0]


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
