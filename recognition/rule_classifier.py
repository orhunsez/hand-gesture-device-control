"""Rule-based static gesture classifier: the baseline of experiment E1 (report sections 4.6, 5.3).

No machine learning: a gesture is a pattern of fingers that are extended or bent, plus the direction
of the thumb for thumb up and thumb down.

    gesture      thumb      index  middle  ring  pinky   thumb direction
    open_palm    extended   ext    ext     ext   ext     -
    fist         bent       bent   bent    bent  bent    -
    thumb_up     extended   bent   bent    bent  bent    up
    thumb_down   extended   bent   bent    bent  bent    down
    point        (any)      ext    bent    bent  bent    -
    anything else -> no_gesture

How "extended" is measured (hand_features.py, normalised landmarks):
  - index to pinky: the sum of the bend angles at the three joints of the finger (radians). About
    0.3 when straight and about 3 to 4 in a fist on the pilot data, so the rule is not delicate.
  - thumb: the distance from the thumb tip to the base of the index finger (hand sizes). The bend
    angles of the thumb separate less well, because the thumb folds sideways over the fingers.
  - thumb direction: the vertical part of the unit vector thumb base -> thumb tip in the picture;
    "up" needs the thumb within about 60 degrees of vertical.
All of these are unchanged when the hand is turned in the picture plane, except the thumb direction,
which is meant to change.

Soft rules and confidence: every yes/no test is a smooth step (logistic function) instead of a hard
threshold, so each finger gets a value between 0 (clearly bent) and 1 (clearly extended). A gesture's
score is the product of the values its pattern asks for; the best gesture wins if its score is at
least MIN_SCORE, otherwise the answer is no_gesture. The score is the confidence used by the
false-trigger stage (4.9). It is a heuristic, not a calibrated probability.

Thresholds: the defaults below were read off the pilot data of one person. fit() sets them from
training data instead: for each finger it takes the gap between the frames where the finger should be
extended and where it should be bent (5th / 95th percentiles) and puts the threshold in the middle.
Fit on training users only.
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import numpy as np

import hand_features as hf

FINGERS = hf.FINGERS                      # thumb, index, middle, ring, pinky
# +1: a larger measure means more extended (thumb distance); -1: larger means more bent (bend angles).
SIGN = np.array([+1, -1, -1, -1, -1], dtype=np.float64)
DEFAULT_THR = np.array([0.55, 1.6, 1.6, 1.6, 1.6])       # pilot data, one person
DEFAULT_SCALE = np.array([0.03, 0.4, 0.4, 0.4, 0.4])
THUMB_VERTICAL = 0.5      # |vertical part| of the thumb direction for "up"/"down" (cos 60 deg)
THUMB_VERTICAL_SCALE = 0.1
MIN_SCORE = 0.5

# 1 = extended, 0 = bent, None = does not matter. Second value: required thumb direction.
TEMPLATES = {
    "open_palm": ((1, 1, 1, 1, 1), None),
    "fist": ((0, 0, 0, 0, 0), None),
    "thumb_up": ((1, 0, 0, 0, 0), "up"),
    "thumb_down": ((1, 0, 0, 0, 0), "down"),
    "point": ((None, 1, 0, 0, 0), None),
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def finger_measures(Q, use_z=True):
    """(N, 5): bend sums for index..pinky and the thumb tip to index base distance (see module text)."""
    Q = np.asarray(Q, dtype=np.float64)
    bends = hf.joint_bends(Q, use_z).reshape(*Q.shape[:-2], 5, 3).sum(axis=-1)
    thumb = hf.distances(Q, use_z)[..., hf.DIST_PAIRS.index((hf.THUMB_TIP, hf.INDEX_MCP))]
    out = bends.copy()
    out[..., 0] = thumb
    return out


class RuleClassifier:
    """Same interface as train_static.LandmarkSVM: fit(Q, y), predict(Q), predict_with_confidence(Q).

    Q: normalised landmarks (N, 21, 3).
    """

    def __init__(self, min_score=MIN_SCORE, use_z=True):
        self.min_score = min_score
        self.use_z = use_z
        self.thr = DEFAULT_THR.copy()
        self.scale = DEFAULT_SCALE.copy()
        self.gesture_names = list(TEMPLATES)
        self.classes_ = np.array(sorted(self.gesture_names + ["no_gesture"]), dtype=object)

    # ---------------------------------------------------------------- thresholds
    def fit(self, Q, y, groups=None, low=5, high=95):
        """Set one threshold per finger from labelled frames. Labels without a template are ignored.

        A finger keeps its default if the training data lacks either state for it.
        groups is accepted (and not needed) so that the call matches LandmarkSVM.fit.
        """
        M = finger_measures(Q, self.use_z)
        y = np.asarray(y)
        for k in range(5):
            want = np.array([TEMPLATES[g][0][k] if g in TEMPLATES else None for g in y], dtype=object)
            ext = M[want == 1, k]
            bent = M[want == 0, k]
            ext, bent = ext[~np.isnan(ext)], bent[~np.isnan(bent)]
            if len(ext) < 5 or len(bent) < 5:
                continue
            # Edge of each group that faces the other group.
            if SIGN[k] > 0:          # extended = large values
                e_edge, b_edge = np.percentile(ext, low), np.percentile(bent, high)
            else:                    # extended = small values
                e_edge, b_edge = np.percentile(ext, high), np.percentile(bent, low)
            if SIGN[k] * (e_edge - b_edge) > 0:      # the groups are separated: middle of the gap
                self.thr[k] = (e_edge + b_edge) / 2
                self.scale[k] = max(abs(e_edge - b_edge) / 6, 1e-3)
            else:                                    # they overlap: middle of the medians
                me, mb = np.median(ext), np.median(bent)
                self.thr[k] = (me + mb) / 2
                self.scale[k] = max(abs(me - mb) / 8, 1e-3)
        return self

    def describe(self):
        rows = [f"{f}: extended if {'>' if SIGN[k] > 0 else '<'} {self.thr[k]:.3f} (step width {self.scale[k]:.3f})"
                for k, f in enumerate(FINGERS)]
        rows.insert(0, "thumb measure = tip to index base (hand sizes); others = sum of 3 joint bends (rad)")
        return "\n".join(rows)

    # ---------------------------------------------------------------- prediction
    def extended_prob(self, Q):
        """(N, 5) soft 'finger is extended' values in 0..1."""
        M = finger_measures(Q, self.use_z)
        return _sigmoid(SIGN * (M - self.thr) / self.scale)

    def gesture_scores(self, Q):
        """(N, number of templates) scores in 0..1, columns in the order of self.gesture_names."""
        Q = np.asarray(Q, dtype=np.float64)
        E = self.extended_prob(Q)
        dir_y = hf.thumb_direction(Q)[..., 1]
        up = _sigmoid((-dir_y - THUMB_VERTICAL) / THUMB_VERTICAL_SCALE)
        down = _sigmoid((dir_y - THUMB_VERTICAL) / THUMB_VERTICAL_SCALE)
        cols = []
        for g in self.gesture_names:
            pattern, direction = TEMPLATES[g]
            s = np.ones(E.shape[:-1])
            for k, want in enumerate(pattern):
                if want == 1:
                    s = s * E[..., k]
                elif want == 0:
                    s = s * (1 - E[..., k])
            if direction == "up":
                s = s * up
            elif direction == "down":
                s = s * down
            cols.append(s)
        return np.stack(cols, axis=-1)

    def predict_with_confidence(self, Q):
        """Labels and confidences, both shape (N,). A frame without a hand is no_gesture with confidence 0."""
        Q = np.asarray(Q, dtype=np.float64)
        single = Q.ndim == 2
        if single:
            Q = Q[None]
        S = self.gesture_scores(Q)
        missing = np.isnan(S).any(axis=1)
        S = np.nan_to_num(S, nan=0.0)
        best = S.argmax(axis=1)
        conf = S.max(axis=1)
        labels = np.array(self.gesture_names, dtype=object)[best]
        none = conf < self.min_score
        labels[none] = "no_gesture"
        conf = np.where(none, 1 - conf, conf)
        conf[missing] = 0.0
        if single:
            return labels[0], float(conf[0])
        return labels, conf

    def predict(self, Q):
        return self.predict_with_confidence(Q)[0]
