"""Frame-by-frame static branch: raw landmarks -> what the trigger gate needs (report 4.1, 4.5, 4.9).

    analyzer = FrameAnalyzer(classifier, profile)        # profile: calibration.UserProfile or None
    r = analyzer.process(landmarks, t, width, height)    # one camera frame (live)
    gate.update(r["t"], r["label"], r["conf"], r["calm"], r["moving"])

For every frame: 1-euro smoothing of the raw landmarks (landmark_smoothing.py), normalisation
(landmark_features.py), the static classifier (rules or LandmarkSVM, anything with
predict_with_confidence), and two motion values of the palm centre:
    speed  palm-centre speed in hand sizes per second; the hand size is the median of the last
           SIZE_WINDOW frames, because the hand size of a single frame is noisy
    pose   mean change of the normalised landmarks since the previous frame (hand sizes)
    calm   hand present, speed <= v_max and pose <= pose_max (the stability check of the static branch)
    moving speed > tau_on (the swipe onset: the speed gate of A.12 rule 1)
The thresholds come from the calibration profile, or from calibration.DEFAULTS without one.
For the swipe segmentation (swipe_segmentation.py) every result also carries
    centre   palm centre (mean of landmarks 0, 5, 9, 13, 17) in pixels, (x, y)
    size     the hand size in pixels that the speed is divided by (median of the last SIZE_WINDOW frames)
    smoothed the smoothed landmarks (21, 3) in the stored units
all NaN in a frame without a hand.

Only past frames are used (the live system cannot look ahead), unlike the stable-frame selection of
dataset_loader.py, which looks at a window around each recorded frame. The time a gesture has to be
held is counted by the trigger gate, not here.

process_clip() runs a recorded clip and gives exactly the same numbers as feeding its frames one by one
to process(); it only classifies all frames in one call, which is faster.
A frame without a hand: label "no_gesture", confidence 0, not calm, not moving; the smoothing and the
motion values start again when the hand comes back.

The live camera loop (live demo, live calibration) is not written yet: postponed with the live system.
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
from collections import deque

import numpy as np

import calibration as cal
import landmark_features as lf
import landmark_smoothing as ls

PALM = [0, 5, 9, 13, 17]
SIZE_WINDOW = 10


class _Motion:
    """Speed and pose change of the current frame from the previous one (past frames only)."""

    def __init__(self, size_window=SIZE_WINDOW):
        self.sizes = deque(maxlen=size_window)
        self.reset()

    def reset(self):
        self.sizes.clear()
        self.prev = None          # (t, palm centre in px, normalised landmarks)
        self.centre = np.full(2, np.nan)   # palm centre (px) and hand size (px, median of the last
        self.size = np.nan                 # frames) of the latest frame; NaN without a hand

    def update(self, smoothed, t, width=None, height=None):
        if np.isnan(smoothed).any():
            self.reset()
            return np.nan, np.nan
        px = lf.to_pixel_units(smoothed, width, height)
        self.sizes.append(float(lf.hand_size(smoothed, width, height)))
        centre = px[PALM, :2].mean(axis=0)
        size = float(np.median(self.sizes))
        self.centre, self.size = centre, size
        q = lf.normalize(smoothed, width, height)
        speed = pose = np.nan
        if self.prev is not None and t > self.prev[0]:
            speed = float(np.linalg.norm(centre - self.prev[1]) / (t - self.prev[0]) / size)
            pose = float(np.mean(np.linalg.norm(q - self.prev[2], axis=1)))
        self.prev = (t, centre, q)
        return speed, pose


class FrameAnalyzer:
    def __init__(self, classifier, profile=None, size_window=SIZE_WINDOW):
        self.classifier = classifier
        self.profile = profile or cal.UserProfile()
        self.size_window = size_window
        self.reset()

    def reset(self):
        self.smoother = ls.LandmarkSmoother()
        self.motion = _Motion(self.size_window)

    def _flags(self, present, speed, pose):
        p = self.profile
        with np.errstate(invalid="ignore"):
            calm = present & (speed <= p.v_max) & (pose <= p.pose_max)
            moving = present & (speed > p.tau_on)
        return calm, moving

    def process(self, landmarks, t, width=None, height=None):
        """One frame. landmarks: (21, 3) raw stored values, or None / NaN when no hand was found."""
        lm = np.full((21, 3), np.nan) if landmarks is None else np.asarray(landmarks, dtype=np.float64)
        sm = self.smoother.update(lm, t, width, height)
        speed, pose = self.motion.update(sm, t, width, height)
        present = not np.isnan(sm).any()
        if present:
            label, conf = self.classifier.predict_with_confidence(lf.normalize(sm, width, height))
        else:
            label, conf = "no_gesture", 0.0
        calm, moving = self._flags(present, speed, pose)
        return {"t": t, "label": label, "conf": float(conf), "present": present, "speed": speed, "pose": pose,
                "calm": bool(calm), "moving": bool(moving),
                "centre": self.motion.centre.copy(), "size": self.motion.size, "smoothed": sm}

    def process_clip(self, landmarks, times, width=None, height=None):
        """A recorded clip, (T, 21, 3) raw stored values. Returns a dict of arrays, one value per frame.
        Starts from a fresh state (reset)."""
        self.reset()
        lm = np.asarray(landmarks, dtype=np.float64)
        T = len(lm)
        sm = np.stack([self.smoother.update(lm[i], times[i], width, height) for i in range(T)])
        speed, pose = np.full(T, np.nan), np.full(T, np.nan)
        centre, size = np.full((T, 2), np.nan), np.full(T, np.nan)
        for i in range(T):
            speed[i], pose[i] = self.motion.update(sm[i], times[i], width, height)
            centre[i], size[i] = self.motion.centre, self.motion.size
        present = ~np.isnan(sm).any(axis=(1, 2))
        labels = np.full(T, "no_gesture", dtype=object)
        conf = np.zeros(T)
        if present.any():
            lab, c = self.classifier.predict_with_confidence(lf.normalize(sm[present], width, height))
            labels[present], conf[present] = lab, c
        calm, moving = self._flags(present, speed, pose)
        return {"t": np.asarray(times, dtype=np.float64), "label": labels, "conf": conf, "present": present,
                "speed": speed, "pose": pose, "calm": calm, "moving": moving,
                "centre": centre, "size": size, "smoothed": sm}
